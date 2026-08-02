#!/usr/bin/env python3
"""Low-impact dual-camera MJPEG dashboard with asynchronous OCR annotations."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file


PROJECT_DIR = Path(__file__).resolve().parent
SHARED_DIR = Path(os.environ.get("THU_VR_WEB_SHARED_DIR", "/dev/shm/thu_vr_camera_web"))
UPSTREAM_URL = os.environ.get("THU_VR_UPSTREAM_URL", "http://127.0.0.1:5000").rstrip("/")
OCR_SUBMIT_INTERVAL = float(os.environ.get("THU_VR_OCR_SUBMIT_INTERVAL", "10.0"))
RAW_MAX_FPS = float(os.environ.get("THU_VR_RAW_MAX_FPS", "2.0"))
EXPERIMENT_DIR = Path(os.environ.get(
    "THU_VR_EXPERIMENT_DIR",
    str(PROJECT_DIR / "ablation_data"),
))
ARM_STATUS_URL = os.environ.get("THU_VR_ARM_STATUS_URL", "http://127.0.0.1:5080/api/status")
ROW_PREFIX_RE = re.compile(r"^[A-Z][0-9]{2}$")
LABEL_SEQUENCE_START = int(os.environ.get("THU_VR_LABEL_SEQUENCE_START", "10"))
LABEL_SEQUENCE_END = int(os.environ.get("THU_VR_LABEL_SEQUENCE_END", "20"))
if not (0 <= LABEL_SEQUENCE_START <= LABEL_SEQUENCE_END <= 9999):
    raise RuntimeError("invalid THU_VR label sequence")


def labels_for_row(row: str) -> list[str]:
    return [
        f"{row}-{number:04d}"
        for number in range(LABEL_SEQUENCE_START, LABEL_SEQUENCE_END + 1)
    ]
EXPERIMENT_CAPTURE_LOCK = threading.Lock()
ANALYSIS_QUEUE: queue.Queue[Path] = queue.Queue()
ANALYSIS_QUEUED: set[str] = set()
ANALYSIS_QUEUE_LOCK = threading.Lock()

CAMERAS = {
    "head": {
        "label": "头部相机",
        "topic": os.environ.get("THU_VR_HEAD_TOPIC", "/head_rgbd/color/image_raw"),
    },
    "base": {
        "label": "底部相机",
        "topic": os.environ.get("THU_VR_BASE_TOPIC", "/base_rgbd/color/image_raw"),
    },
}


@dataclass
class CameraState:
    key: str
    label: str
    topic: str
    condition: threading.Condition = field(default_factory=threading.Condition)
    raw_jpeg: bytes | None = None
    raw_sequence: int = 0
    received_at: float = 0.0
    width: int = 0
    height: int = 0
    frame_times: deque = field(default_factory=lambda: deque(maxlen=30))
    annotated_jpeg: bytes | None = None
    annotation_sequence: int = 0
    annotation_status: dict = field(default_factory=dict)
    last_submit_monotonic: float = 0.0
    stream_viewers: int = 0
    last_snapshot_request_monotonic: float = 0.0
    activity_event: threading.Event = field(default_factory=threading.Event)

    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        duration = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / duration if duration > 0 else 0.0


STATES = {
    key: CameraState(key=key, label=value["label"], topic=value["topic"])
    for key, value in CAMERAS.items()
}


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.jpg")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def placeholder(text: str) -> bytes:
    image = np.full((720, 1280, 3), 24, dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (80, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (170, 180, 190),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return encoded.tobytes() if ok else b""


PLACEHOLDERS = {
    "raw": placeholder("Waiting for camera frame..."),
    "annotated": placeholder("Waiting for OCR annotation..."),
}


class UpstreamCameraPoller:
    """Read cached JPEGs from the existing camera web service.

    This deliberately avoids creating another ROS image subscription.  The
    existing service has already received and decoded the camera frames, so a
    low-rate localhost snapshot read is the least invasive source for this
    dashboard.
    """

    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for key in STATES:
            thread = threading.Thread(
                target=self._poll_camera,
                args=(key,),
                name=f"upstream-{key}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)
        print(
            f"[SOURCE] 低频读取现有相机网页 {UPSTREAM_URL}，"
            f"每路最多 {RAW_MAX_FPS:g} FPS",
            flush=True,
        )

    def join(self) -> None:
        for thread in self.threads:
            thread.join(timeout=3)

    def _poll_camera(self, key: str) -> None:
        state = STATES[key]
        last_error_log = 0.0
        url = f"{UPSTREAM_URL}/snapshot/{key}"
        while not self.stop_event.is_set():
            started = time.monotonic()
            with state.condition:
                has_viewer = (
                    state.stream_viewers > 0
                    or started - state.last_snapshot_request_monotonic < 3.0
                )
            target_fps = RAW_MAX_FPS if has_viewer else 1.0 / OCR_SUBMIT_INTERVAL
            period = 1.0 / max(target_fps, 0.1)
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "thu-vr-ocr-sidecar/1.0"},
                )
                with urllib.request.urlopen(request, timeout=3.0) as response:
                    payload = response.read()
                if not payload.startswith(b"\xff\xd8"):
                    raise RuntimeError("上游未返回 JPEG 图像")

                now = time.time()
                monotonic_now = time.monotonic()
                width = state.width
                height = state.height
                if not width:
                    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise RuntimeError("JPEG 解码失败")
                    height, width = frame.shape[:2]

                with state.condition:
                    state.raw_jpeg = payload
                    state.raw_sequence += 1
                    state.received_at = now
                    state.frame_times.append(monotonic_now)
                    state.width = width
                    state.height = height
                    state.condition.notify_all()

                if monotonic_now - state.last_submit_monotonic >= OCR_SUBMIT_INTERVAL:
                    atomic_write_bytes(SHARED_DIR / f"{key}_input.jpg", payload)
                    state.last_submit_monotonic = monotonic_now
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                if time.monotonic() - last_error_log >= 10.0:
                    print(f"[SOURCE] {key} 画面暂不可用: {exc}", flush=True)
                    last_error_log = time.monotonic()

            elapsed = time.monotonic() - started
            state.activity_event.wait(max(0.0, period - elapsed))
            state.activity_event.clear()


class AnnotationWatcher(threading.Thread):
    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__(name="annotation-watcher", daemon=True)
        self.stop_event = stop_event
        self.mtimes: dict[str, int] = {}

    def run(self) -> None:
        while not self.stop_event.wait(0.25):
            for key, state in STATES.items():
                image_path = SHARED_DIR / f"{key}_annotated.jpg"
                try:
                    mtime = image_path.stat().st_mtime_ns
                except FileNotFoundError:
                    continue
                if self.mtimes.get(key) == mtime:
                    continue
                try:
                    image_bytes = image_path.read_bytes()
                    status_path = SHARED_DIR / f"{key}_status.json"
                    status = json.loads(status_path.read_text()) if status_path.exists() else {}
                except (OSError, json.JSONDecodeError):
                    continue
                self.mtimes[key] = mtime
                with state.condition:
                    state.annotated_jpeg = image_bytes
                    state.annotation_status = status
                    state.annotation_sequence += 1
                    state.condition.notify_all()


def _analysis_status(trial_dir: Path) -> dict:
    status_path = trial_dir / "analysis_status.json"
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        status = {}
    if (trial_dir / "ablation_results.json").exists():
        status["state"] = "complete"
        status["progress"] = 1.0
    elif not status:
        status = {"state": "not_started", "progress": 0.0}
    return status


def _write_analysis_status(trial_dir: Path, **values) -> None:
    status = _analysis_status(trial_dir)
    status.update(values)
    status["updated_at"] = time.time()
    atomic_write_json(trial_dir / "analysis_status.json", status)


def queue_trial_analysis(trial_dir: Path) -> bool:
    if (trial_dir / "ablation_results.json").exists():
        return False
    with ANALYSIS_QUEUE_LOCK:
        if trial_dir.name in ANALYSIS_QUEUED:
            return False
        ANALYSIS_QUEUED.add(trial_dir.name)
    _write_analysis_status(
        trial_dir,
        state="queued",
        progress=0.0,
        completed_steps=0,
        total_steps=0,
        current_frame=None,
        current_variant=None,
        error=None,
    )
    ANALYSIS_QUEUE.put(trial_dir)
    return True


class AblationAnalysisWorker(threading.Thread):
    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__(name="ablation-analysis", daemon=True)
        self.stop_event = stop_event
        self.process_lock = threading.Lock()
        self.process: subprocess.Popen | None = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                trial_dir = ANALYSIS_QUEUE.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if (trial_dir / "ablation_results.json").exists():
                    _write_analysis_status(trial_dir, state="complete", progress=1.0, error=None)
                    continue
                command = [
                    "/usr/bin/nice",
                    "-n",
                    "9",
                    str(PROJECT_DIR / ".venv" / "bin" / "python"),
                    "-u",
                    str(PROJECT_DIR / "ablation_runner.py"),
                    str(trial_dir),
                    "--status-file",
                    str(trial_dir / "analysis_status.json"),
                ]
                environment = os.environ.copy()
                environment["THU_VR_OCR_THREADS"] = "1"
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_DIR),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with self.process_lock:
                    self.process = process
                while process.poll() is None and not self.stop_event.wait(0.25):
                    pass
                if self.stop_event.is_set() and process.poll() is None:
                    process.terminate()
                try:
                    output, _ = process.communicate(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    output, _ = process.communicate()
                if process.returncode == 0 and (trial_dir / "ablation_results.json").exists():
                    _write_analysis_status(
                        trial_dir,
                        state="complete",
                        progress=1.0,
                        error=None,
                        current_frame=None,
                        current_variant=None,
                    )
                    print(f"[ABLATION] {trial_dir.name} 分析完成", flush=True)
                elif not self.stop_event.is_set():
                    tail = (output or "分析进程异常退出").strip()[-1500:]
                    _write_analysis_status(trial_dir, state="error", error=tail)
                    print(f"[ABLATION] {trial_dir.name} 失败: {tail}", flush=True)
            except Exception as exc:
                if not self.stop_event.is_set():
                    _write_analysis_status(
                        trial_dir,
                        state="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
            finally:
                with self.process_lock:
                    self.process = None
                with ANALYSIS_QUEUE_LOCK:
                    ANALYSIS_QUEUED.discard(trial_dir.name)
                ANALYSIS_QUEUE.task_done()

    def stop(self) -> None:
        with self.process_lock:
            process = self.process
        if process is not None and process.poll() is None:
            process.terminate()


app = Flask(__name__)


def get_state(camera: str) -> CameraState:
    state = STATES.get(camera)
    if state is None:
        abort(404)
    return state


def mjpeg_stream(state: CameraState, kind: str):
    sequence = -1
    with state.condition:
        state.stream_viewers += 1
        state.activity_event.set()
    try:
        while True:
            with state.condition:
                current = state.raw_sequence if kind == "raw" else state.annotation_sequence
                if current == sequence:
                    state.condition.wait(timeout=1.0)
                    current = state.raw_sequence if kind == "raw" else state.annotation_sequence
                sequence = current
                payload = state.raw_jpeg if kind == "raw" else state.annotated_jpeg
            if payload is None:
                payload = PLACEHOLDERS[kind]
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n"
                + payload
                + b"\r\n"
            )
    finally:
        with state.condition:
            state.stream_viewers = max(0, state.stream_viewers - 1)


@app.get("/")
def index():
    return render_template("camera_dashboard.html", cameras=CAMERAS)


@app.get("/stream/<camera>/<kind>")
def stream(camera: str, kind: str):
    if kind not in ("raw", "annotated"):
        abort(404)
    state = get_state(camera)
    return Response(
        mjpeg_stream(state, kind),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/snapshot/<camera>/<kind>.jpg")
def snapshot(camera: str, kind: str):
    if kind not in ("raw", "annotated"):
        abort(404)
    state = get_state(camera)
    with state.condition:
        state.last_snapshot_request_monotonic = time.monotonic()
        state.activity_event.set()
        payload = state.raw_jpeg if kind == "raw" else state.annotated_jpeg
    if payload is None:
        payload = PLACEHOLDERS[kind]
    return Response(payload, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def api_status():
    now = time.time()
    response = {}
    for key, state in STATES.items():
        with state.condition:
            response[key] = {
                "label": state.label,
                "topic": state.topic,
                "online": (
                    state.received_at > 0
                    and now - state.received_at < max(3.0, OCR_SUBMIT_INTERVAL + 2.0)
                ),
                "frame_age_seconds": round(now - state.received_at, 2) if state.received_at else None,
                "fps": round(state.fps(), 1),
                "resolution": [state.width, state.height],
                "raw_sequence": state.raw_sequence,
                "annotation_sequence": state.annotation_sequence,
                "annotation": state.annotation_status,
            }
    return jsonify(response)


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "cameras": list(CAMERAS)})


def _fetch_json(url: str, timeout: float = 3.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "thu-vr-ablation/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_odom_once() -> dict:
    command = """
source /opt/ros/humble/setup.bash
source /home/unix_ai/work/controller/install/setup.bash
set -a
source /home/unix_ai/config/ros_domain_id.env
set +a
timeout 6 ros2 topic echo /odom --once --field pose.pose
"""
    result = subprocess.run(
        ["/usr/bin/bash", "-lc", command],
        capture_output=True,
        text=True,
        timeout=9,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "odom unavailable").strip())
    values = {
        key: float(value)
        for key, value in re.findall(r"^\s*(x|y|z|w):\s*([-+0-9.eE]+)\s*$", result.stdout, re.MULTILINE)
    }
    # The field contains position x/y/z followed by quaternion x/y/z/w, so
    # parse the two named blocks explicitly instead of relying on duplicate keys.
    position_match = re.search(
        r"position:\s*\n\s*x:\s*([-+0-9.eE]+)\s*\n\s*y:\s*([-+0-9.eE]+)\s*\n\s*z:\s*([-+0-9.eE]+)",
        result.stdout,
    )
    orientation_match = re.search(
        r"orientation:\s*\n\s*x:\s*([-+0-9.eE]+)\s*\n\s*y:\s*([-+0-9.eE]+)\s*\n\s*z:\s*([-+0-9.eE]+)\s*\n\s*w:\s*([-+0-9.eE]+)",
        result.stdout,
    )
    if not position_match or not orientation_match:
        raise RuntimeError(f"无法解析 odom: {values}")
    px, py, pz = (float(value) for value in position_match.groups())
    qx, qy, qz, qw = (float(value) for value in orientation_match.groups())
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return {
        "position_m": {"x": px, "y": py, "z": pz},
        "orientation_quaternion": {"x": qx, "y": qy, "z": qz, "w": qw},
        "yaw_deg": round(math.degrees(yaw), 4),
    }


def _recent_trials(limit: int = 20) -> list[dict]:
    if not EXPERIMENT_DIR.exists():
        return []
    trials = []
    for metadata_path in sorted(EXPERIMENT_DIR.glob("*/metadata.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        trial_dir = metadata_path.parent
        result_path = trial_dir / "ablation_results.json"
        if not result_path.exists():
            result_path = trial_dir / "ablation_partial.json"
        result = {}
        try:
            result = json.loads(result_path.read_text())
            summary = result.get("summary", [])
            analyzed_frames = len(result.get("frames", []))
        except (OSError, json.JSONDecodeError):
            summary = []
            analyzed_frames = 0
        target_rows_by_camera = metadata.get("target_rows_by_camera") or {
            "head": metadata.get("target_rows", []),
            "base": [],
        }
        frame_count_by_camera = {"head": 0, "base": 0}
        for frame in metadata.get("frames", []):
            camera = frame.get("camera", "head")
            if camera in frame_count_by_camera:
                frame_count_by_camera[camera] += 1
        analyzed_frames_by_camera = {"head": 0, "base": 0}
        for frame in result.get("frames", []):
            camera = frame.get("camera", "head")
            if camera in analyzed_frames_by_camera:
                analyzed_frames_by_camera[camera] += 1
        camera_results = result.get("cameras") or ({
            "head": {
                "frame_count": analyzed_frames,
                "target_rows": target_rows_by_camera["head"],
                "summary": summary,
                "ground_truth_check": result.get("ground_truth_check"),
            }
        } if summary else {})
        trials.append({
            "trial_id": metadata.get("trial_id", metadata_path.parent.name),
            "captured_at": metadata.get("captured_at"),
            "distance_m": metadata.get("distance_m"),
            "target_rows": metadata.get("target_rows", []),
            "target_rows_by_camera": target_rows_by_camera,
            "head_up_deg": metadata.get("robot", {}).get("head", {}).get("head_up_deg"),
            "frame_count": len(metadata.get("frames", [])),
            "notes": metadata.get("notes", ""),
            "thumbnail_url": f"/api/experiment/{metadata_path.parent.name}/thumbnail.jpg",
            "thumbnail_urls": {
                camera: f"/api/experiment/{metadata_path.parent.name}/thumbnail/{camera}.jpg"
                for camera, count in frame_count_by_camera.items() if count
            },
            "analysis": _analysis_status(trial_dir),
            "analyzed_frames": analyzed_frames,
            "frame_count_by_camera": frame_count_by_camera,
            "analyzed_frames_by_camera": analyzed_frames_by_camera,
            "summary": summary,
            "ground_truth_check": result.get("ground_truth_check"),
            "camera_results": camera_results,
        })
        if len(trials) >= limit:
            break
    return trials


@app.get("/api/experiment/config")
def experiment_config():
    return jsonify({
        "row_prefix_format": "[A-Z][0-9]{2}",
        "label_sequence_start": LABEL_SEQUENCE_START,
        "label_sequence_end": LABEL_SEQUENCE_END,
        "recent_trials": _recent_trials(),
        "default_frame_count": 10,
    })


@app.get("/api/experiment/<trial_id>/thumbnail.jpg")
def experiment_thumbnail(trial_id: str):
    return _experiment_thumbnail(trial_id, "head")


@app.get("/api/experiment/<trial_id>/thumbnail/<camera>.jpg")
def experiment_camera_thumbnail(trial_id: str, camera: str):
    return _experiment_thumbnail(trial_id, camera)


def _experiment_thumbnail(trial_id: str, camera: str):
    if not re.fullmatch(r"[0-9_]+", trial_id):
        abort(404)
    if camera not in CAMERAS:
        abort(404)
    trial_dir = EXPERIMENT_DIR / trial_id
    metadata_path = trial_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
        frame = next(
            item for item in metadata["frames"]
            if item.get("camera", "head") == camera
        )
        filename = frame["file"]
    except (OSError, json.JSONDecodeError, KeyError, IndexError, StopIteration):
        abort(404)
    image_path = trial_dir / filename
    if image_path.parent != trial_dir or not image_path.exists():
        abort(404)
    return send_file(image_path, mimetype="image/jpeg", max_age=0)


@app.post("/api/experiment/capture")
def experiment_capture():
    payload = request.get_json(silent=True) or {}
    try:
        distance_m = float(payload["distance_m"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "distance_m 必须是正数"}), 400
    if not (0.05 <= distance_m <= 20.0):
        return jsonify({"ok": False, "error": "distance_m 超出 0.05–20 m 范围"}), 400
    raw_rows_by_camera = payload.get("target_rows_by_camera")
    if raw_rows_by_camera is None:
        legacy_rows = payload.get("target_rows") or []
        raw_rows_by_camera = {camera: legacy_rows for camera in CAMERAS}
    if not isinstance(raw_rows_by_camera, dict):
        return jsonify({"ok": False, "error": "target_rows_by_camera 格式错误"}), 400
    target_rows_by_camera = {}
    for camera in CAMERAS:
        rows = raw_rows_by_camera.get(camera) or []
        if isinstance(rows, str):
            rows = [
                value.strip().upper()
                for value in re.split(r"[,，\s/]+", rows)
                if value.strip()
            ]
        if (
            not isinstance(rows, list)
            or not rows
            or len(rows) > 20
            or any(not isinstance(value, str) or not ROW_PREFIX_RE.fullmatch(value.upper()) for value in rows)
        ):
            return jsonify({
                "ok": False,
                "error": f"请填写{CAMERAS[camera]['label']}可见行，格式如 A05、B12、N03",
            }), 400
        normalized_rows = []
        for value in rows:
            value = value.upper()
            if value not in normalized_rows:
                normalized_rows.append(value)
        target_rows_by_camera[camera] = normalized_rows
    frame_count = int(payload.get("frame_count", 10))
    if not (3 <= frame_count <= 30):
        return jsonify({"ok": False, "error": "frame_count 必须在 3–30 之间"}), 400
    notes = str(payload.get("notes", "")).strip()[:500]

    if not EXPERIMENT_CAPTURE_LOCK.acquire(blocking=False):
        return jsonify({"ok": False, "error": "已有一次采样正在进行"}), 409
    try:
        now = datetime.now(timezone.utc)
        trial_id = now.astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        trial_dir = EXPERIMENT_DIR / trial_id
        trial_dir.mkdir(parents=True, exist_ok=False)

        try:
            head_status = _fetch_json(ARM_STATUS_URL)
            pitch_deg = head_status.get("head_pitch_deg")
            if pitch_deg is not None:
                # Robot feedback uses negative pitch for looking upward. Keep
                # the raw value and add an intuitive positive-up value.
                head_status["head_up_deg"] = round(-float(pitch_deg), 4)
        except Exception as exc:
            head_status = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            odom = _read_odom_once()
        except Exception as exc:
            odom = {"error": f"{type(exc).__name__}: {exc}"}

        frames = []
        camera_metadata = {}
        live_annotation_snapshots = {}
        for camera in CAMERAS:
            camera_frames = []
            seen_hashes: set[str] = set()
            attempts = 0
            while len(camera_frames) < frame_count and attempts < frame_count * 15:
                attempts += 1
                request_upstream = urllib.request.Request(
                    f"{UPSTREAM_URL}/snapshot/{camera}",
                    headers={"User-Agent": "thu-vr-ablation/2.0"},
                )
                try:
                    with urllib.request.urlopen(request_upstream, timeout=4.0) as response:
                        jpeg = response.read()
                except Exception:
                    time.sleep(0.2)
                    continue
                digest = hashlib.sha256(jpeg).hexdigest()
                if not jpeg.startswith(b"\xff\xd8") or digest in seen_hashes:
                    time.sleep(0.2)
                    continue
                seen_hashes.add(digest)
                filename = f"{camera}_{len(camera_frames):03d}.jpg"
                (trial_dir / filename).write_bytes(jpeg)
                frame = {
                    "camera": camera,
                    "file": filename,
                    "sha256": digest,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "bytes": len(jpeg),
                }
                camera_frames.append(frame)
                frames.append(frame)
                time.sleep(0.5)
            if len(camera_frames) != frame_count:
                raise RuntimeError(
                    f"{CAMERAS[camera]['label']}只取得 {len(camera_frames)}/{frame_count} 张独立图像"
                )
            with STATES[camera].condition:
                live_annotation_snapshots[camera] = dict(STATES[camera].annotation_status)
                resolution = [STATES[camera].width, STATES[camera].height]
            camera_metadata[camera] = {
                "name": camera,
                "label": CAMERAS[camera]["label"],
                "topic": CAMERAS[camera]["topic"],
                "resolution": resolution,
                "upstream": UPSTREAM_URL,
                "frame_count": len(camera_frames),
            }
        metadata = {
            "schema_version": 2,
            "trial_id": trial_id,
            "captured_at": now.isoformat(),
            "distance_m": distance_m,
            "target_rows_by_camera": target_rows_by_camera,
            "expected_labels_by_camera": {
                camera: [
                    label for row in rows for label in labels_for_row(row)
                ]
                for camera, rows in target_rows_by_camera.items()
            },
            "notes": notes,
            "frames": frames,
            "cameras": camera_metadata,
            "robot": {"head": head_status, "odom": odom},
            "live_annotation_snapshots": live_annotation_snapshots,
        }
        temporary = trial_dir / "metadata.tmp.json"
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, trial_dir / "metadata.json")
        queue_trial_analysis(trial_dir)
        return jsonify({
            "ok": True,
            "trial_id": trial_id,
            "frame_count": len(frames),
            "frame_count_by_camera": {
                camera: sum(frame.get("camera") == camera for frame in frames)
                for camera in CAMERAS
            },
            "distance_m": distance_m,
            "target_rows_by_camera": target_rows_by_camera,
            "head_pitch_deg": head_status.get("head_pitch_deg"),
            "head_up_deg": head_status.get("head_up_deg"),
            "head_yaw_deg": head_status.get("head_yaw_deg"),
            "odom": odom,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
    finally:
        EXPERIMENT_CAPTURE_LOCK.release()


def start_ocr_worker(shared_dir: Path, camera: str) -> subprocess.Popen:
    python = PROJECT_DIR / ".venv" / "bin" / "python"
    worker = PROJECT_DIR / "live_ocr_worker.py"
    if not python.exists():
        raise RuntimeError(f"OCR 虚拟环境不存在: {python}")
    return subprocess.Popen(
        [
            str(python), "-u", str(worker),
            "--shared-dir", str(shared_dir),
            "--camera", camera,
        ],
        cwd=str(PROJECT_DIR),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="双相机实时原画/标注网页")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5090)
    args = parser.parse_args()

    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    poller = UpstreamCameraPoller(stop_event)
    poller.start()
    watcher = AnnotationWatcher(stop_event)
    watcher.start()
    workers = [start_ocr_worker(SHARED_DIR, camera) for camera in CAMERAS]
    analysis_worker = AblationAnalysisWorker(stop_event)
    analysis_worker.start()
    for metadata_path in sorted(EXPERIMENT_DIR.glob("*/metadata.json")):
        queue_trial_analysis(metadata_path.parent)
    cleanup_lock = threading.Lock()
    cleaned_up = False

    def cleanup() -> None:
        nonlocal cleaned_up
        with cleanup_lock:
            if cleaned_up:
                return
            cleaned_up = True
        stop_event.set()
        for state in STATES.values():
            state.activity_event.set()
        analysis_worker.stop()
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            if worker.poll() is None:
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
        poller.join()
        analysis_worker.join(timeout=10)

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *_: raise_exit())
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        cleanup()


def raise_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    main()
