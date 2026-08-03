#!/usr/bin/env bash

set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/unix_ai/orbbec_ws/install/setup.bash
set -a
source /home/unix_ai/config/ros_domain_id.env
set +a
set -u

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///home/unix_ai/config/cyclonedds.xml}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/dev/shm/head_rgbd_ros_log}"
unset ROS_LOCALHOST_ONLY
mkdir -p "$ROS_LOG_DIR"

readonly CAMERA_NAME="head_rgbd"
readonly SERIAL_NUMBER="${HEAD_ORBBEC_SERIAL:-AY6G65300DD}"
readonly LOCK_FILE="${HEAD_CAMERA_LOCK_FILE:-/dev/shm/bookbot_head_rgbd.lock}"
readonly COLOR_WIDTH="${HEAD_COLOR_WIDTH:-1920}"
readonly COLOR_HEIGHT="${HEAD_COLOR_HEIGHT:-1080}"
readonly COLOR_FPS="${HEAD_COLOR_FPS:-5}"
readonly DEPTH_WIDTH="${HEAD_DEPTH_WIDTH:-640}"
readonly DEPTH_HEIGHT="${HEAD_DEPTH_HEIGHT:-400}"
readonly DEPTH_FPS="${HEAD_DEPTH_FPS:-30}"
readonly UVC_BACKEND="${HEAD_UVC_BACKEND:-libuvc}"

if [[ ! "$SERIAL_NUMBER" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf '[%s] invalid serial number: %s\n' "$CAMERA_NAME" "$SERIAL_NUMBER" >&2
    exit 64
fi
for value in \
    "$COLOR_WIDTH" "$COLOR_HEIGHT" "$COLOR_FPS" \
    "$DEPTH_WIDTH" "$DEPTH_HEIGHT" "$DEPTH_FPS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf '[%s] invalid positive integer profile value: %s\n' "$CAMERA_NAME" "$value" >&2
        exit 64
    fi
done
if [[ "$UVC_BACKEND" != "libuvc" ]] && [[ "$UVC_BACKEND" != "v4l2" ]]; then
    printf '[%s] invalid UVC backend: %s (expected libuvc or v4l2)\n' \
        "$CAMERA_NAME" "$UVC_BACKEND" >&2
    exit 64
fi

readonly PROFILE="${COLOR_WIDTH}:${COLOR_HEIGHT}:${COLOR_FPS}:${DEPTH_WIDTH}:${DEPTH_HEIGHT}:${DEPTH_FPS}"
if [[ "$PROFILE" != "1920:1080:5:640:400:30" ]] && \
   [[ "${HEAD_ALLOW_EXPERIMENTAL_PROFILE:-0}" != "1" ]]; then
    printf '[%s] refusing unvalidated profile %s; set HEAD_ALLOW_EXPERIMENTAL_PROFILE=1 only for an isolated test\n' \
        "$CAMERA_NAME" "$PROFILE" >&2
    exit 64
fi

if ! command -v flock >/dev/null 2>&1; then
    printf '[%s] flock is required for single-owner enforcement\n' "$CAMERA_NAME" >&2
    exit 69
fi
umask 0077
exec 9>"$LOCK_FILE"
if ! flock --exclusive --nonblock 9; then
    printf '[%s] another sanctioned launcher owns %s\n' "$CAMERA_NAME" "$LOCK_FILE" >&2
    exit 75
fi

readonly PROCESS_PATTERN="([o]rbbec_camera.*(camera_name:=${CAMERA_NAME}|serial_number:=${SERIAL_NUMBER})|[c]omponent_container([^ ]*) .*__ns:=/${CAMERA_NAME}([[:space:]]|$))"
existing_processes="$(pgrep -af -- "$PROCESS_PATTERN" || true)"
if [[ -n "$existing_processes" ]]; then
    printf '[%s] refusing to race an existing launch or orphan container:\n%s\n' \
        "$CAMERA_NAME" "$existing_processes" >&2
    exit 75
fi

printf '[%s] serial=%s RGB=%sx%s@%s/MJPG depth-source=%sx%s@%s backend=%s lock=%s\n' \
    "$CAMERA_NAME" "$SERIAL_NUMBER" \
    "$COLOR_WIDTH" "$COLOR_HEIGHT" "$COLOR_FPS" \
    "$DEPTH_WIDTH" "$DEPTH_HEIGHT" "$DEPTH_FPS" "$UVC_BACKEND" "$LOCK_FILE"

exec ros2 launch orbbec_camera gemini2.launch.py \
    camera_name:="$CAMERA_NAME" \
    serial_number:="$SERIAL_NUMBER" \
    depth_registration:=true \
    align_mode:=HW \
    enable_frame_sync:=true \
    enable_point_cloud:=false \
    enable_colored_point_cloud:=false \
    enable_publish_extrinsic:=true \
    color_width:="$COLOR_WIDTH" \
    color_height:="$COLOR_HEIGHT" \
    color_fps:="$COLOR_FPS" \
    color_format:=MJPG \
    depth_width:="$DEPTH_WIDTH" \
    depth_height:="$DEPTH_HEIGHT" \
    depth_fps:="$DEPTH_FPS" \
    enable_ir:=false \
    enable_accel:=false \
    enable_gyro:=false \
    enable_heartbeat:=false \
    uvc_backend:="$UVC_BACKEND" \
    publish_tf:=false
