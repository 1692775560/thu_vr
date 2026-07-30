"""
真实场景（机器人头部相机）书脊标签识别

针对 720p 头部相机画面特点：
- 书脊标签多为白色长条、竖排文字；
- 距离远时标签分辨率极低，需要强放大 + 锐化；
- 单引擎容易漏检，故同时使用 RapidOCR 与 EasyOCR；
- 先整图 OCR 粗定位，再对候选白色竖条做局部增强精识别。

输出：
- output/<name>_result.jpg：带检测框与标签的可视化图
- output/real_scene_results.json：结构化识别结果
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from library_label_ocr import load_image_normalized, clean_text

from rapidocr_onnxruntime import RapidOCR

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

LABEL_RE = re.compile(r'\b([A-Z])(\d{2})-(\d{3,4})\b')


def enhance_for_ocr(img: np.ndarray, scale: float = 6.0, gray: bool = True,
                    clahe: bool = True, sharp: bool = True, blur_sigma: float = 0):
    """放大 + 可选灰度 + CLAHE + unsharp mask。"""
    if img.size == 0:
        return img
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if blur_sigma > 0:
        big = cv2.GaussianBlur(big, (0, 0), blur_sigma)
    if gray and len(big.shape) == 3:
        big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    if clahe:
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        big = cl.apply(big) if len(big.shape) == 2 else cv2.merge([cl.apply(c) for c in cv2.split(big)])
    if sharp:
        g = cv2.GaussianBlur(big, (0, 0), 2)
        big = cv2.addWeighted(big, 1.6, g, -0.6, 0)
    if len(big.shape) == 2:
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    return big


def rotate(img: np.ndarray, deg: int):
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def ocr_rapid(engine: RapidOCR, img: np.ndarray, scales=(4, 6, 8, 12), rotations=(0, 90, 270)):
    """RapidOCR 多角度多尺度识别，返回原始文本框列表（含图像坐标中心）。"""
    results = []
    for deg in rotations:
        base = rotate(img, deg)
        for s in scales:
            proc = enhance_for_ocr(base, scale=s)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, proc, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            try:
                out = engine(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            if out and out[0]:
                for pts, text, score in out[0]:
                    pts_arr = np.array(pts, dtype=np.float32)
                    # 映射回原始图像坐标（rotate 后再 resize，需先除 scale）
                    pts_orig = pts_arr / s
                    if deg == 90:
                        # 顺时针 90 度映射：原始坐标 (x,y) -> (y, H-1-x)
                        h_orig = img.shape[0]
                        pts_orig = np.stack([pts_orig[:, 1], h_orig - 1 - pts_orig[:, 0]], axis=1)
                    elif deg == 270:
                        # 逆时针 90 度映射：原始坐标 (x,y) -> (W-1-y, x)
                        w_orig = img.shape[1]
                        pts_orig = np.stack([w_orig - 1 - pts_orig[:, 1], pts_orig[:, 0]], axis=1)
                    elif deg == 180:
                        h_orig, w_orig = img.shape[:2]
                        pts_orig = np.stack([w_orig - 1 - pts_orig[:, 0], h_orig - 1 - pts_orig[:, 1]], axis=1)
                    cx, cy = pts_orig[:, 0].mean(), pts_orig[:, 1].mean()
                    results.append({
                        'engine': 'rapid',
                        'rot': deg,
                        'scale': s,
                        'text': text,
                        'score': float(score),
                        'center': (float(cx), float(cy)),
                    })
    return results


def ocr_easy(engine, img: np.ndarray, scales=(4, 6, 8, 12), rotations=(0, 90, 270)):
    """EasyOCR 多角度多尺度识别（含图像坐标中心）。"""
    results = []
    for deg in rotations:
        base = rotate(img, deg)
        for s in scales:
            proc = enhance_for_ocr(base, scale=s)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            outs = engine.readtext(rgb)
            for bbox, text, conf in outs:
                pts_arr = np.array(bbox, dtype=np.float32) / s
                if deg == 90:
                    h_orig = img.shape[0]
                    pts_orig = np.stack([pts_arr[:, 1], h_orig - 1 - pts_arr[:, 0]], axis=1)
                elif deg == 270:
                    w_orig = img.shape[1]
                    pts_orig = np.stack([w_orig - 1 - pts_arr[:, 1], pts_arr[:, 0]], axis=1)
                elif deg == 180:
                    h_orig, w_orig = img.shape[:2]
                    pts_orig = np.stack([w_orig - 1 - pts_arr[:, 0], h_orig - 1 - pts_arr[:, 1]], axis=1)
                else:
                    pts_orig = pts_arr
                cx, cy = pts_orig[:, 0].mean(), pts_orig[:, 1].mean()
                results.append({
                    'engine': 'easy',
                    'rot': deg,
                    'scale': s,
                    'text': text,
                    'score': float(conf),
                    'center': (float(cx), float(cy)),
                })
    return results


def whole_image_ocr(engine, image: np.ndarray, scales=(1.5, 2.0), use_easy: bool = False):
    """对整图只做简单放大，然后 OCR。RapidOCR 与 EasyOCR 兼容封装。"""
    results = []
    for s in scales:
        big = cv2.resize(image, None, fx=s, fy=s)
        h, w = image.shape[:2]
        if use_easy:
            rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
            outs = engine.readtext(rgb)
            for bbox, text, conf in outs:
                pts_arr = np.array(bbox, dtype=np.float32) / s
                cx, cy = pts_arr[:, 0].mean(), pts_arr[:, 1].mean()
                results.append({
                    'engine': 'easy', 'scale': s, 'rot': 0,
                    'text': text, 'score': float(conf),
                    'center': (float(cx), float(cy)),
                })
        else:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, big)
            try:
                out = engine(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            if out and out[0]:
                for pts, text, score in out[0]:
                    pts_arr = np.array(pts, dtype=np.float32) / s
                    cx, cy = pts_arr[:, 0].mean(), pts_arr[:, 1].mean()
                    results.append({
                        'engine': 'rapid', 'scale': s, 'rot': 0,
                        'text': text, 'score': float(score),
                        'center': (float(cx), float(cy)),
                    })
    return results


def extract_labels(texts: list[dict]):
    """
    从 OCR 文本中过滤出符合 A03-0068 / B04-0007 格式的标签，并做 O→0 校正。
    支持从碎片组合：例如 'BOA' + '0015' → 'B04-0015'。
    """
    labels = []
    for item in texts:
        c = clean_text(item['text'])
        for m in LABEL_RE.finditer(c):
            labels.append({
                'label': f"{m.group(1)}{m.group(2)}-{m.group(3).zfill(4)}",
                'score': item['score'],
                'engine': item.get('engine'),
                'rot': item.get('rot'),
                'scale': item.get('scale'),
                'raw_text': item['text'],
                'center': item.get('center'),
            })
    return labels


PREFIX_RE = re.compile(r'^([A-Z])(\d{2})-?$')
NUM_RE = re.compile(r'^(\d{3,4})$')


def fuzzy_clean(text: str) -> str:
    """
    针对书脊标签常见 OCR 错误做更激进的校正：
    O→0, A→4, I→1, S→5, Z→2, B→8（仅在可能为数字时），保留连接符。
    """
    t = text.upper()
    t = re.sub(r'[^A-Z0-9\-]', '', t)
    # 前缀字母后的数字误识别
    t = t.replace('O', '0')
    t = t.replace('A', '4')
    t = t.replace('I', '1')
    t = t.replace('S', '5')
    t = t.replace('Z', '2')
    return t


def combine_fragments(texts: list[dict], distance_thresh: float = 80.0):
    """
    把 OCR 碎片按空间位置组合成完整标签。
    输入 texts 需包含 center (cx, cy) 字段。
    """
    fragments = []
    for item in texts:
        c = fuzzy_clean(item['text'])
        center = item.get('center')
        if center is None:
            continue
        # 完整标签
        for m in LABEL_RE.finditer(c):
            fragments.append({
                'label': f"{m.group(1)}{m.group(2)}-{m.group(3).zfill(4)}",
                'score': item['score'],
                'raw': item['text'],
                'center': center,
            })
        # 前缀碎片
        m = PREFIX_RE.search(c)
        if m:
            fragments.append({
                'kind': 'prefix',
                'letter': m.group(1),
                'prefix_num': m.group(2),
                'score': item['score'],
                'raw': item['text'],
                'center': center,
            })
        # 数字碎片
        m = NUM_RE.search(c)
        if m:
            fragments.append({
                'kind': 'number',
                'code': m.group(1),
                'score': item['score'],
                'raw': item['text'],
                'center': center,
            })

    labels = [f for f in fragments if 'label' in f]

    # 前缀 + 数字 配对
    prefixes = [f for f in fragments if f.get('kind') == 'prefix']
    numbers = [f for f in fragments if f.get('kind') == 'number']
    for p in prefixes:
        for n in numbers:
            dist = ((p['center'][0] - n['center'][0]) ** 2 +
                    (p['center'][1] - n['center'][1]) ** 2) ** 0.5
            if dist < distance_thresh:
                labels.append({
                    'label': f"{p['letter']}{p['prefix_num']}-{n['code'].zfill(4)}",
                    'score': (p['score'] + n['score']) / 2,
                    'raw': f"{p['raw']} + {n['raw']}",
                    'center': ((p['center'][0]+n['center'][0])/2,
                               (p['center'][1]+n['center'][1])/2),
                })
    return labels


def refine_text_regions(engine, image: np.ndarray, texts: list[dict], scales=(10, 15, 20)):
    """
    对整图 OCR 检测到的每个文字框，沿竖直方向扩展成可能的完整标签区域，
    然后在大尺度下再次 OCR，以捕捉前缀或完整编号。
    """
    refined = []
    seen = set()
    h, w = image.shape[:2]
    for item in texts:
        if 'center' not in item:
            continue
        cx, cy = item['center']
        # 估算文字高度（假设竖排标签，向上/下扩展）
        box_h = 40  # 默认文字高度
        # 扩展为包含整个可能标签的竖条：上下各扩展数倍
        y1 = max(0, int(cy - box_h * 4))
        y2 = min(h, int(cy + box_h * 4))
        x1 = max(0, int(cx - box_h * 1.5))
        x2 = min(w, int(cx + box_h * 1.5))
        key = (x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        for s in scales:
            big = cv2.resize(crop, None, fx=s, fy=s)
            cv2.imwrite('/tmp/refine.jpg', big, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            out = engine('/tmp/refine.jpg')
            if out and out[0]:
                for pts, text, score in out[0]:
                    pts_arr = np.array(pts, dtype=np.float32) / s
                    pts_arr[:, 0] += x1
                    pts_arr[:, 1] += y1
                    rcx, rcy = pts_arr[:, 0].mean(), pts_arr[:, 1].mean()
                    refined.append({
                        'engine': 'rapid', 'scale': s, 'rot': 0,
                        'text': text, 'score': float(score),
                        'center': (float(rcx), float(rcy)),
                    })
    return refined


def find_white_label_candidates(image: np.ndarray):
    """
    通过多阈值白色分割 + 形态学闭运算，找出可能的书脊标签（竖直白色长条）。
    返回候选框列表，元素为 (x, y, w, h, aspect)。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = []
    for thr in (150, 170, 190, 210):
        _, binary = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / max(min(w, h), 1)
            # 标签为竖直或水平细长条
            if aspect < 1.8 or aspect > 10:
                continue
            candidates.append((x, y, w, h, aspect, thr))
    # 简单去重（IOU>0.5 只保留最大的）
    candidates = sorted(candidates, key=lambda c: c[2]*c[3], reverse=True)
    filtered = []
    for c in candidates:
        x1, y1, w1, h1 = c[:4]
        dup = False
        for f in filtered:
            x2, y2, w2, h2 = f[:4]
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            iou = (ix * iy) / max(w1*h1, w2*h2, 1)
            if iou > 0.5:
                dup = True
                break
        if not dup:
            filtered.append(c)
    return filtered


def crop_label(image: np.ndarray, bbox, pad: float = 1.25):
    """按 bbox 裁剪标签，并旋转使长边水平（便于 OCR 读取）。"""
    x, y, w, h = bbox[:4]
    cx, cy = x + w/2, y + h/2
    long = max(w, h) * pad
    short = min(w, h) * pad
    # 如果 h > w（竖直标签），逆时针 90 度后文字横排
    if h > w:
        theta = 0
    else:
        theta = 90
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    rot = cv2.warpAffine(
        image, M, (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128)
    )
    x1 = int(cx - long/2)
    y1 = int(cy - short/2)
    x2 = int(cx + long/2)
    y2 = int(cy + short/2)
    return rot[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]


def process_image(rapid_engine, easy_engine, image_path: Path, output_dir: Path):
    image = load_image_normalized(image_path)
    h, w = image.shape[:2]

    all_texts = []

    # 1) 整图放大 OCR（对 720p 头部相机小标签更有效；整图不做强增强，避免破坏文字）
    try:
        all_texts.extend(whole_image_ocr(rapid_engine, image, scales=(1.2, 1.5, 2.0, 2.5, 3.0)))
    except Exception as e:
        print(f"  整图 RapidOCR 失败: {e}")
    if easy_engine is not None:
        try:
            all_texts.extend(whole_image_ocr(easy_engine, image, scales=(1.5, 2.0), use_easy=True))
        except Exception as e:
            print(f"  整图 EasyOCR 失败: {e}")

    # 2) 对整图检测到的文字框做局部精识别（捕捉完整标签编号）
    try:
        refined = refine_text_regions(rapid_engine, image, all_texts, scales=(8, 10, 12, 15, 20, 25))
        all_texts.extend(refined)
    except Exception as e:
        print(f"  局部精识别失败: {e}")

    # 3) 白色标签候选局部精识别（限制数量与尺度，控制耗时）
    candidates = find_white_label_candidates(image)
    candidate_results = []
    for i, (x, y, bw, bh, aspect, thr) in enumerate(candidates[:5]):
        crop = crop_label(image, (x, y, bw, bh))
        texts = []
        try:
            texts.extend(ocr_rapid(rapid_engine, crop, scales=(4, 8), rotations=(0, 180)))
        except Exception as e:
            print(f"  候选 {i} RapidOCR 失败: {e}")
        if easy_engine is not None:
            try:
                texts.extend(ocr_easy(easy_engine, crop, scales=(4, 8), rotations=(0, 180)))
            except Exception as e:
                print(f"  候选 {i} EasyOCR 失败: {e}")
        labels = extract_labels(texts)
        candidate_results.append({
            'candidate_id': i,
            'bbox': [int(x), int(y), int(bw), int(bh)],
            'aspect': float(aspect),
            'threshold': int(thr),
            'text_count': len(texts),
            'texts': texts,
            'labels': labels,
        })
        all_texts.extend(texts)

    # 3) 从所有结果中挑选最佳标签（完整匹配 + 碎片组合）
    labels = extract_labels(all_texts)
    combined = combine_fragments(all_texts, distance_thresh=120.0)
    labels.extend(combined)

    best_label = None
    if labels:
        from collections import Counter
        freq = Counter([l['label'] for l in labels])
        def rank(l):
            return (freq[l['label']], l['score'])
        best_label = max(labels, key=rank)

    # 4) 可视化
    vis = image.copy()
    # 画出所有 OCR 检测框中心
    for t in all_texts:
        if 'center' in t:
            cx, cy = int(t['center'][0]), int(t['center'][1])
            cv2.circle(vis, (cx, cy), 4, (255, 255, 0), -1)
            cv2.putText(vis, t['text'], (cx+5, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1)
    for c in candidate_results:
        x, y, bw, bh = c['bbox']
        cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        if c['labels']:
            label = c['labels'][0]['label']
            cv2.putText(vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if best_label:
        cv2.putText(vis, f"BEST: {best_label['label']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    vis_path = output_dir / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(vis_path), vis)

    return {
        'image': str(image_path),
        'size': [int(w), int(h)],
        'candidate_count': len(candidates),
        'best_label': best_label,
        'candidates': candidate_results,
        'all_labels': labels,
        'all_texts': all_texts,
        'visualization': str(vis_path),
    }


def extract_video_frames(video_path: Path, output_dir: Path, step_frames: int = 5):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step_frames == 0:
            frame_path = output_dir / f"{video_path.stem}_frame_{idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append(frame_path)
        idx += 1
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description='真实场景书脊标签识别（机器人头部相机）')
    parser.add_argument('input', help='图片、目录或视频文件')
    parser.add_argument('--output-dir', default=None, help='输出目录')
    parser.add_argument('--video-step', type=int, default=5, help='视频抽帧间隔（帧）')
    parser.add_argument('--output-json', default='real_scene_results.json', help='JSON 结果文件名')
    parser.add_argument('--no-easy', action='store_true', help='不使用 EasyOCR')
    parser.add_argument('--append', action='store_true', help='追加到已有 JSON，不覆盖')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / 'output'
    output_dir.mkdir(exist_ok=True, parents=True)

    print('加载 RapidOCR...')
    rapid_engine = RapidOCR()

    easy_engine = None
    if not args.no_easy and EASYOCR_AVAILABLE:
        print('加载 EasyOCR（首次较慢）...')
        easy_engine = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    elif not EASYOCR_AVAILABLE:
        print('EasyOCR 不可用，仅使用 RapidOCR')

    json_path = output_dir / args.output_json
    all_results = []
    if args.append and json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)

    if input_path.is_dir():
        image_paths = []
        video_paths = []
        for p in input_path.iterdir():
            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                image_paths.append(p)
            elif p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi'):
                video_paths.append(p)

        for img_path in sorted(image_paths):
            print(f"处理图片: {img_path.name}")
            result = process_image(rapid_engine, easy_engine, img_path, output_dir)
            all_results.append(result)
            print(f"  -> 候选 {result['candidate_count']}, 最佳标签: {result['best_label']}")

        for video_path in sorted(video_paths):
            print(f"处理视频: {video_path.name}")
            frames = extract_video_frames(video_path, output_dir, args.video_step)
            for frame_path in frames:
                result = process_image(rapid_engine, easy_engine, frame_path, output_dir)
                all_results.append(result)
            print(f"  -> 抽帧 {len(frames)} 张")

    elif input_path.suffix.lower() in ('.mp4', '.mov', '.avi'):
        frames = extract_video_frames(input_path, output_dir, args.video_step)
        for frame_path in frames:
            print(f"处理视频帧: {frame_path.name}")
            result = process_image(rapid_engine, easy_engine, frame_path, output_dir)
            all_results.append(result)
            print(f"  -> 候选 {result['candidate_count']}, 最佳标签: {result['best_label']}")
    else:
        result = process_image(rapid_engine, easy_engine, input_path, output_dir)
        all_results.append(result)
        print(f"候选 {result['candidate_count']}, 最佳标签: {result['best_label']}")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {json_path}")
    print(f"可视化图保存在: {output_dir}")


if __name__ == '__main__':
    main()
