"""
图书馆书脊标签识别（离线版）

核心思路：
1. 用 PIL 读取图片并校正 EXIF 方向，让书脊/标签在像素坐标里保持竖直；
2. 用 RapidOCR 做整图粗检测 + 局部放大二次 OCR；
3. 按“前缀+数字”配对恢复标签编号；
4. 按书架行（y 坐标聚类）做前缀一致性校正。
"""
import argparse
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

LABEL_RE = re.compile(r'\b([A-Z])(\d{2})-(\d{3,4})\b')
PREFIX_RE = re.compile(r'^([A-Z])(\d{2})-?$')
NUM_RE = re.compile(r'^(\d{2,4})$')


def load_image_normalized(image_path: Path) -> np.ndarray:
    """用 PIL 读取并校正 EXIF 方向，转成 OpenCV 格式（像素方向已统一）。"""
    pil_img = Image.open(str(image_path))
    pil_img = ImageOps.exif_transpose(pil_img)
    return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)


def clean_text(text: str) -> str:
    t = re.sub(r'[^A-Z0-9-]', '', text.upper())
    # OCR 常把 0 认成 O；在这些标签里不会出现字母 O，直接归一化为 0
    return t.replace('O', '0')


def normalize_code(letter: str, prefix_num: str, code: str) -> str | None:
    if not code.isdigit():
        return None
    code_padded = code.zfill(4)
    if len(code_padded) > 4:
        return None
    return f"{letter}{prefix_num}-{code_padded}"


def parse_text(text: str):
    c = clean_text(text)
    m = LABEL_RE.match(c)
    if m:
        return 'full', m.group(1), m.group(2), m.group(3)
    m = PREFIX_RE.match(c)
    if m:
        return 'prefix', m.group(1), m.group(2), None
    m = NUM_RE.match(c)
    if m:
        return 'number', None, None, m.group(1)
    return None, None, None, None


def box_bounds(pts):
    pts = np.array(pts)
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def box_center(pts):
    x_min, x_max, y_min, y_max = box_bounds(pts)
    return (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min


def preprocess_crop(crop: np.ndarray, scale: float = 2.5) -> np.ndarray:
    """对裁剪区域做放大 + 灰度 + CLAHE 对比度增强。"""
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def cluster_ocr_boxes(ocr_results: list | None, distance_thresh: float = 80.0):
    """按中心点距离做简单聚类（类似 DBSCAN）。"""
    if ocr_results is None:
        return []
    boxes = []
    for item in ocr_results:
        pts, text, score = item
        x_min, x_max, y_min, y_max = box_bounds(pts)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        boxes.append({
            'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
            'cx': cx, 'cy': cy, 'w': x_max - x_min, 'h': y_max - y_min,
            'text': text,
            'score': float(score) if isinstance(score, (int, float, str)) else 0.0,
        })

    clusters = []
    used = set()
    for i, b1 in enumerate(boxes):
        if i in used:
            continue
        cluster = [b1]
        used.add(i)
        queue = [b1]
        while queue:
            cur = queue.pop(0)
            for j, b2 in enumerate(boxes):
                if j in used:
                    continue
                dist = np.hypot(cur['cx'] - b2['cx'], cur['cy'] - b2['cy'])
                if dist < distance_thresh:
                    cluster.append(b2)
                    used.add(j)
                    queue.append(b2)
        clusters.append(cluster)

    regions = []
    for cluster in clusters:
        if not any(re.search(r'[A-Z0-9]', clean_text(b['text'])) for b in cluster):
            continue
        x_min = min(b['x_min'] for b in cluster)
        x_max = max(b['x_max'] for b in cluster)
        y_min = min(b['y_min'] for b in cluster)
        y_max = max(b['y_max'] for b in cluster)
        regions.append({
            'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
            'cx': (x_min + x_max) / 2, 'cy': (y_min + y_max) / 2,
            'w': x_max - x_min, 'h': y_max - y_min,
            'raw_texts': [b['text'] for b in cluster],
        })
    return regions


def zoom_ocr(engine: RapidOCR, image: np.ndarray, region: dict, scale: float = 2.5):
    """对指定区域裁剪、放大后 OCR。"""
    h, w = image.shape[:2]
    pad_x = int(region['w'] * 0.4)
    pad_y = int(region['h'] * 0.4)
    x1 = max(0, int(region['x_min'] - pad_x))
    y1 = max(0, int(region['y_min'] - pad_y))
    x2 = min(w, int(region['x_max'] + pad_x))
    y2 = min(h, int(region['y_max'] + pad_y))
    crop = image[y1:y2, x1:x2]

    proc = preprocess_crop(crop, scale=scale)
    r, _ = engine(proc)
    if not r:
        return [], (x1, y1, x2, y2)

    mapped = []
    for pts, text, score in r:
        new_pts = [[p[0] / scale + x1, p[1] / scale + y1] for p in pts]
        mapped.append((new_pts, text, score))
    return mapped, (x1, y1, x2, y2)


def pair_prefix_number(ocr_results: list) -> list[dict]:
    """把 OCR 文本中的前缀与正下方的数字配对，恢复完整标签。

    书脊标签的格式是上下两行：
        B04-
        0007
    所以数字应该在前缀的正下方，且水平中心对齐。
    """
    boxes = []
    for item in ocr_results:
        pts, text, score = item
        cx, cy, w, h = box_center(pts)
        x_min, x_max, y_min, y_max = box_bounds(pts)
        kind, L, N, code = parse_text(text)
        if kind is None:
            continue
        boxes.append({
            'cx': cx, 'cy': cy, 'w': w, 'h': h,
            'bbox': (x_min, x_max, y_min, y_max),
            'kind': kind, 'L': L, 'N': N, 'code': code,
            'text': text,
            'score': float(score) if isinstance(score, (int, float, str)) else 0.0,
        })

    prefixes = [b for b in boxes if b['kind'] in ('full', 'prefix')]
    numbers = [b for b in boxes if b['kind'] == 'number']

    found = []
    used_nums = set()

    for p in prefixes:
        if p['kind'] == 'full':
            label = normalize_code(p['L'], p['N'], p['code'])
            if label:
                found.append({
                    'label': label,
                    'cx': p['cx'], 'cy': p['cy'],
                    'bbox': p['bbox'],
                    'raw_text': p['text'],
                    'score': p['score'],
                })
            continue

        best_idx = None
        best_score = float('inf')
        for i, n in enumerate(numbers):
            if i in used_nums:
                continue
            dx = n['cx'] - p['cx']
            dy = n['cy'] - p['cy']
            # 数字必须在前缀下方，且不能隔太远
            if dy <= 0 or dy > max(p['h'], n['h']) * 2.8:
                continue
            # 同一标签的两行应该水平对齐
            if abs(dx) > max(p['w'], n['w']) * 0.75:
                continue
            # 优先选垂直距离最近、水平偏移最小的
            score = dy + abs(dx) * 2.0
            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            n = numbers[best_idx]
            label = normalize_code(p['L'], p['N'], n['code'])
            if label:
                px_min, px_max, py_min, py_max = p['bbox']
                nx_min, nx_max, ny_min, ny_max = n['bbox']
                found.append({
                    'label': label,
                    'cx': (p['cx'] + n['cx']) / 2,
                    'cy': (p['cy'] + n['cy']) / 2,
                    'bbox': (min(px_min, nx_min), max(px_max, nx_max),
                             min(py_min, ny_min), max(py_max, ny_max)),
                    'raw_text': f"{p['text']} + {n['text']}",
                    'score': (p['score'] + n['score']) / 2,
                })
            used_nums.add(best_idx)

    return found


def bbox_iou(a: tuple, b: tuple) -> float:
    ax_min, ax_max, ay_min, ay_max = a
    bx_min, bx_max, by_min, by_max = b
    inter_xmin = max(ax_min, bx_min)
    inter_xmax = min(ax_max, bx_max)
    inter_ymin = max(ay_min, by_min)
    inter_ymax = min(ay_max, by_max)
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    inter = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area_a = (ax_max - ax_min) * (ay_max - ay_min)
    area_b = (bx_max - bx_min) * (by_max - by_min)
    return inter / (area_a + area_b - inter + 1e-6)


def deduplicate_by_location(labels: list[dict], iou_thresh: float = 0.35) -> list[dict]:
    """按标签贴纸的包围盒做 NMS，避免一本书被重复配对多次。"""
    if not labels:
        return labels
    labels = sorted(labels, key=lambda x: x.get('score', 0), reverse=True)
    kept = []
    for item in labels:
        bbox = item.get('bbox')
        if bbox is None:
            continue
        if any(bbox_iou(bbox, k['bbox']) > iou_thresh for k in kept):
            continue
        kept.append(item)
    return sorted(kept, key=lambda x: (x['cy'], x['cx']))


def deduplicate_by_label(labels: list[dict], min_dist: float = 40.0) -> list[dict]:
    """同一标签字符串在很近的位置出现多次时，只保留分数最高的一个。"""
    if not labels:
        return labels
    groups = {}
    for item in labels:
        groups.setdefault(item['label'], []).append(item)
    result = []
    for items in groups.values():
        items = sorted(items, key=lambda x: x.get('score', 0), reverse=True)
        kept = []
        for item in items:
            if any(np.hypot(item['cx'] - k['cx'], item['cy'] - k['cy']) < min_dist for k in kept):
                continue
            kept.append(item)
        result.extend(kept)
    return sorted(result, key=lambda x: (x['cy'], x['cx']))


def correct_prefixes_by_row(labels: list[dict], row_height_thresh: float = 80.0) -> list[dict]:
    """按 y 坐标把标签分行，对每行用多数前缀校正异常前缀（如 D04→B04）。"""
    if not labels:
        return labels

    # 按 y 聚类成行
    rows = []
    used = set()
    for i, l1 in enumerate(labels):
        if i in used:
            continue
        row = [l1]
        used.add(i)
        for j, l2 in enumerate(labels):
            if j in used:
                continue
            if abs(l1['cy'] - l2['cy']) < row_height_thresh:
                row.append(l2)
                used.add(j)
        rows.append(row)

    corrected = []
    for row in rows:
        # 解析每行的前缀
        prefixes = []
        for item in row:
            m = re.match(r'^([A-Z]\d{2})-\d{4}$', item['label'])
            if m:
                prefixes.append(m.group(1))
        if not prefixes:
            corrected.extend(row)
            continue
        # 多数票前缀
        majority_prefix, _ = Counter(prefixes).most_common(1)[0]
        for item in row:
            m = re.match(r'^([A-Z]\d{2})-(\d{4})$', item['label'])
            if m and m.group(1) != majority_prefix:
                old = item['label']
                item['label'] = f"{majority_prefix}-{m.group(2)}"
                item['corrected_from'] = old
            corrected.append(item)

    return corrected


def cluster_rows(labels: list[dict], row_height_thresh: float):
    """按 y 坐标把标签分行。"""
    rows = []
    used = set()
    for i, l1 in enumerate(labels):
        if i in used:
            continue
        row = [l1]
        used.add(i)
        for j, l2 in enumerate(labels):
            if j in used:
                continue
            if abs(l1['cy'] - l2['cy']) < row_height_thresh:
                row.append(l2)
                used.add(j)
        rows.append(row)
    return rows


def correct_numbers_by_trend(labels: list[dict], row_height_thresh: float = 80.0,
                             outlier_thresh: float = 2.5) -> list[dict]:
    """对每一行书脊标签，用 Theil-Sen 稳健线性拟合修正明显读错的数字。

    书架同一层上的标签编号通常随 x 坐标单调递增，利用这个空间关系可以
    把像 6000、0000 这类孤立错误以及重复编号修正回合理值。
    """
    if not labels:
        return labels

    for row in cluster_rows(labels, row_height_thresh):
        if len(row) < 3:
            continue
        row = sorted(row, key=lambda x: x['cx'])
        xs = np.array([item['cx'] for item in row])
        nums = np.array([int(re.search(r'-(\d{1,4})$', item['label']).group(1)) for item in row])

        # Theil-Sen 估计斜率（对异常值稳健）
        slopes = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                dx = xs[j] - xs[i]
                if abs(dx) > 5:
                    slopes.append((nums[j] - nums[i]) / dx)
        if not slopes:
            continue
        slope = float(np.median(slopes))
        if slope <= 0:
            continue
        intercept = float(np.median(nums - slope * xs))

        preds = slope * xs + intercept
        existing = {n for n in nums}
        for i, item in enumerate(row):
            pred = int(round(preds[i]))
            residual = abs(nums[i] - preds[i])
            is_duplicate = sum(1 for n in nums if n == nums[i]) > 1
            if pred == nums[i]:
                continue
            if not (0 <= pred <= 9999):
                continue
            # 仅当偏离较大或是重复编号时才修正
            should_correct = False
            if residual > outlier_thresh:
                should_correct = True
            elif is_duplicate and abs(pred - nums[i]) >= 1 and pred not in existing:
                should_correct = True

            if should_correct and 0 <= pred <= 9999:
                old = item['label']
                prefix = item['label'][:3]
                item['label'] = f"{prefix}-{pred:04d}"
                item['corrected_from'] = old
                existing.discard(nums[i])
                existing.add(pred)
    return labels


def detect_labels_in_bands(engine: RapidOCR, image: np.ndarray,
                           max_dim: int = 11000) -> list:
    """把图像分成若干水平条带，每条放大后再 OCR，用于拯救远景里的小标签。

    RapidOCR 对极端宽高比的图片会失效，因此条带高度不能太小；
    同时模型对单边尺寸过大（>~11000）的图片也会降采样，所以放大倍数受限。
    """
    h, w = image.shape[:2]
    band_h = max(int(h * 0.35), int(w * 0.25))
    overlap = int(band_h * 0.30)
    bands = []
    y = 0
    while y + band_h * 0.5 < h:
        y1 = y
        y2 = min(h, y + band_h)
        bands.append((y1, y2))
        y += band_h - overlap

    all_results = []
    for y1, y2 in bands:
        crop = image[y1:y2, :]
        ch, cw = crop.shape[:2]
        scale = min(2.0, max_dim / max(cw, ch))
        if scale < 1.0:
            scale = 1.0
        if scale > 1.1:
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            big = crop

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, big, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        result = engine(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        if result and result[0]:
            for pts, text, score in result[0]:
                mapped = [[p[0] / scale, p[1] / scale + y1] for p in pts]
                all_results.append((mapped, text, score))
    return all_results


def process_image(engine: RapidOCR, image_path: Path, output_dir: Path | None = None) -> dict:
    image = load_image_normalized(image_path)
    h, w = image.shape[:2]

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    coarse_result, elapse = engine(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    # 聚类阈值改小：只把同一本书标签里的“前缀”和“数字”框聚在一起
    cluster_thresh = min(w, h) * 0.012
    regions = cluster_ocr_boxes(coarse_result, distance_thresh=cluster_thresh)

    all_results = list(coarse_result) if coarse_result else []
    region_bboxes = []
    for region in regions:
        zoomed, bbox = zoom_ocr(engine, image, region, scale=3.0)
        all_results.extend(zoomed)
        region_bboxes.append(bbox)

    labels = pair_prefix_number(all_results)
    labels = deduplicate_by_location(labels, iou_thresh=0.35)
    labels = deduplicate_by_label(labels, min_dist=min(w, h) * 0.012)
    labels = correct_prefixes_by_row(labels, row_height_thresh=min(w, h) * 0.018)
    labels = correct_numbers_by_trend(labels, row_height_thresh=min(w, h) * 0.018)

    # 如果整张图几乎没检出标签，尝试按“水平条带”放大再检（应对远景/超远景）
    if len(labels) < 2:
        band_results = detect_labels_in_bands(engine, image)
        all_results.extend(band_results)
        labels = pair_prefix_number(all_results)
        # 远景误检多，只保留置信度较高的结果
        labels = [l for l in labels if l.get('score', 0) >= 0.62]
        labels = deduplicate_by_location(labels, iou_thresh=0.35)
        labels = deduplicate_by_label(labels, min_dist=min(w, h) * 0.012)
        labels = correct_prefixes_by_row(labels, row_height_thresh=min(w, h) * 0.018)
        labels = correct_numbers_by_trend(labels, row_height_thresh=min(w, h) * 0.018)

    labels = sorted(labels, key=lambda x: (x['cy'], x['cx']))

    output = {
        'image': str(image_path),
        'label_count': len(labels),
        'labels': labels,
        'region_count': len(regions),
        'elapsed_ms': {
            'detection': round(elapse[0] * 1000, 1) if elapse and len(elapse) > 0 else None,
            'classification': round(elapse[1] * 1000, 1) if elapse and len(elapse) > 1 else None,
            'recognition': round(elapse[2] * 1000, 1) if elapse and len(elapse) > 2 else None,
        },
    }

    if output_dir is not None:
        vis = image.copy()
        for item in labels:
            cx, cy = int(item['cx']), int(item['cy'])
            color = (0, 255, 255) if 'corrected_from' in item else (0, 255, 0)
            cv2.circle(vis, (cx, cy), 10, color, -1)
            cv2.putText(vis, item['label'], (cx + 14, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for (x1, y1, x2, y2) in region_bboxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        vis_path = output_dir / f"{image_path.stem}_result.jpg"
        cv2.imwrite(str(vis_path), vis)
        output['visualization'] = str(vis_path)

    return output


def main():
    parser = argparse.ArgumentParser(description='图书馆书脊标签离线识别（含 EXIF 校正、局部放大、行前缀校正）')
    parser.add_argument('input', help='图片文件或图片目录')
    parser.add_argument('--visualize', action='store_true', help='生成带标注的可视化图到 output/ 目录')
    parser.add_argument('--output', default='ocr_results.json', help='JSON 结果输出路径')
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        for ext in exts:
            image_paths.extend(input_path.glob(ext))
        image_paths = [p for p in image_paths if not p.name.endswith('_result.jpg')]
        image_paths = sorted(image_paths)
    else:
        image_paths = [input_path]

    output_dir = None
    if args.visualize:
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

    engine = RapidOCR()
    all_results = []

    for img_path in image_paths:
        print(f"处理: {img_path.name}")
        try:
            result = process_image(engine, img_path, output_dir=output_dir)
            all_results.append(result)
            print(f"  -> 发现 {result['label_count']} 个标签 (候选区域 {result['region_count']})")
            for item in result['labels']:
                marker = "*" if 'corrected_from' in item else " "
                print(f"    {marker}{item['label']}")
        except Exception as e:
            print(f"  -> 错误: {e}")
            import traceback
            traceback.print_exc()

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {args.output}")
    if args.visualize:
        print(f"可视化图保存在: {output_dir}/")


if __name__ == '__main__':
    main()
