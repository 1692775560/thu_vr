"""
图书馆书脊标签识别 v2：粗检测 + 局部放大二次 OCR

思路：
1. 先在整图上做一次粗 OCR，定位可能有文字的区域；
2. 对每个候选区域裁剪、放大、增强对比度后再次 OCR；
3. 对竖向书脊自动旋转 90°，让标签横向显示；
4. 最后合并所有二次 OCR 结果，提取标签编号。
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

LABEL_RE = re.compile(r'\b([A-Z])(\d{2})-(\d{3,4})\b')
PREFIX_RE = re.compile(r'^([A-Z])(\d{2})-?$')
NUM_RE = re.compile(r'^(\d{2,4})$')


def clean_text(text: str) -> str:
    return re.sub(r'[^A-Z0-9-]', '', text.upper())


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


def preprocess_crop(crop: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """对裁剪区域做放大 + 灰度 + 自适应二值化。"""
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 自适应二值化能更好处理标签白底黑字
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def cluster_ocr_boxes(ocr_results: list, y_overlap_thresh: float = 0.3, x_gap_thresh: float = 2.5):
    """把 OCR 文本框按空间位置聚类成候选标签区域。"""
    boxes = []
    for item in ocr_results:
        pts, text, score = item
        x_min, x_max, y_min, y_max = box_bounds(pts)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        boxes.append({
            'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
            'cx': cx, 'cy': cy, 'w': w, 'h': h,
            'text': text, 'score': float(score) if isinstance(score, (int, float, str)) else 0.0,
        })

    clusters = []
    used = set()
    for i, b1 in enumerate(boxes):
        if i in used:
            continue
        cluster = [b1]
        used.add(i)
        for j, b2 in enumerate(boxes):
            if j in used:
                continue
            y_overlap = max(0, min(b1['y_max'], b2['y_max']) - max(b1['y_min'], b2['y_min']))
            min_h = min(b1['h'], b2['h'])
            x_gap = max(b1['x_min'], b2['x_min']) - min(b1['x_max'], b2['x_max'])
            if y_overlap / min_h > y_overlap_thresh and x_gap < max(b1['w'], b2['w']) * x_gap_thresh:
                cluster.append(b2)
                used.add(j)
        clusters.append(cluster)

    regions = []
    for cluster in clusters:
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


def zoom_ocr(engine: RapidOCR, image: np.ndarray, region: dict, scale: float = 3.0):
    """对指定区域裁剪、放大、可能旋转后再次 OCR。"""
    h, w = image.shape[:2]
    pad_x = int(region['w'] * 0.3)
    pad_y = int(region['h'] * 0.3)
    x1 = max(0, int(region['x_min'] - pad_x))
    y1 = max(0, int(region['y_min'] - pad_y))
    x2 = min(w, int(region['x_max'] + pad_x))
    y2 = min(h, int(region['y_max'] + pad_y))
    crop = image[y1:y2, x1:x2]

    results = []
    # 情况 1：原方向
    proc = preprocess_crop(crop, scale=scale)
    r1, _ = engine(proc)
    if r1:
        results.extend(r1)

    # 情况 2：如果区域高大于宽，可能是竖排书脊，旋转 90° 再试
    if region['h'] > region['w'] * 1.2:
        rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        proc_rot = preprocess_crop(rotated, scale=scale)
        r2, _ = engine(proc_rot)
        if r2:
            # 把旋转后的坐标转回原图坐标系太麻烦，这里只取文本结果
            results.extend(r2)

    return results, (x1, y1, x2, y2)


def pair_prefix_number(ocr_results: list) -> list[dict]:
    """把 OCR 文本中的前缀与最近的数字配对，恢复完整标签。"""
    boxes = []
    for item in ocr_results:
        pts, text, score = item
        cx, cy, w, h = box_center(pts)
        kind, L, N, code = parse_text(text)
        if kind is None:
            continue
        boxes.append({
            'cx': cx, 'cy': cy, 'w': w, 'h': h,
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
                    'raw_text': p['text'],
                    'score': p['score'],
                })
            continue

        best_idx = None
        best_dist = float('inf')
        for i, n in enumerate(numbers):
            if i in used_nums:
                continue
            dist = np.hypot(p['cx'] - n['cx'], p['cy'] - n['cy'])
            thresh = max(p['w'], p['h'], n['w'], n['h']) * 3.0
            if dist < best_dist and dist < thresh:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            n = numbers[best_idx]
            label = normalize_code(p['L'], p['N'], n['code'])
            if label:
                found.append({
                    'label': label,
                    'cx': (p['cx'] + n['cx']) / 2,
                    'cy': (p['cy'] + n['cy']) / 2,
                    'raw_text': f"{p['text']} + {n['text']}",
                    'score': (p['score'] + n['score']) / 2,
                })
            used_nums.add(best_idx)

    return found


def deduplicate_labels(labels: list[dict], min_dist: float = 30.0) -> list[dict]:
    if not labels:
        return labels
    groups = {}
    for item in labels:
        groups.setdefault(item['label'], []).append(item)

    result = []
    for label, items in groups.items():
        items = sorted(items, key=lambda x: x.get('score', 0), reverse=True)
        kept = []
        for item in items:
            too_close = any(
                np.hypot(item['cx'] - k['cx'], item['cy'] - k['cy']) < min_dist
                for k in kept
            )
            if not too_close:
                kept.append(item)
        result.extend(kept)
    return sorted(result, key=lambda x: (x['cy'], x['cx']))


def process_image(engine: RapidOCR, image_path: Path, output_dir: Path | None = None) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 阶段 1：粗 OCR 定位候选区域
    coarse_result, elapse = engine(str(image_path))
    regions = cluster_ocr_boxes(coarse_result)

    # 阶段 2：对每个候选区域放大后二次 OCR
    all_zoomed = []
    region_bboxes = []
    for region in regions:
        # 只处理可能是标签的区域（包含字母或数字）
        if not any(re.search(r'[A-Z0-9]', clean_text(t)) for t in region['raw_texts']):
            continue
        zoomed, bbox = zoom_ocr(engine, image, region, scale=3.0)
        # 把 zoomed 的相对坐标转换回原图坐标
        x1, y1, x2, y2 = bbox
        for item in zoomed:
            pts, text, score = item
            new_pts = [[p[0] + x1, p[1] + y1] for p in pts]
            all_zoomed.append((new_pts, text, score))
        region_bboxes.append(bbox)

    # 合并粗 OCR 和二次 OCR 结果
    combined = coarse_result + all_zoomed
    labels = pair_prefix_number(combined)
    labels = deduplicate_labels(labels)

    output = {
        'image': str(image_path),
        'label_count': len(labels),
        'labels': labels,
        'region_count': len(regions),
        'elapsed_ms': {
            'detection': round(elapse[0] * 1000, 1) if len(elapse) > 0 else None,
            'classification': round(elapse[1] * 1000, 1) if len(elapse) > 1 else None,
            'recognition': round(elapse[2] * 1000, 1) if len(elapse) > 2 else None,
        },
    }

    if output_dir is not None:
        vis = image.copy()
        for item in labels:
            cx, cy = int(item['cx']), int(item['cy'])
            cv2.circle(vis, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(vis, item['label'], (cx + 14, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 画出二次 OCR 的候选区域框（红色细框）
        for (x1, y1, x2, y2) in region_bboxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        vis_path = output_dir / f"{image_path.stem}_result_v2.jpg"
        cv2.imwrite(str(vis_path), vis)
        output['visualization'] = str(vis_path)

    return output


def main():
    parser = argparse.ArgumentParser(description='图书馆书脊标签离线识别 v2（粗检测 + 局部放大）')
    parser.add_argument('input', help='图片文件或图片目录')
    parser.add_argument('--visualize', action='store_true', help='生成带标注的可视化图到 output_v2/ 目录')
    parser.add_argument('--output', default='ocr_results_v2.json', help='JSON 结果输出路径')
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        for ext in exts:
            image_paths.extend(input_path.glob(ext))
        image_paths = [p for p in image_paths if not p.name.endswith('_result_v2.jpg')]
        image_paths = sorted(image_paths)
    else:
        image_paths = [input_path]

    output_dir = None
    if args.visualize:
        output_dir = Path('output_v2')
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
                print(f"     {item['label']}")
        except Exception as e:
            print(f"  -> 错误: {e}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {args.output}")
    if args.visualize:
        print(f"可视化图保存在: {output_dir}/")


if __name__ == '__main__':
    main()
