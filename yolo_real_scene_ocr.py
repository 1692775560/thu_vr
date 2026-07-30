"""
YOLO 检测书脊 + OCR 识别标签的端到端流程。

步骤：
1. YOLO 检测书的位置（低置信度阈值以召回小目标）；
2. 裁剪出书脊区域并放大；
3. 对书脊区域做整图 OCR + 局部精识别；
4. 碎片组合并输出标签。
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO


def load_image(path: Path):
    return cv2.imread(str(path))


def resize_linear(img: np.ndarray, scale: float):
    return cv2.resize(img, None, fx=scale, fy=scale)


def whole_image_ocr(engine, image: np.ndarray, scales=(1.5, 2.0, 2.5)):
    """对整图只做简单放大，然后 OCR。"""
    results = []
    for s in scales:
        big = resize_linear(image, s)
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


def refine_text_regions(engine, image: np.ndarray, texts: list[dict], scales=(10, 15, 20)):
    """对检测到的文字框扩展后精识别。"""
    refined = []
    seen = set()
    h, w = image.shape[:2]
    for item in texts:
        if 'center' not in item:
            continue
        cx, cy = item['center']
        box_h = 40
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
            big = resize_linear(crop, s)
            cv2.imwrite('/tmp/refine.jpg', big)
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


import re
LABEL_RE = re.compile(r'\b([A-Z])(\d{2})-(\d{3,4})\b')
PREFIX_RE = re.compile(r'^([A-Z])(\d{2})-?$')
NUM_RE = re.compile(r'^(\d{3,4})$')


def fuzzy_clean(text: str) -> str:
    t = text.upper()
    t = re.sub(r'[^A-Z0-9\-]', '', t)
    t = t.replace('O', '0')
    t = t.replace('A', '4')
    t = t.replace('I', '1')
    t = t.replace('S', '5')
    t = t.replace('Z', '2')
    return t


def extract_labels(texts):
    labels = []
    for item in texts:
        c = fuzzy_clean(item['text'])
        for m in LABEL_RE.finditer(c):
            labels.append({
                'label': f"{m.group(1)}{m.group(2)}-{m.group(3).zfill(4)}",
                'score': item['score'],
                'raw_text': item['text'],
                'center': item.get('center'),
            })
    return labels


def combine_fragments(texts, distance_thresh=120.0):
    fragments = []
    for item in texts:
        c = fuzzy_clean(item['text'])
        center = item.get('center')
        if center is None:
            continue
        for m in LABEL_RE.finditer(c):
            fragments.append({
                'label': f"{m.group(1)}{m.group(2)}-{m.group(3).zfill(4)}",
                'score': item['score'],
                'raw': item['text'],
                'center': center,
            })
        m = PREFIX_RE.search(c)
        if m:
            fragments.append({
                'kind': 'prefix', 'letter': m.group(1), 'prefix_num': m.group(2),
                'score': item['score'], 'raw': item['text'], 'center': center,
            })
        m = NUM_RE.search(c)
        if m:
            fragments.append({
                'kind': 'number', 'code': m.group(1),
                'score': item['score'], 'raw': item['text'], 'center': center,
            })

    labels = [f for f in fragments if 'label' in f]
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


def process_image_with_yolo(yolo_model, ocr_engine, image_path: Path, output_dir: Path, conf_thr=0.05):
    image = load_image(image_path)
    h, w = image.shape[:2]

    # YOLO 检测书
    results = yolo_model(str(image_path), verbose=False, conf=conf_thr)
    boxes = results[0].boxes

    all_texts = []
    book_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        # 对书框加 padding，避免标签被切掉
        pad_x = max(10, int((x2 - x1) * 0.15))
        pad_y = max(20, int((y2 - y1) * 0.25))
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))
        book_boxes.append((x1, y1, x2, y2))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        texts = whole_image_ocr(ocr_engine, crop, scales=(1.5, 2.0, 2.5))
        # 坐标映射回原图
        for t in texts:
            cx, cy = t['center']
            t['center'] = (cx + x1, cy + y1)
        refined = refine_text_regions(ocr_engine, crop, texts, scales=(10, 15, 20))
        for t in refined:
            cx, cy = t['center']
            t['center'] = (cx + x1, cy + y1)
        all_texts.extend(texts)
        all_texts.extend(refined)

    labels = extract_labels(all_texts)
    labels.extend(combine_fragments(all_texts))

    best_label = None
    if labels:
        from collections import Counter
        freq = Counter([l['label'] for l in labels])
        best_label = max(labels, key=lambda l: (freq[l['label']], l['score']))

    # 可视化
    vis = image.copy()
    for x1, y1, x2, y2 in book_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for t in all_texts:
        if 'center' in t:
            cx, cy = int(t['center'][0]), int(t['center'][1])
            cv2.circle(vis, (cx, cy), 3, (255, 255, 0), -1)
            cv2.putText(vis, t['text'], (cx+5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    if best_label:
        cv2.putText(vis, f"BEST: {best_label['label']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    vis_path = output_dir / f"{image_path.stem}_yolo_result.jpg"
    cv2.imwrite(str(vis_path), vis)

    return {
        'image': str(image_path),
        'book_boxes': book_boxes,
        'best_label': best_label,
        'all_labels': labels,
        'all_texts': all_texts,
        'visualization': str(vis_path),
    }


def main():
    parser = argparse.ArgumentParser(description='YOLO+OCR 书脊标签识别')
    parser.add_argument('input', help='图片、目录或视频文件')
    parser.add_argument('--output-dir', default=None, help='输出目录')
    parser.add_argument('--video-step', type=int, default=5, help='视频抽帧间隔')
    parser.add_argument('--conf', type=float, default=0.05, help='YOLO 置信度阈值')
    parser.add_argument('--yolo-model', default='/Users/wujie/Desktop/code/图书馆视觉/runs/detect/train/weights/best.pt', help='YOLO 模型路径')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / 'output'
    output_dir.mkdir(exist_ok=True, parents=True)

    print('加载 YOLO 模型...')
    yolo_model = YOLO(args.yolo_model)
    print('加载 RapidOCR...')
    ocr_engine = RapidOCR()

    all_results = []

    def process_one(p: Path):
        print(f"处理: {p.name}")
        r = process_image_with_yolo(yolo_model, ocr_engine, p, output_dir, conf_thr=args.conf)
        all_results.append(r)
        print(f"  -> 书框 {len(r['book_boxes'])}, 最佳标签: {r['best_label']}")

    if input_path.is_dir():
        for p in sorted(input_path.iterdir()):
            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                process_one(p)
        for p in sorted(input_path.iterdir()):
            if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi'):
                cap = cv2.VideoCapture(str(p))
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx % args.video_step == 0:
                        frame_path = output_dir / f"{p.stem}_frame_{idx:04d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        process_one(frame_path)
                    idx += 1
                cap.release()
    elif input_path.suffix.lower() in ('.mp4', '.mov', '.avi'):
        cap = cv2.VideoCapture(str(input_path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % args.video_step == 0:
                frame_path = output_dir / f"{input_path.stem}_frame_{idx:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                process_one(frame_path)
            idx += 1
        cap.release()
    else:
        process_one(input_path)

    json_path = output_dir / 'yolo_ocr_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {json_path}")


if __name__ == '__main__':
    main()
