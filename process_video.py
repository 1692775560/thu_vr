"""
处理视频：逐帧 OCR + 可视化标签 + 输出带预测视频 + 测速。

流程：
1. 读取视频每一帧；
2. 对整图做多尺度 OCR（RapidOCR）；
3. 对检测到的文字框做局部精识别；
4. 碎片组合成标签；
5. 在帧上绘制标签和检测框；
6. 写入输出视频；
7. 统计每帧耗时、总耗时、FPS。
"""
import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import imageio
from rapidocr_onnxruntime import RapidOCR


LABEL_RE = re.compile(r'\b([A-Z])(\d{2})-(\d{3,4})\b')
PREFIX_RE = re.compile(r'^([A-Z])(\d{2})-?$')
NUM_RE = re.compile(r'^(\d{3,4})$')


def fuzzy_clean(text: str) -> str:
    """通用清洗：保留字母和数字，用于完整标签匹配。"""
    t = text.upper()
    t = re.sub(r'[^A-Z0-9\-]', '', t)
    # 只替换最容易混淆的 O→0，其他字母保留给前缀
    return t.replace('O', '0')


def clean_number(text: str) -> str:
    """数字专用清洗：允许把 OCR 误读字母转回数字。"""
    t = fuzzy_clean(text)
    return t.replace('A', '4').replace('I', '1').replace('S', '5').replace('Z', '2')


def resize_linear(img: np.ndarray, scale: float):
    return cv2.resize(img, None, fx=scale, fy=scale)


def whole_image_ocr(engine, image: np.ndarray, scales=(1.5, 2.0)):
    results = []
    for s in scales:
        big = resize_linear(image, s)
        out = engine(big)
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


def refine_text_regions(engine, image: np.ndarray, texts: list[dict], scales=(10, 15)):
    refined = []
    seen = set()
    h, w = image.shape[:2]
    # 只 refine 前 8 个最可能是标签的候选（按 score 排序）
    sorted_texts = sorted(texts, key=lambda t: t.get('score', 0), reverse=True)[:8]
    for item in sorted_texts:
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
            out = engine(big)
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
        cn = clean_number(item['text'])
        center = item.get('center')
        if center is None:
            continue
        # 完整标签匹配
        for m in LABEL_RE.finditer(c):
            fragments.append({
                'label': f"{m.group(1)}{m.group(2)}-{m.group(3).zfill(4)}",
                'score': item['score'],
                'raw': item['text'],
                'center': center,
            })
        # 前缀碎片：用保守清洗（不把字母转数字）
        m = PREFIX_RE.search(c)
        if m:
            fragments.append({
                'kind': 'prefix', 'letter': m.group(1), 'prefix_num': m.group(2),
                'score': item['score'], 'raw': item['text'], 'center': center,
            })
        # 数字碎片：先匹配原始文本中的纯数字；如果该文本是前缀碎片，则跳过
        m = NUM_RE.search(item['text'].upper())
        if not m and not PREFIX_RE.match(c):
            m = NUM_RE.search(cn)
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


def process_frame(engine, frame: np.ndarray):
    texts = whole_image_ocr(engine, frame)
    texts.extend(refine_text_regions(engine, frame, texts))
    labels = extract_labels(texts)
    labels.extend(combine_fragments(texts))

    best_label = None
    if labels:
        from collections import Counter
        freq = Counter([l['label'] for l in labels])
        best_label = max(labels, key=lambda l: (freq[l['label']], l['score']))

    return texts, labels, best_label


def draw_results(frame: np.ndarray, texts, labels, best_label):
    vis = frame.copy()
    # 画出所有 OCR 碎片（黄色小字）
    for t in texts:
        if 'center' in t:
            cx, cy = int(t['center'][0]), int(t['center'][1])
            cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)
            cv2.putText(vis, t['text'], (cx + 5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 对 labels 做空间去重（同一位置只保留最高分的）
    seen = {}
    for l in labels:
        cx, cy = int(l['center'][0]), int(l['center'][1])
        key = (cx // 30, cy // 30)  # 30px 网格去重
        if key not in seen or l['score'] > seen[key]['score']:
            seen[key] = l

    # 画出去重后的标签（红色大字）
    for l in seen.values():
        cx, cy = int(l['center'][0]), int(l['center'][1])
        cv2.putText(vis, l['label'], (cx - 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if best_label:
        cv2.putText(vis, f"BEST: {best_label['label']}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    return vis


def main():
    parser = argparse.ArgumentParser(description='视频逐帧 OCR + 输出带预测视频')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('--output-video', default=None, help='输出视频路径')
    parser.add_argument('--output-json', default=None, help='每帧 OCR 结果 JSON')
    parser.add_argument('--frame-skip', type=int, default=3, help='跳帧处理，1 表示每帧都处理，3 表示每 3 帧处理一次')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output_video = args.output_video or str(video_path.parent / f"{video_path.stem}_predicted.mp4")
    output_json = args.output_json or str(video_path.parent / f"{video_path.stem}_predicted.json")

    print('加载 RapidOCR...')
    engine = RapidOCR()

    cap = cv2.VideoCapture(str(video_path))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'输入: {video_path.name}, {w}x{h}, {fps_in:.2f} fps, {total_frames} 帧')

    # macOS 上 mp4v 有时不稳定，优先尝试 H264，失败回退到 mp4v
    # 用 imageio 写 mp4，比 cv2.VideoWriter 在 macOS 上稳定
    frames_out = []

    frame_results = []
    times = []
    idx = 0
    processed = 0
    success = 0

    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % args.frame_skip != 0:
            frames_out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
            continue

        t0 = time.time()
        texts, labels, best_label = process_frame(engine, frame)
        elapsed = time.time() - t0
        times.append(elapsed)
        processed += 1
        if best_label:
            success += 1

        vis = draw_results(frame, texts, labels, best_label)
        # 在左上角显示帧耗时和进度
        cv2.putText(vis, f"frame {idx+1}/{total_frames} {elapsed:.2f}s",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frames_out.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        frame_results.append({
            'frame_idx': idx,
            'elapsed_s': elapsed,
            'best_label': best_label,
            'labels': labels,
            'texts': texts,
        })

        if processed % 5 == 0 or idx == total_frames - 1:
            print(f'  frame {idx+1}/{total_frames} elapsed={elapsed:.2f}s best={best_label}', flush=True)

        idx += 1

    cap.release()

    print('写入视频...')
    imageio.mimwrite(output_video, frames_out, fps=fps_in, codec='libx264')

    total_time = time.time() - t_start
    avg_frame_time = sum(times) / max(len(times), 1)
    fps_proc = processed / total_time if total_time > 0 else 0

    stats = {
        'input_video': str(video_path),
        'output_video': output_video,
        'total_frames': total_frames,
        'processed_frames': processed,
        'success_frames': success,
        'total_time_s': total_time,
        'avg_frame_time_s': avg_frame_time,
        'effective_fps': fps_proc,
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({'stats': stats, 'frames': frame_results}, f, ensure_ascii=False, indent=2)

    print('\n=== 处理完成 ===')
    print(f'输出视频: {output_video}')
    print(f'结果 JSON: {output_json}')
    print(f'总帧数: {total_frames}, 处理帧数: {processed}, 成功识别: {success}')
    print(f'总耗时: {total_time:.2f}s, 平均每帧: {avg_frame_time:.2f}s, 处理 FPS: {fps_proc:.2f}')


if __name__ == '__main__':
    main()
