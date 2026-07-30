"""
真实场景书脊标签识别实验脚本
对比 RapidOCR / EasyOCR 在不同预处理下的效果
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

LABEL_RE = re.compile(r'\b([A-Z])(\d{2})-(\d{3,4})\b')


def load_image(path: Path):
    return cv2.imread(str(path))


def enhance(img: np.ndarray, scale: float = 6.0, sharp: bool = True, clahe: bool = True, gray: bool = True):
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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


def try_rapid(rapid, img: np.ndarray, scales=(4, 6, 8, 10), rotations=(0, 90, 270)):
    results = []
    for deg in rotations:
        if deg == 0:
            base = img
        elif deg == 90:
            base = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif deg == 270:
            base = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for s in scales:
            proc = enhance(base, scale=s)
            path = f'/tmp/rapid_{deg}_{s}.jpg'
            cv2.imwrite(path, proc)
            out = rapid(path)
            if out and out[0]:
                for pts, text, score in out[0]:
                    results.append({'engine': 'rapid', 'rot': deg, 'scale': s, 'text': text, 'score': score})
    return results


def try_easy(easy, img: np.ndarray, scales=(4, 6, 8, 10), rotations=(0, 90, 270)):
    results = []
    for deg in rotations:
        if deg == 0:
            base = img
        elif deg == 90:
            base = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif deg == 270:
            base = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for s in scales:
            proc = enhance(base, scale=s)
            # EasyOCR 输入 numpy array (RGB)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            outs = easy.readtext(rgb)
            for bbox, text, conf in outs:
                results.append({'engine': 'easy', 'rot': deg, 'scale': s, 'text': text, 'score': conf})
    return results


def extract_labels(results):
    labels = []
    for r in results:
        c = re.sub(r'[^A-Z0-9-]', '', r['text'].upper()).replace('O', '0')
        m = LABEL_RE.search(c)
        if m:
            labels.append({
                'label': f"{m.group(1)}{m.group(2)}-{m.group(3).zfill(4)}",
                **r,
                'cleaned': c,
            })
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='输入图片路径')
    parser.add_argument('--x', type=int, default=None)
    parser.add_argument('--y', type=int, default=None)
    parser.add_argument('--w', type=int, default=None)
    parser.add_argument('--h', type=int, default=None)
    args = parser.parse_args()

    img = load_image(Path(args.image))
    if args.x is not None:
        crop = img[args.y:args.y+args.h, args.x:args.x+args.w]
    else:
        crop = img

    print('crop shape:', crop.shape)
    cv2.imwrite('/tmp/experiment_crop.jpg', crop)

    rapid = RapidOCR()
    rapid_res = try_rapid(rapid, crop)
    print('\n--- RapidOCR raw ---')
    for r in rapid_res:
        print(r)
    print('\n--- RapidOCR labels ---')
    for l in extract_labels(rapid_res):
        print(l)

    if EASYOCR_AVAILABLE:
        easy = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        easy_res = try_easy(easy, crop)
        print('\n--- EasyOCR raw ---')
        for r in easy_res:
            print(r)
        print('\n--- EasyOCR labels ---')
        for l in extract_labels(easy_res):
            print(l)
    else:
        print('EasyOCR not available')


if __name__ == '__main__':
    main()
