"""
基于当前 OCR 结果中的成功帧，自动生成书的伪标签，准备 YOLO 训练数据。
"""
import json
import shutil
from pathlib import Path

import cv2


def main():
    json_path = Path('/Users/wujie/Desktop/code/图书馆视觉/真实场景图/output/real_scene_results.json')
    out_dir = Path('/Users/wujie/Desktop/code/图书馆视觉/yolo_book_dataset')
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / 'images').mkdir(exist_ok=True)
    (out_dir / 'labels').mkdir(exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    count = 0
    for r in results:
        if not r['best_label']:
            continue
        img_path = Path(r['image'])
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        cx, cy = r['best_label']['center']

        # 以标签中心为书脊中心，估算书的位置
        # 竖直书，宽度约 60px，高度约 180px（在 1280x720 图中）
        bw, bh = 60, 180
        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(w, int(cx + bw / 2))
        y2 = min(h, int(cy + bh / 2))

        # YOLO 格式：class cx cy w h（归一化）
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        # 复制图片
        dst_img = out_dir / 'images' / img_path.name
        shutil.copy(img_path, dst_img)

        # 写入标注
        label_name = img_path.stem + '.txt'
        with open(out_dir / 'labels' / label_name, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        count += 1

    print(f'生成了 {count} 张伪标签图片到 {out_dir}')


if __name__ == '__main__':
    main()
