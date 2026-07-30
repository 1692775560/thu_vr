"""
根据多帧投票结果，在每张原图上画出稳定标签。
"""
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    base_dir = Path('/Users/wujie/Desktop/code/图书馆视觉/真实场景图/output')
    with open(base_dir / 'voted_results.json', 'r', encoding='utf-8') as f:
        voted = json.load(f)
    with open(base_dir / 'real_scene_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 为每个 voted label 找到属于它的帧
    for v in voted:
        v['frames'] = []
        for r in results:
            for l in r.get('all_labels', []):
                cx, cy = l.get('center') or (0, 0)
                if l.get('label') == v['label']:
                    dist = ((cx - v['center'][0]) ** 2 + (cy - v['center'][1]) ** 2) ** 0.5
                    if dist < 100:
                        v['frames'].append(r['image'])

    # 在每张原图上画稳定标签
    for r in results:
        img_path = Path(r['image'])
        vis_path = base_dir / f"{img_path.stem}_voted.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for v in voted:
            if r['image'] in v['frames']:
                cx, cy = int(v['center'][0]), int(v['center'][1])
                cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(img, f"VOTED: {v['label']}", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(vis_path), img)

    print(f'已在 {len(results)} 张图上绘制投票标签，保存到 *_voted.jpg')


if __name__ == '__main__':
    main()
