"""
对视频多帧 OCR 结果做时空投票，提高标签识别稳定性。

思路：
- 同一本书在相邻帧中位置变化很小；
- 把不同帧的 OCR 检测框按空间距离聚类；
- 对每个空间簇里的候选标签投票，取出现次数最多且平均 score 最高的标签。
"""
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


def cluster_boxes(all_items: list[dict], distance_thresh: float = 60.0):
    """把所有帧的 OCR 框按空间距离聚类。"""
    clusters = []
    used = set()
    for i, item in enumerate(all_items):
        if i in used:
            continue
        cx, cy = item['center']
        cluster = [item]
        used.add(i)
        queue = [item]
        while queue:
            cur = queue.pop(0)
            cur_cx, cur_cy = cur['center']
            for j, other in enumerate(all_items):
                if j in used:
                    continue
                ocx, ocy = other['center']
                dist = ((cur_cx - ocx) ** 2 + (cur_cy - ocy) ** 2) ** 0.5
                if dist < distance_thresh:
                    cluster.append(other)
                    used.add(j)
                    queue.append(other)
        clusters.append(cluster)
    return clusters


def vote_cluster(cluster: list[dict]):
    """对一个空间簇里的标签投票。"""
    from collections import Counter
    labels = []
    for item in cluster:
        c = re.sub(r'[^A-Z0-9\-]', '', item['text'].upper())
        c = c.replace('O', '0').replace('A', '4').replace('I', '1').replace('S', '5').replace('Z', '2')
        m = re.search(r'\b([A-Z])(\d{2})-(\d{3,4})\b', c)
        if m:
            labels.append({
                'label': f"{m.group(1)}{m.group(2)}-{m.group(3).zfill(4)}",
                'score': item['score'],
                'image': item.get('image'),
            })
    if not labels:
        return None
    freq = Counter(l['label'] for l in labels)
    # 按频次、平均 score 排序
    avg_score = defaultdict(list)
    for l in labels:
        avg_score[l['label']].append(l['score'])
    best_label = max(labels, key=lambda l: (freq[l['label']], sum(avg_score[l['label']]) / len(avg_score[l['label']])))
    return {
        'label': best_label['label'],
        'votes': freq[best_label['label']],
        'total': len(labels),
        'avg_score': sum(avg_score[best_label['label']]) / len(avg_score[best_label['label']]),
        'center': tuple(np.mean([it['center'] for it in cluster], axis=0).tolist()),
    }


def main():
    json_path = Path('/Users/wujie/Desktop/code/图书馆视觉/真实场景图/output/real_scene_results.json')
    out_path = json_path.parent / 'voted_results.json'

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 收集所有帧的所有文本框（优先用 all_labels 里的完整标签，其次用 all_texts 碎片）
    all_items = []
    for r in results:
        img = r['image']
        for l in r.get('all_labels', []):
            item = {
                'text': l['label'],
                'score': l['score'],
                'center': l.get('center') or l.get('box_center'),
                'image': img,
            }
            if item['center']:
                all_items.append(item)
        # 如果没有完整标签，也加入原始碎片用于空间聚类
        if not r.get('all_labels'):
            for t in r.get('all_texts', []):
                if 'center' in t:
                    all_items.append({**t, 'image': img})

    clusters = cluster_boxes(all_items, distance_thresh=80.0)
    print(f'共 {len(clusters)} 个空间簇')

    voted = []
    for i, cluster in enumerate(clusters):
        v = vote_cluster(cluster)
        if v:
            v['cluster_id'] = i
            v['box_count'] = len(cluster)
            voted.append(v)

    voted = sorted(voted, key=lambda x: x['votes'], reverse=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(voted, f, ensure_ascii=False, indent=2)

    print(f'\n投票结果已保存到: {out_path}')
    print('Top voted labels:')
    for v in voted[:10]:
        print(f"  {v['label']}: votes={v['votes']}/{v['total']} center=({v['center'][0]:.0f}, {v['center'][1]:.0f})")


if __name__ == '__main__':
    main()
