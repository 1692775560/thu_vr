"""
汇总真实场景 OCR 结果，输出可读报告。
"""
import json
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='/Users/wujie/Desktop/code/图书馆视觉/真实场景图/output/real_scene_results.json')
    args = parser.parse_args()

    with open(args.json, 'r', encoding='utf-8') as f:
        results = json.load(f)

    total = len(results)
    success = sum(1 for r in results if r['best_label'])
    print(f'总样本数: {total}')
    print(f'成功识别: {success} ({success/total*100:.1f}%)')
    print('\n成功样本:')
    for r in results:
        if r['best_label']:
            print(f"  {Path(r['image']).name}: {r['best_label']['label']} "
                  f"(score={r['best_label']['score']:.3f}, raw={r['best_label'].get('raw', '')})")

    print('\n失败样本:')
    for r in results:
        if not r['best_label']:
            print(f"  {Path(r['image']).name}: 候选 {r['candidate_count']} 个")


if __name__ == '__main__':
    main()
