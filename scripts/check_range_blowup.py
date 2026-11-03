# ==============================================================
# [新規] 生成済み C コードに範囲解析爆発フィルタを当てて確認するスクリプト
#
# 使い方 (scripts ディレクトリで):
#   python check_range_blowup.py <ディレクトリ or .c ファイル> [...]
#     例: python check_range_blowup.py ../experiments/sec_eval/350m-lora-single-ep7-r8-a16-sec-4
#   -v を付けると全ファイルの重みを表示
#
# 除外対象 (重み >= MAX_RANGE_WEIGHT) と、重みが大きい順の上位 15 件を表示する。
# 上位に「除外されないが重みが大きい」ファイルがあれば、しきい値の余裕の確認に使える。
# ==============================================================
import os
import sys

from range_blowup_filter import max_range_weight, MAX_RANGE_WEIGHT


def iter_c_files(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for dirpath, _, filenames in os.walk(p):
            for fn in sorted(filenames):
                if fn.endswith('.c'):
                    yield os.path.join(dirpath, fn)


def main():
    args = [a for a in sys.argv[1:] if a != '-v']
    verbose = '-v' in sys.argv[1:]
    if not args:
        args = ['../experiments/sec_eval']

    results = []
    for path in iter_c_files(args):
        with open(path, errors='replace') as f:
            src = f.read()
        var, w = max_range_weight(src)
        results.append((w, var, path))

    print(f'rule: 範囲の重み >= {MAX_RANGE_WEIGHT} -> 除外')
    if verbose:
        for w, var, path in results:
            mark = 'EXCLUDE' if w >= MAX_RANGE_WEIGHT else 'ok     '
            print(f'[{mark}] weight={w:>6} var={var}  {path}')
        print()

    excluded = [r for r in results if r[0] >= MAX_RANGE_WEIGHT]
    print(f'scanned C files: {len(results)}, excluded: {len(excluded)}')
    for w, var, path in excluded:
        print(f'  [EXCLUDE] weight={w:>6} var={var}  {path}')

    print('\ntop 15 by weight:')
    for w, var, path in sorted(results, key=lambda r: -r[0])[:15]:
        mark = 'EXCLUDE' if w >= MAX_RANGE_WEIGHT else 'ok     '
        print(f'  [{mark}] weight={w:>6} var={var}  {path}')


if __name__ == '__main__':
    main()