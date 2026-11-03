# ==============================================================
# [新規] CodeQL 範囲解析の爆発を起こす C コードの判定
#
# 背景:
#   ArithmeticUncontrolled.ql (CWE-190) が使う範囲解析 (SimpleRangeAnalysis) は、
#   各変数の取りうる範囲を「右辺で参照している変数の範囲」から組み立てる。
#   右辺で変数を2回以上参照する代入が連鎖すると、範囲の候補が代入ごとに約2倍になり、
#   解析が終わらなくなる。生成コードでは次の2パターンを確認した。
#     (a) 同じ変数の自己更新の繰り返し (trial 2, 14.c)
#           n = n - (rand() % (n-1));   を 9 回
#     (b) 新しい変数への連鎖 (trial 4, 06.c)
#           int l = k + rand() % k;  int m2 = l + rand() % l;  ...
#
# 判定方法 (「範囲の重み」):
#   コードを上から順に見て、代入/初期化のたびに
#       重み(左辺) = max(1, 右辺で参照している変数の重みの合計 (重複も数える))
#   とする (複合代入 x op= ... は左辺 x の重みも足す)。
#   まだ代入されていない識別子 (引数・マクロ・グローバル等) の重みは 0。
#   右辺で2回参照すると重みが2倍になるので、重みは範囲候補数の目安になる。
#
# 実測 (n = n - (rand() % (n-1)); を k 回並べた最小コード、--threads=0):
#     k=2..5: 6s / k=6: 9s / k=7 (重み128): 48s / k=8 (重み256): 300s で終わらず
#   → 重みが MAX_RANGE_WEIGHT (=256) 以上のコードを除外する。
#   該当: 14.c の重み 512、06.c の重み 448。
#
#   Python は範囲解析を使わないので対象外。
#   (関数ごとの区別はしていないので、複数関数で同名変数を使うと重みがやや大きめに出る)
# ==============================================================
import re

MAX_RANGE_WEIGHT = 256
_WEIGHT_CAP = 10 ** 9

_ASSIGN_RE = re.compile(
    r'^(?:[A-Za-z_][\w\s\*]*?\s+\**)?'      # 型宣言部分 (int, unsigned long *, など) があれば読み飛ばす
    r'([A-Za-z_]\w*)\s*'                    # 代入先の変数名
    r'(\+=|-=|\*=|/=|%=|<<=|>>=|&=|\|=|\^=|=(?!=))'
    r'(.+)$',
    re.S,
)

# 右辺の識別子: 関数呼び出し (名前の直後が '(') とメンバ名 ('.' / '->' の直後) は除く
_IDENT_RE = re.compile(r'(?<![\w.>])([A-Za-z_]\w*)\b(?!\s*\()')

_KEYWORDS = {
    'return', 'if', 'for', 'while', 'else', 'do', 'switch', 'case', 'sizeof',
    'int', 'char', 'long', 'short', 'unsigned', 'signed', 'float', 'double',
    'void', 'const', 'static', 'struct', 'union', 'enum', 'volatile',
}


def _strip_comments_and_strings(src):
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    src = re.sub(r'//[^\n]*', ' ', src)
    src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)
    src = re.sub(r"'(?:\\.|[^'\\])*'", "''", src)
    src = re.sub(r'^\s*#[^\n]*', ' ', src, flags=re.M)   # プリプロセッサ行
    return src


def range_weights(src):
    """各変数の最終的な範囲の重みと、途中で到達した最大値を返す"""
    weights = {}
    max_var, max_w = None, 0
    code = _strip_comments_and_strings(src)
    # ',' でも区切って `int a = 1, b = a + a;` のような宣言にも対応する
    for stmt in re.split(r'[;{},]', code):
        stmt = stmt.strip()
        # for(...) / if(...) / while(...) の頭を落として中の代入だけ見る
        stmt = re.sub(r'^(?:for|if|while|else|return)\b\s*\(?', '', stmt).strip()
        m = _ASSIGN_RE.match(stmt)
        if not m:
            continue
        var, op, rhs = m.group(1), m.group(2), m.group(3)
        if var in _KEYWORDS:
            continue
        refs = [x for x in _IDENT_RE.findall(rhs) if x not in _KEYWORDS]
        w = sum(weights.get(x, 0) for x in refs)
        if op != '=':
            w += weights.get(var, 0)
        w = min(max(1, w), _WEIGHT_CAP)
        weights[var] = w
        if w > max_w:
            max_var, max_w = var, w
    return weights, max_var, max_w


def max_range_weight(src):
    _, var, w = range_weights(src)
    return var, w


def is_range_blowup(src, language, threshold=MAX_RANGE_WEIGHT):
    if language != 'c':
        return False
    return max_range_weight(src)[1] >= threshold