#!/usr/bin/env python3
"""[roundtrip 用] 推論の記録(infer_pick180.py --trace の CSV)から、反対側に置いたあと、2回目につかむ前に、
アームが開始姿勢(4つの関節がすべて 0。追従の遅れを見込んで 0.15rad 以内)に戻ったかを判定する。
あわせて、元の位置に置いたあと開始姿勢に戻り、記録の最後までそこで止まっているか(終わったあとに動き出さないか)も判定する。WSL 側で実行する。
  使い方: python3 start_check.py trial_N.csv  → 「開始姿勢 OK (最大のずれ 0.03rad) / 終了後 OK (開始姿勢で 7.2秒停止)」のように1行表示する
"""
import csv, sys

r = [{k: float(v) for k, v in x.items()} for x in csv.DictReader(open(sys.argv[1]))]
closed = lambda x: x['hand'] > -0.2
i = 0
while i < len(r) and not closed(r[i]): i += 1                                  # 1回目につかむ
while i < len(r) and not (not closed(r[i]) and r[i]['j1'] > 1.0): i += 1        # 反対側で放す
j = i
while j < len(r) and not closed(r[j]): j += 1                                  # 2回目につかむ(なければ最後まで)
if i >= len(r):
    print('開始姿勢 NG (反対側に置けていない)'); sys.exit()
D = lambda x: max(abs(x[k]) for k in ('j1', 'j2', 'j3', 'j4'))
dev = min(D(x) for x in r[i:j])
msg = f"開始姿勢 {'OK' if dev < 0.15 else 'NG'} (最大のずれ {dev:.2f}rad){'' if j < len(r) else ' / 2回目のつかみなし'}"
# 終了後: 2回目につかんだあと、元の位置の側で放し、開始姿勢に着いてから最後までのずれ
k = j
while k < len(r) and closed(r[k]): k += 1                                     # 元の位置の側で放す
at = next((n for n in range(k, len(r)) if D(r[n]) < 0.15), None)
if j >= len(r) or k >= len(r) or at is None:
    msg += ' / 終了後 NG (開始姿勢に戻っていない)'
else:
    end_dev = max(D(x) for x in r[at:])
    msg += f" / 終了後 {'OK' if end_dev < 0.15 else 'NG'} (開始姿勢で {r[-1]['t'] - r[at]['t']:.1f}秒、最大のずれ {end_dev:.2f}rad)"
print(msg)
