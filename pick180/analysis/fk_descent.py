#!/usr/bin/env python3
"""
推論の記録(roundtrip_infer.sh の TRACE_DIR で保存した trial_N.csv)から、関節角度で指先の位置を計算し(URDF の寸法による順運動学)、
指先が低い位置(既定 10cm 未満。環境変数 THR で変更)まで下りた区間ごとに、高さ・位置・ブロックまでの距離・ハンドを閉じたかを表示する。
ブロックのない所へつかみに行っていないか(空振り)の確認に使う。

使い方: THR=0.13 python3 analysis/fk_descent.py results/traces/rec_1317 results/traces/fail_ph2
"""
import csv,sys,glob,math
def tip(x):
    p2=x['j2']; p3=p2-x['j3']; p4=p3-x['j4']
    r=0.083*math.sin(p2)+0.0935*math.sin(p3)+0.045*math.sin(p4)
    z=0.0454+0.026+0.083*math.cos(p2)+0.0935*math.cos(p3)+0.045*math.cos(p4)
    return r*math.cos(x['j1']), r*math.sin(x['j1']), z
for d in sys.argv[1:]:
    print(d)
    for f in sorted(glob.glob(d+'/trial_*.csv')):
        r=[{k:float(v) for k,v in x.items()} for x in csv.DictReader(open(f))]
        for x in r: x['tx'],x['ty'],x['tz']=tip(x)
        # descents: contiguous runs with tz < thr
        thr=float(__import__("os").environ.get("THR","0.10")); ev=[]; i=0
        while i<len(r):
            if r[i]['tz']<thr:
                j=i
                while j<len(r) and r[j]['tz']<thr: j+=1
                seg=r[i:j]; m=min(seg,key=lambda x:x['tz'])
                dist=math.hypot(m['tx']-m['cube_x'],m['ty']-m['cube_y'])
                cl=max(x['a_hand'] for x in seg)>-0.2
                ev.append(f"t{seg[0]['t']:.1f}-{seg[-1]['t']:.1f} 高さ{m['tz']*100:.1f}cm 先端({m['tx']:+.3f},{m['ty']:+.3f}) ブロックまで{dist*100:.1f}cm {'閉' if cl else '開のみ'}")
                i=j
            else: i+=1
        print(' ',f.split('/')[-1],'\n    '+'\n    '.join(ev))
