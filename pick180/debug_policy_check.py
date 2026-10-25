#!/usr/bin/env python3
"""
[roundtrip 用] 教材の scripts/debug_policy.py を元にした、方策のデバッグ(見本の観測を与えて、予測と見本の動作を比べる)。
変更点:
- 複数のエピソードをまとめて評価し、区間ごと(1回目に放すまで / 2回目に放すまで / そのあと)の平均誤差を出す。
- --time-scale: environment_state の時間の値(最後の列)だけを何倍かにして与える(推論が遅いときの観測を再現する)。
- --from-sec: その秒数以降の区間を別に集計する(見本の最後に足した「止まっている区間」などを見る)。
- 図は画面に出さず、--out に保存する。
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy

GRIP_THR = -0.2  # ハンドの指令がこれより大きければ「閉じる」


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--repo-id', required=True)
    p.add_argument('--policy-path', required=True)
    p.add_argument('--episodes', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--time-scale', type=float, default=1.0)
    p.add_argument('--from-sec', type=float, default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()

    ds = LeRobotDataset(a.repo_id, root=a.dataset_root)
    policy = ACTPolicy.from_pretrained(a.policy_path).to(a.device).eval()
    fps = ds.fps
    seg_err = {k: [] for k in ('1回目に放すまで', '2回目に放すまで', 'そのあと')}
    late, grip_miss, total = [], 0, 0
    plot = None
    for ep in a.episodes:
        policy.reset()
        lo, hi = ds.episode_data_index["from"][ep].item(), ds.episode_data_index["to"][ep].item()
        pred, gt = [], []
        for i in range(lo, hi):
            s = ds[i]
            env = s["observation.environment_state"].clone().float()
            env[-1] = env[-1] * a.time_scale
            obs = {"observation.state": s["observation.state"].float().unsqueeze(0).to(a.device),
                   "observation.environment_state": env.unsqueeze(0).to(a.device)}
            with torch.no_grad():
                pred.append(policy.select_action(obs).squeeze(0).cpu().numpy())
            gt.append(s["action"].float().numpy())
        pred, gt = np.array(pred), np.array(gt)
        err = np.linalg.norm(pred - gt, axis=1)
        # 見本の動作から、放した時刻(閉→開)を求めて区間に分ける
        closed = gt[:, 4] > GRIP_THR
        rel = [k for k in range(1, len(gt)) if closed[k - 1] and not closed[k]]
        b1 = rel[0] if rel else len(gt)
        b2 = rel[1] if len(rel) > 1 else len(gt)
        seg_err['1回目に放すまで'] += list(err[:b1])
        seg_err['2回目に放すまで'] += list(err[b1:b2])
        seg_err['そのあと'] += list(err[b2:])
        if a.from_sec is not None:
            late += list(err[int(a.from_sec * fps):])
        grip_miss += int(np.sum((pred[:, 4] > GRIP_THR) != closed)); total += len(gt)
        print(f"エピソード {ep}: {len(gt)} フレーム、平均誤差 {err.mean():.4f}、最大 {err.max():.4f}(t={err.argmax() / fps:.1f}s)、放した時刻 {[round(k / fps, 1) for k in rel]}")
        if plot is None:
            plot = (ep, pred, gt, err)
    print(f"時間の倍率 {a.time_scale}: 全体の平均誤差 {np.mean(sum(seg_err.values(), [])):.4f}、ハンドの開閉が見本と食い違うフレーム {grip_miss}/{total}")
    for k, v in seg_err.items():
        if v:
            print(f"  {k}: 平均誤差 {np.mean(v):.4f}、最大 {np.max(v):.4f}({len(v)} フレーム)")
    if late:
        print(f"  {a.from_sec}秒以降: 平均誤差 {np.mean(late):.4f}、最大 {np.max(late):.4f}({len(late)} フレーム)")
    if a.out and plot:
        ep, pred, gt, err = plot
        t = np.arange(len(gt)) / fps
        names = ['j1', 'j2', 'j3', 'j4', 'hand']
        fig, ax = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
        for j in range(5):
            ax[j].plot(t, gt[:, j], '--', label='GT')
            ax[j].plot(t, pred[:, j], label='pred')
            ax[j].set_ylabel(names[j])
        ax[0].legend(loc='upper right')
        ax[5].plot(t, err, color='k'); ax[5].set_ylabel('L2 error'); ax[5].set_xlabel('time [s]')
        fig.suptitle(f"{a.policy_path.split('/')[-1]}  episode {ep}  time x{a.time_scale}")
        fig.tight_layout(); fig.savefig(a.out, dpi=80)
        print(f"図: {a.out}")


if __name__ == "__main__":
    main()
