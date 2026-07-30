"""
双资源实验入口 — 对比单网络 vs 多模块市场在双资源冲突环境下的表现

用法:
    python main_dual.py                  # 完整对比实验
    python main_dual.py --mode single    # 仅跑单网络
    python main_dual.py --mode market    # 仅跑市场
    python main_dual.py --antitrust_off  # 关闭反垄断做对照
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import DualConfig
from train_dual import (train_dual_single, train_dual_market,
                         evaluate_dual_single, evaluate_dual_market)


def run_dual_experiment(config, mode="both", verbose=True):
    print("=" * 60)
    print("  双资源生存博弈 — 内部思想市场")
    print(f"  环境: {config.grid_size}x{config.grid_size}, "
          f"食物x{config.n_foods}, 水源x{config.n_waters}, 危险x{config.n_dangers}")
    print(f"  耗尽: 能量{config.step_energy_cost}/步, 水{config.step_water_cost}/步")
    print(f"  训练: {config.n_episodes} 回合, seed={config.seed}")
    if config.n_modules >= 4:
        print(f"  模块: 食物偏(1.5) | 水源偏(1.5) | 避险偏(1.5) | 平衡(1.2)")
    print(f"  反垄断: {'开' if config.antitrust_enabled else '关'}")
    print("=" * 60)

    results = {}

    if mode in ("single", "both"):
        print("\n>>> 模式 1: 单网络基线 (Single Agent)")
        t0 = time.time()
        agent, hist_s, eval_s = train_dual_single(config, verbose=verbose)
        t1 = time.time()
        print(f"\n  [评估结果 - Single]")
        print(f"    平均奖励:     {eval_s['avg_reward']:8.1f} ± {eval_s['std_reward']:.1f}")
        print(f"    平均食物/水:   {eval_s['avg_foods']:.1f} / {eval_s['avg_waters']:.1f}")
        print(f"    平均生存步数:  {eval_s['avg_steps']:8.1f} ± {eval_s['std_steps']:.1f}")
        print(f"    存活率(满步):  {eval_s['survival_rate']:8.1%}")
        print(f"    训练耗时:      {t1-t0:8.1f}s")
        results["single"] = {
            "eval": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in eval_s.items()},
            "train_time": round(t1 - t0, 1),
        }

    if mode in ("market", "both"):
        print("\n>>> 模式 2: 多模块竞标市场 (Market)")
        t0 = time.time()
        modules, market, hist_m, eval_m = train_dual_market(config, verbose=verbose)
        t1 = time.time()
        print(f"\n  [评估结果 - Market]")
        print(f"    平均奖励:     {eval_m['avg_reward']:8.1f} ± {eval_m['std_reward']:.1f}")
        print(f"    平均食物/水:   {eval_m['avg_foods']:.1f} / {eval_m['avg_waters']:.1f}")
        print(f"    平均生存步数:  {eval_m['avg_steps']:8.1f} ± {eval_m['std_steps']:.1f}")
        print(f"    存活率(满步):  {eval_m['survival_rate']:8.1%}")
        print(f"    训练耗时:      {t1-t0:8.1f}s")

        stats = market.get_stats()
        print(f"\n  [模块状态]")
        print(f"    活跃: {stats['active_modules']}/{config.n_modules}  "
              f"均资:{stats['avg_capital']:.1f} 均誉:{stats['avg_reputation']:.3f}  "
              f"革命:{stats.get('revolutions',0)}次")
        for d in stats["module_details"]:
            s = "●" if d["active"] else "○"
            print(f"    {s} M{d['id']}: 资本={d['capital']:6.2f} 信誉={d['reputation']:.3f} "
                  f"赢/输={d['wins']:3d}/{d['losses']:3d} 被选={d['selected']:4d}")

        if "module_selections" in eval_m:
            sel = eval_m["module_selections"]
            total_sel = sum(sel.values())
            if total_sel > 0:
                print(f"\n  [评估模块选择分布]")
                for mid in sorted(sel.keys()):
                    pct = sel[mid] / total_sel * 100
                    bar = "█" * int(pct / 5)
                    label = ["食物偏","水源偏","避险偏","平衡"][mid] if mid < 4 else f"M{mid}"
                    print(f"    {label:6s} M{mid}: {sel[mid]:4d} ({pct:5.1f}%) {bar}")

        # 窗口分布
        if stats.get("window_distribution"):
            wd = stats["window_distribution"]
            print(f"\n  [反垄断窗口] 冷却={stats.get('cooldown',0)}步")
            for mid in sorted(wd.keys()):
                label = ["食物偏","水源偏","避险偏","平衡"][mid] if mid < 4 else f"M{mid}"
                pct = wd[mid] * 100
                bar = "█" * int(pct / 5)
                print(f"    {label:6s}: {pct:5.1f}% {bar}")

        results["market"] = {
            "eval": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in eval_m.items() if k != "module_selections"},
            "market_stats": stats,
            "train_time": round(t1 - t0, 1),
        }

    if mode == "both":
        print("\n" + "=" * 60)
        print("  对比总结")
        print("=" * 60)
        s, m = results["single"]["eval"], results["market"]["eval"]
        print(f"  {'指标':<20} {'Single':>10} {'Market':>10} {'差值':>10}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
        for key, label in [("avg_reward", "平均奖励"), ("avg_steps", "平均步数"),
                            ("avg_foods", "平均食物"), ("avg_waters", "平均水"),
                            ("survival_rate", "存活率")]:
            sv, mv = s[key], m[key]
            diff = mv - sv
            if key == "survival_rate":
                print(f"  {label:<20} {sv:10.1%} {mv:10.1%} {diff:+10.1%}")
            else:
                print(f"  {label:<20} {sv:10.1f} {mv:10.1f} {diff:+10.1f}")
        print(f"  {'训练耗时(s)':<20} {results['single']['train_time']:10.1f} "
              f"{results['market']['train_time']:10.1f}")

        path = os.path.join(os.path.dirname(__file__), "results_dual.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  结果已保存至: {path}")

    return results


def main():
    p = argparse.ArgumentParser(description="双资源生存博弈实验")
    p.add_argument("--mode", default="both", choices=["single", "market", "both"])
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--antitrust_off", action="store_true", help="关闭反垄断做对照")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    config = DualConfig()
    if args.episodes:
        config.n_episodes = args.episodes
    if args.seed is not None:
        config.seed = args.seed
    if args.antitrust_off:
        config.antitrust_enabled = False

    run_dual_experiment(config, mode=args.mode, verbose=not args.quiet)


if __name__ == "__main__":
    main()
