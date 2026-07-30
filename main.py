"""
主实验入口 — 对比单网络 vs 多模块市场

用法:
    python main.py              # 使用默认配置运行完整实验
    python main.py --small      # 使用简化配置快速测试
    python main.py --mode both  # (默认) 两种模式都跑
    python main.py --mode single --episodes 300
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# 确保导入路径
sys.path.insert(0, os.path.dirname(__file__))

from config import Config, SmallConfig
from trainer import train_single, train_market


def run_experiment(config, mode="both", verbose=True):
    """运行实验"""
    print("=" * 56)
    print(f"  生存博弈实验 — 内部思想市场")
    print(f"  环境: {config.grid_size}x{config.grid_size} 网格, "
          f"{config.n_foods}食物, {config.n_dangers}危险")
    print(f"  训练: {config.n_episodes} 回合, seed={config.seed}")
    print("=" * 56)

    results = {}

    if mode in ("single", "both"):
        print("\n>>> 模式 1: 单网络基线 (Single Agent)")
        t0 = time.time()
        agent, hist_single, eval_single = train_single(config, verbose=verbose)
        t1 = time.time()
        print(f"\n  [评估结果 - Single]")
        print(f"    平均奖励:     {eval_single['avg_reward']:8.1f} ± {eval_single['std_reward']:.1f}")
        print(f"    平均生存步数:  {eval_single['avg_steps']:8.1f} ± {eval_single['std_steps']:.1f}")
        print(f"    平均最终能量:  {eval_single['avg_final_energy']:8.1f}")
        print(f"    存活率(满步):  {eval_single['survival_rate']:8.1%}")
        print(f"    训练耗时:      {t1-t0:8.1f}s")
        results["single"] = {
            "eval": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in eval_single.items()},
            "train_time": round(t1 - t0, 1),
        }

    if mode in ("market", "both"):
        print("\n>>> 模式 2: 多模块竞标市场 (Market)")
        t0 = time.time()
        modules, market, hist_market, eval_market = train_market(config, verbose=verbose)
        t1 = time.time()
        print(f"\n  [评估结果 - Market]")
        print(f"    平均奖励:     {eval_market['avg_reward']:8.1f} ± {eval_market['std_reward']:.1f}")
        print(f"    平均生存步数:  {eval_market['avg_steps']:8.1f} ± {eval_market['std_steps']:.1f}")
        print(f"    平均最终能量:  {eval_market['avg_final_energy']:8.1f}")
        print(f"    存活率(满步):  {eval_market['survival_rate']:8.1%}")
        print(f"    训练耗时:      {t1-t0:8.1f}s")

        # 模块分化统计
        stats = market.get_stats()
        print(f"\n  [模块状态]")
        print(f"    活跃模块: {stats['active_modules']}/{config.n_modules}")
        print(f"    平均资本: {stats['avg_capital']:.2f}")
        print(f"    平均信誉: {stats['avg_reputation']:.3f}")
        print(f"    总赢/总输: {stats['total_wins']}/{stats['total_losses']}")
        for d in stats["module_details"]:
            status = "●" if d["active"] else "○"
            print(f"    {status} M{d['id']}: 资本={d['capital']:6.2f} "
                  f"信誉={d['reputation']:.3f} "
                  f"赢/输={d['wins']:3d}/{d['losses']:3d} "
                  f"被选={d['selected']:3d}")

        # 反垄断/革命统计
        if "revolutions" in stats:
            print(f"\n  [反垄断机制]")
            print(f"    革命次数: {stats['revolutions']}")
            print(f"    冷却剩余: {stats.get('cooldown', 0)} 步")
            if stats.get("window_distribution"):
                wd = stats["window_distribution"]
                print(f"    当前窗口选择分布:")
                for mid in sorted(wd.keys()):
                    pct = wd[mid] * 100
                    bar = "█" * int(pct / 5)
                    print(f"      M{mid}: {pct:5.1f}% {bar}")

        # 评估中的模块选择分布
        if "module_selections" in eval_market:
            sel = eval_market["module_selections"]
            total_sel = sum(sel.values())
            if total_sel > 0:
                print(f"\n  [评估期模块选择分布]")
                for mid in sorted(sel.keys()):
                    pct = sel[mid] / total_sel * 100
                    bar = "█" * int(pct / 5)
                    print(f"    M{mid}: {sel[mid]:4d} ({pct:5.1f}%) {bar}")

        results["market"] = {
            "eval": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in eval_market.items() if k != "module_selections"},
            "market_stats": stats,
            "train_time": round(t1 - t0, 1),
        }

    # 对比总结
    if mode == "both":
        print("\n" + "=" * 56)
        print("  对比总结")
        print("=" * 56)
        s = results["single"]["eval"]
        m = results["market"]["eval"]
        print(f"  {'指标':<20} {'Single':>10} {'Market':>10} {'差值':>10}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
        print(f"  {'平均奖励':<20} {s['avg_reward']:10.1f} {m['avg_reward']:10.1f} "
              f"{m['avg_reward']-s['avg_reward']:+10.1f}")
        print(f"  {'平均生存步数':<20} {s['avg_steps']:10.1f} {m['avg_steps']:10.1f} "
              f"{m['avg_steps']-s['avg_steps']:+10.1f}")
        print(f"  {'存活率':<20} {s['survival_rate']:10.1%} {m['survival_rate']:10.1%} "
              f"{m['survival_rate']-s['survival_rate']:+10.1%}")
        print(f"  {'训练耗时(s)':<20} {results['single']['train_time']:10.1f} "
              f"{results['market']['train_time']:10.1f}")

        # 保存结果
        output_path = os.path.join(os.path.dirname(__file__), "results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  结果已保存至: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="生存博弈实验 — 多模块内部思想市场 vs 单网络基线"
    )
    parser.add_argument("--small", action="store_true",
                        help="使用简化配置快速测试")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["single", "market", "both"],
                        help="实验模式 (default: both)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="训练回合数")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--quiet", action="store_true",
                        help="减少打印输出")
    parser.add_argument("--modules", type=int, default=None,
                        help="市场模块数量")

    args = parser.parse_args()

    if args.small:
        config = SmallConfig()
        print("使用 SmallConfig 快速测试模式")
    else:
        config = Config()

    if args.episodes is not None:
        config.n_episodes = args.episodes
    if args.seed is not None:
        config.seed = args.seed
    if args.modules is not None:
        config.n_modules = args.modules

    run_experiment(config, mode=args.mode, verbose=not args.quiet)


if __name__ == "__main__":
    main()
