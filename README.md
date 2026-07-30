# Internal Agent Markets

**Why Internal Agent Markets Fail at Small Scale: Five Hypotheses Tested on a Multi-Objective GridWorld**

## 概述

一个最小可行的实验框架——多模块神经网络在内部市场（含资本、清算、反垄断革命）中竞争行动执行权。核心问题：**多个有差异化的"专家"通过内部竞争，能否产生优于单一统一网络的集体决策？**

## 文件结构

```
survival_market/
├── config.py          # 实验配置 (Config / SmallConfig / DualConfig)
├── env.py             # 单资源 GridWorld 环境
├── env_dual.py        # 双资源 GridWorld 环境 (能量 + 水)
├── modules.py         # 策略模块 (MLP + 资本/信誉 + 感知偏差)
├── modules_dual.py    # 双资源模块工厂 (乐观偏差 / 感知盲区)
├── market.py          # 内部市场 (仲裁 / 清算 / 复活 / 反垄断革命 / 可学习仲裁器)
├── trainer.py         # 单资源训练器 (PPO 批量训练)
├── train_dual.py      # 双资源训练器 (支持可学习仲裁器)
├── main.py            # 单资源实验入口
├── main_dual.py       # 双资源实验入口
├── requirements.txt
└── paper/
    └── paper.tex      # 预印本 LaTeX 源码
```

## 快速开始

```bash
pip install -r requirements.txt

# 单资源实验 (快速)
python main.py --small --mode both

# 双资源实验 (含可学习仲裁器)
python main_dual.py --mode both

# 关闭反垄断做对照
python main_dual.py --mode market --antitrust_off
```

## 五个假说

| 假说 | 结论 |
|---|---|
| H1: 市场竞争 → 自组织协作 | 否定 — 天然垄断 |
| H2: 革命机制打破垄断 | 证实 — 69~116次革命 |
| H3: 状态感知仲裁选对专家 | 部分 — 食物收集 6× 提升 |
| H4: 感知偏差 > 奖励扭曲 | 部分 — 架构更干净 |
| H5: 可学习仲裁器反超基线 | 部分 — 首次生存步数反超 (+13%, 3.5×更稳定) |

## 引用

```bibtex
@article{li2026internalmarkets,
  title={Why Internal Agent Markets Fail at Small Scale:
         Five Hypotheses Tested on a Multi-Objective GridWorld},
  author={Li, Duan},
  year={2026},
  note={Preprint}
}
```

## 许可

CC BY 4.0
