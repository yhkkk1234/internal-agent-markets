"""
双资源模块工厂 — 通过乐观偏差(obs_bias)实现策略分化

每个模块看到完整世界，但对其非专长资源有乐观偏差:
  M0 (食物专家): 水指标偏高 — 倾向于觉得"水还够"，优先找食物
  M1 (水源专家): 能量指标偏高 — 倾向于觉得"吃饱了"，优先找水
  M2 (避险专家): 危险视觉模糊 — 危险感知迟钝，但学会更谨慎
  M3 (全能者):   无偏差 — 看到世界的本来面目

所有模块使用相同的真实奖励训练，差异化纯粹来自感知偏差。
"""
import sys
import os
import numpy as np
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from modules import PolicyModule
from env_dual import DualResourceGridWorld


def create_dual_modules(config):
    temp_env = DualResourceGridWorld(config)
    obs_dim = temp_env.obs_dim
    n_actions = temp_env.n_actions

    # obs 结构: vision(5*5*6=150) + [x, y, energy_ratio, water_ratio, steps] = 155
    # idx 152 = energy_ratio (0~1, 1=满), idx 153 = water_ratio (0~1, 1=满)

    modules_specs = [
        # M0: 食物专家 — 水指标乐观偏差 +0.35 (只有水<35%时才感知到渴)
        {
            "mask": np.ones(obs_dim, dtype=np.float32),
            "bias": np.zeros(obs_dim, dtype=np.float32),
            "personality": {"food_bias": 1.5, "water_bias": 0.3, "danger_aversion": 1.0},
        },
        # M1: 水源专家 — 能量指标乐观偏差 +0.35
        {
            "mask": np.ones(obs_dim, dtype=np.float32),
            "bias": np.zeros(obs_dim, dtype=np.float32),
            "personality": {"food_bias": 0.3, "water_bias": 1.5, "danger_aversion": 1.0},
        },
        # M2: 避险专家 — 危险视觉通道衰减
        {
            "mask": np.ones(obs_dim, dtype=np.float32),
            "bias": np.zeros(obs_dim, dtype=np.float32),
            "personality": {"food_bias": 1.0, "water_bias": 1.0, "danger_aversion": 1.5},
        },
        # M3: 全能者 — 无偏差
        {
            "mask": np.ones(obs_dim, dtype=np.float32),
            "bias": np.zeros(obs_dim, dtype=np.float32),
            "personality": {"food_bias": 1.0, "water_bias": 1.0, "danger_aversion": 1.0},
        },
    ]

    # 设置偏差
    modules_specs[0]["bias"][153] = 0.35  # M0: 水看起来总多35%
    modules_specs[1]["bias"][152] = 0.35  # M1: 能量看起来总多35%
    # M2: 危险通道 (每个视觉单元的类型4) 衰减到20%
    for cell in range(25):
        danger_idx = cell * 6 + 4
        modules_specs[2]["mask"][danger_idx] = 0.2

    modules = nn.ModuleList()
    for i in range(min(config.n_modules, len(modules_specs))):
        spec = modules_specs[i]
        module = PolicyModule(obs_dim, n_actions, config.hidden_dim_module,
                              i, config, spec["personality"],
                              obs_mask=spec["mask"], obs_bias=spec["bias"])
        modules.append(module)

    return modules, obs_dim, n_actions
