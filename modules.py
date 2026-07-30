"""
策略模块 — 内部市场的"意识体"

每个模块包含：神经网络 (策略+价值) + 市场属性 (资本/信誉/赌注)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyModule(nn.Module):
    """单个策略模块：输出动作 logits 和状态价值，同时维护市场属性"""

    def __init__(self, obs_dim, n_actions, hidden_dim, module_id, config, personality=None, obs_mask=None, obs_bias=None):
        super().__init__()
        self.module_id = module_id
        self.n_actions = n_actions
        self.config = config
        self.obs_dim = obs_dim

        self.personality = {
            'food_bias': personality.get('food_bias', 1.0) if personality else 1.0,
            'water_bias': personality.get('water_bias', 1.0) if personality else 1.0,
            'danger_aversion': personality.get('danger_aversion', 1.0) if personality else 1.0,
        }

        self.register_buffer('obs_mask',
            torch.tensor(obs_mask, dtype=torch.float32) if obs_mask is not None
            else torch.ones(obs_dim, dtype=torch.float32))
        self.register_buffer('obs_bias',
            torch.tensor(obs_bias, dtype=torch.float32) if obs_bias is not None
            else torch.zeros(obs_dim, dtype=torch.float32))

        # 神经网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # 市场属性 (非梯度，由市场机制更新)
        self.capital = config.init_capital       # 资本（内部经济中的"生命值"）
        self.reputation = config.init_reputation  # 信誉 (0~1)
        self.is_active = True                     # 是否活跃

        # 统计数据
        self.total_bets_won = 0
        self.total_bets_lost = 0
        self.total_selected = 0                   # 被选中次数
        self.stake_history = []                   # 赌注历史

        # 本次 step 的暂存
        self.last_logits = None
        self.last_value = None
        self.last_stake = 0.0
        self.last_action = None
        self.last_confidence = 0.0

    def forward(self, obs):
        """前向传播：自动应用感知掩码和乐观偏差"""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        obs = obs * self.obs_mask + self.obs_bias
        features = self.net(obs)
        logits = self.action_head(features)
        value = self.value_head(features)
        return logits, value

    def propose(self, obs):
        """提出行动方案：返回 (action, stake_amount, logits, value, confidence)"""
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max().item()  # 用最大概率作为置信度
        action = probs.argmax().item()

        # 计算赌注：confidence * capital * base_ratio，但不超过上限
        stake = confidence * self.capital * self.config.stake_base_ratio
        stake = min(stake, self.capital * self.config.capital_stake_max_ratio)
        stake = max(stake, 0.01)  # 最低赌注

        # 暂存本次状态
        self.last_logits = logits.detach()
        self.last_value = value.detach()
        self.last_stake = stake
        self.last_action = action
        self.last_confidence = confidence

        return action, stake, logits, value, confidence

    def update_capital(self, delta):
        """更新资本 (delta 可正可负)"""
        self.capital = max(0.0, min(self.capital + delta, 50.0))
        if self.capital <= 0.0:
            self.is_active = False

    def update_reputation(self, delta):
        """更新信誉"""
        self.reputation = np.clip(self.reputation + delta, 0.0, 1.0)

    def record_stake_outcome(self, outcome, stake):
        """记录赌注结果"""
        self.stake_history.append((outcome, stake))
        if len(self.stake_history) > 1000:
            self.stake_history = self.stake_history[-500:]

    def get_stats(self):
        """获取模块统计"""
        return {
            "id": self.module_id,
            "active": self.is_active,
            "capital": round(self.capital, 2),
            "reputation": round(self.reputation, 3),
            "wins": self.total_bets_won,
            "losses": self.total_bets_lost,
            "selected": self.total_selected,
        }


def create_modules(config):
    """创建一组策略模块，各自具有不同的"人格"偏好"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from env import GridWorldSurvival

    temp_env = GridWorldSurvival(config)
    obs_dim = temp_env.obs_dim
    n_actions = temp_env.n_actions

    # 为不同模块预设不同的"人格"偏好
    personalities = [
        {"food_bias": 1.5, "danger_aversion": 0.5, "explore_bonus": 0.0},   # 激进觅食型
        {"food_bias": 0.5, "danger_aversion": 1.5, "explore_bonus": 0.0},   # 谨慎避险型
        {"food_bias": 1.0, "danger_aversion": 1.0, "explore_bonus": 0.3},   # 探索型
        {"food_bias": 1.0, "danger_aversion": 1.0, "explore_bonus": 0.0},   # 均衡型
    ]

    modules = nn.ModuleList()
    for i in range(config.n_modules):
        personality = personalities[i % len(personalities)]
        module = PolicyModule(obs_dim, n_actions, config.hidden_dim_module, i, config, personality)
        modules.append(module)

    return modules, obs_dim, n_actions
