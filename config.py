"""
实验配置参数
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ============ 环境参数 ============
    grid_size: int = 12                # 网格大小
    n_foods: int = 4                   # 食物数量
    n_dangers: int = 3                 # 危险区域数量
    vision_range: int = 2              # 视野半径 (5x5窗口)
    max_steps: int = 200               # 最大步数
    max_agent_energy: int = 100        # Agent 能量上限
    init_agent_energy: int = 50        # Agent 初始能量
    food_energy_gain: int = 15         # 采集食物获得的能量
    danger_energy_loss: int = 20       # 踩到危险损失的能量
    step_energy_cost: int = 1          # 每步消耗能量
    food_respawn_prob: float = 0.3     # 采集后食物重生概率
    proximity_reward_scale: float = 0.3  # 接近食物奖励缩放

    # ============ 模块参数 ============
    n_modules: int = 6                 # 模块数量
    hidden_dim_single: int = 64        # 单网络模式隐藏层维度
    hidden_dim_module: int = 24        # 市场模式每个模块隐藏层维度
    init_reputation: float = 0.5       # 初始信誉
    init_capital: float = 10.0         # 初始资本（内部能量）

    # ============ 市场参数 ============
    stake_base_ratio: float = 0.15     # 基础赌注比例
    capital_stake_max_ratio: float = 0.3  # 最大赌注占资本比例
    exploration_epsilon: float = 0.25  # 探索概率 (market模式softmax温度)
    rep_update_rate: float = 0.1       # 信誉更新速率

    # ============ 清算参数 ============
    reward_threshold_good: float = 0.0     # 奖励 > 此值视为好结果
    reward_threshold_bad: float = -5.0     # 奖励 < 此值视为坏结果
    liquidation_multiplier: float = 1.5    # 清算乘数
    capital_per_step_cost: float = 0.02    # 每步模块资本消耗

    # ============ 训练参数 ============
    n_episodes: int = 500              # 训练回合数
    learning_rate: float = 1e-3        # 学习率
    gamma: float = 0.95                # 折扣因子
    entropy_coef: float = 0.01         # 熵正则化系数
    grad_clip: float = 1.0             # 梯度裁剪
    batch_episodes: int = 8            # 每批收集的 episode 数
    ppo_epochs: int = 4               # PPO 更新轮数
    ppo_clip: float = 0.2             # PPO 裁剪范围

    # ============ 复活/复制参数 ============
    revive_check_interval: int = 10    # 复活检查间隔
    revive_capital: float = 8.0        # 复活时赋予的资本
    mutation_std: float = 0.05         # 复制时变异标准差

    # ============ 反垄断/革命参数 ============
    antitrust_enabled: bool = True      # 是否启用反垄断机制
    antitrust_window: int = 50          # 监控窗口大小(步)
    antitrust_threshold: float = 0.65   # 选择率超过此比例触发革命
    antitrust_tax_rate: float = 0.30    # 垄断者资本征收比例
    antitrust_rep_penalty: float = 0.3  # 垄断者信誉惩罚
    antitrust_rebel_bonus: float = 0.2  # 革命者临时信誉加成
    antitrust_cooldown: int = 30        # 革命冷却期(步)

    # ============ 实验参数 ============
    seed: int = 42                     # 随机种子
    eval_episodes: int = 20            # 评估回合数
    print_interval: int = 50           # 打印间隔
    modes: tuple = ("single", "market")  # 实验模式


# 快速测试用的简化配置
@dataclass
class SmallConfig(Config):
    grid_size: int = 8
    n_foods: int = 4
    n_dangers: int = 1
    max_steps: int = 150
    init_agent_energy: int = 100
    n_modules: int = 4
    n_episodes: int = 800
    batch_episodes: int = 8
    hidden_dim_single: int = 64
    hidden_dim_module: int = 32
    print_interval: int = 80
    learning_rate: float = 3e-4
    entropy_coef: float = 0.05
    food_respawn_prob: float = 0.5
    proximity_reward_scale: float = 0.3
    eval_episodes: int = 30


# 双资源实验配置
@dataclass
class DualConfig(Config):
    grid_size: int = 8
    n_foods: int = 3
    n_waters: int = 3
    n_dangers: int = 1
    max_steps: int = 200
    init_agent_energy: int = 80
    max_agent_energy: int = 100
    init_water: int = 80
    max_water: int = 100
    food_energy_gain: int = 20
    water_gain: int = 20
    danger_energy_loss: int = 20
    step_energy_cost: int = 1
    step_water_cost: int = 1
    food_respawn_prob: float = 0.5
    water_respawn_prob: float = 0.5
    proximity_reward_scale: float = 0.3

    n_modules: int = 4
    n_episodes: int = 3000
    batch_episodes: int = 8
    hidden_dim_single: int = 64
    hidden_dim_module: int = 32
    print_interval: int = 300
    learning_rate: float = 3e-4
    entropy_coef: float = 0.05
    eval_episodes: int = 30

    antitrust_enabled: bool = True
    antitrust_threshold: float = 0.60

    # ============ 可学习仲裁器 ============
    use_learnable_arbitrator: bool = True
    arbitrator_hidden: int = 32
    arbitrator_lr: float = 3e-4
    arbitrator_temperature: float = 0.5
