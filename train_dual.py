"""
双资源训练器 - 通过感知盲区(obs_mask)实现模块分化
差异化在感知层完成，奖励保持真实。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from env_dual import DualResourceGridWorld
from modules_dual import create_dual_modules
from market import InternalMarket, LearnableArbitrator
from trainer import _ppo_update, compute_gae


class DualSingleAgent(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        f = self.net(obs)
        return self.action_head(f), self.value_head(f)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy(), logits.detach(), value


def train_dual_single(config, verbose=True):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = DualResourceGridWorld(config)
    agent = DualSingleAgent(env.obs_dim, env.n_actions, config.hidden_dim_single)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    buffer = {"obs": [], "actions": [], "old_log_probs": [],
              "rewards": [], "values": [], "dones": []}
    stats = []
    episode = 0

    while episode < config.n_episodes:
        np.random.seed(config.seed + episode * 1000)
        obs = env.reset()
        foods = 0
        waters = 0
        done = False
        while not done:
            a, lp, ent, logits, val = agent.get_action(obs)
            obs_n, reward, done, info = env.step(a)
            if info.get("event") == "collected_food":
                foods += 1
            elif info.get("event") == "collected_water":
                waters += 1

            buffer["obs"].append(obs)
            buffer["actions"].append(a)
            buffer["old_log_probs"].append(lp.item())
            buffer["rewards"].append(reward)
            buffer["values"].append(val.item())
            buffer["dones"].append(float(done))
            obs = obs_n

        stats.append({"ep": episode, "reward": env.total_reward,
                       "steps": env.step_count, "foods": foods, "waters": waters})
        episode += 1

        if episode % config.batch_episodes == 0:
            adv, ret = compute_gae(buffer["rewards"], buffer["values"], buffer["dones"], config.gamma)
            buffer["advantages"] = adv
            buffer["returns"] = ret
            _ppo_update(agent, optimizer, buffer, config)
            for k in buffer:
                buffer[k] = []

        if verbose and episode % config.print_interval == 0:
            recent = stats[-config.print_interval:]
            print(f"  [Single] Ep {episode:4d} | "
                  f"Reward: {np.mean([s['reward'] for s in recent]):+7.1f} | "
                  f"F:{np.mean([s['foods'] for s in recent]):.1f} "
                  f"W:{np.mean([s['waters'] for s in recent]):.1f} | "
                  f"Best: {max(s['reward'] for s in stats):+7.1f}")

    return agent, stats, evaluate_dual_single(agent, config)


def train_dual_market(config, verbose=True):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = DualResourceGridWorld(config)
    modules, obs_dim, n_actions = create_dual_modules(config)

    # 创建可学习仲裁器
    arbitrator = None
    if config.use_learnable_arbitrator:
        arbitrator = LearnableArbitrator(
            obs_dim, config.n_modules, config.arbitrator_hidden,
            temperature=config.arbitrator_temperature)
        arbitrator.optimizer = torch.optim.Adam(
            arbitrator.parameters(), lr=config.arbitrator_lr)

    market = InternalMarket(config, modules, arbitrator=arbitrator)

    optimizers = [torch.optim.Adam(m.parameters(), lr=config.learning_rate) for m in modules]
    buffers = {m.module_id: {"obs": [], "actions": [], "old_log_probs": [],
                              "rewards": [], "values": [], "dones": []} for m in modules}

    stats = []
    episode = 0

    while episode < config.n_episodes:
        np.random.seed(config.seed + episode * 1000)
        obs = env.reset()
        foods = 0
        waters = 0
        done = False

        while not done:
            action, winner_idx, stake, ledger = market.select_action(obs)
            obs_n, env_reward, done, info = env.step(action)

            if info.get("event") == "collected_food":
                foods += 1
            elif info.get("event") == "collected_water":
                waters += 1

            settlement = market.settle(env_reward, ledger)
            winner_lr = 0.0
            ally_lr = 0.0
            if settlement:
                winner_lr = settlement["winner_liquidation_reward"]
                ally_lr = settlement["ally_liquidation_reward"]

            # 所有模块使用真实奖励 (差异化在感知层完成)
            for idx, mod in enumerate(modules):
                if not mod.is_active:
                    continue

                adj_reward = env_reward

                if idx == winner_idx:
                    adj_reward += winner_lr
                elif any(a["idx"] == idx for a in ledger.get("allies", [])):
                    adj_reward += ally_lr

                obs_t = torch.from_numpy(obs).float()
                logits, value = mod(obs_t)
                probs = F.softmax(logits, dim=-1)
                lp = torch.distributions.Categorical(probs).log_prob(torch.tensor(action))

                buffers[idx]["obs"].append(obs)
                buffers[idx]["actions"].append(action)
                buffers[idx]["old_log_probs"].append(lp.item())
                buffers[idx]["rewards"].append(adj_reward)
                buffers[idx]["values"].append(value.item())
                buffers[idx]["dones"].append(float(done))

            obs = obs_n

        episode += 1
        market.revive_check()

        # 仲裁器: 记录本 episode 的表现
        if arbitrator is not None:
            arbitrator.finalize_episode(env.total_reward, env.step_count)

        # 资本防爆：周期性归一化
        if episode % 50 == 0:
            active_caps = [m.capital for m in modules if m.is_active]
            if active_caps and max(active_caps) > 100:
                total = sum(active_caps)
                for m in modules:
                    if m.is_active and total > 0:
                        m.capital = m.capital / total * config.n_modules * config.init_capital

        stats.append({"ep": episode, "reward": env.total_reward,
                       "steps": env.step_count, "foods": foods, "waters": waters,
                       "active": sum(1 for m in modules if m.is_active)})

        if episode % config.batch_episodes == 0:
            for mod in modules:
                buf = buffers[mod.module_id]
                if len(buf["obs"]) < 4:
                    continue
                adv, ret = compute_gae(buf["rewards"], buf["values"], buf["dones"], config.gamma)
                buf["advantages"] = adv
                buf["returns"] = ret
                _ppo_update(mod, optimizers[mod.module_id], buf, config)
                for k in buf:
                    buf[k] = []

            # 仲裁器训练 (在模块更新之后)
            if arbitrator is not None:
                arb_loss = arbitrator.train_step(gamma=config.gamma)
                if verbose and episode % config.print_interval == 0:
                    pass  # 仲裁器 loss 在下个日志周期打印

        if verbose and episode % config.print_interval == 0:
            recent = stats[-config.print_interval:]
            print(f"  [Market] Ep {episode:4d} | "
                  f"R:{np.mean([s['reward'] for s in recent]):+7.1f} | "
                  f"F:{np.mean([s['foods'] for s in recent]):.1f} "
                  f"W:{np.mean([s['waters'] for s in recent]):.1f} | "
                  f"Best:{max(s['reward'] for s in stats):+7.1f}")

    return modules, market, stats, evaluate_dual_market(market, config)


def evaluate_dual_single(agent, config):
    torch.manual_seed(config.seed + 999)
    np.random.seed(config.seed + 999)
    env = DualResourceGridWorld(config)
    rewards, steps, foods, waters = [], [], [], []

    for _ in range(config.eval_episodes):
        obs = env.reset()
        er, f, w, s = 0.0, 0, 0, 0
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = agent(obs)
                a = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            obs, r, done, info = env.step(a)
            er += r
            s += 1
            if info.get("event") == "collected_food":
                f += 1
            elif info.get("event") == "collected_water":
                w += 1
        rewards.append(er)
        steps.append(s)
        foods.append(f)
        waters.append(w)

    return {"avg_reward": np.mean(rewards), "std_reward": np.std(rewards),
            "avg_steps": np.mean(steps), "std_steps": np.std(steps),
            "avg_foods": np.mean(foods), "avg_waters": np.mean(waters),
            "survival_rate": np.mean([1.0 if s >= config.max_steps else 0.0 for s in steps])}


def evaluate_dual_market(market, config):
    torch.manual_seed(config.seed + 999)
    np.random.seed(config.seed + 999)
    env = DualResourceGridWorld(config)
    rewards, steps, foods, waters = [], [], [], []
    sel = defaultdict(int)

    for m in market.modules:
        m.is_active = True
        m.capital = max(m.capital, config.revive_capital)
        m.reputation = max(m.reputation, config.init_reputation * 0.5)

    for _ in range(config.eval_episodes):
        obs = env.reset()
        er, f, w, s = 0.0, 0, 0, 0
        done = False
        while not done:
            action, wid, stake, ledger = market.select_action(obs)
            if wid is not None:
                sel[wid] += 1
            obs, r, done, info = env.step(action)
            er += r
            s += 1
            if info.get("event") == "collected_food":
                f += 1
            elif info.get("event") == "collected_water":
                w += 1
            market.revive_check()
        rewards.append(er)
        steps.append(s)
        foods.append(f)
        waters.append(w)

    return {"avg_reward": np.mean(rewards), "std_reward": np.std(rewards),
            "avg_steps": np.mean(steps), "std_steps": np.std(steps),
            "avg_foods": np.mean(foods), "avg_waters": np.mean(waters),
            "survival_rate": np.mean([1.0 if s >= config.max_steps else 0.0 for s in steps]),
            "module_selections": dict(sel)}
