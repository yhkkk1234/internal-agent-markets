"""
训练器 — PPO 风格的批量训练，支持 single 和 market 两种模式对比
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from env import GridWorldSurvival
from modules import create_modules
from market import InternalMarket


def compute_gae(rewards, values, dones, gamma, lam=0.95):
    """计算 GAE 优势函数"""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


class SingleAgent(nn.Module):
    """单网络智能体 (基线)"""

    def __init__(self, obs_dim, n_actions, hidden_dim):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        features = self.net(obs)
        logits = self.action_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action(self, obs):
        """采样动作"""
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), logits.detach(), value


def _ppo_update(agent, optimizer, batch_data, config):
    """PPO 更新 (通用，single 和 market 共用)"""
    if len(batch_data["obs"]) == 0:
        return

    obs_t = torch.tensor(np.array(batch_data["obs"]), dtype=torch.float32)
    actions_t = torch.tensor(batch_data["actions"], dtype=torch.long)
    old_log_probs_t = torch.tensor(batch_data["old_log_probs"], dtype=torch.float32)
    advantages_t = torch.tensor(batch_data["advantages"], dtype=torch.float32)
    returns_t = torch.tensor(batch_data["returns"], dtype=torch.float32)

    # 归一化优势
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    dataset_size = len(obs_t)
    mini_batch_size = max(32, dataset_size // 4)

    for _ in range(config.ppo_epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, mini_batch_size):
            end = min(start + mini_batch_size, dataset_size)
            idx = indices[start:end]

            logits, values = agent(obs_t[idx])
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t[idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
            adv = advantages_t[idx]

            # PPO clipped objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - config.ppo_clip, 1 + config.ppo_clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns_t[idx])

            loss = policy_loss + 0.5 * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.grad_clip)
            optimizer.step()


def train_single(config, verbose=True):
    """训练单网络基线 (PPO 批量更新)"""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = GridWorldSurvival(config)
    agent = SingleAgent(env.obs_dim, env.n_actions, config.hidden_dim_single)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    batch_buffer = {"obs": [], "actions": [], "old_log_probs": [],
                    "rewards": [], "values": [], "dones": []}
    stats_history = []
    best_reward = -float('inf')
    episode = 0
    batch_foods = []

    while episode < config.n_episodes:
        # 每 episode 使用不同种子，避免位置记忆
        np.random.seed(config.seed + episode * 1000)
        obs = env.reset()
        ep_obs, ep_actions, ep_log_probs = [], [], []
        ep_rewards, ep_values, ep_dones = [], [], []
        foods = 0
        done = False

        while not done:
            action, log_prob, entropy, logits, value = agent.get_action(obs)
            obs_next, reward, done, info = env.step(action)
            if info.get("event") == "collected_food":
                foods += 1

            ep_obs.append(obs)
            ep_actions.append(action)
            ep_log_probs.append(log_prob.item())
            ep_rewards.append(reward)
            ep_values.append(value.item())
            ep_dones.append(float(done))

            obs = obs_next

        # 加入缓冲
        batch_buffer["obs"].extend(ep_obs)
        batch_buffer["actions"].extend(ep_actions)
        batch_buffer["old_log_probs"].extend(ep_log_probs)
        batch_buffer["rewards"].extend(ep_rewards)
        batch_buffer["values"].extend(ep_values)
        batch_buffer["dones"].extend(ep_dones)
        batch_foods.append(foods)

        stats_history.append({
            "episode": episode,
            "total_reward": sum(ep_rewards),
            "steps": len(ep_rewards),
            "foods": foods,
            "final_energy": env.agent_energy,
        })

        episode += 1

        # 每 batch_episodes 个 episode 做一次 PPO 更新
        if episode % config.batch_episodes == 0:
            # 计算 GAE
            advantages, returns = compute_gae(
                batch_buffer["rewards"], batch_buffer["values"],
                batch_buffer["dones"], config.gamma)

            batch_buffer["advantages"] = advantages
            batch_buffer["returns"] = returns

            _ppo_update(agent, optimizer, batch_buffer, config)

            # 清空缓冲
            for k in batch_buffer:
                batch_buffer[k] = []

        if verbose and episode % config.print_interval == 0:
            recent = stats_history[-config.print_interval:]
            avg_reward = np.mean([s["total_reward"] for s in recent])
            avg_steps = np.mean([s["steps"] for s in recent])
            avg_foods = np.mean([s["foods"] for s in recent])
            if stats_history[-1]["total_reward"] > best_reward:
                best_reward = stats_history[-1]["total_reward"]
            print(f"  [Single] Ep {episode:4d} | "
                  f"Reward: {avg_reward:+7.1f} | Steps: {avg_steps:6.1f} | "
                  f"Foods: {avg_foods:.1f} | Best: {max(s['total_reward'] for s in stats_history):+7.1f}")

    eval_results = evaluate_single(agent, config)
    return agent, stats_history, eval_results


def train_market(config, verbose=True):
    """训练多模块市场 (PPO 批量更新)"""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = GridWorldSurvival(config)
    modules, obs_dim, n_actions = create_modules(config)
    market = InternalMarket(config, modules)

    optimizers = [
        torch.optim.Adam(m.parameters(), lr=config.learning_rate)
        for m in modules
    ]

    # 每个模块独立的缓冲
    buffers = {m.module_id: {"obs": [], "actions": [], "old_log_probs": [],
                              "rewards": [], "values": [], "dones": []}
               for m in modules}

    stats_history = []
    episode = 0
    update_counter = 0

    while episode < config.n_episodes:
        np.random.seed(config.seed + episode * 1000)
        obs = env.reset()
        ep_rewards = []
        foods = 0
        done = False

        while not done:
            action, winner_idx, stake, ledger = market.select_action(obs)
            obs_next, env_reward, done, info = env.step(action)
            if info.get("event") == "collected_food":
                foods += 1

            settlement = market.settle(env_reward, ledger)
            winner_lr = 0.0
            ally_lr = 0.0
            if settlement:
                winner_lr = settlement["winner_liquidation_reward"]
                ally_lr = settlement["ally_liquidation_reward"]

            # 所有活跃模块都从此经验中学习 (各自用自己的人格偏好调整奖励)
            for idx, mod in enumerate(modules):
                if not mod.is_active:
                    continue

                # 应用人格偏好
                pers = mod.personality
                adj_reward = env_reward
                if info.get("event") == "collected_food":
                    adj_reward *= pers['food_bias']
                elif info.get("event") == "hit_danger":
                    adj_reward *= pers['danger_aversion']

                # 只有获胜者和盟友获得清算奖励
                if idx == winner_idx:
                    adj_reward += winner_lr
                elif any(a["idx"] == idx for a in ledger.get("allies", [])):
                    adj_reward += ally_lr

                obs_t = torch.from_numpy(obs).float()
                logits, value = mod(obs_t)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(torch.tensor(action))

                buffers[idx]["obs"].append(obs)
                buffers[idx]["actions"].append(action)
                buffers[idx]["old_log_probs"].append(log_prob.item())
                buffers[idx]["rewards"].append(adj_reward)
                buffers[idx]["values"].append(value.item())
                buffers[idx]["dones"].append(float(done))

            ep_rewards.append(env_reward)
            obs = obs_next

        episode += 1
        market.revive_check()

        stats_history.append({
            "episode": episode,
            "total_reward": sum(ep_rewards),
            "steps": len(ep_rewards),
            "foods": foods,
            "final_energy": env.agent_energy,
            "active_modules": sum(1 for m in modules if m.is_active),
        })

        # 每 batch_episodes 个 episode 更新一次
        if episode % config.batch_episodes == 0:
            for mod in modules:
                buf = buffers[mod.module_id]
                if len(buf["obs"]) < 4:
                    continue

                advantages, returns = compute_gae(
                    buf["rewards"], buf["values"], buf["dones"], config.gamma)
                buf["advantages"] = advantages
                buf["returns"] = returns

                _ppo_update(mod, optimizers[mod.module_id], buf, config)

                # 清空该模块的缓冲
                for k in buf:
                    buf[k] = []

            update_counter += 1

        if verbose and episode % config.print_interval == 0:
            recent = stats_history[-config.print_interval:]
            avg_reward = np.mean([s["total_reward"] for s in recent])
            avg_steps = np.mean([s["steps"] for s in recent])
            avg_foods = np.mean([s["foods"] for s in recent])
            active = sum(1 for m in modules if m.is_active)
            best = max(s["total_reward"] for s in stats_history)
            print(f"  [Market] Ep {episode:4d} | "
                  f"Reward: {avg_reward:+7.1f} | Steps: {avg_steps:6.1f} | "
                  f"Foods: {avg_foods:.1f} | Active: {active}/{config.n_modules} | "
                  f"Best: {best:+7.1f}")


    eval_results = evaluate_market(market, config)
    return modules, market, stats_history, eval_results


def evaluate_single(agent, config):
    """评估单网络"""
    torch.manual_seed(config.seed + 999)
    np.random.seed(config.seed + 999)

    env = GridWorldSurvival(config)
    total_rewards, total_steps, final_energies, total_foods = [], [], [], []

    for _ in range(config.eval_episodes):
        obs = env.reset()
        ep_reward, steps, foods = 0.0, 0, 0
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = agent(obs)
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            if info.get("event") == "collected_food":
                foods += 1

        total_rewards.append(ep_reward)
        total_steps.append(steps)
        final_energies.append(env.agent_energy)
        total_foods.append(foods)

    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_steps": np.mean(total_steps),
        "std_steps": np.std(total_steps),
        "avg_final_energy": np.mean(final_energies),
        "avg_foods": np.mean(total_foods),
        "survival_rate": np.mean([1.0 if s >= config.max_steps else 0.0
                                   for s in total_steps]),
    }


def evaluate_market(market, config):
    """评估市场模式"""
    torch.manual_seed(config.seed + 999)
    np.random.seed(config.seed + 999)

    eval_env = GridWorldSurvival(config)
    total_rewards, total_steps, final_energies = [], [], []
    total_foods = []
    module_selections = defaultdict(int)

    # 评估前强制所有模块激活并赋予均等机会
    for m in market.modules:
        m.is_active = True
        m.capital = max(m.capital, config.revive_capital)
        m.reputation = max(m.reputation, config.init_reputation * 0.5)

    for ep in range(config.eval_episodes):
        obs = eval_env.reset()
        ep_reward, steps, foods = 0.0, 0, 0
        done = False
        while not done:
            action, winner_idx, stake, ledger = market.select_action(obs)
            if winner_idx is not None:
                module_selections[winner_idx] += 1
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward
            steps += 1
            if info.get("event") == "collected_food":
                foods += 1
            market.revive_check()

        total_rewards.append(ep_reward)
        total_steps.append(steps)
        final_energies.append(eval_env.agent_energy)
        total_foods.append(foods)

    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_steps": np.mean(total_steps),
        "std_steps": np.std(total_steps),
        "avg_final_energy": np.mean(final_energies),
        "avg_foods": np.mean(total_foods),
        "survival_rate": np.mean([1.0 if s >= config.max_steps else 0.0
                                   for s in total_steps]),
        "module_selections": dict(module_selections),
    }
