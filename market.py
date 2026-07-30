"""
内部竞标市场 — 多模块博弈的核心机制

负责：
1. 收集模块提案并仲裁选出行动
2. 根据环境反馈清算赌注
3. 复活休眠模块
4. 反垄断监控与革命机制
"""
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableArbitrator(nn.Module):
    """可学习仲裁器 — 一个微型 MLP 学会'何时选哪个模块'

    输入: 环境观察 (obs)
    输出: 各模块的 logits (softmax 后得到选择概率)

    用 REINFORCE 训练: 每个 episode 的仲裁决策 → 环境反馈 → 更新
    仲裁器学的不是'谁最可信'，而是'在什么状态选谁最好'
    """

    def __init__(self, obs_dim, n_modules, hidden_dim, temperature=0.5):
        super().__init__()
        self.n_modules = n_modules
        self.temperature = temperature

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modules),
        )

        # 训练缓冲
        self.trajectory = {"obs": [], "actions": [], "log_probs": [],
                            "rewards": [], "dones": []}
        self.optimizer = None  # 外部设置

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        logits = self.net(obs)
        return logits

    def select_module(self, obs, active_modules):
        """选择模块: 只在活跃模块中选"""
        logits = self.forward(obs)
        # 将非活跃模块的 logit 设为 -inf
        mask = torch.full((self.n_modules,), -float('inf'))
        for i, mod in enumerate(active_modules):
            if mod.is_active:
                mask[i] = 0.0
        masked_logits = logits + mask

        probs = F.softmax(masked_logits / self.temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        winner_idx = dist.sample()
        return winner_idx.item(), dist.log_prob(winner_idx), probs.detach()

    def record(self, obs, winner_idx, log_prob):
        """记录一步仲裁决策"""
        self.trajectory["obs"].append(obs.copy())
        self.trajectory["actions"].append(winner_idx)
        self.trajectory["log_probs"].append(log_prob)

    def finalize_episode(self, episode_reward, total_steps):
        """标记 episode 结束，填充 reward (仅对新增步骤)"""
        current = len(self.trajectory["log_probs"])
        recorded = len(self.trajectory["rewards"])
        new_steps = current - recorded
        if new_steps <= 0:
            return

        self.trajectory["rewards"].extend([0.0] * new_steps)
        self.trajectory["dones"].extend([0.0] * new_steps)
        # 最后一步获得 episode 回报
        self.trajectory["rewards"][-1] = episode_reward
        self.trajectory["dones"][-1] = 1.0

    def train_step(self, gamma=0.95):
        """REINFORCE 更新仲裁器"""
        buf = self.trajectory
        if len(buf["log_probs"]) == 0:
            return 0.0

        rewards = buf["rewards"]
        dones = buf["dones"]
        log_probs = torch.stack(buf["log_probs"])

        # 计算折扣回报 (简单的 MC return)
        returns = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - d)
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # 归一化
        if returns_t.std() > 0:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # REINFORCE loss
        loss = -(log_probs * returns_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空缓冲
        for k in buf:
            buf[k] = []

        return loss.item()


class InternalMarket:
    """内部思想市场 (含反垄断革命机制 + 可学习仲裁器)"""

    def __init__(self, config, modules, arbitrator=None):
        self.config = config
        self.modules = modules
        self.n_modules = len(modules)
        self.total_steps = 0

        # 可学习仲裁器
        self.arbitrator = arbitrator

        # 反垄断机制
        self.selection_history = deque(maxlen=config.antitrust_window)
        self.antitrust_cooldown = 0
        self.revolution_count = 0
        self.revolution_log = []

        # 革命者临时信誉加成
        self.rebel_bonus = {}

    def select_action(self, obs):
        """
        收集所有活跃模块的提案，仲裁选出最终行动
        返回: (action, winner_idx, winner_stake, stake_ledger)
        """
        proposals = []

        for idx, module in enumerate(self.modules):
            if not module.is_active:
                continue

            action, stake, logits, value, confidence = module.propose(obs)
            proposals.append({
                "idx": idx,
                "module": module,
                "action": action,
                "stake": stake,
                "confidence": confidence,
                "logits": logits,
                "value": value,
            })

        # 所有模块都休眠了：随机选择一个模块强制激活
        if not proposals:
            module = self.modules[0]
            module.is_active = True
            module.capital = self.config.revive_capital
            action, stake, logits, value, confidence = module.propose(obs)
            proposals.append({
                "idx": 0,
                "module": module,
                "action": action,
                "stake": stake,
                "confidence": confidence,
                "logits": logits,
                "value": value,
            })

        # ====== 仲裁：可学习仲裁器 或 手工规则 ======
        if self.arbitrator is not None:
            # 使用可学习仲裁器选择模块
            winner_idx, arb_log_prob, _ = self.arbitrator.select_module(
                obs, self.modules)
            # 找到对应的 proposal
            winner = None
            for i, p in enumerate(proposals):
                if p["idx"] == winner_idx:
                    winner_idx_in_proposals = i
                    winner = p
                    break
            if winner is None:
                winner_idx_in_proposals = 0
                winner = proposals[0]
                winner_idx = winner["idx"]

            # 记录仲裁器决策
            self.arbitrator.record(obs, winner_idx, arb_log_prob)
        else:
            # 手工规则仲裁
            obs_len = len(obs)
            energy_ratio = 1.0
            water_ratio = 1.0
            if obs_len == 155:
                energy_ratio = float(obs[152])
                water_ratio = float(obs[153])
            elif obs_len == 129:
                energy_ratio = float(obs[127])

            need_energy = max(0.0, 1.0 - energy_ratio)
            need_water = max(0.0, 1.0 - water_ratio)

            scores = []
            for p in proposals:
                pers = p["module"].personality
                score = p["module"].reputation * p["confidence"]
                state_bonus = 0.0
                if need_energy > 0:
                    state_bonus += need_energy * pers.get("food_bias", 1.0) * 0.6
                if need_water > 0 and obs_len == 155:
                    state_bonus += need_water * pers.get("water_bias", 1.0) * 0.6
                score += state_bonus

                fair_share = max(1, self.total_steps // max(1, len(proposals)))
                if p["module"].total_selected < fair_share:
                    score += 0.5
                rebel_steps = self.rebel_bonus.get(p["module"].module_id, 0)
                if rebel_steps > 0:
                    score += self.config.antitrust_rebel_bonus
                scores.append(score)

            scores_t = torch.tensor(scores, dtype=torch.float32)
            temperature = 0.3 + 0.2 * (self.config.exploration_epsilon)
            probs = F.softmax(scores_t / temperature, dim=-1)
            winner_idx_in_proposals = torch.multinomial(probs, 1).item()
            winner = proposals[winner_idx_in_proposals]

        # 查找联盟：押注相同 action 的其他模块
        allies = [p for p in proposals
                  if p["action"] == winner["action"] and p["idx"] != winner["idx"]]

        # 记录被选中
        winner["module"].total_selected += 1

        # 记录选择历史 (反垄断监控)
        self.selection_history.append(winner["idx"])

        # 构建账本
        stake_ledger = {
            "winner": winner,
            "allies": allies,
            "all_proposals": proposals,
            "total_stake": winner["stake"] + sum(a["stake"] for a in allies),
        }

        # 反垄断检查 (在本次选择完成后触发，影响后续选择)
        revolution_event = self._antitrust_check()
        if revolution_event:
            stake_ledger["revolution"] = revolution_event

        return winner["action"], winner["idx"], winner["stake"], stake_ledger

    def settle(self, env_reward, stake_ledger):
        """
        清算赌注
        返回: settlement_info (包含 winner_liquidation_reward 等)
        """
        if stake_ledger is None:
            return None

        winner = stake_ledger["winner"]
        allies = stake_ledger["allies"]
        module = winner["module"]
        stake = winner["stake"]

        info = {
            "winner_id": module.module_id,
            "env_reward": env_reward,
            "stake": stake,
            "outcome": "neutral",
            "winner_liquidation_reward": 0.0,
            "ally_liquidation_reward": 0.0,
            "dormant_modules": [],
        }

        if env_reward > self.config.reward_threshold_good:
            # ---- 好结果：返还赌注 + 奖金 ----
            bonus = stake * self.config.liquidation_multiplier
            module.update_capital(bonus)
            module.update_reputation(+self.config.rep_update_rate)
            module.total_bets_won += 1
            module.record_stake_outcome("won", stake)
            info["outcome"] = "won"
            info["winner_liquidation_reward"] = bonus

            # 盟友获半奖
            for ally in allies:
                ally_bonus = ally["stake"] * 0.5
                ally["module"].update_capital(ally_bonus)
                ally["module"].update_reputation(+self.config.rep_update_rate * 0.5)
                ally["module"].total_bets_won += 1
            info["ally_liquidation_reward"] = info["winner_liquidation_reward"] * 0.5

        elif env_reward < self.config.reward_threshold_bad:
            # ---- 坏结果：没收赌注 ----
            penalty = stake * self.config.liquidation_multiplier
            module.update_capital(-penalty)
            module.update_reputation(-self.config.rep_update_rate)
            module.total_bets_lost += 1
            module.record_stake_outcome("lost", stake)
            info["outcome"] = "lost"
            info["winner_liquidation_reward"] = -penalty

            if not module.is_active:
                info["dormant_modules"].append(module.module_id)

            # 盟友受轻罚
            for ally in allies:
                ally_penalty = ally["stake"] * 0.5
                ally["module"].update_capital(-ally_penalty)
                ally["module"].update_reputation(-self.config.rep_update_rate * 0.3)
                ally["module"].total_bets_lost += 1
                if not ally["module"].is_active:
                    info["dormant_modules"].append(ally["module"].module_id)
            info["ally_liquidation_reward"] = info["winner_liquidation_reward"] * 0.5

        else:
            # 中性结果：不奖不罚
            info["outcome"] = "neutral"

        # 所有活跃模块的持续资本消耗
        for mod in self.modules:
            if mod.is_active:
                mod.update_capital(-self.config.capital_per_step_cost)

        self.total_steps += 1
        return info

    def _antitrust_check(self):
        """反垄断监控：检测权力集中并触发革命"""
        if not self.config.antitrust_enabled:
            return None

        # 冷却期未结束
        if self.antitrust_cooldown > 0:
            self.antitrust_cooldown -= 1
            self._decay_rebel_bonus()
            return None

        # 窗口未满，不检查
        if len(self.selection_history) < self.config.antitrust_window:
            self._decay_rebel_bonus()
            return None

        # 计算各模块在窗口内的选择比例
        active_modules = [m for m in self.modules if m.is_active]
        if len(active_modules) < 2:
            self._decay_rebel_bonus()
            return None

        history = list(self.selection_history)
        selection_counts = {}
        for mid in history:
            selection_counts[mid] = selection_counts.get(mid, 0) + 1

        total = len(history)
        for module_id, count in selection_counts.items():
            ratio = count / total
            if ratio > self.config.antitrust_threshold:
                return self._execute_revolution(module_id, ratio, history, selection_counts)

        self._decay_rebel_bonus()
        return None

    def _execute_revolution(self, monopolist_id, ratio, history, selection_counts):
        """执行革命：征税垄断者，赋能革命者"""
        monopolist = self.modules[monopolist_id]
        active_others = [m for m in self.modules
                         if m.is_active and m.module_id != monopolist_id]

        if not active_others:
            return None

        # 1. 税收：没收垄断者部分资本
        tax_amount = monopolist.capital * self.config.antitrust_tax_rate
        monopolist.update_capital(-tax_amount)

        # 2. 信誉惩罚：垄断者信誉大幅降低
        monopolist.update_reputation(-self.config.antitrust_rep_penalty)

        # 3. 资本再分配：税收平分给其他活跃模块
        share = tax_amount / len(active_others)
        for mod in active_others:
            mod.update_capital(share)
            mod.update_reputation(+self.config.antitrust_rep_penalty * 0.5)

            # 4. 革命者临时加成：为所有非垄断模块提供持续信誉boost
            self.rebel_bonus[mod.module_id] = self.config.antitrust_cooldown

        # 5. 冷却期
        self.antitrust_cooldown = self.config.antitrust_cooldown
        self.revolution_count += 1

        event = {
            "monopolist": monopolist_id,
            "concentration": round(ratio, 3),
            "tax_amount": round(tax_amount, 2),
            "rebels": [m.module_id for m in active_others],
            "step": self.total_steps,
        }
        self.revolution_log.append(event)
        return event

    def _decay_rebel_bonus(self):
        """衰减革命者临时加成"""
        expired = []
        for mid in list(self.rebel_bonus.keys()):
            self.rebel_bonus[mid] -= 1
            if self.rebel_bonus[mid] <= 0:
                expired.append(mid)
        for mid in expired:
            del self.rebel_bonus[mid]

    def revive_check(self):
        """复活休眠模块（每 N 步检查一次）"""
        if self.total_steps % self.config.revive_check_interval != 0:
            return []

        active_modules = [m for m in self.modules if m.is_active]
        dormant_modules = [m for m in self.modules if not m.is_active]

        if not dormant_modules:
            return []

        revived = []

        if not active_modules:
            # 全部复活
            for m in dormant_modules:
                m.is_active = True
                m.capital = self.config.revive_capital
                m.reputation = self.config.init_reputation * 0.5
                revived.append(m.module_id)
            return revived

        # 选择最好的活跃模块作为"父本"
        best = max(active_modules, key=lambda m: m.capital * m.reputation)

        # 每次复活一个休眠模块
        dormant = dormant_modules[0]
        self._copy_weights_with_mutation(best, dormant)

        dormant.is_active = True
        dormant.capital = self.config.revive_capital
        dormant.reputation = self.config.init_reputation * 0.8
        dormant.total_bets_won = 0
        dormant.total_bets_lost = 0
        dormant.total_selected = 0
        revived.append(dormant.module_id)

        return revived

    def _copy_weights_with_mutation(self, source, target):
        """复制权重并施加微小变异"""
        target.load_state_dict(source.state_dict())
        with torch.no_grad():
            for param in target.parameters():
                noise = torch.randn_like(param) * self.config.mutation_std
                param.add_(noise)

    def get_stats(self):
        """获取市场整体统计"""
        active = sum(1 for m in self.modules if m.is_active)
        if active == 0:
            return {
                "active_modules": 0,
                "avg_capital": 0.0,
                "avg_reputation": 0.0,
                "total_wins": 0,
                "total_losses": 0,
                "module_details": [],
            }

        active_modules = [m for m in self.modules if m.is_active]
        details = [m.get_stats() for m in self.modules]
        # 计算窗口内选择分布
        window_dist = {}
        if len(self.selection_history) > 0:
            hist = list(self.selection_history)
            for mid in set(hist):
                window_dist[mid] = hist.count(mid) / len(hist)

        return {
            "active_modules": active,
            "avg_capital": np.mean([m.capital for m in active_modules]),
            "avg_reputation": np.mean([m.reputation for m in active_modules]),
            "total_wins": sum(m.total_bets_won for m in self.modules),
            "total_losses": sum(m.total_bets_lost for m in self.modules),
            "module_details": details,
            "revolutions": self.revolution_count,
            "cooldown": self.antitrust_cooldown,
            "window_distribution": window_dist,
        }
