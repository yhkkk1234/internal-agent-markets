"""
GridWorld 生存环境

Agent 在网格世界中移动，采集食物维持能量，躲避危险区域。
观察空间：局部视野 (one-hot) + 全局特征
动作空间：上/下/左/右/采集/等待
"""
import numpy as np


class GridWorldSurvival:
    """网格生存环境"""

    # 实体编码
    EMPTY = 0
    WALL = 1
    FOOD = 2
    DANGER = 3
    AGENT = 4
    N_ENTITY_TYPES = 5

    def __init__(self, config):
        self.grid_size = config.grid_size
        self.n_foods = config.n_foods
        self.n_dangers = config.n_dangers
        self.vision_range = config.vision_range
        self.max_steps = config.max_steps
        self.max_agent_energy = config.max_agent_energy
        self.init_agent_energy = config.init_agent_energy
        self.food_energy_gain = config.food_energy_gain
        self.danger_energy_loss = config.danger_energy_loss
        self.step_energy_cost = config.step_energy_cost
        self.food_respawn_prob = config.food_respawn_prob
        self.proximity_reward_scale = config.proximity_reward_scale

        # 用于距离奖励
        self._prev_food_dist = None

        # 观察维度
        self.vision_size = 2 * self.vision_range + 1
        self.n_vision_cells = self.vision_size ** 2
        self.obs_dim = self.n_vision_cells * self.N_ENTITY_TYPES + 4  # 4个全局特征
        self.n_actions = 6

        # 内部状态
        self.grid = None
        self.agent_pos = None
        self.agent_energy = None
        self.food_positions = None
        self.danger_positions = None
        self.step_count = None
        self.total_reward = None

    def reset(self):
        """重置环境"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.step_count = 0
        self.total_reward = 0.0

        # 放置墙壁 (边框)
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        # 放置 Agent (随机位置，不在墙上)
        self.agent_pos = self._random_empty_position(avoid_border=True)
        self.grid[self.agent_pos] = self.AGENT
        self.agent_energy = self.init_agent_energy

        # 放置食物
        self.food_positions = []
        self._spawn_foods(self.n_foods)

        # 放置危险区域
        self.danger_positions = []
        self._spawn_dangers(self.n_dangers)

        # 初始化距离跟踪
        self._prev_food_dist = self._min_food_distance() if self.food_positions else None

        return self._get_observation()

    def step(self, action):
        """执行动作，返回 (obs, reward, done, info)"""
        prev_energy = self.agent_energy
        info = {"action": action, "event": "move"}

        # 移动动作
        if action == 0:      # 上
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 1:    # 下
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 2:    # 左
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == 3:    # 右
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        elif action == 4:    # 采集 (原地)
            new_pos = self.agent_pos
        elif action == 5:    # 等待
            new_pos = self.agent_pos
            info["event"] = "wait"

        # 检查移动合法性
        if action in (0, 1, 2, 3):
            if not self._is_walkable(new_pos):
                new_pos = self.agent_pos
                info["event"] = "blocked"

        # 更新位置
        old_pos = self.agent_pos
        self.grid[old_pos] = self.EMPTY
        self.grid[new_pos] = self.AGENT
        self.agent_pos = new_pos

        # 处理新位置的效果
        cell_type_before = self._get_cell_type(new_pos)

        if cell_type_before == self.FOOD:
            # 采集食物
            self.agent_energy = min(self.agent_energy + self.food_energy_gain, self.max_agent_energy)
            self.food_positions.remove(new_pos)
            info["event"] = "collected_food"
            # 食物可能重生
            if np.random.random() < self.food_respawn_prob:
                self._spawn_single_food()

        elif cell_type_before == self.DANGER:
            # 踩到危险
            self.agent_energy -= self.danger_energy_loss
            info["event"] = "hit_danger"

        # 采集动作 (action==4)：如果在食物上则采集
        if action == 4 and cell_type_before == self.FOOD:
            pass  # 已在上面处理
        elif action == 4:
            info["event"] = "collect_empty"

        # 每步消耗能量
        self.agent_energy -= self.step_energy_cost
        self.step_count += 1

        # 距离奖励：接近食物时获得小额正奖励
        proximity_reward = 0.0
        if self.proximity_reward_scale > 0 and self.food_positions:
            cur_dist = self._min_food_distance()
            if self._prev_food_dist is not None:
                dist_diff = self._prev_food_dist - cur_dist
                proximity_reward = dist_diff * self.proximity_reward_scale
            self._prev_food_dist = cur_dist

        # 采集到食物时跳过距离奖励 (避免覆盖+15正信号)
        if info["event"] == "collected_food":
            proximity_reward = 0.0

        reward = self.agent_energy - prev_energy + proximity_reward

        self.total_reward += reward

        # 终止判断
        done = False
        if self.agent_energy <= 0:
            self.agent_energy = 0
            done = True
            info["event"] = "starved"
        elif self.step_count >= self.max_steps:
            done = True
            info["event"] = "timeout"

        info["agent_energy"] = self.agent_energy
        info["step"] = self.step_count

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """获取 Agent 观察"""
        # 局部视野: vision_size x vision_size, one-hot 编码
        vision = np.zeros((self.vision_size, self.vision_size, self.N_ENTITY_TYPES), dtype=np.float32)

        r_center, c_center = self.agent_pos
        for dr in range(-self.vision_range, self.vision_range + 1):
            for dc in range(-self.vision_range, self.vision_range + 1):
                vr, vc = dr + self.vision_range, dc + self.vision_range
                gr, gc = r_center + dr, c_center + dc
                if 0 <= gr < self.grid_size and 0 <= gc < self.grid_size:
                    entity = self.grid[gr, gc]
                else:
                    entity = self.WALL  # 越界视为墙
                vision[vr, vc, entity] = 1.0

        # 全局特征
        global_features = np.array([
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.agent_energy / self.max_agent_energy,
            self.step_count / self.max_steps,
        ], dtype=np.float32)

        # 合并
        obs = np.concatenate([vision.flatten(), global_features])
        return obs

    def _random_empty_position(self, avoid_border=False):
        """随机找一个空位"""
        while True:
            if avoid_border:
                r = np.random.randint(1, self.grid_size - 1)
                c = np.random.randint(1, self.grid_size - 1)
            else:
                r = np.random.randint(0, self.grid_size)
                c = np.random.randint(0, self.grid_size)
            if self.grid[r, c] == self.EMPTY:
                return (r, c)

    def _is_walkable(self, pos):
        """检查位置是否可行走"""
        r, c = pos
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return False
        return self.grid[r, c] != self.WALL

    def _get_cell_type(self, pos):
        """获取某位置上的实体类型"""
        for food_pos in self.food_positions:
            if pos == food_pos:
                return self.FOOD
        for danger_pos in self.danger_positions:
            if pos == danger_pos:
                return self.DANGER
        return self.EMPTY

    def _min_food_distance(self):
        """计算到最近食物的曼哈顿距离"""
        if not self.food_positions:
            return 0
        return min(
            abs(self.agent_pos[0] - f[0]) + abs(self.agent_pos[1] - f[1])
            for f in self.food_positions
        )

    def _spawn_foods(self, n):
        """批量放置食物"""
        for _ in range(n):
            self._spawn_single_food()

    def _spawn_single_food(self):
        """放置单个食物"""
        pos = self._random_empty_position()
        if pos:
            self.food_positions.append(pos)
            self.grid[pos] = self.FOOD

    def _spawn_dangers(self, n):
        """批量放置危险区域"""
        for _ in range(n):
            pos = self._random_empty_position()
            if pos:
                self.danger_positions.append(pos)
                self.grid[pos] = self.DANGER

    def render(self):
        """渲染当前网格状态"""
        symbol_map = {
            self.EMPTY: '.',
            self.WALL: '#',
            self.FOOD: 'F',
            self.DANGER: 'X',
            self.AGENT: 'A',
        }
        lines = []
        for r in range(self.grid_size):
            row = ''
            for c in range(self.grid_size):
                row += symbol_map.get(self.grid[r, c], '?') + ' '
            lines.append(row)
        lines.append(f"Energy: {self.agent_energy}/{self.max_agent_energy} | Step: {self.step_count}")
        return '\n'.join(lines)
