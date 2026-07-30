"""
双资源 GridWorld 生存环境

Agent 需要同时维护能量(energy)和水(hydration)两种资源。
任一资源归零即死亡。两种资源在不同位置，产生真正的策略冲突：
- 先补食物还是先补水？
- 保守平衡还是激进冒险？
"""
import numpy as np


class DualResourceGridWorld:
    """双资源网格生存环境"""

    EMPTY = 0
    WALL = 1
    FOOD = 2
    WATER = 3
    DANGER = 4
    AGENT = 5
    N_ENTITY_TYPES = 6

    def __init__(self, config):
        self.grid_size = config.grid_size
        self.n_foods = config.n_foods
        self.n_waters = getattr(config, 'n_waters', 3)
        self.n_dangers = config.n_dangers
        self.vision_range = config.vision_range
        self.max_steps = config.max_steps
        self.max_energy = config.max_agent_energy
        self.init_energy = config.init_agent_energy
        self.max_water = getattr(config, 'max_water', 100)
        self.init_water = getattr(config, 'init_water', 60)
        self.food_energy_gain = config.food_energy_gain
        self.water_gain = getattr(config, 'water_gain', 15)
        self.danger_loss = config.danger_energy_loss
        self.step_energy_cost = config.step_energy_cost
        self.step_water_cost = getattr(config, 'step_water_cost', 1)
        self.food_respawn_prob = config.food_respawn_prob
        self.water_respawn_prob = getattr(config, 'water_respawn_prob', 0.3)
        self.proximity_reward_scale = config.proximity_reward_scale

        self.vision_size = 2 * self.vision_range + 1
        self.n_vision_cells = self.vision_size ** 2
        self.obs_dim = self.n_vision_cells * self.N_ENTITY_TYPES + 5
        self.n_actions = 6

        self.grid = None
        self.agent_pos = None
        self.energy = None
        self.water = None
        self.food_positions = None
        self.water_positions = None
        self.danger_positions = None
        self.step_count = None
        self.total_reward = None
        self._prev_food_dist = None
        self._prev_water_dist = None

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.step_count = 0
        self.total_reward = 0.0

        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        self.agent_pos = self._random_empty_position(avoid_border=True)
        self.grid[self.agent_pos] = self.AGENT
        self.energy = self.init_energy
        self.water = self.init_water

        self.food_positions = []
        self._spawn_resources(self.n_foods, self.FOOD, self.food_positions)
        self.water_positions = []
        self._spawn_resources(self.n_waters, self.WATER, self.water_positions)
        self.danger_positions = []
        self._spawn_resources(self.n_dangers, self.DANGER, self.danger_positions)

        self._prev_food_dist = self._min_dist(self.food_positions) if self.food_positions else None
        self._prev_water_dist = self._min_dist(self.water_positions) if self.water_positions else None

        return self._get_observation()

    def step(self, action):
        prev_energy, prev_water = self.energy, self.water

        info = {
            "action": action, "event": "move",
            "food_delta": 0, "water_delta": 0, "danger_delta": 0,
            "proximity_food": 0, "proximity_water": 0,
            "step_energy_cost": -self.step_energy_cost,
            "step_water_cost": -self.step_water_cost,
        }

        if action == 0:
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 1:
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 2:
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == 3:
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        else:
            new_pos = self.agent_pos

        if action in (0, 1, 2, 3) and not self._is_walkable(new_pos):
            new_pos = self.agent_pos
            info["event"] = "blocked"

        self.grid[self.agent_pos] = self.EMPTY
        self.grid[new_pos] = self.AGENT
        self.agent_pos = new_pos

        cell_type = self._get_cell_type(new_pos)

        if cell_type == self.FOOD:
            self.energy = min(self.energy + self.food_energy_gain, self.max_energy)
            self.food_positions.remove(new_pos)
            info["event"] = "collected_food"
            info["food_delta"] = self.food_energy_gain
            if np.random.random() < self.food_respawn_prob:
                self._spawn_single(self.FOOD, self.food_positions)

        elif cell_type == self.WATER:
            self.water = min(self.water + self.water_gain, self.max_water)
            self.water_positions.remove(new_pos)
            info["event"] = "collected_water"
            info["water_delta"] = self.water_gain
            if np.random.random() < self.water_respawn_prob:
                self._spawn_single(self.WATER, self.water_positions)

        elif cell_type == self.DANGER:
            self.energy = max(0, self.energy - self.danger_loss)
            self.water = max(0, self.water - self.danger_loss)
            info["event"] = "hit_danger"
            info["danger_delta"] = -self.danger_loss

        self.energy -= self.step_energy_cost
        self.water -= self.step_water_cost
        self.step_count += 1

        if self.food_positions:
            d = self._min_dist(self.food_positions)
            if self._prev_food_dist is not None:
                info["proximity_food"] = (self._prev_food_dist - d) * self.proximity_reward_scale
            self._prev_food_dist = d
        if self.water_positions:
            d = self._min_dist(self.water_positions)
            if self._prev_water_dist is not None:
                info["proximity_water"] = (self._prev_water_dist - d) * self.proximity_reward_scale
            self._prev_water_dist = d

        if info["event"] in ("collected_food", "collected_water"):
            info["proximity_food"] = 0
            info["proximity_water"] = 0

        reward = (info["food_delta"] + info["water_delta"] + info["danger_delta"]
                  + info["proximity_food"] + info["proximity_water"]
                  + info["step_energy_cost"] + info["step_water_cost"])
        self.total_reward += reward

        done = False
        if self.energy <= 0 or self.water <= 0:
            self.energy = max(0, self.energy)
            self.water = max(0, self.water)
            done = True
            if self.energy <= 0 and self.water <= 0:
                info["event"] = "starved_and_dehydrated"
            elif self.energy <= 0:
                info["event"] = "starved"
            else:
                info["event"] = "dehydrated"
        elif self.step_count >= self.max_steps:
            done = True
            info["event"] = "timeout"

        info["energy"] = self.energy
        info["water"] = self.water
        info["step"] = self.step_count
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        vision = np.zeros((self.vision_size, self.vision_size, self.N_ENTITY_TYPES), dtype=np.float32)
        rc, cc = self.agent_pos
        for dr in range(-self.vision_range, self.vision_range + 1):
            for dc in range(-self.vision_range, self.vision_range + 1):
                vr, vc = dr + self.vision_range, dc + self.vision_range
                gr, gc = rc + dr, cc + dc
                if 0 <= gr < self.grid_size and 0 <= gc < self.grid_size:
                    e = self.grid[gr, gc]
                else:
                    e = self.WALL
                vision[vr, vc, e] = 1.0

        global_feat = np.array([
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.energy / self.max_energy,
            self.water / self.max_water,
            self.step_count / self.max_steps,
        ], dtype=np.float32)

        return np.concatenate([vision.flatten(), global_feat])

    def _random_empty_position(self, avoid_border=False):
        while True:
            lo, hi = (1, self.grid_size - 1) if avoid_border else (0, self.grid_size)
            r = np.random.randint(lo, hi)
            c = np.random.randint(lo, hi)
            if self.grid[r, c] == self.EMPTY:
                return (r, c)

    def _is_walkable(self, pos):
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size and self.grid[r, c] != self.WALL

    def _get_cell_type(self, pos):
        if pos in self.food_positions:
            return self.FOOD
        if pos in self.water_positions:
            return self.WATER
        if pos in self.danger_positions:
            return self.DANGER
        return self.EMPTY

    def _min_dist(self, positions):
        if not positions:
            return 0
        return min(abs(self.agent_pos[0] - p[0]) + abs(self.agent_pos[1] - p[1]) for p in positions)

    def _spawn_resources(self, n, entity_type, pos_list):
        for _ in range(n):
            self._spawn_single(entity_type, pos_list)

    def _spawn_single(self, entity_type, pos_list):
        pos = self._random_empty_position()
        if pos:
            pos_list.append(pos)
            self.grid[pos] = entity_type

    def render(self):
        sym = {self.EMPTY: '.', self.WALL: '#', self.FOOD: 'F',
               self.WATER: 'W', self.DANGER: 'X', self.AGENT: 'A'}
        lines = [''.join(sym.get(self.grid[r, c], '?') + ' ' for c in range(self.grid_size))
                 for r in range(self.grid_size)]
        lines.append(f"Energy:{self.energy}/{self.max_energy} Water:{self.water}/{self.max_water} Step:{self.step_count}")
        return '\n'.join(lines)
