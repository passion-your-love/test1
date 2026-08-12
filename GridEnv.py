# GridEnv.py
import gym
from gym import spaces
import numpy as np
from grid_obstacle_const import OBSTACLE_COORDS

class GridEnv(gym.Env):
    def __init__(self, goal=(20, 15), max_steps=200):
        super(GridEnv, self).__init__()
        
        self.rows = 21
        self.cols = 16
        self.goal = goal          # 固定目标，也可在reset时随机指定
        self.max_steps = max_steps
        
        # 障碍集合（快速查找）
        self.obstacles = set(OBSTACLE_COORDS)
        
        # 动作空间：0-上, 1-下, 2-左, 3-右
        self.action_space = spaces.Discrete(4)
        # 观测空间：离散坐标 (row, col)，这里用Box表示连续值，也可以使用Discrete扁平化
        self.observation_space = spaces.Box(low=0, high=max(self.rows, self.cols), 
                                            shape=(2,), dtype=np.int32)
        
        self.state = None
        self.steps = 0
        self.collision_count = 0   # 用于记录碰撞次数（可选）
        
    def reset(self, start=None):
        """重置环境，若未指定起点则随机生成（避开障碍和目标）"""
        if start is None:
            # 随机采样有效起点（非障碍、非目标）
            while True:
                r = np.random.randint(0, self.rows)
                c = np.random.randint(0, self.cols)
                if (r, c) not in self.obstacles and (r, c) != self.goal:
                    break
            start = (r, c)
        self.state = start
        self.steps = 0
        self.collision_count = 0
        return np.array(self.state, dtype=np.int32)
    
    def step(self, action):
        """执行动作，返回 (next_obs, reward, done, info)"""
        r, c = self.state
        # 计算下一位置（边界内）
        if action == 0:    # 上
            nr, nc = max(r-1, 0), c
        elif action == 1:  # 下
            nr, nc = min(r+1, self.rows-1), c
        elif action == 2:  # 左
            nr, nc = r, max(c-1, 0)
        elif action == 3:  # 右
            nr, nc = r, min(c+1, self.cols-1)
        else:
            raise ValueError("Invalid action")
        
        # 碰撞检测（障碍物）
        if (nr, nc) in self.obstacles:
            # 撞障碍：原地不动，给予负奖励，但为了保持MDP，也可选择停在原地
            nr, nc = r, c   # 回退到原位置
            collision = True
        else:
            collision = False
        
        self.state = (nr, nc)
        self.steps += 1
        if collision:
            self.collision_count += 1
        
        # 奖励函数设计（可根据要求调整）
        reward = self._compute_reward(nr, nc, collision)
        
        # 终止判定
        done = False
        if (nr, nc) == self.goal:
            done = True
            reward += 10.0   # 额外到达奖励（已在compute中可能包含）
        elif self.steps >= self.max_steps:
            done = True
        
        info = {
            'collision': collision,
            'steps': self.steps,
            'is_success': (nr, nc) == self.goal
        }
        return np.array(self.state, dtype=np.int32), reward, done, info
    
    def _compute_reward(self, r, c, collision):
        """奖励函数：到达+10，每步-0.1，碰撞-5，距离惩罚（可选）"""
        reward = -0.1   # 每步代价
        if collision:
            reward -= 5.0
        # 目标到达奖励在step中额外加，也可以在这里加
        if (r, c) == self.goal:
            reward += 10.0
        # 可选：距离接近奖励（为了加速收敛），但简单任务可不用
        return reward
    
    def render(self, mode='human', save_path=None):
        """绘制当前状态（文本或图形），具体实现见grid_render.py，这里调用外部函数"""
        # 为保持Gym风格，建议渲染实现放在grid_render.py，此处可留空或调用的封装
        pass
    
    def close(self):
        pass
