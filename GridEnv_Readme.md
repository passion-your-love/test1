# GridEnv 接口文档

## 环境描述
- 尺寸：21行 × 16列，坐标 (row, col)，row∈[0,20], col∈[0,15]
- 障碍物：硬编码于 `grid_obstacle_const.py`
- 目标：固定 (20,15)，可在初始化时修改

## 状态空间
- `observation_space`: Box(2,)，表示 (row, col)，整型

## 动作空间
- 0: 上, 1: 下, 2: 左, 3: 右（边界内移动，越界则停在原地）

## 接口
- `reset(start=None)` -> obs (ndarray)
  - 若未指定起点，随机生成非障碍、非目标位置
- `step(action)` -> (obs, reward, done, info)
  - reward: 每步-0.1，碰撞-5，到达目标+10（额外）
  - info: 包含 `collision`(bool), `steps`, `is_success`

## 依赖
- gym, numpy, matplotlib, imageio

## 快速测试
运行 `python env_test_demo.py` 即可验证环境基本功能。
