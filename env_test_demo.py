# env_test_demo.py
from GridEnv import GridEnv
import numpy as np
from grid_render import render_grid, save_trajectory_gif

def test_env():
    env = GridEnv(goal=(20, 15))
    obs = env.reset()
    print(f"初始位置: {obs}")
    
    trajectory = [tuple(obs)]
    done = False
    total_reward = 0
    step_count = 0
    
    # 随机策略测试（或简单人工指定）
    while not done:
        action = np.random.randint(0, 4)  # 随机动作
        obs, reward, done, info = env.step(action)
        trajectory.append(tuple(obs))
        total_reward += reward
        step_count += 1
        print(f"Step {step_count}: action={action}, obs={obs}, reward={reward:.2f}, done={done}, info={info}")
        if step_count > 300:  # 安全保护
            break
    
    print(f"总步数: {step_count}, 总奖励: {total_reward:.2f}, 成功: {info['is_success']}")
    
    # 渲染最终地图并保存轨迹GIF
    render_grid(agent_pos=obs, goal=env.goal, trajectory=trajectory, save_path='final_state.png')
    save_trajectory_gif(trajectory, env.goal, gif_path='trajectory.gif')
    
    # 测试边界碰撞：尝试走出边界（应被钳制）
    env.reset(start=(0,0))
    obs, _, _, _ = env.step(0)  # 向上，应停在(0,0)
    assert tuple(obs) == (0,0), "边界碰撞测试失败"
    print("边界碰撞测试通过")
    
    # 测试障碍碰撞：如果起点旁边是障碍，尝试走入障碍应原地不动
    # 假设 (2,3) 是障碍，从 (2,2) 向右走
    env.reset(start=(2,2))
    obs, _, _, info = env.step(3)  # 右
    if (2,3) in env.obstacles:
        assert tuple(obs) == (2,2), "障碍碰撞测试失败"
        assert info['collision'] == True
        print("障碍碰撞测试通过")
    else:
        print("注意：示例中(2,3)不是障碍，请调整测试坐标")

if __name__ == "__main__":
    test_env()
