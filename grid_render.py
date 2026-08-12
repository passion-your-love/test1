# grid_render.py
import matplotlib.pyplot as plt
import numpy as np
from grid_obstacle_const import OBSTACLE_COORDS
import imageio
import os

def render_grid(agent_pos=None, goal=(20,15), trajectory=None, save_path=None):
    """
    绘制栅格地图，可显示当前智能体位置、目标、轨迹。
    - agent_pos: (row, col) 或 None
    - trajectory: list of (row, col) 历史路径
    - save_path: 若提供则保存为png，否则显示
    """
    rows, cols = 21, 16
    grid = np.zeros((rows, cols))
    # 标记障碍
    for r, c in OBSTACLE_COORDS:
        grid[r, c] = 1  # 黑色
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # 显示栅格：0白色，1黑色
    ax.imshow(grid, cmap='gray', origin='upper', interpolation='none')
    
    # 绘制轨迹（从起点到当前点）
    if trajectory:
        traj_arr = np.array(trajectory)
        ax.plot(traj_arr[:, 1], traj_arr[:, 0], 'b-', linewidth=2, alpha=0.7, label='轨迹')
    
    # 绘制目标（绿色星）
    ax.plot(goal[1], goal[0], 'g*', markersize=15, label='目标')
    
    # 绘制当前智能体（红色圆）
    if agent_pos:
        ax.plot(agent_pos[1], agent_pos[0], 'ro', markersize=10, label='智能体')
    
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)
    ax.legend()
    ax.set_title('Grid World (21x16)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片保存至 {save_path}")
    else:
        plt.show()
    plt.close()

def save_trajectory_gif(trajectory, goal, gif_path='trajectory.gif', fps=2):
    """
    将轨迹保存为GIF动画，每帧显示智能体逐步移动。
    """
    frames = []
    rows, cols = 21, 16
    for i in range(1, len(trajectory)+1):
        fig, ax = plt.subplots(figsize=(8, 6))
        grid = np.zeros((rows, cols))
        for r, c in OBSTACLE_COORDS:
            grid[r, c] = 1
        ax.imshow(grid, cmap='gray', origin='upper', interpolation='none')
        # 绘制已经过的轨迹
        traj_part = trajectory[:i]
        if traj_part:
            arr = np.array(traj_part)
            ax.plot(arr[:, 1], arr[:, 0], 'b-', linewidth=2, alpha=0.7)
        # 目标
        ax.plot(goal[1], goal[0], 'g*', markersize=15)
        # 当前位置
        cur = trajectory[i-1]
        ax.plot(cur[1], cur[0], 'ro', markersize=10)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.set_xlim(-0.5, cols-0.5)
        ax.set_ylim(rows-0.5, -0.5)
        ax.tick_params(which='minor', size=0)
        plt.close(fig)
        # 转换为图像
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"GIF已保存至 {gif_path}")

# 简单测试渲染（独立运行）
if __name__ == "__main__":
    # 显示空白地图
    render_grid()
    # 显示带有模拟轨迹的图
    traj = [(0,0), (1,0), (2,0), (3,1), (4,2), (5,3)]
    render_grid(agent_pos=(5,3), goal=(20,15), trajectory=traj, save_path='demo_map.png')
