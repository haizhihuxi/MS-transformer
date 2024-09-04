import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


for i in range(0, 10):
    cluster = i
    # 加载数据
    data = pd.read_pickle(f'./clusters/train/cluster_{cluster}_data.pkl')

    # 提取所有轨迹的经纬度坐标
    trajectories = [np.array(d['traj'])[:, :2] for d in data if 'traj' in d]

    # 确保输出文件夹存在
    output_folder = f'clusters/trajectory_plots_{cluster}/train'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 为每个轨迹绘图并保存
    for i, traj in enumerate(trajectories):
        plt.figure(figsize=(10, 6))
        plt.plot(traj[:, 1], traj[:, 0], marker='o')  # 绘制轨迹，注意经纬度的顺序
        plt.title(f'Trajectory {i+1}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)

        # 保存图像到文件夹
        plt.savefig(os.path.join(output_folder, f'trajectory_{i+1}.png'))
        plt.close()

    # 验证加载数据
    data = pd.read_pickle(f'./clusters/validation/cluster_{cluster}_data.pkl')

    # 提取所有轨迹的经纬度坐标
    trajectories = [np.array(d['traj'])[:, :2] for d in data if 'traj' in d]

    # 确保输出文件夹存在
    output_folder = f'clusters/trajectory_plots_{cluster}/valid'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 为每个轨迹绘图并保存
    for i, traj in enumerate(trajectories):
        plt.figure(figsize=(10, 6))
        plt.plot(traj[:, 1], traj[:, 0], marker='o')  # 绘制轨迹，注意经纬度的顺序
        plt.title(f'Trajectory {i+1}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)

        # 保存图像到文件夹
        plt.savefig(os.path.join(output_folder, f'trajectory_{i+1}.png'))
        plt.close()  # 关闭图形窗口以节省资源

    # 测试加载数据
    data = pd.read_pickle(f'./clusters/test/cluster_{cluster}_data.pkl')

    # 提取所有轨迹的经纬度坐标
    trajectories = [np.array(d['traj'])[:, :2] for d in data if 'traj' in d]

    # 确保输出文件夹存在
    output_folder = f'clusters/trajectory_plots_{cluster}/test'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 为每个轨迹绘图并保存
    for i, traj in enumerate(trajectories):
        plt.figure(figsize=(10, 6))
        plt.plot(traj[:, 1], traj[:, 0], marker='o')  # 绘制轨迹，注意经纬度的顺序
        plt.title(f'Trajectory {i + 1}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)

        # 保存图像到文件夹
        plt.savefig(os.path.join(output_folder, f'trajectory_{i + 1}.png'))
        plt.close()  # 关闭图形窗口以节省资源
