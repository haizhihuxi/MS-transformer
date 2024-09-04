from datetime import datetime
import math
import pandas as pd
import torch
import os
from config import Config
import gc
import numpy as np

def get_all_lengths(lst):
    if isinstance(lst, list):
        lengths = [len(lst)]
        child_lengths = [get_all_lengths(item) for item in lst]
        # 将子列表的长度拼接成一个更全面的列表
        for child in child_lengths:
            if child:  # 非空子列表长度
                lengths.append(child)
        return lengths
    else:
        return []



def select_trajectories_for_comparison(real_trajectories, all_trajectories, threshold):
    """
    根据起始点的距离选择用于比较的轨迹。如果所有轨迹的前两个点中任何一个与真实轨迹的对应点的距离大于阈值，则不比较这些轨迹。

    参数:
    - real_trajectories (np.array): 真实轨迹数组，形状为 (32, 150, 2)。
    - all_trajectories (list of np.array): 所有待比较轨迹的数组，形状为 (N, M, 2)。
    - threshold (float): 选择轨迹的距离阈值。

    返回:
    - list of np.array: 从 all_trajectories 中选出的距离在阈值之内的轨迹数组。
    """
    # 收集 all_trajectories 中每个轨迹的前两个点
    all_start_points = np.array([traj[4] for traj in all_trajectories])  # (9144, 2)
    # 提取真实轨迹和所有轨迹的起始点
    real_start_points = real_trajectories[:, 4, :2]  # torch.Size([32, 2])

    # 计算所有起始点与每个真实起始点之间的距离
    distance_squared = (real_start_points[:, np.newaxis, :] - all_start_points[np.newaxis, :, :]) ** 2
    distances = np.sqrt(distance_squared.sum(axis=2))  # torch.Size([32, 9144])
    # 找到 all_trajectories 中每条轨迹到任何真实轨迹的最小距离
    min_distances = np.min(distances.numpy(), axis=0)  # (9144,)

    # 选择最小距离小于或等于阈值的轨迹
    selected_indices = np.where(min_distances <= threshold)[0]

    selected_trajectories = [all_trajectories[i] for i in selected_indices]
    retain_ratio = 0.1  # 你想要保留的数据点的比例

    #新的轨迹列表，用于存储处理后的轨迹
    retained_trajectories = []

    retain_count = int(len(selected_trajectories) * retain_ratio)
    # 生成随机但不重复的索引以选择数据点
    retained_indices = np.random.choice(len(selected_trajectories), size=retain_count, replace=False)
    # 将处理后的轨迹添加到新列表中
    for i in retained_indices:
        retained_trajectories.append(selected_trajectories[i])

    return retained_trajectories


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


# 计算离散弗雷歇距离的递归函数
def frechet_dist_rec_recursive(curve1, curve2, i, j, memo):
    if memo[i, j] > -1:
        return memo[i, j]
    elif i == 0 and j == 0:
        memo[i, j] = euclidean_distance(curve1[0], curve2[0])
    elif i > 0 and j == 0:
        memo[i, j] = max(frechet_dist_rec_recursive(curve1, curve2, i - 1, 0, memo),
                         euclidean_distance(curve1[i], curve2[0]))
    elif i == 0 and j > 0:
        memo[i, j] = max(frechet_dist_rec_recursive(curve1, curve2, 0, j - 1, memo),
                         euclidean_distance(curve1[0], curve2[j]))
    elif i > 0 and j > 0:
        memo[i, j] = max(min(frechet_dist_rec_recursive(curve1, curve2, i - 1, j, memo),
                             frechet_dist_rec_recursive(curve1, curve2, i - 1, j - 1, memo),
                             frechet_dist_rec_recursive(curve1, curve2, i, j - 1, memo)),
                         euclidean_distance(curve1[i], curve2[j]))
    else:
        memo[i, j] = float('inf')
    return memo[i, j]


# 计算离散弗雷歇距离
def discrete_frechet_distance(curve1, curve2):
    curve1 = np.asarray(curve1)
    curve2 = np.asarray(curve2)
    m, n = len(curve1), len(curve2)
    memo = np.full((m, n), -1.0)
    return frechet_dist_rec_recursive(curve1, curve2, m - 1, n - 1, memo)


def compare_trajectories(trajectory1, trajectory2, interval=20, threshold=0.1):
    change_detected = False  # 标记是否检测到突变

    for i in range(0, len(trajectory1), interval):
        # 确保每个段落结束时都有足够的数据点
        segment1 = trajectory1[i:i + interval]
        segment2 = trajectory2[i:i + interval]
        if len(segment1) < interval or len(segment2) < interval:
            continue
        # 筛选非本条轨迹的比较轨迹，如果起始点差距过大则跳过
        if euclidean_distance(trajectory1[1], trajectory2[1]) > 0.05:
            continue
        distance = discrete_frechet_distance(segment1, segment2)

        # 检测突变：如果这是第一个距离，就跳过突变检测
        if i == 0:
            prev_distance = distance
            continue

        # 根据突变的定义来设置change_detected标志
        if (distance > threshold >= prev_distance) or (
                distance <= threshold < prev_distance):
            change_detected = True

        prev_distance = distance

    # 根据是否检测到突变返回不同的结果
    if change_detected:
        return calculate_frechet_distances_5(trajectory1, trajectory2)
    else:
        # 根据trajectory1的长度返回适当大小的0填充数值列表
        result_length = int(np.ceil(len(trajectory1) / 5))
        # 返回一个长度为result_length的0值列表
        return [1] * result_length


def calculate_frechet_distances_5(trajectory1, trajectory2, interval=5):
    distances = []  # 存储每个间隔的离散弗雷歇距离

    # 确保轨迹1足够长
    if len(trajectory1) < interval:
        raise ValueError("Trajectory1 is too short for the given interval")

    last_distance = None
    for i in range(0, len(trajectory1), interval):
        segment1 = trajectory1[i:i + interval]
        if i + interval <= len(trajectory2):
            segment2 = trajectory2[i:i + interval]
        else:
            # 如果轨迹2长度不足，使用轨迹2的最后一段进行比较
            segment2 = trajectory2[-interval:]

        # 计算离散弗雷歇距离
        distance = discrete_frechet_distance(segment1, segment2)
        last_distance = distance

        # 将计算出的距离添加到列表中
        distances.append(distance)

    # 如果轨迹1还有剩余段未比较，则用最后一次比较的距离填充
    while len(distances) * interval < len(trajectory1):
        distances.append(last_distance)

    return distances


def interpolate_trajectories_np(trajectories, target_length):
    """
    轨迹插值，统一长度
    """
    interpolated_trajectories = []
    for trajectory in trajectories:
        # 现在假设trajectory已经是numpy数组，不需要再调用.numpy()
        current_length = trajectory.shape[0]
        x = np.linspace(0, current_length - 1, current_length)
        x_new = np.linspace(0, current_length - 1, target_length)
        # 直接使用trajectory，而不是调用trajectory.numpy()
        interpolated_trajectory_x = np.interp(x_new, x, trajectory[:, 0])
        interpolated_trajectory_y = np.interp(x_new, x, trajectory[:, 1])
        interpolated_trajectory = np.stack((interpolated_trajectory_x, interpolated_trajectory_y), axis=-1)
        interpolated_trajectories.append(torch.from_numpy(interpolated_trajectory))
        result = torch.stack(interpolated_trajectories)
    # print(f"成功插值，结果的维度是{result.shape}")

    return result


def self_adaption_weight(real_x):
    # cpu上进行插值处理，和挑选符合条件的轨迹
    real_x_cpu = real_x.cpu()
    datapath_train = os.path.join(Config.datadir, Config.trainset_name)
    data = pd.read_pickle(datapath_train)
    moving_threshold = 0.05  # 移除速度在0.05以下的轨迹
    for V in data:
        try:
            moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]  # 返回元组形式
        except:
            moving_idx = len(V["traj"]) - 1  # This track will be removed这条轨迹会被移除
        V["traj"] = V["traj"][moving_idx:, :]

    Data = [x for x in data if not np.isnan(x["traj"]).any() and len(x["traj"]) > Config.min_seqlen]
    # #  data 是包含所有轨迹的列表
    traj_arrays = [np.array(Data[idx]["traj"][:, [0, 1]], dtype=np.float32) for idx in range(len(Data))]

    del Data, data
    gc.collect()
    # 选择用于比较的轨迹
    selected_trajectories = select_trajectories_for_comparison(real_x_cpu, traj_arrays, Config.traj_compary_threshold)
    # 插值统一长度
    print(f"选择的轨迹数目：{len(selected_trajectories)}")
    seqlen = min(real_x_cpu.shape[1], Config.max_seqlen)
    interpolated_trajectories = interpolate_trajectories_np(selected_trajectories, seqlen)

    # 直接在主线程中顺序处理每条轨迹
    interpolated_trajectories = np.array(interpolated_trajectories)

    start_time = datetime.now()
    start_formatted_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start_formatted_time}] 开始计算权重")

    list_length = math.ceil(real_x_cpu.size(1) / 5)

    results = np.zeros((len(real_x_cpu), len(interpolated_trajectories), list_length))
    # 遍历第一个轨迹数组中的每条轨迹
    for i in range(len(real_x_cpu)):
        traj1 = real_x_cpu[i]  # 第一个数组中的第i条轨迹

        # for i, real_traj in enumerate(real_x_cpu):
        #
        #     plt.figure()
        #     # 绘制真实轨迹和选中轨迹
        #     plt.plot(real_traj[:, 0], real_traj[:, 1], 'r', label='Real Trajectory')
        #     plt.scatter(real_traj[:1, 0], real_traj[:1, 1], c='blue', label='Real First 5 Points', zorder=5)
        #
        #     # 设置图像标题和图例
        #     plt.title(f'Comparison: Real Trajectory {i + 1}')
        #     plt.xlabel('X Coordinate')
        #     plt.ylabel('Y Coordinate')
        #     plt.xlim([-0.05, 1.05])
        #     plt.ylim([-0.05, 1.05])
        #     plt.legend()
        #
        #     # 保存图像
        #     plt.savefig(f'E:/代码/traisformer修改/测试results/trajectory_plots/real_{i + 1}.png')
        #     plt.close()

        # 遍历第二个轨迹数组中的每条轨迹
        for j in range(len(interpolated_trajectories)):
            traj2 = interpolated_trajectories[j]  # 第二个数组中的第j条轨迹

            # 比较两条轨迹并存储结果
            results[i, j] = compare_trajectories(traj1[:, 2], traj2)

    summed_array = np.sum(results, axis=1)

    transformed_array = np.zeros_like(summed_array)
    # 遍历每条轨迹
    for i in range(len(summed_array)):
        for j in range(1, len(summed_array[i])):
            if summed_array[i, j] != summed_array[i, j - 1]:
                # 如果分母不为零，则正常计算变化率
                change_rate = abs(summed_array[i, j] - summed_array[i, j - 1]) / summed_array[i, j - 1]
            else:
                # 如果分子为零，则将变化率设置为0或其他适当的值
                change_rate = 1
            transformed_array[i, j] = change_rate
        # 前两个点相同
        transformed_array[i, 0] = transformed_array[i, 1]
        transformed_array[i, j] = transformed_array[i, j - 1]

    # 归一化处理
    min_vals = transformed_array.min(axis=1, keepdims=True)
    max_vals = transformed_array.max(axis=1, keepdims=True)

    # 计算分子
    numerator = transformed_array - min_vals + np.finfo(float).eps
    # 计算分母
    denominator = max_vals - min_vals + np.finfo(float).eps
    # 检查哪些列的最大值和最小值相等
    cols_equal = max_vals == min_vals
    # 对于最大值和最小值相等的列，设置为1；对于其他列，进行正常的归一化计算
    transformed_array = np.where(cols_equal, 1, numerator / denominator)

    end_time = datetime.now()
    end_formatted_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{end_formatted_time}] 全部轨迹的权重计算完毕")

    return transformed_array
