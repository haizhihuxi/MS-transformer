import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


single_traj = np.array(
 [[0.4538512, 0.93111295],
  [0.447278, 0.9208315],
  [0.44378227, 0.9065539],
  [0.4404194, 0.8923535],
  [0.43635, 0.8796989],
  [0.4293567, 0.8702535],
  [0.4227772, 0.8607026],
  [0.41609713, 0.8511952],
  [0.4096128, 0.8416452],
  [0.402936, 0.8320185],
  [0.3963792, 0.82250077],
  [0.3900592, 0.81302965],
  [0.3837368, 0.80342776],
  [0.3770348, 0.7937826],
  [0.3704502, 0.7839283],
  [0.3638066, 0.77414924],
  [0.356874, 0.7641011],
  [0.349756, 0.75375295],
  [0.34234625, 0.7435738],
  [0.33515716, 0.7334231],
  [0.32805935, 0.7229923],
  [0.32015637, 0.71567315],
  [0.3103541, 0.7169852],
  [0.3006908, 0.7175537],
  [0.2914768, 0.71976036],
  [0.2849136, 0.7333536],
  [0.27834392, 0.74845487],
  [0.27189332, 0.76448965],
  [0.2653708, 0.7798474],
  [0.2592728, 0.7944426],
  [0.2531148, 0.8089],
  [0.2469912, 0.82234627],
  [0.24006425, 0.8352142],
  [0.2328182, 0.84743166],
  [0.225556, 0.85985667],
  [0.21717988, 0.8618164],
  [0.20868778, 0.86240965],
  [0.1996468, 0.86195195],
  [0.19023341, 0.8608249],
  [0.1808208, 0.8595493],
  [0.17134, 0.8623381],
  [0.1617972, 0.8657704], ])

# 模型需要的输入序列长度
seq_length = 18

# 假设你之前的归一化范围是这样的：
lat_min, lat_max = 10.3, 13.0
lon_min, lon_max = 55.5, 58.0


# 反归一化函数，根据你的归一化方式进行反归一化
def denormalize_coords(coords, min_val, max_val):
    return coords * (max_val - min_val) + min_val


# 创建保存图片的文件夹
output_folder = '../picture-results/第二组/'
os.makedirs(output_folder, exist_ok=True)

# 遍历每个簇，加载对应的模型并进行预测和绘图
for i in range(18):
    trajectory_input = single_traj[:18].reshape(1, seq_length, 2)
    # 加载训练好的模型
    model_path = f'best_gtrans_model_{i}.keras'  # 根据簇编号加载模型
    model = load_model(model_path)

    # 使用模型进行预测
    predicted_output = model.predict(trajectory_input)

    # 反归一化输入和预测结果
    inputs_lat = denormalize_coords(single_traj[:19, 1], lat_min, lat_max)
    inputs_lon = denormalize_coords(single_traj[:19, 0], lon_min, lon_max)
    pre_tru_lat = denormalize_coords(single_traj[18:, 1], lat_min, lat_max)  # 使用最后6个真实值作为 ground truth
    pre_tru_lon = denormalize_coords(single_traj[18:, 0], lon_min, lon_max)
    pred_lat = denormalize_coords(predicted_output[0, :, 1], lat_min, lat_max)
    pred_lon = denormalize_coords(predicted_output[0, :, 0], lon_min, lon_max)

    # 创建绘图并保存
    plt.figure(figsize=(10,6),dpi=150)
    plt.plot(inputs_lat, inputs_lon, 'g-', label='Input')  # 绘制输入轨迹
    plt.plot(pre_tru_lat, pre_tru_lon, 'b-', label='True')  # 绘制真实轨迹
    plt.plot(pred_lat, pred_lon, 'r--', label='Predicted')  # 绘制预测轨迹
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.tight_layout()

    savepath = os.path.join(output_folder, f'GRU-CNN-trans predict 2_{i}.png')
    plt.savefig(savepath)
    plt.close()

    print(f'Cluster {i} 预测结果已保存到 {savepath}')
