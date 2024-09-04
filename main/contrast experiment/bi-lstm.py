import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from haversine import haversine, Unit
import pandas as pd
from sklearn.metrics import mean_squared_error

reset = True

# Load the provided data files
test_data_path = '../data/test_filter.pkl'
train_data_path = '../data/train_erp_dtw.pkl'
valid_data_path = '../data/valid_filter.pkl'

test_data = pd.read_pickle(test_data_path)
train_data = pd.read_pickle(train_data_path)
valid_data = pd.read_pickle(valid_data_path)


# 帮助函数，用于创建序列
def create_sequences(data, seq_length=18, y_length=24):
    xs, ys = [], []
    for i in range(len(data) - seq_length - y_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length:i + seq_length + y_length]
        xs.append(x)
        ys.append(y)
    xs = np.array([np.array(xi) for xi in xs])
    ys = np.array([np.array(yi) for yi in ys])
    return xs, ys

# 准备训练数据
train_data = [x for x in train_data if not np.isnan(x["traj"]).any() and len(x["traj"]) > 42]
test_data = [x for x in test_data if not np.isnan(x["traj"]).any() and len(x["traj"]) > 42]
train_trajectories = [traj['traj'][:, 0:2] for traj in train_data]

# 在创建序列之前验证每个轨迹的维度
for i, traj in enumerate(train_trajectories):
    print(f"轨迹 {i} 的形状: {traj.shape}")

# 创建训练序列
train_sequences = [create_sequences(traj, 18) for traj in train_trajectories]
X_train = np.concatenate([seq[0] for seq in train_sequences if len(seq[0]) > 0])
y_train = np.concatenate([seq[1] for seq in train_sequences if len(seq[1]) > 0])

# 准备验证数据
valid_trajectories = [traj['traj'][:, 0:2] for traj in test_data]

# 在创建序列之前验证每个轨迹的维度
for i, traj in enumerate(valid_trajectories):
    print(f"轨迹 {i} 的形状: {traj.shape}")

# 创建验证序列
valid_sequences = [create_sequences(traj, 18) for traj in valid_trajectories]
X_valid = np.concatenate([seq[0] for seq in valid_sequences if len(seq[0]) > 0])
y_valid = np.concatenate([seq[1] for seq in valid_sequences if len(seq[1]) > 0])

# 确保所有序列具有相同的特征数
assert X_train.shape[2] == X_valid.shape[2], "训练数据和验证数据的特征数不匹配"

if reset:
    model = Sequential([
        Input(shape=(18, X_train.shape[2])),
        Bidirectional(LSTM(64, activation='relu', return_sequences=True)),
        LSTM(64, activation='relu', return_sequences=False),
        Dense(24 * X_train.shape[2])
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # 将y_train和y_valid的形状调整为模型的输出形状
    y_train = y_train.reshape((y_train.shape[0], 24 * y_train.shape[2]))
    y_valid = y_valid.reshape((y_valid.shape[0], 24 * y_valid.shape[2]))

    # 定义ModelCheckpoint回调以保存验证集上表现最好的模型
    checkpoint_callback = ModelCheckpoint(
        filepath='best_ship_trajectory_bilstm_model.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=1,
        batch_size=32,
        callbacks=[checkpoint_callback]
    )

# 加载最好的模型
best_model = load_model('best_ship_trajectory_bilstm_model.keras')

# 使用最佳模型进行预测和误差计算
y_pred = best_model.predict(X_valid)

# 将预测结果的形状调整回原始形状
y_pred = y_pred.reshape((y_pred.shape[0], 24, -1))


# 计算每个预测点与真实点的哈弗塞恩地理距离误差
def calculate_haversine_distance(y_true, y_pred):
    distances = []
    for i in range(y_true.shape[0]):
        traj_true = y_true[i]
        traj_pred = y_pred[i]
        traj_distances = []
        for j in range(traj_true.shape[0]):
            coord_true = (traj_true[j, 0], traj_true[j, 1])
            coord_pred = (traj_pred[j, 0], traj_pred[j, 1])
            distance = haversine(coord_true, coord_pred, unit=Unit.METERS)
            traj_distances.append(distance)
        distances.append(traj_distances)
    return np.array(distances)

# 假设经纬度在特征的前两列
y_valid_coords = y_valid.reshape((y_valid.shape[0], 24, -1))
errors = calculate_haversine_distance(y_valid_coords, y_pred)


# 计算每个样本的前24个点的RMSE误差
def calculate_rmse(y_true, y_pred):
    rmses = []
    for i in range(y_true.shape[0]):
        rmse = mean_squared_error(y_true[i], y_pred[i], squared=False)
        rmses.append(rmse)
    return np.array(rmses)

# 计算前24个点的RMSE误差
rmse_errors = calculate_rmse(y_valid_coords, y_pred)

# 计算整体的平均RMSE误差
overall_avg_rmse = np.mean(rmse_errors)

# 打印前24个点的平均RMSE误差

print(f"整体平均RMSE误差: {overall_avg_rmse:.2f} meters")

# 可视化预测与真实轨迹的差异
plt.figure(figsize=(15, 10))

for i in range(5):  # 可视化前5个样本
    plt.subplot(5, 1, i+1)
    plt.plot(y_valid_coords[i, :, 0], y_valid_coords[i, :, 1], 'b-', label='True')
    plt.plot(y_pred[i, :, 0], y_pred[i, :, 1], 'r--', label='Predicted')
    plt.title(f'Sample {i+1}, Avg Error: {np.mean(errors[i]):.2f} meters')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()

plt.tight_layout()
plt.show()

# 将误差从米转换为千米
errors_km = errors / 1000

# 计算每个点的平均误差和整体平均误差
pointwise_avg_errors_km = np.mean(errors_km, axis=0)
overall_avg_error_km = np.mean(errors_km)

# 打印每个点的平均误差
print("每个点的平均误差（千米）：")
for i, error in enumerate(pointwise_avg_errors_km):
    print(f"Point {i+1}: {error:.2f} kilometers")

print(f"整体平均哈弗塞恩距离误差: {overall_avg_error_km:.2f} kilometers")

# 创建结果文件夹
results_folder = 'lstm-results/bilstm'
os.makedirs(results_folder, exist_ok=True)
rmse_file_path = os.path.join(results_folder, 'average_rmse_error.txt')
with open(rmse_file_path, 'w') as file:
    file.write(f"整体平均RMSE误差: {overall_avg_rmse:.2f} meters\n")
    file.write(f"整体平均哈弗塞恩距离误差: {overall_avg_error_km:.4f} kilometers\n")

# 将误差保存到CSV文件
errors_df = pd.DataFrame({
    'Point Index': range(1, len(pointwise_avg_errors_km) + 1),
    'Average Haversine Distance Error (kilometers)': pointwise_avg_errors_km
})
csv_path = os.path.join(results_folder, 'average_haversine_distance_errors.csv')
errors_df.to_csv(csv_path, index=False)

# 绘制每个点的平均误差并保存为PNG文件
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(pointwise_avg_errors_km) + 1), pointwise_avg_errors_km, 'bo-')

# 每六个点绘制一个标记，并标注出值
for i in range(0, len(pointwise_avg_errors_km), 6):
    plt.annotate(f'{pointwise_avg_errors_km[i]:.2f}',
                 (i + 1, pointwise_avg_errors_km[i]),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center',
                 fontsize=8)
    plt.plot(i + 1, pointwise_avg_errors_km[i], 'ro')  # 标记点

plt.xlabel('Point Index')
plt.ylabel('Average Haversine Distance Error (kilometers)')
plt.title('Average Haversine Distance Error for Each Point')
plt.grid(True)
png_path = os.path.join(results_folder, 'average_haversine_distance_errors.png')
plt.savefig(png_path)
plt.show()
