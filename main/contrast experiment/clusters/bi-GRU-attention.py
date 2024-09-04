import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from haversine import haversine, Unit
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

reset = True

# Load the provided data files
test_data_path = '../data/ct_dma/ct_dma_test_剔除无效数据.pkl'
train_data_path = '../data/ct_dma/ct_dma_train_erp.pkl'
valid_data_path = '../data/ct_dma/ct_dma_valid_剔除无效数据.pkl'

test_data = pd.read_pickle(test_data_path)
train_data = pd.read_pickle(train_data_path)
valid_data = pd.read_pickle(valid_data_path)


def create_sequences(data, seq_length, y_length=24):
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

# 验证每个轨迹的维度
for i, traj in enumerate(train_trajectories):
    print(f"轨迹 {i} 的形状: {traj.shape}")

# 创建训练序列
train_sequences = [create_sequences(traj, 18) for traj in train_trajectories]
X_train = np.concatenate([seq[0] for seq in train_sequences if len(seq[0]) > 0])
y_train = np.concatenate([seq[1] for seq in train_sequences if len(seq[1]) > 0])

# 准备验证数据
valid_trajectories = [traj['traj'][:, 0:2] for traj in valid_data]

# 验证每个轨迹的维度
for i, traj in enumerate(valid_trajectories):
    print(f"轨迹 {i} 的形状: {traj.shape}")

# 创建验证序列
valid_sequences = [create_sequences(traj, 18) for traj in valid_trajectories]
X_valid = np.concatenate([seq[0] for seq in valid_sequences if len(seq[0]) > 0])
y_valid = np.concatenate([seq[1] for seq in valid_sequences if len(seq[1]) > 0])

# 确保所有序列具有相同的特征数
assert X_train.shape[2] == X_valid.shape[2], "训练数据和验证数据的特征数不匹配"


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers_encoder, n_layers_decoder):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.l_encoder = n_layers_encoder
        self.l_decoder = n_layers_decoder

        # 定义双向GRU编码器，输入维度为input_dim，隐藏层维度为hidden_dim，层数为n_layers_encoder
        self.encoder = nn.GRU(input_dim, hidden_dim, n_layers_encoder, bidirectional=True, batch_first=True)
        # 定义GRU解码器，输入维度为hidden_dim*2（因为编码器是双向的，输出维度加倍），隐藏层维度为hidden_dim*2，层数为n_layers_decoder
        self.decoder = nn.GRU(hidden_dim * 2, hidden_dim * 2, n_layers_decoder, batch_first=True)
        # 定义全连接层，将解码器输出的hidden_dim*2维度映射到output_dim维度
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.fc_con = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        # 定义注意力机制的权重参数
        self.attention = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.v = nn.Parameter(torch.rand(hidden_dim * 2))

    def forward(self, x):
        h0_encoder = torch.zeros(self.l_encoder * 2, x.size(0), self.hidden_dim).to(x.device)
        # 将输入x和初始化隐藏状态h0_encoder传入编码器，得到编码器输出out_encoder和隐藏状态hidden_encoder
        out_encoder, hidden_encoder = self.encoder(x, h0_encoder)

        # 使用编码器最后一个时间步的输出作为解码器的初始输入
        decoder_input = out_encoder[:, -1, :].unsqueeze(1)
        # 分离双向GRU的隐藏状态，并将其合并用于初始化解码器的隐藏状态
        hidden_encoder_forward = hidden_encoder[0:hidden_encoder.size(0):2]
        hidden_encoder_backward = hidden_encoder[1:hidden_encoder.size(0):2]
        hidden_decoder = torch.cat((hidden_encoder_forward, hidden_encoder_backward), dim=2)

        # 保存解码器每个时间步的输出
        outputs = []

        for t in range(24):  # 预测接下来的24个时间步
            # 计算注意力权重
            attn_weights = self.calculate_attention(hidden_decoder[-1], out_encoder)
            context_vector = attn_weights.bmm(out_encoder)  # [batch_size, 1, hidden_dim * 2]

            # 将上下文向量和解码器输入拼接，并传递给解码器
            decoder_input_combined = torch.cat((decoder_input, context_vector), dim=2)
            decoder_input_combined = self.fc_con(decoder_input_combined)
            out_decoder, hidden_decoder = self.decoder(decoder_input_combined, hidden_decoder)
            out_step = self.fc(out_decoder.squeeze(1))
            outputs.append(out_step.unsqueeze(1))
            # 将当前时间步的输出作为下一个时间步的输入
            decoder_input = out_decoder

        # 将所有时间步的输出连接起来
        outputs = torch.cat(outputs, dim=1)

        return outputs

    def calculate_attention(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1)  # [batch_size, 1, hidden_dim * 2]
        # 将解码器隐藏状态与编码器输出拼接，并计算注意力得分
        energy = torch.tanh(self.attention(torch.cat((hidden.expand(encoder_outputs.size(0), encoder_outputs.size(1), -1), encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)  # [batch_size, hidden_dim * 2, seq_len]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [batch_size, 1, hidden_dim * 2]
        attention_weights = torch.bmm(v, energy).squeeze(1)  # [batch_size, seq_len]
        return F.softmax(attention_weights, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]




input_dim = 2
hidden_dim = 50
output_dim = 2
n_layers_encoder = 3
n_layers_decoder = 3
batch_size = 32
epochs = 50
learning_rate = 0.001

model = GRUModel(input_dim, hidden_dim, output_dim, n_layers_encoder, n_layers_decoder)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32))
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if reset:
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        val_loss /= len(valid_loader)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ship_trajectory_bigru_attention_model.pth')
            print("Model saved!")

# 加载最好的模型
model.load_state_dict(torch.load('best_ship_trajectory_bigru_attention_model.pth'))

# 使用最佳模型进行预测和误差计算
model.eval()
y_pred = []
with torch.no_grad():
    for X_batch, _ in valid_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        y_pred.append(output.cpu().numpy())
y_pred = np.concatenate(y_pred, axis=0)

# 将预测结果的形状调整回原始形状
y_pred = y_pred.reshape((y_pred.shape[0], 24, X_train.shape[2]))


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
print(f"整体平均RMSE误差: {overall_avg_rmse:.4f} meters")

# 可视化预测与真实轨迹的差异
plt.figure(figsize=(15, 10))

for i in range(5):  # 可视化前5个样本
    plt.subplot(5, 1, i + 1)
    plt.plot(y_valid_coords[i, :, 0], y_valid_coords[i, :, 1], 'b-', label='True')
    plt.plot(y_pred[i, :, 0], y_pred[i, :, 1], 'r--', label='Predicted')
    plt.title(f'Sample {i + 1}, Avg Error: {np.mean(errors[i]):.2f} meters')
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
    print(f"Point {i + 1}: {error:.4f} kilometers")

print(f"整体平均哈弗塞恩距离误差: {overall_avg_error_km:.4f} kilometers")

# 创建结果文件夹
results_folder = 'gru-results/bigru-attention'
os.makedirs(results_folder, exist_ok=True)

# 将误差保存到CSV文件
errors_df = pd.DataFrame({
    'Point Index': range(1, len(pointwise_avg_errors_km) + 1),
    'Average Haversine Distance Error (kilometers)': pointwise_avg_errors_km
})
csv_path = os.path.join(results_folder, 'average_haversine_distance_errors.csv')
errors_df.to_csv(csv_path, index=False)
rmse_file_path = os.path.join(results_folder, 'average_rmse_error.txt')
with open(rmse_file_path, 'w') as file:
    file.write(f"整体平均RMSE误差: {overall_avg_rmse:.4f} meters\n")
    file.write(f"整体平均哈弗塞恩距离误差: {overall_avg_error_km:.4f} kilometers\n")

# 绘制每个点的平均误差并保存为PNG文件
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(pointwise_avg_errors_km) + 1), pointwise_avg_errors_km, 'bo-')

# 每六个点绘制一个标记，并标注出值
for i in range(0, len(pointwise_avg_errors_km), 6):
    plt.annotate(f'{pointwise_avg_errors_km[i]:.2f}',
                 (i + 1, pointwise_avg_errors_km[i]),
                 textcoords="offset points",
                 xytext=(0, 10),
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
