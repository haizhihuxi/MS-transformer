import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from haversine import haversine, Unit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

reset = False
# Load the provided data files
test_data_path = '../data/test_filter.pkl'
train_data_path = '../data/train_erp_dtw.pkl'
valid_data_path = '../data/valid_filter.pkl'

test_data = pd.read_pickle(test_data_path)
train_data = pd.read_pickle(train_data_path)
valid_data = pd.read_pickle(valid_data_path)


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


# Prepare training data
train_data = [x for x in train_data if not np.isnan(x["traj"]).any() and len(x["traj"]) > 42]
test_data = [x for x in test_data if not np.isnan(x["traj"]).any() and len(x["traj"]) > 42]
train_trajectories = [traj['traj'][:, 0:2] for traj in train_data]

# Create training sequences
train_sequences = [create_sequences(traj, 18) for traj in train_trajectories]
X_train = np.concatenate([seq[0] for seq in train_sequences if len(seq[0]) > 0])
y_train = np.concatenate([seq[1] for seq in train_sequences if len(seq[1]) > 0])

# Prepare validation data
valid_trajectories = [traj['traj'][:, 0:2] for traj in valid_data]

# Create validation sequences
valid_sequences = [create_sequences(traj, 18) for traj in valid_trajectories]
X_valid = np.concatenate([seq[0] for seq in valid_sequences if len(seq[0]) > 0])
y_valid = np.concatenate([seq[1] for seq in valid_sequences if len(seq[1]) > 0])


class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TrajectoryDataset(X_train, y_train)
valid_dataset = TrajectoryDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


class BiLSTMWithAttention(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(BiLSTMWithAttention, self).__init__()
        self.encoder_bilstm = nn.LSTM(2, 64, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  # 添加dropout层
        self.attention_dense = nn.Linear(128, 24)
        self.decoder_lstm = nn.LSTM(128, 128, batch_first=True)
        self.output_dense = nn.Linear(128, 2)

    def forward(self, x):
        encoder_outputs, (h_n, c_n) = self.encoder_bilstm(x)
        encoder_outputs = self.dropout(encoder_outputs)  # 在encoder输出上应用dropout

        state_h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        state_c = torch.cat((c_n[-2], c_n[-1]), dim=1)

        attention_scores = self.attention_dense(encoder_outputs)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_output = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)

        attention_output = self.dropout(attention_output)  # 对注意力机制的输出应用dropout
        decoder_output, _ = self.decoder_lstm(attention_output, (state_h.unsqueeze(0), state_c.unsqueeze(0)))
        decoder_output = self.dropout(decoder_output)  # 在decoder输出上应用dropout

        final_output = self.output_dense(decoder_output)
        return final_output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMWithAttention(dropout_rate=0.5).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item() * X_batch.size(0)

        valid_loss /= len(valid_loader.dataset)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), 'best_ship_trajectory_GRUCNN-transformer_model.pth')
            print(f'Saved model with Validation Loss: {best_val_loss:.4f}')


if reset:
    train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=100)

model.load_state_dict(torch.load('best_ship_trajectory_GRUCNN-transformer_model.pth'))

# 计算哈弗赛因距离
def calculate_haversine_distance(y_true, y_pred):
    distances = []
    for i in range(y_true.shape[0]):
        traj_true = y_true[i]
        traj_pred = y_pred[i]
        traj_distances = []
        for j in range(traj_true.shape[0]):
            coord_true = (traj_true[j, 0].item(), traj_true[j, 1].item())
            coord_pred = (traj_pred[j, 0].item(), traj_pred[j, 1].item())
            distance = haversine(coord_true, coord_pred, unit=Unit.METERS)
            traj_distances.append(distance)
        distances.append(traj_distances)
    return np.array(distances)

# 计算均方根误差
def calculate_rmse(y_true, y_pred):
    rmses = []
    for i in range(y_true.shape[0]):
        rmse = mean_squared_error(y_true[i].cpu().numpy(), y_pred[i].cpu().numpy(), squared=False)
        rmses.append(rmse)
    return np.array(rmses)

# 模型预测
model.eval()
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_tensor = []
    for X_batch, _ in tqdm(valid_loader, desc="Predicting"):
        X_batch = X_batch.to(device)
        y_pred_batch = model(X_batch)
        y_pred_tensor.append(y_pred_batch)

y_pred_tensor = torch.cat(y_pred_tensor, dim=0)

errors = calculate_haversine_distance(y_valid_tensor.cpu().numpy(), y_pred_tensor.cpu().numpy())
rmse_errors = calculate_rmse(y_valid_tensor, y_pred_tensor)
overall_avg_rmse = np.mean(rmse_errors)

print(f"整体平均RMSE误差: {overall_avg_rmse:.4f} meters")

# 保存每个样本预测与真实轨迹的单独图
results_folder = 'lstm-results/bilstm_attention/result_picture'
os.makedirs(results_folder, exist_ok=True)

for i in range(y_valid_tensor.shape[0]):
    print(f'绘制第{i}条轨迹，一共{y_valid_tensor.shape[0]}条')
    plt.figure()
    plt.plot(X_valid[i, :, 1], X_valid[i, :, 0], 'g-', label='Input')  # 绘制输入值
    plt.plot(y_valid_tensor[i].cpu().numpy()[:, 1], y_valid_tensor[i].cpu().numpy()[:, 0], 'b-', label='True')
    plt.plot(y_pred_tensor[i].cpu().numpy()[:, 1], y_pred_tensor[i].cpu().numpy()[:, 0], 'r--', label='Predicted')
    plt.title(f'Sample {i + 1}, Avg Error: {np.mean(errors[i]):.2f} meters')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.tight_layout()
    sample_plot_path = os.path.join(results_folder, f'sample_{i + 1}.png')
    plt.savefig(sample_plot_path)
    plt.close()

errors_km = errors / 1000
pointwise_avg_errors_km = np.mean(errors_km, axis=0)
overall_avg_error_km = np.mean(errors_km)

print("每个点的平均误差（千米）：")
for i, error in enumerate(pointwise_avg_errors_km):
    print(f"Point {i + 1}: {error:.2f} kilometers")

print(f"整体平均哈弗塞恩距离误差: {overall_avg_error_km:.4f} kilometers")

results_folder = 'gtrans-results'
os.makedirs(results_folder, exist_ok=True)
rmse_file_path = os.path.join(results_folder, 'average_rmse_error.txt')
with open(rmse_file_path, 'w') as file:
    file.write(f"整体平均RMSE误差: {overall_avg_rmse:.4f} meters\n")
    file.write(f"整体平均哈弗塞恩距离误差: {overall_avg_error_km:.4f} kilometers\n")

errors_df = pd.DataFrame({
    'Point Index': range(1, len(pointwise_avg_errors_km) + 1),
    'Average Haversine Distance Error (kilometers)': pointwise_avg_errors_km
})
csv_path = os.path.join(results_folder, 'average_haversine_distance_errors.csv')
errors_df.to_csv(csv_path, index=False)

# plt.figure(figsize=(12, 6))
# plt.plot(range(1, len(pointwise_avg_errors_km) + 1), pointwise_avg_errors_km, 'bo-')
#
# for i in range(0, len(pointwise_avg_errors_km), 6):
#     plt.annotate(f'{pointwise_avg_errors_km[i]:.2f}',
#                  (i + 1, pointwise_avg_errors_km[i]),
#                  textcoords="offset points",
#                  xytext=(0, 10),
#                  ha='center',
#                  fontsize=8)
#     plt.plot(i + 1, pointwise_avg_errors_km[i], 'ro')
#
# plt.xlabel('Point Index')
# plt.ylabel('Average Haversine Distance Error (kilometers)')
# plt.title('Average Haversine Distance Error for Each Point')
# plt.grid(True)
# png_path = os.path.join(results_folder, 'average_haversine_distance_errors.png')
# plt.savefig(png_path)
# plt.show()
