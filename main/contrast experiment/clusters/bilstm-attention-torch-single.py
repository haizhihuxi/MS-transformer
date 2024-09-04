import torch
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine, Unit
from torch import nn

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


# 转换单条轨迹数据
def prepare_single_trajectory(traj, seq_length=18):
    if len(traj) < seq_length:
        raise ValueError("Trajectory length must be at least seq_length")
    traj = traj[:seq_length]
    traj = np.expand_dims(traj, axis=0)  # 扩展维度以适应模型输入
    return torch.tensor(traj, dtype=torch.float32)

def denormalize_coords(coords, min_val, max_val):
    return coords * (max_val - min_val) + min_val

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
model.load_state_dict(torch.load('best_ship_trajectory_GRUCNN-transformer_model.pth'))
model.eval()

# 准备单条轨迹数据
X_single = prepare_single_trajectory(single_traj, seq_length=18).to(device)

# 预测
with torch.no_grad():
    y_single_pred = model(X_single).cpu().numpy()

# 定义经度和纬度的最小值和最大值
lat_min, lat_max = 10.3, 13.0
lon_min, lon_max = 55.5, 58.0
# 反归一化处理
inputs_lat = denormalize_coords(single_traj[:19, 1], lat_min, lat_max)
inputs_lon = denormalize_coords(single_traj[:19, 0], lon_min, lon_max)
true_lat = denormalize_coords(single_traj[18:, 1], lat_min, lat_max)
true_lon = denormalize_coords(single_traj[18:, 0], lon_min, lon_max)
pred_lat = denormalize_coords(y_single_pred[0, :, 1], lat_min, lat_max)
pred_lon = denormalize_coords(y_single_pred[0, :, 0], lon_min, lon_max)
# 展示结果
savepath = '../picture-results/第二组/Bi-LSTM-Attention predict 2.png'
plt.figure(figsize=(10,6),dpi=150)
plt.plot(inputs_lat, inputs_lon, 'g-', label='Input')  # 绘制输入轨迹
plt.plot(true_lat, true_lon, 'b-', label='True')  # 绘制预测点真实轨迹
plt.plot(pred_lat, pred_lon, 'r--', label='Predicted')  # 绘制预测轨迹
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.tight_layout()
plt.savefig(savepath)
plt.show()

