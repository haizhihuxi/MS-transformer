import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GRU, Dense, Input, Concatenate, TimeDistributed, Add, UpSampling1D, Conv1D, \
    MaxPooling1D, Reshape, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from haversine import haversine, Unit
import pandas as pd
from tensorflow.keras.layers import GRU, MultiHeadAttention, Add, LayerNormalization, ZeroPadding1D
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Add, Softmax

import tensorflow as tf


def feature_extraction(inputs):
    x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    return x


def upsample(x, target_size):
    x = UpSampling1D(size=2)(x)
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    return x


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

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (size, size)



def G_block(x, units, target_length=24):
    # 计算需要填充的长度
    padding_length = target_length - x.shape[1]
    if padding_length > 0:
        x = ZeroPadding1D(padding=(0, padding_length))(x)  # 在序列的末尾填充
    gru_output = GRU(units, return_sequences=True)(x)
    multi_head_output = MultiHeadAttention(num_heads=4, key_dim=units)(gru_output, gru_output)
    add_output = Add()([gru_output, multi_head_output])
    norm_output = LayerNormalization()(add_output)
    return norm_output


def transformer_encoder_layer(query, key, value, mask=None, num_units=64):
    # Multi-Head Attention (Masked)
    attention_out = MultiHeadAttention(num_heads=4, key_dim=64)(query, key, value, attention_mask=mask)
    attention_out = Add()([query, attention_out])  # Add & Norm
    attention_out = LayerNormalization()(attention_out)

    # Feed Forward
    ff_out = Dense(64 * 4, activation='relu')(attention_out)
    ff_out = Dense(64)(ff_out)
    ff_out = Add()([attention_out, ff_out])  # Add & Norm
    ff_out = LayerNormalization()(ff_out)

    # Linear and Scaled Softmax
    final_output = Dense(num_units)(ff_out)  # Linear transformation
    scaled_output = final_output / tf.sqrt(tf.cast(num_units, tf.float32))
    softmax_output = Softmax()(scaled_output)  # Apply softmax

    return ff_out

def transformer_decoder_layer(query, key, value, pos, mask=None):
    # Multi-Head Attention (Masked)
    query = Add()([query, pos])
    attention_out_mask = MultiHeadAttention(num_heads=4, key_dim=64)(query, query, query, attention_mask=mask)
    attention_out_mask = Add()([query, attention_out_mask])  # Add & Norm
    attention_out_mask = LayerNormalization()(attention_out_mask)

    attention_out = MultiHeadAttention(num_heads=4, key_dim=64)(attention_out_mask, key, value)
    attention_out = Add()([attention_out_mask, attention_out])  # Add & Norm
    attention_out = LayerNormalization()(attention_out)

    # Feed Forward
    ff_out = Dense(64 * 4, activation='relu')(attention_out)
    ff_out = Dense(64)(ff_out)
    ff_out = Add()([attention_out, ff_out])  # Add & Norm
    ff_out = LayerNormalization()(ff_out)

    return ff_out

def build_gtrans_model(input_shape):
    encoder_inputs = Input(shape=input_shape)

    x = feature_extraction(encoder_inputs)
    x = upsample(x, 64)  # 恢复维度

    x = Conv1D(64, 3, padding='same', activation='relu')(x)

    # Reshape for Dense layer
    x = Reshape((x.shape[1] * x.shape[2],))(x)
    x = Dense(24 * 64, activation='relu')(x)
    x = Reshape((24, 64))(x)

    # # G_block_output with G_block
    # g_block_output = G_block(encoder_inputs, 64)
    # g_block_output = Add()([g_block_output, x])
    #
    # # Positional Encoding
    # position_encoded = positional_encoding(24, 64)
    # position_encoded = tf.expand_dims(position_encoded, 0)
    # position_encoded = tf.tile(position_encoded, [tf.shape(g_block_output)[0], 1, 1])
    #
    # # Transformer Decoder
    # look_ahead_mask = create_look_ahead_mask(tf.shape(g_block_output)[1])
    # encoder_output = transformer_encoder_layer(g_block_output, g_block_output, g_block_output)
    # decoder_output = transformer_decoder_layer(g_block_output, encoder_output, encoder_output, position_encoded,
    #                                            mask=look_ahead_mask)
    #
    # final_output = TimeDistributed(Dense(2))(decoder_output)

    # Initial Positional Encoding
    position_encoded = positional_encoding(24, 64)
    position_encoded = tf.expand_dims(position_encoded, 0)
    position_encoded = tf.tile(position_encoded, [tf.shape(x)[0], 1, 1])

    # Stack G_block, Encoder, and Decoder 4 times
    output = x
    for i in range(4):
        g_block_output = G_block(output, 64)
        print("g_block_output", g_block_output.shape)
        print("output", output.shape)
        g_block_output = Add()([g_block_output, output])

        look_ahead_mask = create_look_ahead_mask(tf.shape(g_block_output)[1])
        encoder_output = transformer_encoder_layer(g_block_output, g_block_output, g_block_output)
        output = transformer_decoder_layer(g_block_output, encoder_output, encoder_output, position_encoded,
                                           mask=look_ahead_mask)

    final_output = TimeDistributed(Dense(2))(output)

    model = Model(inputs=encoder_inputs, outputs=final_output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def positional_encoding(seq_len, d_model):
    angle_rads = get_angles(np.arange(seq_len)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


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


# 数据加载
reset = True
for i in range(0, 18):
    cluster = i
    test_data_path = f'clusters/test/cluster_{cluster}_data.pkl'
    train_data_path = f'clusters/train/cluster_{cluster}_data.pkl'
    valid_data_path = f'clusters/validation/cluster_{cluster}_data.pkl'

    test_data = pd.read_pickle(test_data_path)
    train_data = pd.read_pickle(train_data_path)
    valid_data = pd.read_pickle(valid_data_path)

    # 准备训练数据
    train_data = [x for x in train_data if not np.isnan(x["traj"]).any() and len(x["traj"]) > 42]
    test_data = [x for x in test_data if not np.isnan(x["traj"]).any() and len(x["traj"]) > 42]
    train_trajectories = [traj['traj'][:, 0:2] for traj in train_data]

    train_sequences = [create_sequences(traj, 18) for traj in train_trajectories]
    X_train = np.concatenate([seq[0]for seq in train_sequences if len(seq[0]) > 0])
    y_train = np.concatenate([seq[1] for seq in train_sequences if len(seq[1]) > 0])

    # 准备验证数据
    valid_trajectories = [traj['traj'][:, 0:2] for traj in valid_data]
    valid_sequences = [create_sequences(traj, 18) for traj in valid_trajectories]

    X_valid = np.concatenate([seq[0] for seq in valid_sequences if len(seq[0]) > 0])
    y_valid = np.concatenate([seq[1] for seq in valid_sequences if len(seq[1]) > 0])



    if reset:
        input_shape = (18, 2)
        model = build_gtrans_model(input_shape)
        model.summary()

        y_train = y_train.reshape((y_train.shape[0], 24, input_shape[1]))
        y_valid = y_valid.reshape((y_valid.shape[0], 24, input_shape[1]))

        checkpoint_callback = ModelCheckpoint(
            filepath=f'best_gtrans_model_{cluster}.keras',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=20,
            batch_size=32,
            callbacks=[checkpoint_callback]
        )

    # 加载最佳模型
    best_model = load_model(f'best_gtrans_model_{cluster}.keras')
    y_pred = best_model.predict(X_valid)
    y_pred = y_pred.reshape((y_pred.shape[0], 24, X_train.shape[2]))

    y_valid_coords = y_valid.reshape((y_valid.shape[0], 24, -1))
    errors = calculate_haversine_distance(y_valid, y_pred)

    # Calculate RMSE for each sample
    def calculate_rmse(y_true, y_pred):
        rmses = []
        for i in range(y_true.shape[0]):
            rmse = mean_squared_error(y_true[i], y_pred[i], squared=False)
            rmses.append(rmse)
        return np.array(rmses)


    rmse_errors = calculate_rmse(y_valid_coords, y_pred)
    overall_avg_rmse = np.mean(rmse_errors)

    # Print overall average RMSE error
    print(f"整体平均RMSE误差: {overall_avg_rmse:.4f} meters")

    # Save individual plots of each sample's predictions vs true trajectories
    results_folder = 'gtrans-results/gtans/result picture'
    os.makedirs(results_folder, exist_ok=True)


    def denormalize_coords(coords, min_val, max_val):
        return coords * (max_val - min_val) + min_val

    for j in range(y_valid_coords.shape[0]):
        print(f'绘制{i}簇第{j}条轨迹，一共{y_valid_coords.shape[0]}条')
        lat_min, lat_max = 10.3, 13.0
        lon_min, lon_max = 55.5, 58.0

        # 展示结果
        plt.figure()
        inputs_lat = denormalize_coords(X_valid[j, :, 1], lat_min, lat_max)
        inputs_lon = denormalize_coords(X_valid[j, :, 0], lon_min, lon_max)
        pre_tru_lat = denormalize_coords(y_valid_coords[j, :, 1], lat_min, lat_max)
        pre_tru_lon = denormalize_coords(y_valid_coords[j, :, 0], lon_min, lon_max)
        pred_lat = denormalize_coords(y_pred[j, :, 1], lat_min, lat_max)
        pred_lon = denormalize_coords(y_pred[j, :, 0], lon_min, lon_max)
        plt.plot(inputs_lat, inputs_lon, 'g-', label='Input')  # Plot input values
        plt.plot(pre_tru_lat, pre_tru_lon, 'b-', label='True')
        plt.plot(pred_lat, pred_lon, 'r--', label='Predicted')
        plt.title(f'Sample {j + 1}, Avg Error: {np.mean(errors[i]):.2f} meters')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.tight_layout()
        sample_plot_path = os.path.join(results_folder, f'xluster_{i} sample_{j + 1}.png')
        plt.savefig(sample_plot_path)
        plt.close()

    errors_km = errors / 1000
    pointwise_avg_errors_km = np.mean(errors_km, axis=0)
    overall_avg_error_km = np.mean(errors_km)

    print("每个点的平均误差（千米）：")
    for i, error in enumerate(pointwise_avg_errors_km):
        print(f"Point {i + 1}: {error:.2f} kilometers")
    print(f"整体平均哈弗塞恩距离误差: {overall_avg_error_km:.2f} kilometers")

    results_folder = 'gtrans-results'
    os.makedirs(results_folder, exist_ok=True)
    rmse_file_path = os.path.join(results_folder, f'average_rmse_error_{cluster}.txt')
    with open(rmse_file_path, 'w') as file:
        file.write(f"整体平均RMSE误差: {overall_avg_rmse:.4f} meters\n")
        file.write(f"整体平均哈弗塞恩距离误差: {overall_avg_error_km:.4f} kilometers\n")
    errors_df = pd.DataFrame({
        'Point Index': range(1, len(pointwise_avg_errors_km) + 1),
        'Average Haversine Distance Error (kilometers)': pointwise_avg_errors_km
    })
    csv_path = os.path.join(results_folder, f'average_haversine_distance_errors_{cluster}.csv')
    errors_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(pointwise_avg_errors_km) + 1), pointwise_avg_errors_km, 'bo-')
    for i in range(0, len(pointwise_avg_errors_km), 6):
        plt.annotate(f'{pointwise_avg_errors_km[i]:.2f}', (i + 1, pointwise_avg_errors_km[i]), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8)
        plt.plot(i + 1, pointwise_avg_errors_km[i], 'ro')
    plt.xlabel('Point Index')
    plt.ylabel('Average Haversine Distance Error (kilometers)')
    plt.title('Average Haversine Distance Error for Each Point')
    plt.grid(True)
    png_path = os.path.join(results_folder, f'average_haversine_distance_errors_{cluster}.png')
    plt.savefig(png_path)
    plt.show()
