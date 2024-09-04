
import pandas as pd
import numpy as np
import os
from sklearn_extra.cluster import KMedoids
from joblib import dump, load

# 定义一个函数来加载数据并提取特征
def load_data_and_features(file_path):
    data = pd.read_pickle(file_path)
    trajectories = [np.array(d['traj'])[:, :2] for d in data if 'traj' in d]
    features = np.array([np.mean(traj, axis=0) for traj in trajectories])
    return data, features

# 定义一个函数来保存聚类结果
def save_cluster_results(data, labels, output_folder, n_clusters):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(n_clusters):
        cluster_data = [data[idx] for idx, label in enumerate(labels) if label == i]
        pd.to_pickle(cluster_data, os.path.join(output_folder, f'cluster_{i}_data.pkl'))

# 训练集
train_data, train_features = load_data_and_features('../data/train_erp_dtw.pkl')
kmedoids = KMedoids(n_clusters=18, method='pam', init='k-medoids++')
kmedoids.fit(train_features)
dump(kmedoids, 'kmedoids_model.joblib')  # 保存模型
save_cluster_results(train_data, kmedoids.labels_, './clusters/train', kmedoids.n_clusters)

# 测试集
test_data, test_features = load_data_and_features('../data/test_filter.pkl')
kmedoids = load('kmedoids_model.joblib')  # 加载模型
test_labels = kmedoids.predict(test_features)
save_cluster_results(test_data, test_labels, './clusters/test', kmedoids.n_clusters)

# 验证集
validation_data, validation_features = load_data_and_features('../data/valid_filter.pkl')
validation_labels = kmedoids.predict(validation_features)
save_cluster_results(validation_data, validation_labels, './clusters/validation', kmedoids.n_clusters)

