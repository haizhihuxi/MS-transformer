
import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets_SEA
import similarity_models
import similarity_train
import utils_SEA
from config_SEA import Config

cf = Config()
logger = logging.getLogger(__name__)

# make deterministic 统一
utils_SEA.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device  # 设定gpu
    init_seqlen = cf.init_seqlen  # 18

    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils_SEA.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}

    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1
            V["traj"] = V["traj"][moving_idx:, :]

        Data[phase] = [x for x in l_pred_errors if
                       not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        aisdatasets[phase] = datasets_SEA.AISDataset_sim(Data[phase],
                                                     max_seqlen=cf.max_seqlen + 1,
                                                     device=cf.device)
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    # 相似度模型=============================
    sim_model = similarity_models.TrajectoryMatchingNetwork(cf)
    if cf.sim_half_train:
        sim_model.load_state_dict(torch.load(cf.ckpt_path_sim))
    sim_trainer = similarity_train.Trainer(sim_model, aisdatasets["train"], aisdatasets["valid"], cf)
    if cf.similarity_retrain:
        sim_trainer.train()

    ## Evaluation
    # ===============================
    # 评估模型
    # 加载最好的模型权重
    if cf.sim_test:
        sim_model.load_state_dict(torch.load(cf.ckpt_path_sim))
        sim_model.eval()
        w_10 = []
        w_5 = []
        w_15 = []
        w_20 = []
        losses = []
        pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
        print("==========================testing==========================")
        count = 0
        for it, (seqs, masks, seqlens) in pbar:
            kendall_tau_scores_w_10 = []
            kendall_tau_scores_w_5 = []
            kendall_tau_scores_w_15 = []
            kendall_tau_scores_w_20 = []
            for index_1 in range(len(seqs) - 1):
                a = len(seqs) - 1
                part_w_10 = []
                part_w_5 = []
                part_w_15 = []
                part_w_20 = []
                for index_2 in range(index_1 + 1, len(seqs)):
                    seq1 = seqs[index_1]  # (maxseqlen, 2)
                    seq2 = seqs[index_2]
                    with torch.no_grad():
                        loss, sim_scores_w_5, true_scores_w_5, sim_scores_w_10, true_scores_w_10, sim_scores_w_15, true_scores_w_15, sim_scores_w_20, true_scores_w_20 = sim_model(
                            seq1, seq2, count, index_1, index_2, masks=masks)
                        # 转换为排名序列
                        rank_true_scores_w_10 = utils_SEA.rankdata(true_scores_w_10)
                        rank_sim_scores_w_10 = utils_SEA.rankdata(sim_scores_w_10)
                        rank_true_scores_w_5 = utils_SEA.rankdata(true_scores_w_5)
                        rank_sim_scores_w_5 = utils_SEA.rankdata(sim_scores_w_5)
                        rank_true_scores_w_15 = utils_SEA.rankdata(true_scores_w_15)
                        rank_sim_scores_w_15 = utils_SEA.rankdata(sim_scores_w_15)
                        rank_true_scores_w_20 = utils_SEA.rankdata(true_scores_w_20)
                        rank_sim_scores_w_20 = utils_SEA.rankdata(sim_scores_w_20)
                        # 计算 Kendall's Tau
                        tau_w_10 = utils_SEA.kendall_tau(rank_true_scores_w_10, rank_sim_scores_w_10)
                        tau_w_5 = utils_SEA.kendall_tau(rank_true_scores_w_5, rank_sim_scores_w_5)
                        tau_w_15 = utils_SEA.kendall_tau(rank_true_scores_w_15, rank_sim_scores_w_15)
                        tau_w_20 = utils_SEA.kendall_tau(rank_true_scores_w_20, rank_sim_scores_w_20)
                        # 记录指标
                        part_w_10.append(tau_w_10)
                        part_w_5.append(tau_w_5)
                        part_w_15.append(tau_w_15)
                        part_w_20.append(tau_w_20)
                        losses.append(loss)
                # 计算平均 Kendall's Tau
                if part_w_10:
                    ave_tau_w_10 = np.mean(part_w_10)
                    kendall_tau_scores_w_10.append(ave_tau_w_10)
                if part_w_5:
                    ave_tau_w_5 = np.mean(part_w_5)
                    kendall_tau_scores_w_5.append(ave_tau_w_5)
                if part_w_15:
                    ave_tau_w_15 = np.mean(part_w_15)
                    kendall_tau_scores_w_15.append(ave_tau_w_15)
                if part_w_20:
                    ave_tau_w_20 = np.mean(part_w_20)
                    kendall_tau_scores_w_20.append(ave_tau_w_20)
            if kendall_tau_scores_w_10:
                ave_w_10_score = np.mean(kendall_tau_scores_w_10)
                w_10.append(ave_w_10_score)
            if kendall_tau_scores_w_5:
                ave_w_5_score = np.mean(kendall_tau_scores_w_5)
                w_5.append(ave_w_5_score)
            if kendall_tau_scores_w_15:
                ave_w_15_score = np.mean(kendall_tau_scores_w_15)
                w_15.append(ave_w_15_score)
            if kendall_tau_scores_w_20:
                ave_w_20_score = np.mean(kendall_tau_scores_w_20)
                w_20.append(ave_w_20_score)

            count += 1
        # 计算测试集上的平均 Kendall's Tau
        average_w_10 = np.mean(w_10)
        average_w_5 = np.mean(w_5)
        average_w_15 = np.mean(w_15)
        average_w_20 = np.mean(w_20)
        test_loss = float(np.mean(losses))
        logger.info("")  # 插入一个空行
        logger.info(f"Average Kendall's Tau for w_5: {average_w_5}")
        logger.info(f"Average Kendall's Tau for w_10: {average_w_10}")
        logger.info(f"Average Kendall's Tau for w_15: {average_w_15}")
        logger.info(f"Average Kendall's Tau for w_20: {average_w_20}")
        logger.info(f"test loss: {test_loss}")
        logger.info("")

