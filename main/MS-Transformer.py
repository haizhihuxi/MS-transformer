import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from tqdm import tqdm
import logging
import torch
from torch.utils.data import DataLoader
import models, trainers, datasets, utils
from config import Config


cf = Config()
logger = logging.getLogger(__name__)


utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}

    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)

        Data[phase] = [x for x in l_pred_errors if
                       not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]  # 36

        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        if phase == "train":
            aisdatasets[phase] = datasets.AISDataset_train(Data[phase],
                                                         max_seqlen=cf.max_seqlen + 1,
                                                         device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
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
    model = models.TrAISformer(cf)
    if cf.half_train:
        model.load_state_dict(torch.load(cf.ckpt_path))

    trainer = trainers.Trainer(model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir,
                               device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

    if cf.retrain:
        trainer.train()

    model.load_state_dict(torch.load(cf.ckpt_path))
    v_ranges = torch.tensor([2.5, 2.7, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * 10
    model.eval()
    l_errors, l_masks = [], []
    l_errors_rmse, l_masks_rmse = [], []
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    count = -1
    with torch.no_grad():
        for it, (seqs, target, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            masks_rmse = masks[:, :42]
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(
                cf.device)
            error_ens_rmse = torch.zeros((batchsize, 24, cf.n_samples)).to(
                cf.device)
            for i_sample in range(cf.n_samples):
                preds = trainers.sample(model,
                                        target,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                preds_reduced = preds[:, :42, :2]
                inputs_reduced = inputs[:, :42, :2]
                errors_squared = (preds_reduced - inputs_reduced) ** 2
                errors_squared = errors_squared.sum(dim=(2))
                masked_errors = errors_squared * masks_rmse

                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]
                error_ens_rmse[:, :, i_sample] = masked_errors[:, cf.init_seqlen:]

            l_errors.append(error_ens.min(dim=-1))
            l_errors_rmse.append(error_ens_rmse.min(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])
            l_masks_rmse.append(masks_rmse[:, cf.init_seqlen:])


            def denormalize_coords(coords, min_val, max_val):
                return coords * (max_val - min_val) + min_val


            if cf.plot_result:
                count += 1
                cmap = plt.colormaps.get_cmap("jet")
                preds_np = preds.detach().cpu().numpy()
                inputs_np = seqs.detach().cpu().numpy()
                n_bs = inputs_np.shape[0]
                n_sb = inputs_np.shape[1]
                lat_min, lat_max = 10.3, 13.0
                lon_min, lon_max = 55.5, 58.0
                for idx in range(n_bs):
                    plt.figure(figsize=(10, 6), dpi=150)
                    c = cmap(float(idx) / n_sb)
                    try:
                        seqlen = seqlens[idx].item()
                    except:
                        continue
                    inputs_lat = denormalize_coords(inputs_np[idx][:42, 1], lat_min, lat_max)
                    inputs_lon = denormalize_coords(inputs_np[idx][:42, 0], lon_min, lon_max)
                    preds_lat = denormalize_coords(preds_np[idx][:42, 1], lat_min, lat_max)
                    preds_lon = denormalize_coords(preds_np[idx][:42, 0], lon_min, lon_max)

                    plt.plot(inputs_lat[:init_seqlen + 1], inputs_lon[:init_seqlen + 1], "g-",
                             label='Input')
                    plt.plot(inputs_lat[init_seqlen:42], inputs_lon[init_seqlen:42], 'b-', label='True')
                    plt.plot(preds_lat[init_seqlen:42], preds_lon[init_seqlen:42], "r--", label='Predicted')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.legend()
                    img_path = os.path.join(cf.savedir_result_picture, f'traj_{count :03d}--number_{idx + 1:03d}.jpg')
                    plt.savefig(img_path)
                    plt.close()

    l_ = [x.values for x in l_errors]
    m_masks = torch.cat(l_masks, dim=0)
    errors = torch.cat(l_, dim=0) * m_masks
    pred_errors = errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()
    l_rmse = [x.values for x in l_errors_rmse]
    m_masks_rmse = torch.cat(l_masks_rmse, dim=0)
    errors_rmse = torch.cat(l_rmse, dim=0) * m_masks_rmse
    pred_errors_rmse = errors_rmse.sum(dim=0) / (m_masks_rmse.sum(dim=0) * 2)
    pred_errors_rmse = torch.sqrt(pred_errors_rmse)
    pred_errors_rmse = pred_errors_rmse.detach().cpu().numpy()

    dili_4 = pred_errors[:24]
    rmse_4 = pred_errors_rmse[:24]
    dli_mean = dili_4.mean()
    rmse_mean = rmse_4.mean()
    print("整体平均RMSE误差", rmse_mean.item())
    print("平均哈弗塞恩误差", dli_mean.item())

    errors_df = pd.DataFrame({
        'Point Index': range(1, len(dili_4) + 1),
        'Average Haversine Distance Error (kilometers)': dili_4
    })
    csv_path = os.path.join(cf.savedir, 'average_haversine_distance_errors.csv')
    errors_df.to_csv(csv_path, index=False)

    rmse_file_path = os.path.join(cf.savedir, 'average_rmse_error 1.txt')
    with open(rmse_file_path, 'w') as file:
        file.write(f"整体平均RMSE误差: {rmse_mean:.4f} km\n")
        file.write(f"平均哈弗塞恩误差: {dli_mean:.4f} km\n")

    ## Plot
    # ===============================
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    timestep = 6
    plt.plot(1, pred_errors[timestep], "o")
    plt.plot([1, 1], [0, pred_errors[timestep]], "r")
    plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 12
    plt.plot(2, pred_errors[timestep], "o")
    plt.plot([2, 2], [0, pred_errors[timestep]], "r")
    plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 18
    plt.plot(3, pred_errors[timestep], "o")
    plt.plot([3, 3], [0, pred_errors[timestep]], "r")
    plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 24
    plt.plot(4, pred_errors[timestep], "o")
    plt.plot([4, 4], [0, pred_errors[timestep]], "r")
    plt.plot([0, 4], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(4.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 30
    plt.plot(5, pred_errors[timestep], "o")
    plt.plot([5, 5], [0, pred_errors[timestep]], "r")
    plt.plot([0, 5], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(5.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 36
    plt.plot(6, pred_errors[timestep], "o")
    plt.plot([6, 6], [0, pred_errors[timestep]], "r")
    plt.plot([0, 6], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(6.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 42
    plt.plot(7, pred_errors[timestep], "o")
    plt.plot([7, 7], [0, pred_errors[timestep]], "r")
    plt.plot([0, 7], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(7.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 48
    plt.plot(8, pred_errors[timestep], "o")
    plt.plot([8, 8], [0, pred_errors[timestep]], "r")
    plt.plot([0, 8], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(8.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
    plt.xlabel("Time (hours)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 12])
    plt.ylim([0, 20])
    plt.savefig(cf.savedir + "prediction_error.png")
