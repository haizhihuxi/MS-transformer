import os
import torch


class Config():
    sim_half_train = False
    similarity_retrain = True
    sim_test = True
    sim_train_plot_result = False
    device = torch.device("cuda:0")
    #     device = torch.device("cpu")

    max_epochs = 50
    max_sim_epochs = 15
    batch_size = 32
    n_samples = 16

    init_seqlen = 18
    max_seqlen = 140
    min_seqlen = 36

    input_dim = 2
    hidden_dim = 256
    output_dim = 10
    window_size_20 = 20
    window_size_10 = 10
    window_size_15 = 15
    window_size_5 = 5


    # ===========================================================================
    # Model and sampling flags
    mode = "pos"
    sample_mode = "pos_vicinity"
    top_k = 10  # int or None
    r_vicinity = 40  # int
    frechet_judge = False
    dtw_judge = False
    erp_judge = False
    cos_judge = False
    hausdorff_judge = True

    show_parameter = False

    # Data flags
    # ===================================================
    datadir = f"./data/"
    trainset_name = f"train.pkl"
    validset_name = f"valid.pkl"
    testset_name = f"test.pkl"

    # model parameters
    # ===================================================
    traj_compary_threshold = 0.05
    threathod_10 = 0.2
    threathod_5 = 0.2
    weight_10 = 0.5
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1


    # optimization parameters
    # ===================================================
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    grad_norm_clip_sim = 0.5
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original 模型在训练初期更快地收敛，然后在训练后期进行更精细的参数调整。
    lr_decay = True
    warmup_tokens = 512 * 20  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    num_workers = 6  # for DataLoader

    filename = f"hausdorff_judge" \
               + f"-window_size-{window_size_5}"\
               + f"-window_size-{window_size_10}"\
               + f"-window_size-{window_size_15}"\
               + f"-window_size-{window_size_20}"
    savedir = "./re-results/" + filename + "/"
    savedir_test = "./results/" + filename + "/"
    sim_outfile = 'evaluation_results.txt'

    ckpt_path = os.path.join(savedir, "model.pt")
    ckpt_path_sim = os.path.join(savedir, "model_sim.pt")
