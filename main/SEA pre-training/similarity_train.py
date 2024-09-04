
import math
import logging

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim



class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, device=torch.device("cpu")):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = device
        self.model = model.to(device)


    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path_sim}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path_sim)

    def train(self):
        model, config = self.model, self.config
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)  # 设置模型为训练模式或验证模式
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            # 训练模式创建进度条
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            count = 0 # 绘图标签
            for it, (seqs, masks, seqlens) in pbar:

                # 将数据放置在正确的设备上
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)

                # 确保每次从批次中取出两条轨迹
                for index_1 in range(0, len(seqs) - 1):
                    for index_2 in range(index_1, len(seqs)):
                        seq1 = seqs[index_1]  # (maxseqlen , 2)
                        seq2 = seqs[index_2]

                        # 前向传播
                        with torch.set_grad_enabled(is_train):  # 训练模式启动梯度计算
                            loss, _, _, _, _, _, _, _, _ = model(seq1, seq2, count, index_1, index_2, masks=masks)
                        losses.append(loss.item())

                        d_loss += loss.item()
                        d_n += 1
                        if is_train:
                            model.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip_sim)  # 1
                            optimizer.step()

                            # decay the learning rate based on our progress
                            if config.lr_decay:
                                self.tokens += (
                                        seqs >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                                if self.tokens < config.warmup_tokens:
                                    # 处理的tokens数量小于预热tokens（config.warmup_tokens），则执行线性预热策略。 linear warmup
                                    lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                                else:
                                    # cosine learning rate decay 余弦学习率衰减策略
                                    progress = float(self.tokens - config.warmup_tokens) / float(
                                        max(1, config.final_tokens - config.warmup_tokens))
                                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                                lr = config.learning_rate * lr_mult
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = lr
                            else:
                                lr = config.learning_rate

                            # report progress 输出训练集结果
                            pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}, "
                                                 f"similarity model.")
                count += 1

            if is_train:
                logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}, similarity model.")
            else:
                logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, similarity model.")

            if not is_train:
                test_loss = float(np.mean(losses))
                #                 logging.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        for epoch in range(config.max_sim_epochs):

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path_sim is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch + 1)


