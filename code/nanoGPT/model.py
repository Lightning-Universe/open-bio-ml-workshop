import os
import math 

import torch
from lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset, DataLoader
import numpy as np

from gpt import GPT

learning_rate = 6e-4  # max learning rate
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


class LitGPT(LightningModule):
    def __init__(self, config, block_size):
        super().__init__()
        self.gpt = GPT(config)

        # crop down the model block size if desired
        if block_size < self.gpt.config.block_size:
            self.gpt.crop_block_size(block_size)

    def training_step(self, batch):
        X, Y = batch
        _, loss = self.gpt(X, Y)
        return loss

    def configure_optimizers(self):
        return self.gpt.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))

    def on_before_optimizer_step(self, optimizer):
        # clip the gradient
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

    def on_train_batch_start(self, batch, batch_idx):
        # determine the learning rate for this iteration
        if decay_lr:
            lr = self.get_lr(self.global_step)
            for param_group in self.optimizers().param_groups:
                param_group["lr"] = lr
        else:
            lr = learning_rate

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, iter):
        # 1) linear warmup for warmup_iters steps
        if iter < warmup_iters:
            return learning_rate * iter / warmup_iters
        # 2) if iter > lr_decay_iters, return min learning rate
        if iter > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
        


class TextDataset(Dataset):
    def __init__(self, path, block_size):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        x = torch.from_numpy((self.data[i: i + self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[i + 1: i + 1 + self.block_size]).astype(np.int64))
        return x, y


class LitDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, block_size):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.train_data = os.path.join(data_dir, "train.bin")
        self.val_data = os.path.join(data_dir, "val.bin")

    def train_dataloader(self):
        return DataLoader(
            dataset=TextDataset(self.train_data, self.block_size),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=TextDataset(self.val_data, self.block_size),
            batch_size=self.batch_size,
            shuffle=False,
        )
