import os

import torch
from lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset, DataLoader
import numpy as np

from gpt import GPT


class LitGPT(LightningModule):
    def __init__(self, config, block_size):
        super().__init__()
        self.gpt = GPT(config)
        self.weight_decay = 1e-2
        self.learning_rate = 6e-4
        self.betas = (0.9, 0.95)
        self.grad_clip = 1.0

        # crop down the model block size if desired
        if block_size < self.gpt.config.block_size:
            self.gpt.crop_block_size(block_size)

    def training_step(self, batch):
        X, Y = batch
        _, loss = self.gpt(X, Y)
        return loss

    def configure_optimizers(self):
        return self.gpt.configure_optimizers(self.weight_decay, self.learning_rate, self.betas)

    def on_before_optimizer_step(self, optimizer):
        # clip the gradient
        if self.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)


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
