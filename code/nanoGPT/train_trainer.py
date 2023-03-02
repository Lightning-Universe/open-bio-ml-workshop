"""
This training script is a reduced version of the original nanoGPT, but has been converted to Trainer.
The full version can be found here: https://github.com/Lightning-AI/nanoGPT
"""
from dataclasses import asdict

import os

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelSummary

from model import LitDataModule, LitGPT
from gpt import GPTConfig


trainer = Trainer(
    accelerator="cuda",
    precision="bf16",
    devices=1,
    max_steps=600000, 
    accumulate_grad_batches=1, 
    val_check_interval=2000,
    limit_val_batches=200,
    num_sanity_val_steps=0,
)

torch.manual_seed(1337)

# init a new model from scratch
print("Initializing a new model from scratch")
model = LitGPT()
datamodule = LitDataModule()

# requires PyTorch 2.0
# model = torch.compile(model)

# training loop
trainer.fit(model, datamodule)
