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


# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
# data
dataset = "shakespeare"
gradient_accumulation_steps = 5  # used to simulate larger batch sizes
vocab_size = 50257
batch_size = 6  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

compile = False  # use PyTorch 2.0 to compile the model to be faster


trainer = Trainer(
    accelerator="cuda",
    # precision="bf16",
    devices=1,
    max_steps=600000, 
    accumulate_grad_batches=5, 
    val_check_interval=eval_interval,
    limit_val_batches=eval_iters,
    callbacks=ModelSummary()
)

torch.manual_seed(1337)

# init a new model from scratch
print("Initializing a new model from scratch")
model = LitGPT()
datamodule = LitDataModule()


# compile the model
if compile:
    print("compiling the model ...")
    model = torch.compile(model)  # requires PyTorch 2.0

# training loop
trainer.fit(model, datamodule)
