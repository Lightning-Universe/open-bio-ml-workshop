"""
This training script is a reduced version of the original nanoGPT, but has been converted to Fabric.
The full version can be found here: https://github.com/Lightning-AI/nanoGPT

On CPU (slow):
$ lightning run model train_fabric.py

Single GPU:
$ lightning run model --accelerator=cuda train_fabric.py

Mixed precision:
$ lightning run model --accelerator=cuda --precision=bf16 train_fabric.py

Multi-GPU (DDP):
$ lightning run model --accelerator=cuda --precision=bf16 --devices=4 train_fabric.py
"""
from dataclasses import asdict

import os
import time
import math

import numpy as np
import torch
from lightning.fabric import Fabric

from model import GPTConfig, GPT

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
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
compile = False  # use PyTorch 2.0 to compile the model to be faster

# Initialize Fabric
# It will receive configuration from the command line via `lightning rum model ...`
fabric = Fabric()

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

data_dir = os.path.join("data", dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    x, y = x.to(fabric.device), y.to(fabric.device)
    return x, y


iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict()

# init a new model from scratch
fabric.print("Initializing a new model from scratch")

gptconf = GPTConfig(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout,
    vocab_size=vocab_size,
    bias=bias,
)
model = GPT(gptconf)

# crop down the model block size if desired
if block_size < model.config.block_size:
    model.crop_block_size(block_size)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))

# compile the model
if compile:
    fabric.print("compiling the model ...")
    model = torch.compile(model)  # requires PyTorch 2.0


model, optimizer = fabric.setup(model, optimizer)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(iter):
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


# training loop
t0 = time.time()
while True:
    # determine the learning rate for this iteration
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    else:
        lr = learning_rate

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num > 0 and iter_num % eval_interval == 0:
        fabric.print("running validation")
        losses = estimate_loss()
        fabric.print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": model,
                    "optimizer": optimizer,
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": asdict(gptconf),
                }
                fabric.print(f"saving checkpoint to {out_dir}")
                fabric.save(os.path.join(out_dir, "ckpt.pt"), checkpoint)

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        with fabric.no_backward_sync(model, enabled=(micro_step < gradient_accumulation_steps - 1)):
            # fetch a batch
            X, Y = get_batch("train")
            logits, loss = model(X, Y)
            # backward pass
            fabric.backward(loss)

    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer
    optimizer.step()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item()
        fabric.print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
