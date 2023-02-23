"""
This training script is a reduced version of the original nanoGPT, but has been converted to Fabric.
The full version can be found here: https://github.com/Lightning-AI/nanoGPT

On CPU (slow):
$ lightning run model train_fabric_lm.py

Single GPU:
$ lightning run model --accelerator=cuda train_fabric_lm.py

Mixed precision:
$ lightning run model --accelerator=cuda --precision=bf16 train_fabric_lm.py

Multi-GPU (DDP):
$ lightning run model --accelerator=cuda --precision=bf16 --devices=4 train_fabric_lm.py
"""
from dataclasses import asdict

import os
import time

import torch
from lightning.fabric import Fabric
from lightning.pytorch.callbacks import ModelSummary

from model import LitDataModule, LitGPT


# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
always_save_checkpoint = True  # if True, always save a checkpoint after each eval

gradient_accumulation_steps = 5  # used to simulate larger batch sizes
max_iters = 600000  # total number of training iterations

compile = False  # use PyTorch 2.0 to compile the model to be faster

# Initialize Fabric
# It will receive configuration from the command line via `lightning rum model ...`
fabric = Fabric(callbacks=ModelSummary())

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

iter_num = 0
best_val_loss = 1e9

# init a new model from scratch
fabric.print("Initializing a new model from scratch")


model = LitGPT()
datamodule = LitDataModule()


# compile the model
if compile:
    fabric.print("compiling the model ...")
    model = torch.compile(model)  # requires PyTorch 2.0


model, optimizer = fabric.setup(model, model.configure_optimizers())
train_dataloader, val_dataloader = fabric.setup_dataloaders(datamodule.train_dataloader(), datamodule.val_dataloader())


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k, batch in enumerate(val_dataloader):
            X, Y = batch
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            if k == eval_iters - 1:
                break
        out[split] = losses.mean()
    model.train()
    return out


# training loop
t0 = time.time()
train_iter = iter(train_dataloader)

fabric.call("on_fit_start", fabric, model)

while True:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        batch = next(train_iter)

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
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": asdict(model.gptconf),
                }
                fabric.print(f"saving checkpoint to {out_dir}")
                fabric.save(os.path.join(out_dir, "ckpt.pt"), checkpoint)

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        with fabric.no_backward_sync(model, enabled=(micro_step < gradient_accumulation_steps - 1)):
            # fetch a batch
            loss = model.training_step(batch)
            # backward pass
            fabric.backward(loss)

    model.on_before_optimizer_step(optimizer)

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
