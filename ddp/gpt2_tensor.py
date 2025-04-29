#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
torchrun --standalone --nproc-per-node 2 gpt2_tp.py
Runs GPT-2 small with 1-D tensor parallelism on 2 CPU ranks.
"""
import os, torch, torch.distributed as dist
from transformers import GPT2Config, GPT2LMHeadModel

# --- TP imports
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module
)

def shard_gpt2(model, mesh):
    """
    Apply Colwise / Rowwise TP to every Linear that matches GPT-2's
    naming convention.  Keeps the example short; for production
    you’d build an explicit dict(plan) instead.
    """
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            # QKV and feed-forward expanders: shard columns (input_dim kept, out_dim / tp)
            if name.endswith(("c_attn", "c_fc")):
                parallelize_module(mod, mesh, ColwiseParallel())
            # Output projections: shard rows (input_dim / tp, out_dim kept)
            elif name.endswith(("c_proj",)):
                parallelize_module(mod, mesh, RowwiseParallel())
    return model


def main():
    # 1) distributed init -----------------------------------------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "gloo"                       # CPU backend
    dist.init_process_group(backend)

    # 2) device mesh for TP ---------------------------------------------------
    # One-dimensional mesh over the world; works on CPU the same way it does on GPUs
    tp_mesh = init_device_mesh("cpu", (world_size,))

    # 3) build & shard the model ---------------------------------------------
    config = GPT2Config()                  # 124 M params
    model  = GPT2LMHeadModel(config)
    model  = shard_gpt2(model, tp_mesh)    # <- tensor parallelism
    model  = model.to(torch.device("cpu"))

    # 4) toy data -------------------------------------------------------------
    bs, seq = 4, 128
    vocab   = config.vocab_size
    x = torch.randint(0, vocab, (bs, seq), dtype=torch.long)
    y = torch.randint(0, vocab, (bs, seq), dtype=torch.long)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for step in range(5):
        opt.zero_grad()
        loss = model(input_ids=x, labels=y).loss
        loss.backward()
        opt.step()
        if local_rank == 0:
            print(f"[TP-rank 0] step {step}  loss={loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
