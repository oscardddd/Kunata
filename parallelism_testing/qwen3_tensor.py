#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
torchrun --standalone --nproc-per-node 2 qwen3_tensor.py
Runs **Qwen‑3 0.6B** with 1‑D tensor parallelism on 2 CPU ranks.

Requirements
------------
    pip install "transformers>=4.51.0" accelerate
    # 1‑D TP utilities ship with PyTorch ≥ 2.2
"""
import os
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

# ── 1‑D Tensor Parallel imports ────────────────────────────────────────────────
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


def shard_qwen3(model: torch.nn.Module, mesh):
    """Apply Col‑/Row‑wise TP to Qwen‑3 linear layers."""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            # Attention and MLP *input* projections → shard columns
            if name.endswith((
                "q_proj", "k_proj", "v_proj",  # self‑attention QKV
                "gate_proj", "up_proj",         # gated‑MLP expanders
            )):
                parallelize_module(mod, mesh, ColwiseParallel())
            # Attention & MLP *output* projections → shard rows
            elif name.endswith(("o_proj", "down_proj")):
                parallelize_module(mod, mesh, RowwiseParallel())
    return model


def main():
    # 1) ── distributed init ───────────────────────────────────────────────────
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "gloo"            # CPU backend (switch to "nccl" for GPUs)
    dist.init_process_group(backend)

    # 2) ── create 1‑D device mesh ─────────────────────────────────────────────
    tp_mesh = init_device_mesh("cpu", (world_size,))

    # 3) ── load & shard model ─────────────────────────────────────────────────
    model_name = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32,  # keep fp32 on CPU; use fp16 on GPUs
        device_map=None,           # avoid automatic map so we can shard first
    )

    # 🔍 Print the full module hierarchy once (rank 0 only)
    if local_rank == 0:
        print(model)

    model = shard_qwen3(model, tp_mesh)
    model = model.to(torch.device("cpu"))

    # 4) ── synthetic data ─────────────────────────────────────────────────────
    batch_size, seq_len = 4, 128
    print(vocab_size)
    vocab_size = config.vocab_size
    inp = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 5) ── tiny training loop ─────────────────────────────────────────────────
    for step in range(5):
        optim.zero_grad()
        loss = model(input_ids=inp, labels=tgt).loss  # cross‑entropy
        loss.backward()
        optim.step()
        print(f"[TP‑rank {local_rank}] step {step}\tloss = {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
