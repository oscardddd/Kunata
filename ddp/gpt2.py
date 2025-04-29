#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
torchrun --standalone --nproc-per-node 2 gpt2_ddp.py
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel


def main():

    # ---------- 1. 读取 torchrun 环境变量 ----------
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = int(os.environ["RANK"])

    # ---------- 2. 设备与后端 ----------
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cpu")
    backend  = "nccl" if use_cuda else "gloo"

    dist.init_process_group(backend)

    # ---------- 3. 构建 GPT-2 small ----------
    config = GPT2Config()                 # 默认 = gpt2-small 124 M
    model  = GPT2LMHeadModel(config).to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    # ---------- 4. 伪造一批随机 token ----------
    batch_size, seq_len = 4, 128          # 放大请注意显存 / 内存1
    vocab_size = config.vocab_size
    inputs  = torch.randint(0, vocab_size, (batch_size, seq_len),
                            dtype=torch.long, device=device)
    labels  = torch.randint(0, vocab_size, (batch_size, seq_len),
                            dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=5e-5)

    for step in range(5):
        optimizer.zero_grad()
        loss = ddp_model(input_ids=inputs, labels=labels).loss   # CE loss
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"[rank 0] step {step}, loss = {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()