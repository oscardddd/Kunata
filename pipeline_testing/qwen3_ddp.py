#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
torchrun --standalone --nproc-per-node 2 qwen3_ddp.py
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def main():
    # 1️⃣ torchrun environment
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = int(os.environ["RANK"])

    # 2️⃣ device & backend
    use_cuda = torch.cuda.is_available()
    device   = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    backend  = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend)


    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 👇 先加载 config，然后手动补字段
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Qwen3 特有：必须显式设置 attention 实现，否则 transformers 默认是 None
    if not hasattr(config, "attn_implementation") or config.attn_implementation is None:
        config.attn_implementation = "sdpa"  # or "flash_attention_2" if installed

    # 👇 再加载模型，确保 config 是你手动处理过的
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)


    ddp_model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    # 4️⃣ fake random tokens
    batch_size, seq_len = 4, 128
    vocab_size = config.vocab_size
    inputs  = torch.randint(0, vocab_size, (batch_size, seq_len),
                            dtype=torch.long, device=device)
    labels  = torch.randint(0, vocab_size, (batch_size, seq_len),
                            dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=5e-5)

    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = ddp_model(input_ids=inputs, labels=labels).loss
        loss.backward()
        optimizer.step()
        print(f"[rank {rank}] step {step}, loss = {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
