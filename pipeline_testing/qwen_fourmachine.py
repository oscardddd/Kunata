#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
四机两卡（共 8 GPU）示例：
--------------------------------------------------
假设 node‑0 的 IP 是 10.0.0.1；其余三台机器能够 SSH 互通。
在 **每台机器** 上分别执行（只需改 node_rank）：

```bash
export NCCL_IB_DISABLE=1          # 如果没有 InfiniBand 建议关掉
export CUDA_VISIBLE_DEVICES=0,1   # 每台 2 张 GPU

torchrun \
  --nnodes 4 \
  --nproc-per-node 2 \
  --node_rank <0‑3> \
  --master_addr 10.0.0.1 \
  --master_port 29500 \
  qwen3_ddp_multinode.py
```

> ⚠️ `torchrun` 会在每个进程里注入 `LOCAL_RANK / RANK / WORLD_SIZE / MASTER_*` 等环境变量；脚本只需读取即可。
"""
from __future__ import annotations
import os, socket, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Utility: pretty log
# ---------------------------------------------------------------------------

def log(msg: str):
    rank = int(os.environ.get("RANK", -1))
    host = socket.gethostname().split(".")[0]
    print(f"[rank {rank:>2} | {host}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    # 1️⃣ torchrun environment -------------------------------------------------
    local_rank = int(os.environ["LOCAL_RANK"])            # rank within node
    global_rank = int(os.environ["RANK"])                 # unique across cluster
    world_size = int(os.environ["WORLD_SIZE"])            # total processes

    # 2️⃣ device & backend -----------------------------------------------------
    use_cuda = torch.cuda.is_available()
    device   = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    backend  = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)
    torch.manual_seed(42 + global_rank)

    # 3️⃣ load Qwen‑3‑0.6B ------------------------------------------------------
    model_name = "Qwen/Qwen3-0.6B"  # 👉 可替换成 2‑B / 4‑B 等更大模型
    log("loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    log("loading model … (may take 1‑2 min)")
    config = AutoConfig.from_pretrained(model_name)
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    ).to(device)

    # Wrap with DistributedDataParallel --------------------------------------
    ddp_model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    # 4️⃣ synthetic data -------------------------------------------------------
    batch_size, seq_len = 4, 128
    vocab_size = config.vocab_size
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=5e-5)

    log("start training …")
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = ddp_model(input_ids=inputs, labels=labels).loss
        loss.backward()
        optimizer.step()
        if global_rank == 0:
            log(f"step {step:02d} | loss = {loss.item():.4f}")

    dist.barrier()  # make sure all ranks finish before cleanup
    dist.destroy_process_group()
    log("training finished ✅")


if __name__ == "__main__":
    main()
