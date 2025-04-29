#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手动 PipelineStage：rank-0 持有 Embedding + h[0-5]，
rank-1 持有 h[6-11] + ln_f + lm_head，并负责计算因果 LM 损失
"""

import os, time, argparse
import torch, torch.nn as nn, torch.distributed as dist
from transformers import GPT2Config, GPT2LMHeadModel
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

# ──────────────────────────────────────────────────────────────────────────────
# 1. 手动切分 GPT-2 small
# ──────────────────────────────────────────────────────────────────────────────
class GPT2Stage0(nn.Module):
    """Embedding + 前 6 个 block"""
    def __init__(self, full):
        super().__init__()
        self.wte  = full.transformer.wte
        self.wpe  = full.transformer.wpe
        self.drop = full.transformer.drop
        self.blocks = nn.ModuleList(full.transformer.h[:6])

    def forward(self, input_ids, labels):
        B, S = input_ids.size()
        device = input_ids.device
        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)

        hidden = self.wte(input_ids) + self.wpe(pos_ids)
        hidden = self.drop(hidden)

        for blk in self.blocks:
            hidden = blk(hidden)[0]          # 取 hidden_states

        return hidden, labels                # 把 labels 一并传下游


class GPT2Stage1(nn.Module):
    """后 6 个 block + ln_f + lm_head（输出 logits）"""
    def __init__(self, full):
        super().__init__()
        self.blocks = nn.ModuleList(full.transformer.h[6:])
        self.ln_f   = full.transformer.ln_f
        self.lm_head = full.lm_head          # 不再与 wte 共享，但 demo 足够

    def forward(self, hidden_and_labels):
        hidden, labels = hidden_and_labels
        for blk in self.blocks:
            hidden = blk(hidden)[0]
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        return logits, labels                # 供 loss_fn 计算


# ──────────────────────────────────────────────────────────────────────────────
# 2. 因果语言模型损失（labels 左移一位，与 HuggingFace 内部一致）
# ──────────────────────────────────────────────────────────────────────────────
def causal_lm_loss(tuple_in, _=None):
    logits, labels = tuple_in               # logits: (B,S,V)  labels:(B,S)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab = logits.size(-1)
    return nn.functional.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1)
    )

# ──────────────────────────────────────────────────────────────────────────────
# 3. 训练循环
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    device = torch.device("cpu")

    cfg   = GPT2Config()
    full  = GPT2LMHeadModel(cfg)

    local_mod = GPT2Stage0(full) if rank == 0 else GPT2Stage1(full)
    local_mod.to(device)

    # micro-batch 样例
    mb = args.batch_size // args.num_microbatches
    dummy_ids  = torch.zeros(mb, args.seq_len, dtype=torch.long)
    dummy_lbls = torch.zeros(mb, args.seq_len, dtype=torch.long)
    dummy_mb   = (dummy_ids, dummy_lbls)

    stage = PipelineStage(
        submodule   = local_mod,
        stage_index = rank,
        num_stages  = world,
        device      = device,
        input_args  = dummy_mb
    )

    # 随机输入
    tok = torch.randint(0, cfg.vocab_size,
                        (args.batch_size, args.seq_len),
                        dtype=torch.long, device=device)
    lbl = torch.randint(0, cfg.vocab_size,
                        (args.batch_size, args.seq_len),
                        dtype=torch.long, device=device)

    sched = ScheduleGPipe(stage,
                          n_microbatches=args.num_microbatches,
                          loss_fn=causal_lm_loss,
                        )

    for it in range(args.iterations):
        tic = time.time()

        if rank == 0:
            sched.step((tok, lbl))           # 传 tuple
        else:
            sched.step()                     # 等待上游

        if rank == world - 1:
            losses = []
            sched.step(target=None, losses=losses)
            print(f"[rank-{rank}] iter {it}  "
                  f"loss={sum(losses)/len(losses):.4f}  "
                  f"({time.time()-tic:.2f}s)")

    dist.destroy_process_group()

# ──────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--iterations", type=int, required=True)
    pa.add_argument("--batch_size", type=int, default=4)
    pa.add_argument("--seq_len",   type=int, default=128)
    pa.add_argument("--num_microbatches", type=int, default=2)
    train(pa.parse_args())
