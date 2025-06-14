#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
torchrun --standalone --nproc-per-node 2 test1.py
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def main():
    # 1️⃣ 读取 torchrun 注入的环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank       = int(os.environ["RANK"])

    print(f"rank {rank} started")
    # 2️⃣ 根据环境选择设备 & 后端
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cpu")
    backend  = "nccl" if use_cuda else "gloo"

    # 3️⃣ 初始化进程组
    dist.init_process_group(backend)

    # 4️⃣ 构建模型
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    loss_fn  = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)

    # 5️⃣ 伪造一批数据跑一次前后向
    inputs = torch.randn(20, 10, device=device)
    labels = torch.randn(20,  5, device=device)

    for i in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"[rank 0] Finished iteration {i}, loss={loss.item():.4f}")
        elif rank == 1:
            print(f"[rank 1] Finished iteration {i}, loss={loss.item():.4f}")


    # 6️⃣ 结束
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
