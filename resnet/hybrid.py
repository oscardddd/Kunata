#!/usr/bin/env python3

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipeline.sync import Pipe
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# 1. Initialization
# ----------------------------
def init_process(rank, world_size, backend='nccl'):
    """
    Initializes the default process group for distributed training.
    """
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] process initialized.")

# ----------------------------
# 2. Model Definition
# ----------------------------

class Stage1(nn.Sequential):
    """
    First part of the model. For demonstration, we'll define
    a simple CNN block with a single Conv layer.
    """
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

class Stage2(nn.Sequential):
    """
    Second part of the model. Another block of layers and a linear layer.
    """
    def __init__(self):
        super().__init__(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 10)  # For MNIST-like classification
        )

def build_pipeline():
    """
    Build a simple two-stage pipeline model.
    We assign:
      - Stage 1 to CUDA device 0
      - Stage 2 to CUDA device 1
    Then we wrap the sequential model in Pipe.
    """
    # Create the sequential container on the CPU initially
    model = nn.Sequential(
        Stage1().to('cuda:0'),
        Stage2().to('cuda:1'),
    )
    
    # Wrap in Pipe for pipeline parallelism
    # chunks=8 => micro-batches
    # balance=[1, 1] => each stage is a single module in the sequential
    # devices=[0, 1] => stage 1 on GPU 0, stage 2 on GPU 1
    pipeline_model = Pipe(
        model,
        chunks=8,
        balance=[1, 1],
        devices=[0, 1]
    )
    
    return pipeline_model

# ----------------------------
# 3. Training Function
# ----------------------------
def train(rank, world_size):
    """
    Main training loop for a single process.
    - rank: The process rank.
    - world_size: Total number of processes.
    """
    init_process(rank, world_size)

    # Build the pipeline-parallel model (on rank 0 or rank 1, etc.)
    pipeline_model = build_pipeline()

    # Wrap the pipeline model with DistributedDataParallel
    # This replicates the pipeline across all processes, enabling data parallelism
    ddp_model = DDP(
        pipeline_model,
        device_ids=[rank],      # This process handles 'rank' GPU
        output_device=rank      # Output on the same GPU
    )

    # Define an optimizer and loss function
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Sample training loop
    for epoch in range(3):
        # Random input (batch=32, 1 channel, 28x28) to simulate MNIST
        inputs = torch.randn(32, 1, 28, 28).cuda(rank)
        labels = torch.randint(0, 10, (32,)).cuda(rank)

        # Forward
        optimizer.zero_grad()
        outputs = ddp_model(inputs)

        # Compute loss and backprop
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print loss from rank 0
        if rank == 0:
            print(f"[Epoch {epoch}] Loss = {loss.item():.4f}")

    # Cleanup
    dist.destroy_process_group()

# ----------------------------
# 4. Main Entry Point
# ----------------------------
def main():
    """
    Entry point to launch multiple processes (ranks) for distributed training.
    """
    # Here we set world_size=2 for illustration:
    #   - We expect 2 processes, each assigned to one GPU.
    #   - In a real system with more GPUs or multiple nodes, increase world_size.
    world_size = 2

    # Spawn the given 'train' function in 'world_size' processes
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
