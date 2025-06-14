# hybrid_cpu_parallel.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# Dummy dataset
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 128)
        self.labels = torch.randn(size, 128)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

# Split model into two logical stages
class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 256)

    def forward(self, x):
        return torch.relu(self.fc(x))

class Stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        return self.fc(x)

# Combined model with logical model parallelism
class SplitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = Stage1().to("cpu")
        self.stage2 = Stage2().to("cpu")

    def forward(self, x):
        x = x.to("cpu")
        x = self.stage1(x)
        x = self.stage2(x)
        return x

def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    dataset = SimpleDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    model = SplitModel()
    ddp_model = DDP(model)

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)

    for epoch in range(3):
        sampler.set_epoch(epoch)
        for i, (x, y) in enumerate(dataloader):
            x = x.to("cpu")
            y = y.to("cpu")

            out = ddp_model(x)
            loss = nn.functional.mse_loss(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0 and rank == 0:
                print(f"[Rank {rank}] Epoch {epoch}, Step {i}, Loss {loss.item():.4f}")
    dist.destroy_process_group() 

def main():
    world_size = 2  # Use 2 CPU processes
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    

if __name__ == "__main__":
    main()
