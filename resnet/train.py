import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
import os
import time
import argparse
# Minimum effort to run this example:
# $ torchrun --nproc-per-node 2 train.py
from torchvision.models import resnet50

def run(args, model):
    split_spec = {
        "layer1": SplitPoint.END,
    }
    rank = int(os.environ["RANK"])  # Get the rank from environment
    world_size = int(os.environ["WORLD_SIZE"])  # Get the world size from environment
    
    batch_size = args.batch_size
    # random input data
    input_data = torch.randn(batch_size, 3, 224, 224).to(device)
    microbatch_size = input_data.size(0) // args.chunks  # Calculate microbatch size
    example_input = torch.rand(microbatch_size, input_data.size(1), input_data.size(2),  input_data.size(3)).to(device)

    model.to(args.device)
    model.eval() 
    # Create pipeline
    pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec)
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    stage = pipe.build_stage(rank, device)
    loss_fn = nn.CrossEntropyLoss()
    schedule = ScheduleGPipe(stage, args.chunks, loss_fn=loss_fn)
    target = torch.randint(0, 1000, (batch_size,), device=device)

    # Create local optimizer (only for parameters on this stage)
    local_params = list(stage.submod.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    start_time = time.time()  # Record initial start time

    # Simple training iteration
    for i in range(1000):
        print(f"iteration {i}")  
        optimizer.zero_grad()
        if rank == 0:
            # Rank 0 feeds the input and target
            schedule.step(input_data)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Rank {rank}: Iteration {i} stage 0 completed in {elapsed:.4f} seconds")
            optimizer.step()

        elif rank == world_size - 1:
            losses = []
            schedule.step(target=target, losses=losses)
            
            print(len(losses))
            end_time = time.time()
            elapsed = end_time - start_time
            optimizer.step()
            combined_loss = sum(losses) / len(losses)
            print(f"Rank {rank}: Loss at iteration {i} = {combined_loss}")
            print(f"Rank {rank}: iteration {i} final stage completed in {elapsed:.4f} seconds")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 2)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batches', type=int, default=1)

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")
    
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print("load the model")

    model = resnet50()

    run(args, model)
