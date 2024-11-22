# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_resnet50.py

import argparse
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint

import torchvision.models as models

def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters())

def generate_inputs_for_resnet(batch_size, device):
    input_size = (batch_size, 3, 224, 224)
    inputs = torch.randn(input_size, device=device)
    return (inputs,)

def get_current_time():
    # Define the timezone (e.g., 'Asia/Shanghai' for CST)
    timezone = ZoneInfo('Asia/Shanghai')  # Adjust as needed
    now = datetime.now(timezone)
    formatted_time = now.strftime("%b %d, %Y %H:%M:%S.%f %Z")
    return formatted_time

def run(args):
    # Model configs
    print("Using device:", args.device)

    # Create model
    resnet = models.resnet50()
    resnet.to(args.device)
    resnet.eval()
    
    print(resnet)

    # Example microbatch inputs
    example_mb = generate_inputs_for_resnet(args.batch_size // args.chunks, args.device)

    # Split points
    split_spec = {
        'layer2': SplitPoint.BEGINNING,
    }

    # Create pipeline
    pipe = pipeline(
        resnet,
        mb_args=example_mb,
        split_spec=split_spec,
    )

    assert pipe.num_stages == args.world_size, f"nstages = {pipe.num_stages} nranks = {args.world_size}"
    smod = pipe.get_stage_module(args.rank)

    # Calculate and print parameter counts
    params_in_millions = get_number_of_params(smod) / 1e6
    print(f"Pipeline stage {args.rank} {params_in_millions:.2f}M params")

    # Create schedule runtime
    stage = pipe.build_stage(
        args.rank,
        device=args.device,
    )

    # Attach to a schedule
    schedule = ScheduleGPipe(stage, args.chunks)

    # Full batch inputs
    inputs = generate_inputs_for_resnet(args.batch_size, args.device)

    for i in range(1000):
        print(f"rank {args.rank}, iteration {i+1}")
        # Run
        if args.rank == 0:
            schedule.step(*inputs)
        else:
            schedule.step()
        
        # Sleep for 2 seconds
        time.sleep(2)
        
        # Get and print the current timestamp
        current_time = get_current_time()
        print(f"{current_time} - Rank {args.rank} completed iteration {i+1}")

    dist.barrier()
    dist.destroy_process_group()
    print(f"Rank {args.rank} completes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batches', type=int, default=1)

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run(args)
