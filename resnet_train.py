import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
import os
import time
import argparse
# Minimum effort to run this example:
# $ torchrun --nproc-per-node 3 resnet_train.py
from torchvision.models import resnet50
def run(args, model):
    # model.train()
    # for name, module in model.named_modules():
    #     print(name, ":", module)

    # Example splitting function: Suppose we split after layer2.
    # Identify layers by printing model and checking names of submodules.
    split_spec = {
    "layer2": SplitPoint.END
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
    optimizer = optim.SGD(local_params, lr=0.001)

    # Simple training iteration
    for i in range(50):
        print(f"iteration {i}")  
        optimizer.zero_grad()
        if rank == 0:
            # Rank 0 feeds the input and target
            schedule.step(input_data)
        else:
            losses = []
            schedule.step(target=target, losses=losses)
            
# The last stage computes the loss internally and does backward,
# The gradients should propagate automatically if pipeline is set up correctly.

# After backward pass is done, we can step the optimizer on each stage
            optimizer.step()

        if rank == world_size - 1:
            print("Training step completed on last stage.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 2)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batches', type=int, default=1)

    args = parser.parse_args()


    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")
    

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print("load the model")

    model = resnet50(pretrained=True)

    run(args, model)
