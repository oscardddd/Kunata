import torch
import torch.nn as nn
import time
import argparse
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe, PipelineStage
import os
import torch.distributed as dist
from transformers import FNetModel, FNetConfig
from hf_utils import generate_inputs_for_model, get_number_of_params


def run(args):

    config = FNetConfig()
    # Create model
    model_class = FNetModel
    model_name = "FNetModel"
    fnet = model_class(config)
    fnet.to(args.device)
    fnet.eval()
    if args.rank == 0:
        print(fnet.config)
        print(f"Total number of params = {get_number_of_params(fnet) // 10 ** 6}M")
        print(fnet)

    
    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, fnet, model_name, args.batch_size, args.device)
    input_ids = example_inputs["input_ids"]
    print("######## The example inputs looks like?", input_ids)

    # Split things
    layers_per_rank = fnet.config.num_hidden_layers // args.world_size
    split_spec = {
        f"encoder.layer.{i * layers_per_rank}": SplitPoint.BEGINNING
        for i in range(1, args.world_size)
    }

    pipe = pipeline(
        fnet,
        mb_args=(input_ids, ),
        split_spec=split_spec,
    )
    assert pipe.num_stages == args.world_size, f"nstages = {pipe.num_stages} nranks = {args.world_size}"
    smod = pipe.get_stage_module(args.rank)
    print(f"Pipeline stage {args.rank} {get_number_of_params(smod) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        pipe,
        args.rank,
        num_stages=4,
        input_args=(input_ids,),
        device=args.device,
    )

    # Attach to a schedule
    schedule = ScheduleGPipe(stage, args.chunks)

    # Run
    if args.rank == 0:
        schedule.step(input_ids)
    else:
        out = schedule.step()

    dist.barrier()
    dist.destroy_process_group()
    print(f"Rank {args.rank} completes")



# Main function to parse arguments and run the benchmark
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batches', type=int, default=1)
    args = parser.parse_args()
    args.device = torch.device("cpu")

    backend = "nccl" if args.cuda else "gloo"

    dist.init_process_group(
    backend=backend,
    rank=args.rank,
    world_size=args.world_size,
    )
    run(args)


if __name__ == "__main__":
    main()
