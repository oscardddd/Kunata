import torch
import torch.nn as nn
import time
import argparse
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe, PipelineStage
import os
import torch.distributed as dist

torch.set_num_threads(20)
# Transformer Block definition
import torch
import torch.nn as nn

# Transformer Block definition
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.linear1(x)
        ffn_output = self.relu(ffn_output)
        ffn_output = self.linear2(ffn_output)
        x = self.layernorm2(x + ffn_output)
        return x

# Define the model wrapper
class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, d_ffn, num_heads):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ffn, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


rank = int(os.environ["RANK"])  # Get the rank from environment
world_size = int(os.environ["WORLD_SIZE"])  # Get the world size from environment

print(rank, world_size)



# Function to benchmark model for a specified duration
def benchmark_model(model, input_tensor, num_iterations, num_layers, num_microbatch = 1):
    # start_time = time.time()
    elapsed_time = 0
    iterations = 0
    real_start = 0
    print(type(input_tensor))

    # Assign the device based on rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    microbatch_size = input_tensor.size(0) // num_microbatch  # Calculate microbatch size

    # 这个是microbatch下的 tensor size
    example_input = torch.rand(microbatch_size, input_tensor.size(1), input_tensor.size(2)).to(device)
    # Move the model to the respective device
    model.to(device)
    # model.eval()


    split_spec = {
        "layers.2": SplitPoint.BEGINNING
    }
    chunks = num_microbatch
    print(f"The number of microbatches is: {chunks}")


    # Create the pipeline object
    pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec)
    print(pipe)
    # Initialize distributed environment
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # Define the pipeline stage
    stage = PipelineStage(pipe, rank, num_stages=2, input_args=(example_input,), device=device)
    schedule = ScheduleGPipe(stage, chunks)
    print(f"{rank} Finished stage creation")
    for _ in range(5):
        if rank == 0:
            schedule.step(input_tensor)
        else:
            output = schedule.step()
        if rank == world_size - 1:
            reference_output = model(input_tensor)
    iterations = 0
    real_start = time.time()

    print(f"Rank {rank}, real start is {time.time()-real_start}")
    while iterations < num_iterations:
        if rank == 0:
            schedule.step(input_tensor)
            print(f"{rank}, Time: {time.time()-real_start}")
        else:
            output = schedule.step()
        if rank == world_size - 1:

            # reference_output = model(input_tensor)
            # print("Pipeline parallel model ran successfully!".center(80, "*"))
            iterations += 1

            print(f"Rank {rank}, Time: {time.time()-real_start}: Iteration {iterations} completed.")

    total_time = time.time() - real_start
    avg_time_per_iteration = (total_time) / iterations
    print(f"Benchmark completed: {iterations} iterations in {total_time:.4f} seconds")
    print(f"Average time per iteration: {avg_time_per_iteration:.4f} seconds")
    return avg_time_per_iteration


# Configuration parameters for different models
configs = {
    'base': {'d_model': 768, 'd_ffn': 3072, 'num_heads': 12, 'num_layers': 4},
    'xxlarge': {'d_model': 4096, 'd_ffn': 16384, 'num_heads': 32, 'num_layers': 4},
    'GPT-3': {'d_model': 12288, 'd_ffn': 49152, 'num_heads': 96, 'num_layers': 2},
    'Ours': {'d_model': 4096, 'd_ffn': 16384, 'num_heads': 32, 'num_layers': 2}  # 3 layers per pipeline stage
}


# Main function to parse arguments and run the benchmark
def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models.")
    parser.add_argument("--model", type=str, choices=['base', 'xxlarge', 'GPT-3', 'Ours'], required=True,
                        help="Model type to benchmark.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for input tensor.")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for input tensor.")
    parser.add_argument("--num_microbatches", type=int, default=1, help="Number of microbatches for pipeline parallelism.")

    args = parser.parse_args()

    # Get the selected configuration
    config = configs[args.model]

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        d_ffn=config['d_ffn'],
        num_heads=config['num_heads']
    ).to(device)

    # Create input tensor (batch_size, seq_len, d_model)
    input_tensor = torch.rand(args.batch_size, args.seq_len, config['d_model']).to(device)
    # print(f"Input dimention: {input_tensor}" )

    # Run benchmark
    print(f"Benchmarking {args.model} version for {args.iterations} iterations...")
    benchmark_model(model, input_tensor, num_iterations = args.iterations, num_layers=config['num_layers'], num_microbatch=args.num_microbatches)


if __name__ == "__main__":
    main()
