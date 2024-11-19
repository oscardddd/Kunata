import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import os
import time
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe, PipelineStage


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
        ffn_output = self.relu(self.linear1(x))
        ffn_output = self.linear2(ffn_output)
        x = self.layernorm2(x + ffn_output)
        return x


# Complex Model definition with convolutional and transformer layers
class ComplexModel(nn.Module):
    def __init__(self, num_conv_layers, d_model, num_filters, kernel_size, num_transformer_layers, d_ffn, num_heads):
        super(ComplexModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 64  # Starting with feature size 64 for Conv1d input

        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.BatchNorm1d(num_filters))
            in_channels = num_filters

        self.fc_projection = nn.Linear(num_filters, d_model)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(d_model, d_ffn, num_heads) for _ in range(num_transformer_layers)]
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1)  # Change to (batch_size, d_model, seq_len) for Conv1d
        for layer in self.conv_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, num_filters)
        x = self.fc_projection(x)  # Map num_filters to d_model
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        return x


# Initialize distributed environment
def setup_distributed(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


# Benchmarking the model
def benchmark_model(model, input_tensor, num_iterations, num_microbatches):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    microbatch_size = input_tensor.size(0) // num_microbatches
    example_input = input_tensor[:microbatch_size]

    split_spec = {
        "conv_layers.1": SplitPoint.BEGINNING,
        "transformer_layers.1": SplitPoint.BEGINNING,
        "transformer_layers.3": SplitPoint.BEGINNING,
    }

    pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec)
    stage = PipelineStage(pipe, rank, num_stages=4, input_args=(example_input,), device=device)
    schedule = ScheduleGPipe(stage, num_microbatches)

    iterations = 0
    start_time = time.time()

    while iterations < num_iterations:
        if rank == 0:
            schedule.step(input_tensor)
        else:
            output = schedule.step()
        iterations += 1
        if rank == world_size - 1:
            print(f"Rank {rank}, Iteration {iterations} completed.")

    total_time = time.time() - start_time
    print(f"Rank {rank}: Benchmark completed: {iterations} iterations in {total_time:.4f} seconds.")
    return total_time / iterations


# Model configurations
configs = {
    "base": {"d_model": 768, "d_ffn": 3072, "num_heads": 12, "num_layers": 8, "num_conv_layers": 2, "num_filters": 512, "kernel_size": 3},
    "xxlarge": {"d_model": 4096, "d_ffn": 16384, "num_heads": 32, "num_layers": 4, "num_conv_layers": 4, "num_filters": 2048, "kernel_size": 3},
    "GPT-3": {"d_model": 12288, "d_ffn": 49152, "num_heads": 96, "num_layers": 2, "num_conv_layers": 6, "num_filters": 4096, "kernel_size": 3},
    "Ours": {"d_model": 4096, "d_ffn": 16384, "num_heads": 32, "num_layers": 2, "num_conv_layers": 3, "num_filters": 1024, "kernel_size": 3},
}


# Main function
def main():
    parser = argparse.ArgumentParser(description="Distributed Inference Benchmark")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process")
    parser.add_argument("--world_size", type=int, required=True, help="Total number of processes")
    parser.add_argument("--model", type=str, choices=["base", "xxlarge", "GPT-3", "Ours"], required=True, help="Model type")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the input tensor")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num_microbatches", type=int, default=1, help="Number of microbatches")

    args = parser.parse_args()

    setup_distributed(args.rank, args.world_size)

    config = configs[args.model]
    model = ComplexModel(
        num_conv_layers=config["num_conv_layers"],
        d_model=config["d_model"],
        num_filters=config["num_filters"],
        kernel_size=config["kernel_size"],
        num_transformer_layers=config["num_layers"],
        d_ffn=config["d_ffn"],
        num_heads=config["num_heads"],
    )

    input_tensor = torch.rand(args.batch_size, args.seq_len, config["d_model"])
    avg_time = benchmark_model(model, input_tensor, args.iterations, args.num_microbatches)
    print(f"Rank {args.rank}: Average time per iteration: {avg_time:.4f} seconds.")


if __name__ == "__main__":
    main()
