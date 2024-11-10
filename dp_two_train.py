import torch
import torch.nn as nn
import torch.distributed as dist
import time
import argparse
import os
from torch.distributed.pipelining  import pipeline, SplitPoint, ScheduleGPipe, PipelineStage

torch.set_num_threads(20)

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
    def __init__(self, num_layers, d_model, d_ffn, num_heads, num_classes):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ffn, num_heads) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0, :]  # Use the first token's representation for classification
        x = self.output_layer(x)
        return x

def generate_split_spec(num_layers, world_size):
    split_spec = {}
    layers_per_stage = num_layers // world_size
    remainder = num_layers % world_size
    layer_idx = 0
    for stage in range(world_size - 1):
        layer_idx += layers_per_stage
        if stage < remainder:
            layer_idx += 1
        layer_name = f"layers.{layer_idx}"
        split_spec[layer_name] = SplitPoint.BEGINNING
    return split_spec

# Function to train the model
def train_model(model, input_tensor, target, num_iterations, num_layers, num_microbatch=1):
    rank = int(os.environ["RANK"])  # Get the rank from environment
    world_size = int(os.environ["WORLD_SIZE"])  # Get the world size from environment

    # Assign the device based on rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    microbatch_size = input_tensor.size(0) // num_microbatch  # Calculate microbatch size

    # This is the microbatch size
    example_input = torch.rand(microbatch_size, input_tensor.size(1), input_tensor.size(2)).to(device)

    # Move the model to the respective device
    model.to(device)

    # Generate split_spec
    split_spec = generate_split_spec(num_layers, world_size)

    chunks = num_microbatch
    print(f"Rank {rank}: The number of microbatches is: {chunks}")

    # Create the pipeline object
    pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec)
    print(f"Rank {rank}: Pipeline created")

    # Initialize distributed environment
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # Define the pipeline stage
    stage = PipelineStage(pipe, rank, num_stages=world_size, input_args=(example_input,), device=device)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Create the schedule with the loss function
    schedule = ScheduleGPipe(stage, chunks, loss_fn=loss_fn)
    print(f"Rank {rank}: Schedule created")

    # Get the local submodule
    local_submodule = getattr(pipe.split_gm, f"submod_{rank}")

    # Create an optimizer for the local parameters
    optimizer = torch.optim.SGD(local_submodule.parameters(), lr=0.01)

    # Training loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        if rank == 0:
            schedule.step(input_tensor)
        else:
            schedule.step()
        if rank == world_size - 1:
            losses = []
            output = schedule.step(target=target, losses=losses)
            print(f"Iteration {iteration}, Loss: {losses}")
    

        optimizer.step()

        if rank == world_size - 1:
            print(f"Iteration {iteration} completed.")

    print(f"Rank {rank}: Training completed.")

# Configuration parameters for different models
configs = {
    'base': {'d_model': 768, 'd_ffn': 3072, 'num_heads': 12, 'num_layers': 8},
    'xxlarge': {'d_model': 4096, 'd_ffn': 16384, 'num_heads': 32, 'num_layers': 4},
    'GPT-3': {'d_model': 12288, 'd_ffn': 49152, 'num_heads': 96, 'num_layers': 2},
    'Ours': {'d_model': 4096, 'd_ffn': 16384, 'num_heads': 32, 'num_layers': 2}  # Adjust as needed
}

# Main function to parse arguments and run the training
def main():
    parser = argparse.ArgumentParser(description="Train Transformer models.")
    parser.add_argument("--model", type=str, choices=['base', 'xxlarge', 'GPT-3', 'Ours'], required=True,
                        help="Model type to train.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for input tensor.")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for input tensor.")
    parser.add_argument("--num_microbatches", type=int, default=1, help="Number of microbatches for pipeline parallelism.")

    args = parser.parse_args()

    rank = int(os.environ["RANK"])  # Get the rank from environment
    world_size = int(os.environ["WORLD_SIZE"])  # Get the world size from environment

    # Get the selected configuration
    config = configs[args.model]

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10  # Number of classes for classification

    model = TransformerModel(
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        d_ffn=config['d_ffn'],
        num_heads=config['num_heads'],
        num_classes=num_classes
    ).to(device)

    # Create input tensor (batch_size, seq_len, d_model)
    input_tensor = torch.rand(args.batch_size, args.seq_len, config['d_model']).to(device)

    # Create target labels
    target = torch.randint(0, num_classes, (args.batch_size,), device=device)

    # Run training
    
    train_model(model, input_tensor, target, num_iterations=args.iterations, num_layers=config['num_layers'], num_microbatch=args.num_microbatches)

if __name__ == "__main__":
    main()
