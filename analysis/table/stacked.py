import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_stacked_bars(csv_file):
    # Lists to store our data
    layers = []
    fwd_data = []
    bwd_data = []
    
    # Read the CSV file
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        # Skip the header row
        headers = next(reader)
        
        # Parse each row
        for row in reader:
            layer_name = row[0]
            fwd_mb = float(row[2])   # "Fwd Activ (MB, FP32)"
            bwd_mb = float(row[3])   # "Bwd Grad (MB, FP32)"
            
            layers.append(layer_name)
            fwd_data.append(fwd_mb)
            bwd_data.append(bwd_mb)
    
    # Convert lists to numpy arrays (helpful for stacking)
    fwd_data = np.array(fwd_data)
    bwd_data = np.array(bwd_data)
    
    # X-axis positions
    x = np.arange(len(layers))
    
    # Plot settings
    bar_width = 0.6
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw stacked bars:
    #  - first bar is the forward data
    #  - second bar is the backward data on top (bottom=fwd_data)
    ax.bar(x, fwd_data, bar_width, label='Forward Activ', color='skyblue')
    ax.bar(x, bwd_data, bar_width, bottom=fwd_data, label='Backward Grad', color='salmon')
    
    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_xlabel('ResNet-50 Layer')
    ax.set_ylabel('Data Size (MB, FP32)')
    ax.set_title('Forward + Backward Activation Sizes by Layer (Stacked)')
    
    # Add legend
    ax.legend()
    
    # Add a bit of layout adjustment
    plt.tight_layout()
    
    # Show the plot (or you can save to file with plt.savefig('out.png'))
    plt.savefig("./stacked.png")

# Run the function
if __name__ == "__main__":
    csv_file_path = "resnet50.csv"  # <-- Replace with your actual file path
    plot_stacked_bars(csv_file_path)
