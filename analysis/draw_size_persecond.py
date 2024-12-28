import pandas as pd
import matplotlib.pyplot as plt

def plot_data_size_by_second(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Compute the left edge of each bar
    left_edges = df['Time (s)'] - 1  # Each bar starts at t-1
    heights = df['Data Size (MB)']

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw bars so each spans [t-1, t], with a darker blue frame
    ax.bar(
        left_edges, 
        heights, 
        width=1, 
        align='edge', 
        color='skyblue',      # Bar fill color
        edgecolor='darkblue'  # Bar frame color (darker blue)
    )

    # Add a horizontal line at 50 MB
    ax.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='50 MB')

    # Labeling axes
    ax.set_xlabel('Time (s) - Each bar covers previous 1 second')
    ax.set_ylabel('Data Size (MB)')
    ax.set_title('Data Size by Second (Bar = [t-1, t])')

    # Add legend for the horizontal line
    ax.legend(loc='upper right')

    # If the earliest time is 0, the first bar starts at -1.
    # You can clip the x-axis at 0 if you want:
    ax.set_xlim(left=0)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig("./datasize_persecond")

# Usage example
csv_file = 'processed_data_by_second.csv'  # Replace with your CSV file path
plot_data_size_by_second(csv_file)
