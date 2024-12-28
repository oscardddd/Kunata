import matplotlib.pyplot as plt

def plot_bar_chart(data_dict):
    # Extract keys and values from the dictionary
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot a bar chart
    ax.bar(labels, values, color='skyblue')
    
    # Set labels and title
    ax.set_xlabel("Connection Speed")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Connection Speed")

    bars = ax.bar(labels, values, color='skyblue')

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        # Round to 2 decimal places in the f-string:
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height + 1,  # a small vertical offset
            f"{height:.2f}",
            ha='center', va='bottom'
        )
    # Optionally rotate x labels if needed
    # ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Adjust layout so labels/titles fit
    plt.tight_layout()
    
    # Show the plot
    plt.savefig("./latency.png")

if __name__ == "__main__":
    directory = {
        "10Gbps": 11.2495,
        "400 Mbps": 84.2869,
        "100 Mbps": 311.0725 
    }
    plot_bar_chart(directory)
