import pandas as pd
import matplotlib.pyplot as plt

# List of file names and labels (you can add more files as needed)
file_names = ['data_transfer_rate.csv']
labels = ['2 machine']

print("####")

def plot_rx_data_transfer_rate(file_names, labels):
    plt.figure(figsize=(12, 6))
    
    # Check that file names and labels match in length
    if len(file_names) != len(labels):
        raise ValueError("The number of files and labels must be the same.")
    
    # Iterate over each file and plot the RX rate
    for i, file_name in enumerate(file_names):
        # Read the CSV file
        df = pd.read_csv(f'./{file_name}')
        
        # Plot RX rate with the corresponding label, adding transparency with alpha
        plt.plot(df['Timestamp_s'], df['RX_Rate_Mbps'], label=labels[i], alpha=0.7)
    
    # Labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('RX Data Transfer Rate (Mbps)')
    plt.title('Receive (RX) Data Transfer Rate Comparison')
    
    # Legend and grid
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image file (optional)
    # plt.savefig('resnet.png')
    print("####")
    # Show the plot
    # plt.show()
    plt.savefig("interface_monitor.png")

# Call the function with the list of file names and labels
plot_rx_data_transfer_rate(file_names, labels)
