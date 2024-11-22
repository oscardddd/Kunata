import pandas as pd
import matplotlib.pyplot as plt

# Your data as a multiline string (replace this with reading from a file if needed)


# Read the data into a pandas DataFrame
from io import StringIO
df = pd.read_csv("processed_data_l2.csv")

# Sort the DataFrame by 'Time (s)' to ensure chronological order
df = df.sort_values('Time (s)').reset_index(drop=True)

# Initialize a list to hold the intervals
intervals = []

# Iterate over the DataFrame to identify intervals ending at times when data is sent
for i in range(1, len(df)):
    current_time = df.loc[i, 'Time (s)']
    previous_time = df.loc[i - 1, 'Time (s)']
    data_size = df.loc[i, 'Data Size (MB)']
    
    if data_size > 0:
        # Data was sent in the interval from previous_time to current_time
        intervals.append({
            'start_time': previous_time,
            'end_time': current_time,
            'data_size': data_size
        })

# Create the plot
plt.figure(figsize=(14, 7))

for interval in intervals:
    start = interval['start_time']
    end = interval['end_time']
    data_size = interval['data_size']
    duration = end - start

    # Draw the bar for the interval
    plt.bar(x=start + duration / 2, height=data_size, width=duration, align='center',
            color='skyblue', edgecolor='blue')

    # Add labels
    plt.text(start + duration / 2, data_size + 0.2,
             f"{data_size:.2f} MB\n({start:.2f}s - {end:.2f}s)",
             ha='center', va='bottom', fontsize=8)

# Set labels and title
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Data Size (MB)', fontsize=12)
plt.title('Data Transfer Over Time', fontsize=14)
plt.grid(axis='y', linestyle='--', linewidth=0.5)

# Adjust x-axis limits
plt.xlim(df['Time (s)'].min() - 1, df['Time (s)'].max() + 1)

# Show the plot
plt.tight_layout()
plt.savefig("data_size_l2.png")
