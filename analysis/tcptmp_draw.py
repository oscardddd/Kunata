import pandas as pd
import matplotlib.pyplot as plt

# Example iteration completion times extracted from your log
# iteration_times2 = [1735332848.1506894, 1735332849.6177952, 1735332851.0719066, 1735332852.5352502, 1735332853.920412, 1735332855.3313034, 1735332856.7420602, 1735332858.4252732, 1735332859.9091892, 1735332861.3122706]

# A second set of iteration completion times (dummy data for illustration)
iteration_times = [1735347365.2718134, 1735347366.4605253, 1735347367.7654736, 1735347369.02472, 1735347370.195167, 1735347371.3265624, 1735347372.4645348, 1735347373.6455452, 1735347374.864465, 1735347375.978047]
# Read the CSV file
df = pd.read_csv("processed_data_test.csv")

# Sort the DataFrame by 'Time (s)' and reset index
df = df.sort_values('Time (s)').reset_index(drop=True)

# Identify the intervals
intervals = []
for i in range(1, len(df)):
    current_time = df.loc[i, 'Time (s)']
    previous_time = df.loc[i - 1, 'Time (s)']
    data_size = df.loc[i, 'Data Size (MB)']

    if data_size > 0:
        intervals.append({
            'start_time': previous_time,
            'end_time': current_time,
            'data_size': data_size
        })

# Create the plot
plt.figure(figsize=(14, 7))

# Plot the bars for data transfers
for interval in intervals:
    start = interval['start_time']
    end = interval['end_time']
    data_size = interval['data_size']
    duration = end - start

    # Draw the bar for the interval
    plt.bar(
        x=start + duration / 2,
        height=data_size,
        width=duration,
        align='center',
        color='skyblue',
        edgecolor='blue'
    )

# Add vertical lines (and text labels) for the *first* iteration times
# for idx, t in enumerate(iteration_times):
#     plt.axvline(x=t, color='red', linestyle='--', linewidth=1)
#     plt.text(
#         t,
#         plt.ylim()[1] * 0.95,  # place near the top of the plot
#         f"rank 1 {idx}",
#         rotation=90,
#         color='red',
#         ha='right',
#         va='top',
#         fontsize=9
#     )

# Add vertical lines (and text labels) for the *second* iteration times in a different color
for idx, t in enumerate(iteration_times):
    plt.axvline(x=t, color='orange', linestyle='--', linewidth=1)
    plt.text(
        t,
        plt.ylim()[1] * 0.90,  # slightly lower than the red labels
        f"rank 0{idx}",
        rotation=90,
        color='orange',
        ha='right',
        va='top',
        fontsize=9
    )

# Set labels and title
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Data Size (MB)', fontsize=12)
plt.title('Data Transfer Over Time', fontsize=14)
plt.grid(axis='y', linestyle='--', linewidth=0.5)

# Optional: adjust x-axis limits if you want some padding
# plt.xlim(df['Time (s)'].min() - 1, df['Time (s)'].max() + 1)

plt.tight_layout()
plt.savefig("data_size_test.png")
plt.show()
