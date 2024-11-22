import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the processed data CSV file
csv_file = 'processed_data.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Ensure the data is sorted by 'Time (s)'
df.sort_values('Time (s)', inplace=True)

# Convert 'Time (s)' and 'Data Size (MB)' to numeric types if necessary
df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
df['Data Size (MB)'] = pd.to_numeric(df['Data Size (MB)'], errors='coerce')

# Drop rows with missing values in essential columns
df.dropna(subset=['Time (s)', 'Data Size (MB)'], inplace=True)

# Calculate the duration between each time point for bar widths
df['Next_Time'] = df['Time (s)'].shift(-1)
df['Width'] = df['Next_Time'] - df['Time (s)']

# Handle the last bar width
default_width = df['Width'].median()
if np.isnan(default_width) or default_width <= 0:
    default_width = 1.0  # Or any other appropriate default
df['Width'].fillna(default_width, inplace=True)
df['Width'] = df['Width'].apply(lambda x: default_width if x <= 0 else x)

# Plotting the bar graph
plt.figure(figsize=(14, 7))
bars = plt.bar(df['Time (s)'], df['Data Size (MB)'], width=df['Width'], align='edge', edgecolor='black', color='skyblue')

# Setting labels and title
plt.xlabel('Time (s)')
plt.ylabel('Data Size (MB)')
plt.title('Data Size Over Time')

# Adjust x-axis limits to include the full range
x_min = df['Time (s)'].min()
x_max = df['Time (s)'].max() + df['Width'].iloc[-1]
plt.xlim(x_min, x_max)

# Adding grid
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add specific time labels on x-axis
# Format time labels to two decimal places
time_labels = df['Time (s)'].apply(lambda x: f'{x:.2f}')
plt.xticks(df['Time (s)'], labels=time_labels, rotation=45, ha='right')

# Add data size values on top of each bar
for bar, data_size in zip(bars, df['Data Size (MB)']):
    height = bar.get_height()
    if height > 0:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{data_size:.2f}',
            ha='center',
            va='bottom',
            fontsize=8
        )

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Show the plot
plt.savefig("data_time.png")
