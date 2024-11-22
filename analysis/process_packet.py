import pandas as pd
import numpy as np

def process_packet_data(csv_file, output_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Handle 'frame.time_epoch' or parse 'frame.time' if necessary
    if 'frame.time_epoch' in df.columns:
        df['frame.time_epoch'] = pd.to_numeric(df['frame.time_epoch'], errors='coerce')
    else:
        # If 'frame.time_epoch' is not in the CSV, parse 'frame.time'
        df['frame.time'] = df['frame.time'].astype(str).str.strip('"')
        # Adjust the date format to match your 'frame.time' format
        date_format = '%b %d, %Y %H:%M:%S.%f %Z'  # Example: "Nov 22, 2024 12:00:00.000000000 CST"
        df['frame.time_epoch'] = pd.to_datetime(df['frame.time'], format=date_format, errors='coerce').astype('int64') / 1e9  # Convert to seconds
    
    # Drop rows with invalid times
    df.dropna(subset=['frame.time_epoch'], inplace=True)

    # Ensure 'frame.len' is numeric
    df['frame.len'] = pd.to_numeric(df['frame.len'], errors='coerce')
    df.dropna(subset=['frame.len'], inplace=True)

    # Sort the DataFrame by 'frame.time_epoch'
    df.sort_values('frame.time_epoch', inplace=True)

    # Initialize variables
    start_time = df['frame.time_epoch'].iloc[0]
    df['relative_time'] = df['frame.time_epoch'] - start_time  # Time in seconds starting from 0

    # Round relative_time to handle floating-point precision issues
    df['relative_time'] = df['relative_time'].round(6)

    # Initialize lists for time slots and data sizes
    time_slots = []
    data_sizes = []

    # Variables to keep track of the current time slot
    size_accumulator = 0.0

    # Iterate over the DataFrame
    for idx in range(len(df)):
        current_time = df['relative_time'].iloc[idx]
        packet_size = df['frame.len'].iloc[idx]

        # Accumulate packet size
        size_accumulator += packet_size

        # If this is not the last packet, check for time gap
        if idx < len(df) - 1:
            next_time = df['relative_time'].iloc[idx + 1]
            time_diff = next_time - current_time

            if time_diff >= 1.0:
                # Append accumulated size to data_sizes (convert bytes to MB)
                data_size_mb = size_accumulator / (1024 * 1024)
                time_slots.append(current_time)
                data_sizes.append(data_size_mb)

                # Reset accumulator
                size_accumulator = 0.0

                # For the time between current_time and next_time, mark data size as 0
                gap_times = np.arange(current_time + 1.0, next_time, 1.0)
                for gap_time in gap_times:
                    time_slots.append(gap_time)
                    data_sizes.append(0.0)
        else:
            # Last packet
            # Append accumulated size to data_sizes (convert bytes to MB)
            data_size_mb = size_accumulator / (1024 * 1024)
            time_slots.append(current_time)
            data_sizes.append(data_size_mb)

    # Create the output DataFrame
    output_df = pd.DataFrame({
        'Time (s)': time_slots,
        'Data Size (MB)': data_sizes
    })

    # Sort the output DataFrame by 'Time (s)'
    output_df.sort_values('Time (s)', inplace=True)

    # Reset index
    output_df.reset_index(drop=True, inplace=True)

    # Save to CSV
    output_df.to_csv(output_file, index=False)

    return output_df

# Usage example
csv_file = 'packet_info2.csv'         # Replace with your CSV file path
output_file = 'processed_data.csv'   # Output CSV file
output_df = process_packet_data(csv_file, output_file)

# Print the results
print(output_df)
