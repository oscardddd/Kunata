import pandas as pd
import numpy as np

def process_packet_data_by_second(csv_file, output_file):
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
        df['frame.time_epoch'] = (
            pd.to_datetime(df['frame.time'], format=date_format, errors='coerce')
            .astype('int64') / 1e9  # Convert to seconds
        )
    
    # Drop rows with invalid times
    df.dropna(subset=['frame.time_epoch'], inplace=True)

    # Ensure 'frame.len' is numeric
    df['frame.len'] = pd.to_numeric(df['frame.len'], errors='coerce')
    df.dropna(subset=['frame.len'], inplace=True)

    # -------------------------------------------------------------------------
    # 1. Find the earliest (first) arrival time
    earliest_time = df['frame.time_epoch'].min()

    # 2. Cut out any packets that arrive before that time (if any)
    df = df[df['frame.time_epoch'] >= earliest_time]

    # 3. Shift the timeline so that earliest_time -> 0
    df['frame.time_epoch'] = df['frame.time_epoch'] - earliest_time
    # -------------------------------------------------------------------------

    # Create a column that represents the integer second (flooring is common)
    df['second'] = df['frame.time_epoch'].floordiv(1).astype(int)

    # Group by the second, and sum the total size (in bytes) for that second
    grouped = df.groupby('second')['frame.len'].sum().reset_index()

    # Convert bytes to MB
    grouped['Data Size (MB)'] = grouped['frame.len'] / (1024 * 1024)

    # Rename the column for clarity
    grouped.rename(columns={'second': 'Time (s)', 'frame.len': 'Total Bytes'}, inplace=True)

    # Sort by second, just in case
    grouped.sort_values('Time (s)', inplace=True)

    # Save to CSV
    grouped.to_csv(output_file, index=False)

    return grouped

# Usage example
csv_file = 'packet_info_test.csv'          # Replace with your CSV file path
output_file = 'processed_data_by_second.csv'  # Output CSV file
output_df = process_packet_data_by_second(csv_file, output_file)

# Print the results
print(output_df)
