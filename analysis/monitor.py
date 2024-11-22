import time
import pandas as pd

def get_transfer_rate(interface="vlan1216", interval=0.5, duration=800):
    rx_path = f"/sys/class/net/{interface}/statistics/rx_bytes"
    tx_path = f"/sys/class/net/{interface}/statistics/tx_bytes"
    
    # Initialize data storage
    timestamps, rx_rates, tx_rates = [], [], []
    
    # Read initial byte counts
    with open(rx_path, 'r') as f:
        rx_bytes_prev = int(f.read())
    with open(tx_path, 'r') as f:
        tx_bytes_prev = int(f.read())
    
    start_time = time.time()
    
    try:
        while True:
            # Wait for the specified interval
            time.sleep(interval)
            current_time = time.time()
            
            # Read current byte counts
            with open(rx_path, 'r') as f:
                rx_bytes_now = int(f.read())
            with open(tx_path, 'r') as f:
                tx_bytes_now = int(f.read())
            
            # Calculate data transfer rate (megabits per second)
            rx_diff = rx_bytes_now - rx_bytes_prev
            tx_diff = tx_bytes_now - tx_bytes_prev
            
            # Handle counter reset (if applicable)
            if rx_diff < 0:
                rx_diff += 2**64  # Adjust if the counter resets (assuming 64-bit counter)
            if tx_diff < 0:
                tx_diff += 2**64
            
            rx_rate = (rx_diff * 8) / 1_000_000 / interval  # Convert bytes to megabits
            tx_rate = (tx_diff * 8) / 1_000_000 / interval  # Convert bytes to megabits
            
            # Store results
            timestamps.append(current_time - start_time)  # Time elapsed since start
            rx_rates.append(rx_rate)
            tx_rates.append(tx_rate)
            
            # Update previous byte counts
            rx_bytes_prev, tx_bytes_prev = rx_bytes_now, tx_bytes_now

            # Print the transfer rate (optional)
            print(f"Time: {current_time - start_time:.2f}s, RX Rate: {rx_rate:.6f} Mbps, TX Rate: {tx_rate:.6f} Mbps")
            
            # Check if the duration has been reached
            if current_time - start_time >= duration:
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Save results to a CSV file upon stopping
        data = {"Timestamp_s": timestamps, "RX_Rate_Mbps": rx_rates, "TX_Rate_Mbps": tx_rates}
        df = pd.DataFrame(data)
        df.to_csv("data_transfer_rate.csv", index=False)
        print("Data transfer rate recorded and saved to data_transfer_rate.csv")

# Run the function with your desired parameters
if __name__ == "__main__":
    get_transfer_rate(interface="vlan1216", interval=0.1, duration=30)
