#!/bin/bash

# IP address to filter
FILTER_IP="10.10.2.1"  # Replace with your specific IP address

# Output file
OUTPUT_FILE="packet_info.csv"

# Print header to output file
echo "Timestamp,Source IP,Source Port,Destination IP,Destination Port,Protocol" > "$OUTPUT_FILE"

# Use tcpdump to capture packets from the specific IP and record specific info
tcpdump -nn -tttt src "$FILTER_IP" | while read -r line
do
    # Extract timestamp
    TIMESTAMP=$(echo "$line" | awk '{print $1 " " $2}')
    
    # Extract protocol
    PROTOCOL=$(echo "$line" | awk '{print $3}')
    
    # Extract source and destination IP and ports
    SRC_DST=$(echo "$line" | awk '{for (i=4; i<=NF; i++) printf $i " "; print ""}')
    
    # Parse source and destination IP and ports
    SRC_IP_PORT=$(echo "$SRC_DST" | awk -F" > " '{print $1}')
    DST_IP_PORT=$(echo "$SRC_DST" | awk -F" > " '{print $2}' | awk '{print $1}')
    
    SRC_IP=$(echo "$SRC_IP_PORT" | awk -F"." 'BEGIN{OFS="."}{print $1,$2,$3,$4}')
    SRC_PORT=$(echo "$SRC_IP_PORT" | awk -F"." '{print $5}')
    
    DST_IP=$(echo "$DST_IP_PORT" | awk -F"." 'BEGIN{OFS="."}{print $1,$2,$3,$4}')
    DST_PORT=$(echo "$DST_IP_PORT" | awk -F"." '{print $5}')
    
    # Write to output file
    echo "$TIMESTAMP,$SRC_IP,$SRC_PORT,$DST_IP,$DST_PORT,$PROTOCOL" >> "$OUTPUT_FILE"
done
