#!/bin/bash

# Define variables
DEV="enp6s0f0"
IFB_DEV="ifb0"
TOTAL_RATE="100mbit"

# Ensure the script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit
fi

# Load the ifb module
modprobe ifb numifbs=1

# Bring up the ifb interface
ip link set dev $IFB_DEV up 

# Clear existing qdiscs on $DEV and $IFB_DEV
tc qdisc del dev $DEV root 2>/dev/null
tc qdisc del dev $DEV ingress 2>/dev/null
tc qdisc del dev $IFB_DEV root 2>/dev/null

# Set up egress shaping on $DEV
tc qdisc add dev $DEV root handle 1: htb default 10
tc class add dev $DEV parent 1: classid 1:1 htb rate $TOTAL_RATE ceil $TOTAL_RATE
tc class add dev $DEV parent 1:1 classid 1:10 htb rate $TOTAL_RATE ceil $TOTAL_RATE
tc qdisc add dev $DEV parent 1:10 handle 10: fq_codel

# Set up ingress redirection to $IFB_DEV
tc qdisc add dev $DEV handle ffff: ingress
tc filter add dev $DEV parent ffff: protocol all u32 match u32 0 0 \
  action mirred egress redirect dev $IFB_DEV

# Set up ingress shaping on $IFB_DEV
tc qdisc add dev $IFB_DEV root handle 1: htb default 10
tc class add dev $IFB_DEV parent 1: classid 1:1 htb rate $TOTAL_RATE ceil $TOTAL_RATE
tc class add dev $IFB_DEV parent 1:1 classid 1:10 htb rate $TOTAL_RATE ceil $TOTAL_RATE
tc qdisc add dev $IFB_DEV parent 1:10 handle 10: fq_codel

# Display the qdisc configuration
echo "Egress configuration on $DEV:"
tc -s qdisc show dev $DEV

echo "Ingress configuration on $IFB_DEV:"
tc -s qdisc show dev $IFB_DEV
