#!/bin/bash

# Define the primary network interface and associated VLAN interfaces
PRIMARY_INTERFACE="enp6s0f0"
VLAN_INTERFACES=("vlan1140" "vlan1148" "vlan1153" "vlan1156")

echo "Clearing all tc rules on $PRIMARY_INTERFACE and associated VLAN interfaces..."

# Clear root and ingress qdiscs on the primary interface
sudo tc qdisc del dev $PRIMARY_INTERFACE root 2>/dev/null
sudo tc qdisc del dev $PRIMARY_INTERFACE ingress 2>/dev/null

# Loop through each VLAN interface and clear root and ingress qdiscs
for VLAN_IFACE in "${VLAN_INTERFACES[@]}"; do
  echo "Clearing tc rules on $VLAN_IFACE..."
  sudo tc qdisc del dev $VLAN_IFACE root 2>/dev/null
  sudo tc qdisc del dev $VLAN_IFACE ingress 2>/dev/null
done

echo "All tc rules cleared on $PRIMARY_INTERFACE and VLAN interfaces."
