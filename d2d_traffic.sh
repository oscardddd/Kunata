#!/usr/bin/env bash
#
# star-tc.sh ─ hub-and-spoke shaping: 100 Mbit per leaf, 400 Mbit aggregate
#
# spokes (existing VLAN sub-ifaces) – edit to match your setup
VLAN_INTERFACES=(vlan1153 vlan1155 vlan1156 vlan1157)

PHY_DEV="enp6s0f0"
TOTAL_RATE=400mbit
SPOKE_RATE=$(printf "%.0f" "$(( ${TOTAL_RATE%mbit} / ${#VLAN_INTERFACES[@]} ))")mbit   # 100mbit

[[ $EUID -ne 0 ]] && { echo "Run as root"; exit 1; }

# ────────────────── IFB PREP ──────────────────
modprobe ifb numifbs=${#VLAN_INTERFACES[@]}

for idx in "${!VLAN_INTERFACES[@]}"; do
    IFB_DEV="ifb$idx"
    ip link show "$IFB_DEV" &>/dev/null || ip link add "$IFB_DEV" type ifb
    ip link set "$IFB_DEV" up
done

# ────────────────── ROOT CAP ON THE NIC ──────────────────
tc qdisc del dev "$PHY_DEV" root 2>/dev/null
tc qdisc add dev "$PHY_DEV" root handle 1: htb default 999
tc class add dev "$PHY_DEV" parent 1: classid 1:1 htb rate $TOTAL_RATE ceil $TOTAL_RATE
tc qdisc add  dev "$PHY_DEV" parent 1:1 handle 10: fq_codel          # stop bufferbloat

# ────────────────── PER-SPOKE SHAPING ──────────────────
idx=0
for SUB_DEV in "${VLAN_INTERFACES[@]}"; do
    IFB_DEV="ifb$idx"
    ip link set "$SUB_DEV" up                                    # make sure the iface is up

    # egress  (hub → leaf)
    tc qdisc del dev "$SUB_DEV" root 2>/dev/null
    tc qdisc add dev "$SUB_DEV" root handle 1: htb default 10
    tc class add dev "$SUB_DEV" parent 1: classid 1:1  htb rate $SPOKE_RATE ceil $SPOKE_RATE
    tc class add dev "$SUB_DEV" parent 1:1 classid 1:10 htb rate $SPOKE_RATE ceil $SPOKE_RATE
    tc qdisc add  dev "$SUB_DEV" parent 1:10 handle 10: fq_codel

    # ingress (leaf → hub) – mirror traffic into IFB and shape there
    tc qdisc add dev "$SUB_DEV" handle ffff: ingress 2>/dev/null
    tc filter add dev "$SUB_DEV" parent ffff: protocol all matchall \
                   action mirred egress redirect dev "$IFB_DEV"

    tc qdisc del dev "$IFB_DEV" root 2>/dev/null
    tc qdisc add dev "$IFB_DEV" root handle 1: htb default 10
    tc class add dev "$IFB_DEV" parent 1: classid 1:1  htb rate $SPOKE_RATE ceil $SPOKE_RATE
    tc class add dev "$IFB_DEV" parent 1:1 classid 1:10 htb rate $SPOKE_RATE ceil $SPOKE_RATE
    tc qdisc add  dev "$IFB_DEV" parent 1:10 handle 10: fq_codel

    ((idx++))
done

# ────────────────── SHOW WHAT WE BUILT ──────────────────
echo "======== root on $PHY_DEV (cap 400 Mbit) ========"
tc -s qdisc show dev "$PHY_DEV"
echo
for SUB_DEV in "${VLAN_INTERFACES[@]}"; do
    echo "---- egress on $SUB_DEV (cap $SPOKE_RATE) ----"
    tc -s qdisc show dev "$SUB_DEV"
done
echo
for idx in "${!VLAN_INTERFACES[@]}"; do
    echo "---- ingress on ifb$idx (cap $SPOKE_RATE) ----"
    tc -s qdisc show dev "ifb$idx"
done
