#!/usr/bin/env bash
#
# wipe-tc.sh ─ 一键清理所有带宽限速 / IFB 设备 / 过滤器
#
# ⚠️ 请根据实际情况修改 PHY_DEV 与 VLAN_INTERFACES；或直接留空用自动探测。
#

# ====== 手动配置（如果留空则自动探测） ===========================
PHY_DEV="enp6s0f0"                                  # 你的物理网卡
VLAN_INTERFACES=(vlan1153 vlan1155 vlan1156 vlan1157)   # 你的 VLAN 子接口
# =================================================================

[[ $EUID -ne 0 ]] && { echo "必须使用 root 运行"; exit 1; }

# ---------- 自动探测（如果变量为空） ------------------------------
[[ -z $PHY_DEV ]] && PHY_DEV=$(ip -o link show | awk -F': ' '!/lo/ {print $2; exit}')
if ((${#VLAN_INTERFACES[@]}==0)); then
    readarray -t VLAN_INTERFACES < <(ip -o link show | awk -v phy="$PHY_DEV" -F': ' \
        '$2 ~ ("^"phy"\\.") {print $2}')
fi

echo "Primary device : $PHY_DEV"
echo "VLAN devices   : ${VLAN_INTERFACES[*]:-(none detected)}"

# -----------------------------------------------------------------
echo "❌ 删除 $PHY_DEV 上的 root / ingress qdisc ..."
tc qdisc del dev "$PHY_DEV" root    2>/dev/null
tc qdisc del dev "$PHY_DEV" ingress 2>/dev/null

for v in "${VLAN_INTERFACES[@]}"; do
    echo "❌ 删除 $v 上的 root / ingress qdisc ..."
    tc qdisc del dev "$v" root    2>/dev/null
    tc qdisc del dev "$v" ingress 2>/dev/null
done

# -----------------------------------------------------------------
echo "❌ 删除所有 IFB 设备上的 root qdisc ..."
for d in /sys/class/net/ifb*; do
    [ -e "$d" ] || continue      # 如果通配符没匹配到文件，跳过
    dev=$(basename "$d")
    tc qdisc del dev "$dev" root 2>/dev/null
done 2>/dev/null                 # ⚠️ 重定向放在 done 后面


# -----------------------------------------------------------------
read -p "👉 是否删除 IFB 设备并卸载 ifb 模块？[y/N] " yn
if [[ $yn =~ ^[Yy]$ ]]; then
    echo "🗑️  删除 IFB 设备..."
    for d in /sys/class/net/ifb*; do
        [ -e "$d" ] || continue             # 没有匹配时跳过
        dev=$(basename "$d")
        ip link delete "$dev" type ifb 2>/dev/null
    done 2>/dev/null                        # 重定向写在这里

    echo "🔻 卸载 ifb 模块..."
    rmmod ifb 2>/dev/null || echo "  (ifb 仍在使用或已卸载)"
fi

# -----------------------------------------------------------------
echo -e "\n✅ 清理完成。当前 qdisc 状态："
tc qdisc show dev "$PHY_DEV"
for v in "${VLAN_INTERFACES[@]}"; do
    tc qdisc show dev "$v"
done
for d in /sys/class/net/ifb*; do
    tc qdisc show dev "$(basename "$d")"
done 2>/dev/null 
