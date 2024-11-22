tshark -r ./captured/capture_l2.pcap -T fields \
  -e frame.time_epoch \
  -e frame.len \
  -E header=y \
  -E separator=, \
  -E quote=n \
  -E occurrence=f > ./packet_info_l2.csv
