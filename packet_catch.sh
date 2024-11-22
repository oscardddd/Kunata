tshark -r ./capture.pcap -T fields \
  -e frame.time_epoch \
  -e frame.len \
  -E header=y \
  -E separator=, \
  -E quote=n \
  -E occurrence=f > ./analyis/packet_info2.csv
