from scapy.all import *
HOST_MAC = "34:23:87:45:78:21"
HOST_IP = "192.168.31.239"
DST_IP = "192.168.31.1"
eth=Ether(src=HOST_MAC,type=0x0806)
arp=ARP(hwtype=0x0001,ptype=0x0800,op=0x0001,hwsrc=HOST_MAC,psrc=HOST_IP,pdst=DST_IP)
a = eth/arp
for i in range(50):
	recv = srp1(a)
