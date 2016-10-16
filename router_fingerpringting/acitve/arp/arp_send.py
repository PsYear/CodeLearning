import commands
import threading
import time
import os
from scapy.all import *
def chatch():
	commands.getstatusoutput("tcpdump -p arp -l |tee ~/Desktop/arp.txt")


def send():

	#HOST_MAC = "34:23:87:45:78:21"
	# HOST_IP = "192.168.31.239"
	# DST_IP = "192.168.31.1"
	HOST_IP = "10.164.170.111"
	DST_IP = "10.164.175.254"
	HOST_MAC = "34:23:87:45:78:21"
	#HOST_IP = "192.168.1.29"
	#DST_IP = "192.168.1.1"
	eth=Ether(src=HOST_MAC,type=0x0806)
	arp=ARP(hwtype=0x0001,ptype=0x0800,op=0x0001,hwsrc=HOST_MAC,psrc=HOST_IP,pdst=DST_IP)
	a = eth/arp
	for i in range(1000):
		recv = srp(a,timeout=1)
	print "ending!!!!!!!!!!!!!!!!!!!!!!!!!"
	block = 1

def read():
	print "over"
	tamp = open("/home/wendell/Desktop/arp.txt").readlines()
	time_list = []
	flag = 0
	for time in tamp:
		if time[21:26] == "Reque" and flag == 0:
			time_list.append(time[0:15])
			flag = 1
		elif time[21:26] == "Reply" and flag == 1:
			time_list.append(time[0:15])
			flag = 0
		else:
			continue
	if len(time_list) % 2 == 1:
		time_list.pop()
	print len(time_list)
	print time_list
	commands.getstatusoutput("rm ~/Desktop/arp.txt")



if os.path.exists("/home/wendell/Desktop/arp.txt") is True:
	commands.getstatusoutput("rm ~/Desktop/arp.txt")
else:
	print "needn't rm"

t1 = threading.Thread(target = chatch)
t2 = threading.Thread(target = send)
t1.start()
time.sleep(2)
t2.start()
t2.join()
t1.join(5)

read()
t1._Thread__stop()

