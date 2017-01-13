#coding:utf-8
"""
	change the ip in the send function
"""
import commands
import threading
import time
import os
import sys
from scapy.all import *



def chatch():
	commands.getstatusoutput("tcpdump -p arp -l |tee ~/Desktop/arp.txt")


def send():
	global route
	ip_dic = {"xiaomi":["192.168.31.1","192.168.31.239"],
				"xunjie":["192.168.1.1","192.168.1.110"],
				"XJTUWlan":["10.164.175.254","10.164.170.111"],
				"tplink_4f10":["192.168.0.1","192.168.0.108"],
				"tplink_fe1c1a":["192.168.1.253","192.168.1.100"]
	}
	#HOST_MAC = "34:23:87:45:78:21"
	# HOST_IP = "192.168.31.239"
	# DST_IP = "192.168.31.1"
	#HOST_IP = "192.168.31.239"
	#DST_IP = "192.168.31.1"
	HOST_MAC = "34:23:87:45:78:21"
	DST_IP = ip_dic[route][0]
	HOST_IP = ip_dic[route][1]
	eth=Ether(src=HOST_MAC,type=0x0806)
	arp=ARP(hwtype=0x0001,ptype=0x0800,op=0x0001,hwsrc=HOST_MAC,psrc=HOST_IP,pdst=DST_IP)
	a = eth/arp
	for i in range(1000):
		recv = srp(a,timeout=1)
		print "*********"+str(i)+"************"
	print "ending!!!!!!!!!!!!!!!!!!!!!!!!!"
	block = 1

def read():
	print "over"
	global route,file_num
	tamp = open("/home/wendell/Desktop/arp.txt").readlines()
	time_list = []
	fail_fe = []
	flag = 0
	count = 0
	for time in tamp:
		if time[21:26] == "Reque" and flag == 0:
			time_list.append(time[0:15])
			flag = 1
		elif time[21:26] == "Reply" and flag == 1:
			time_list.append(time[0:15])
			flag = 0
			fail_fe.append(count)
			count = 0
		elif time[21:26] == "Reque" and flag == 1:
			count += 1 
		else:
			continue
	if len(time_list) % 2 == 1:
		time_list.pop()

	time_file = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/"+route+"/arp_data_"+str(file_num)+".txt"
	count_file = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/"+route+"/arp_data_count_"+str(file_num)+".txt"     
	data_out = open(time_file,"a")
	data_out_count = open(count_file,"a")
	
	data_out.write("\n")
	for i in time_list:
		data_out.write(i+" ")
	data_out.close()

	data_out_count.write("\n")
	for i in fail_fe:
		data_out_count.write(str(i)+" ")
	data_out_count.close()
	
	commands.getstatusoutput("rm ~/Desktop/arp.txt")





route = sys.argv[1]
file_num = 1
while True:
	filename_bool = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/"+route+"/arp_data_"+str(file_num)+".txt"
	if not os.path.exists(filename_bool):
		break
	else:
		file_num += 1
for i in range(20):
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
	print "*********"+str(i)+"************"

