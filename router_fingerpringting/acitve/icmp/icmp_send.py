#coding:utf-8
"""

"""
import commands
import threading
import time
import os
from scapy.all import *
import sys

route = sys.argv[1]
ip_dic = {"xiaomi":["192.168.31.1","192.168.31.239"],
			"xunjie":["192.168.1.1","192.168.1.110"],
			"XJTUWlan":["10.164.175.254","10.164.170.111"],
			"tplink_4f10":["192.168.0.1","192.168.0.108"],
			"tplink_fe1c1a":["192.168.1.253","192.168.1.100"]
}
file_num = 1
while True:
	filename_bool = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/"+route+"/icmp_data_"+str(file_num)+".txt"
	if not os.path.exists(filename_bool):
		break
	else:
		file_num += 1

ipdst = ip_dic[route][0]



def chatch():
	commands.getstatusoutput("tcpdump icmp -l|tee /home/wendell/Desktop/icmp.txt")


def send():
	global ipdst 
	command = "ping "+ipdst+" -c 150"
	commands.getstatusoutput(command)

def read():

	print "over"
	global route,file_num
	tamp = open("/home/wendell/Desktop/icmp.txt").readlines()
	time_list = []
	fail_fe = []
	flag = 0
	count = 0
	for time in tamp:
		time_echo = re.findall(r'(?<=echo ).+(?=, id)',time)
		if len(time_echo) == 0:
			continue
		time_echo = time_echo[0] 
		if time_echo == "request" and flag == 0:
			time_list.append(time[0:15])
			flag = 1
		elif time_echo == "reply" and flag == 1:
			time_list.append(time[0:15])
			flag = 0
			fail_fe.append(count)
			count = 0
		elif time_echo == "request" and flag ==1:
			count += 1
		else:
			continue


	if len(time_list) % 2 == 1:
		time_list.pop()

	time_file = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/"+route+"/icmp_data_"+str(file_num)+".txt"
	count_file = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/"+route+"/icmp_data_count_"+str(file_num)+".txt"     
	data_out = open(time_file,"a")
	data_out_count = open(count_file,"a")

	data_out.write("\n")
	for i in time_list:
		data_out.write(i+" ")
	data_out.close()

	data_out_count.write("\n")
	for i in fail_fe:
		text = str(i)+" "
		data_out_count.write(text)
	data_out_count.close()


	commands.getstatusoutput("rm /home/wendell/Desktop/icmp.txt")


for i in range(10):
	if os.path.exists("/home/wendell/Desktop/icmp.txt") is True:
		commands.getstatusoutput("rm ~/Desktop/icmp.txt")
	else:
		print "needn't rm"
	print str(i) +"round to ping"
	t1 = threading.Thread(target = chatch)
	t2 = threading.Thread(target = send)
	t1.start()
	time.sleep(2)
	t2.start()
	t2.join()
	t1.join(10)

	read()
	t1._Thread__stop()

