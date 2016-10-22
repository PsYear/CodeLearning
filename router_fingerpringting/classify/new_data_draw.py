#coding:utf-8

import sys,re,os
import numpy as np
from scipy import stats 
import matplotlib.pylab as plt



filepath = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/icmp_data.txt"

filepath_arp_set = [
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/arp_data.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/arp_data_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/arp_data_2.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/XJTUWlan/arp_data.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/XJTUWlan/arp_data_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/arp_data.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/arp_data_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/tplink_4f10/arp_data_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/tplink_fe1c1a/arp_data_1.txt"

]
filepath_icmp_set = [
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/icmp_data.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/icmp_data_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/icmp_data.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/icmp_data_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/XJTUWlan/icmp_data_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/tplink_4f10/icmp_data_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/tplink_4f10/icmp_data_2.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/tplink_fe1c1a/icmp_data_1.txt"

]
filepath_arp_count_set = [
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/arp_data_count_2.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/arp_data_count_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/XJTUWlan/arp_data_count_1.txt"
]
filepath_icmp_count_set = [
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/icmp_data_count_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/icmp_data_count_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/XJTUWlan/icmp_data_count_1.txt"
]


def changetime(str_time):
	dec = str_time.split(":")
	time = int(dec[0])*3600+int(dec[1])*60
	sec = dec[2].split(".")
	time = time + int(sec[0])+int(sec[1])*1e-6
	return time


def readfile_count(filepath):
	data_count_list_set = []
	data_count_list = []
	for j in filepath:
		data_list = open(j,'r').readlines()
		for i in data_list:
			if len(i) != 1:
				data_count = [int(x) for x in i.split(" ")[:-1]]
				data_count_copy = []
				for k in range(len(data_count)):
					if data_count[k] != 0:
						data_count_copy.append(data_count[k])
				if len(data_count_copy) != 0:
					data_count_list += data_count_copy
		if len(data_count_list) != 0:
			data_count_list_set.append(data_count_list)
	return data_count_list_set

def readfile(filepath):
	data_list = open(filepath,'r').readlines()
	data_time_IAT_list = []
	data_time_delta_list = []
	for i in data_list:
		if len(i) != 1:
			data_time =[changetime(x) for x in i.split(" ")[:-1]]
			data_time_IAT = [data_time[num+2] - data_time[num] for num in range(1,len(data_time)-2,2)]
			data_time_delta = [data_time[num+1] - data_time[num] for num in range(0,len(data_time),2)]
			data_time_IAT_list.append(data_time_IAT)
			data_time_delta_list.append(data_time_delta)
	return [data_time_delta_list,data_time_IAT_list]


def readfile_set(filepath):
	data_time_IAT_list = []
	data_time_delta_list = []
	for i in filepath:
		data_time_delta_list.append(readfile(i)[0])
		data_time_IAT_list.append(readfile(i)[1])
	return [data_time_delta_list,data_time_IAT_list]


def draw_kde_single(grade):
	gkde = []
	ind = np.arange(0.,3.,0.01)
	for i in range(len(grade)):
		gkde.append(stats.kde.gaussian_kde(grade[i]))
		plt.plot(ind, gkde[i](ind))
		# plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()
	return gkde

def time_seq(grade):
	co = ['b','k','g','c','m','y','r','w']
	for j in range(len(grade)):
		for i in range(len(grade[j])):
			plt.plot(grade[j][i],color = co[j])
	plt.show()

def draw_kde_mul_icmp(grade):
	gkde = []
	ind = np.arange(0.,3.,0.01)
	co = ['b','k','g','c','m','y','r','w']
	for j in range(len(grade)):
		name =  filepath_icmp_set[j][75:86]
		print name
		for i in range(len(grade[j])):
			gkde.append(stats.kde.gaussian_kde(grade[j][i]))
			if i == 0:
				plt.plot(ind, gkde[i+j*(len(grade[j]))](ind),color = co[j],label = name)
			else:
				plt.plot(ind, gkde[i+j*(len(grade[j]))](ind),color = co[j] )
		# plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()	
	return gkde



def draw_kde_mul_arp(grade):
	gkde = []
	ind = np.arange(0.,.4,0.001)
	co = ['b','k','g','c','m','y','r','w']
	for j in range(len(grade)):
		name =  filepath_arp_set[j][75:86]
		print name
		for i in range(len(grade[j])):
			gkde.append(stats.kde.gaussian_kde(grade[j][i]))
			if i == 0:
				plt.plot(ind, gkde[i+j*(len(grade[j]))](ind),color = co[j%8],label = name )
			else:
				plt.plot(ind, gkde[i+j*(len(grade[j]))](ind),color = co[j%8] )
		# plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()
	return gkde

def draw_kde_mul_count(grade):
	gkde = []
	ind = np.arange(0.,100,0.1)
	co = ['b','k','g','c','m','y','r','w']
	for j in range(len(grade)):
		name =  filepath_arp_set[j][75:86]
		print name
		for i in range(len(grade[j])):
			gkde.append(stats.kde.gaussian_kde(grade[j][i]))
			if i == 0:
				plt.plot(ind, gkde[i+j*(len(grade[j]))](ind),color = co[j],label = name )
			else:
				plt.plot(ind, gkde[i+j*(len(grade[j]))](ind),color = co[j] )
		# plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()
	return gkde

#draw_kde_single(readfile(filepath)[1])
#draw_kde_mul_icmp(readfile_set(filepath_icmp_set)[0])
draw_kde_mul_arp(readfile_set(filepath_arp_set)[1])
#time_seq(readfile_set(filepath_icmp_set)[1])
#draw_kde_mul_count(readfile_count(filepath_arp_count_set))
#print readfile_count(filepath_arp_count_set)
