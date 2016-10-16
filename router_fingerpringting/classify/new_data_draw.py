#coding:utf-8

import sys,re,os
import numpy as np
from scipy import stats 
import matplotlib.pylab as plt



filepath = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/icmp_data.txt"

filepath_arp_set = [
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/arp_data.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/arp_data_1.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/XJTUWlan/arp_data.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/arp_data.txt"
]
filepath_icmp_set = [
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xiaomi/icmp_data.txt",
"/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/xunjie/icmp_data.txt"
]




def changetime(str_time):
	dec = str_time.split(":")
	time = int(dec[0])*3600+int(dec[1])*60
	sec = dec[2].split(".")
	time = time + int(sec[0])+int(sec[1])*1e-6
	return time



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

def draw_kde_mul(grade):
	gkde = []
	ind = np.arange(0.,3.,0.01)
	co = ['b','k','g','c','m','y','r','w']
	for j in range(len(grade)):
		print j
		for i in range(len(grade[j])):
			gkde.append(stats.kde.gaussian_kde(grade[j][i]))
			plt.plot(ind, gkde[i+j*(len(grade[j]))](ind),color = co[j] )
		# plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()
	return gkde

#draw_kde_single(readfile(filepath)[1])
draw_kde_mul(readfile_set(filepath_icmp_set)[1])
#draw_kde_mul(readfile_set(filepath_arp_set)[1])
