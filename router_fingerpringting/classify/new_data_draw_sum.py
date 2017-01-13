import sys,re,os
import numpy as np
from scipy import stats 
import matplotlib.pylab as plt




win_path = "F:\github_workspace\CodeLearning\\router_fingerpringting\classify\\"
ubuntu_path = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/"



filepath_arp_set_before = [
"xiaomi/arp_data.txt",
"xiaomi/arp_data_1.txt",
"xiaomi/arp_data_2.txt",
"XJTUWlan/arp_data.txt",
"XJTUWlan/arp_data_1.txt",
"xunjie/arp_data.txt",
"xunjie/arp_data_1.txt",
"tplink_4f10/arp_data_1.txt",
"tplink_fe1c1a/arp_data_1.txt"
]
filepath_icmp_set_before = [
"xiaomi/icmp_data.txt",
"xiaomi/icmp_data_1.txt",
"xunjie/icmp_data.txt",
"xunjie/icmp_data_1.txt",
"XJTUWlan/icmp_data_1.txt",
"tplink_4f10/icmp_data_1.txt",
"tplink_4f10/icmp_data_2.txt",
"tplink_fe1c1a/icmp_data_1.txt"
]

def filepath_after(os, file):
	file_ppath = []
	for i in file:
		file_ppath.append(os+i)
	return file_ppath



filepath_arp_set = filepath_after(win_path, filepath_arp_set_before)
filepath_icmp_set = filepath_after(win_path, filepath_icmp_set_before)
label_str = []


def abs(num):
	if num >= 0:
		return num
	else:
		return -num


def changetime(str_time):
	dec = str_time.split(":")
	time = int(dec[0])*3600+int(dec[1])*60
	sec = dec[2].split(".")
	time = time + int(sec[0])+int(sec[1])*1e-6
	return time


def draw_sort(grade):
	global label_str
	for i in range(len(grade)):
		plt.plot(grade[i],label=label_str[i])
		# plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()

def readfile_sort(filepath):
	data_list = open(filepath,'r').readlines()
	data_time_IAT_list = []
	data_time_delta_list = []
	data_time_clock_list = []
	data_time_test_list = []
	for i in data_list:
		if len(i) != 1:
			data_time =[changetime(x) for x in i.split(" ")[:-1]]
			data_time_IAT = [data_time[num+2] - data_time[num] for num in range(1,len(data_time)-2,2)]
			data_time_delta = [data_time[num+1] - data_time[num] for num in range(0,len(data_time),2)]
			data_time_clock = [(data_time[num+3] - data_time[num+1])/(data_time[num+2] - data_time[num]) for num in range(0,len(data_time)-2,2)]

			data_time_test = [(data_time[num+3]-data_time[num+1])-(data_time[num+2]-data_time[num+0]) for num in range(0, len(data_time)-2,2)]
			data_time_delta_list += data_time_delta
			data_time_IAT_list += data_time_IAT
			data_time_clock_list += data_time_clock
			data_time_test_list += data_time_test
	data_time_delta_list.sort()
	data_time_test_list.sort()
	data_time_IAT_list.sort()
	data_time_clock.sort()
	return [data_time_delta_list,data_time_IAT_list, data_time_clock,data_time_test_list]

def readfile_sort_set(filepath):
	data_time_IAT_list = []
	data_time_delta_list = []
	data_time_clock_list = []
	data_time_test_list = []
	flag = 0
	global label_str
	label_str.append(filepath[0][75:86])
	for i in range(len(filepath)):
		if i != 0:
			name =  filepath[i][75:86]
			name_pre = filepath[i-1][75:86]
			if name == name_pre:
				flag = 1
			else:
				label_str.append(name)
				flag = 0
		i = filepath[i]
		if flag == 1:
			
			data_time_delta_list[-1]+=readfile_sort(i)[0]
			data_time_IAT_list[-1]+=readfile_sort(i)[1]
			data_time_clock_list[-1]+=readfile_sort(i)[2]
			data_time_test_list[-1]+=readfile_sort(i)[3]
		else:
			data_time_delta_list.append(readfile_sort(i)[0])
			data_time_IAT_list.append(readfile_sort(i)[1])
			data_time_clock_list.append(readfile_sort(i)[2])
			data_time_test_list.append(readfile_sort(i)[3])
	return [data_time_delta_list,data_time_IAT_list,data_time_clock_list,data_time_test_list]

def readfile(filepath):
	data_list = open(filepath,'r').readlines()
	data_time_IAT_list = []
	data_time_delta_list = []
	data_time_clock_list = []
	data_time_test_list = []
	for i in data_list:
		if len(i) != 1:
			data_time =[changetime(x) for x in i.split(" ")[:-1]]
			data_time_IAT = [data_time[num+2] - data_time[num] for num in range(1,len(data_time)-2,2)]
			data_time_delta = [data_time[num+1] - data_time[num] for num in range(0,len(data_time),2)]
			data_time_clock = [(data_time[num+3] - data_time[num+1])/(data_time[num+2] - data_time[num]) for num in range(0,len(data_time)-2,2)]

			data_time_test = [(data_time[num+3]-data_time[num+1])-(data_time[num+2]-data_time[num+0]) for num in range(0, len(data_time)-2,2)]
			data_time_delta_list += data_time_delta
			data_time_IAT_list += data_time_IAT
			data_time_clock_list += data_time_clock
			data_time_test_list += data_time_test
	return [data_time_delta_list,data_time_IAT_list, data_time_clock,data_time_test_list]



def readfile_set(filepath):
	data_time_IAT_list = []
	data_time_delta_list = []
	data_time_clock_list = []
	data_time_test_list = []
	flag = 0
	global label_str
	label_str.append(filepath[0][75:86])
	for i in range(len(filepath)):
		if i != 0:
			name =  filepath[i][75:86]
			name_pre = filepath[i-1][75:86]
			if name == name_pre:
				flag = 1
			else:
				label_str.append(name)
				flag = 0
		i = filepath[i]
		if flag == 1:
			data_time_delta_list[-1]+readfile(i)[0]
			data_time_IAT_list[-1]+readfile(i)[1]
			data_time_clock_list[-1]+readfile(i)[2]
			data_time_test_list[-1]+readfile(i)[3]
		else:
			data_time_delta_list.append(readfile(i)[0])
			data_time_IAT_list.append(readfile(i)[1])
			data_time_clock_list.append(readfile(i)[2])
			data_time_test_list.append(readfile(i)[3])
	return [data_time_delta_list,data_time_IAT_list,data_time_clock_list,data_time_test_list]


def draw_kde_arp_single(grade):
	global label_str
	gkde=[]
	ind = np.arange(0.,0.5,0.001)
	for i in range(len(grade)):
		gkde.append(stats.kde.gaussian_kde(grade[i]))
		plt.plot(ind, gkde[i](ind),label=label_str[i])
		# plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()

def draw_kde_icmp_single(grade):
	global label_str
	gkde=[]
	ind = np.arange(0.,3,0.001)
	for i in range(len(grade)):
		gkde.append(stats.kde.gaussian_kde(grade[i]))
		plt.plot(ind, gkde[i](ind),label=label_str[i])
		# plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()




def draw_single(grade):
	for i in range(len(grade)):
		plt.plot(grade[i],label=label_str[i])
		# plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()

def draw_div(grade):
	co = ['b','k','g','c','m','y','r','w']
	for j in range(len(grade[0])):
		print j
		for i in range(min(len(grade[0][j]),len(grade[1][j]))):
			plt.plot(grade[0][j][i],grade[1][j][i],'+',color = co[j])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()





#x = np.array(readfile_set(filepath_icmp_set))
draw_kde_icmp_single(readfile_set(filepath_icmp_set)[3])
# draw_sort(readfile_sort_set(filepath_icmp_set)[0])

#print readfile_set(filepath_icmp_set)[2]
#draw_kde_icmp_single(readfile_set(filepath_arp_set)[0])
#draw_kde_icmp_single(readfile_set(filepath_icmp_set)[2])
#draw_single(readfile_set(filepath_arp_set)[1])


#draw_single(readfile_set(filepath_icmp_set)[1])
