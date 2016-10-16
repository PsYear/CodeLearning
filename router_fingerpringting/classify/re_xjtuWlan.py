import re

def refile(filename):
	file_open = open(filename,'rb')
	file_text = file_open.readlines()
	timelist = []
	for line in file_text:
		time_text = re.findall(r'.+(?=   E)',line)
		if(len(time_text)>0):
			time_text = time_text[0].split(",")
			time_int = [int(x) for x in time_text[0].split(":")]
			time_int.append(int(time_text[1]))
			time_int.append(int(time_text[2]))
			time = time_int[0]*3600+time_int[1]*60+time_int[2]+time_int[3]*1e-3+time_int[4]*1e-6
			timelist.append(time)
	return timelist

def IAT(timelist):
	IAT = []
	timelist.sort()
	for i in range(len(timelist)-1):
		IAT.append(timelist[i+1] - timelist[i])
	return IAT


import sys,re,os
import numpy as np
from scipy import stats 
import matplotlib.pylab as plt
 
def draw_kde(grade,label_str):
	gkde = []
	ind = np.arange(0.,3.,0.01)
	for i in range(len(grade)):
		gkde.append(stats.kde.gaussian_kde(grade[i]))
		plt.plot(ind, gkde[i](ind), label=label_str[i])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()
	return gkde

from math import log
from numpy import array

def Bayes(test, grade):
	p = 1.0
	for i in range(len(test)):
		p = p*grade(test[i])
		print p
	return p




class JSD:
    def KLD(self,p,q):
        if 0 in q :
            raise ValueError
        return sum(_p * log(_p/_q) for (_p,_q) in zip(p,q) if _p!=0)

    def JSD_core(self,p,q):
        M = [0.5*(_p+_q) for _p,_q in zip(p,q)]
        return 0.5*self.KLD(p,M)+0.5*self.KLD(q,M)

# p = [2,3,4,1,1]
# q = [3,3,2,1,0]

# jsd = JSD()
# print jsd.JSD_core(p,q)
# print jsd.JSD_core(q,p)
 
def add_dns_100_dir(num,data_dir):
	dns_100_dir = "C:\Users\peter\Documents\github_new\CodeLearning\\router_fingerpringting\classify\\xiaomi\dns_100_"
	refile_dir = []
	for i in range(num+1)[1:]:
		dic_name = "dns_100_"+str(i)+"_dir"
		data_dir[dic_name] = dns_100_dir+str(i)+".txt"
		refile_dir.append(data_dir[dic_name])
	return refile_dir

def add_icmp_100_dir(num,data_dir):

	path = "C:\Users\wendell\Desktop\GitHub\CodeLearning\\router_fingerpringting\classify\XJTUWlan\ping_100_"
	

	refile_dir = []
	for i in range(num+1)[1:]:
		dic_name = "icmp_100_"+str(i)+"_dir"
		data_dir[dic_name] = path + str(i) + ".txt"
		refile_dir.append(data_dir[dic_name])
	return refile_dir

def add_icmp_200_dir(num,data_dir):

	path = "C:\Users\wendell\Desktop\GitHub\CodeLearning\\router_fingerpringting\classify\XJTUWlan\ping_200_"
	refile_dir = []
	for i in range(num+1)[1:]:
		dic_name = "icmp_200_"+str(i)+"_dir"
		data_dir[dic_name] = path + str(i) + ".txt"
		refile_dir.append(data_dir[dic_name])
	return refile_dir



def add_arp_1000_dir(num,data_dir):

	path_ubuntu = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/XJTUWlan/arp_1000_"
	refile_dir = []
	for i in range(num+1)[1:]:
		dic_name = "arp_1000_"+str(i)+"_dir"
		data_dir[dic_name] = path_ubuntu + str(i) + ".txt"
		refile_dir.append(data_dir[dic_name])
	return refile_dir

def add_arp_100_dir(num,data_dir):

	path_ubuntu = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/XJTUWlan/arp_50_"
	refile_dir = []
	for i in range(num+1)[1:]:
		dic_name = "arp_100_"+str(i)+"_dir"
		data_dir[dic_name] = path_ubuntu + str(i) + ".txt"
		refile_dir.append(data_dir[dic_name])
	return refile_dir

import sys
if __name__ == '__main__':
	# data_dir = {
	# "dns_dir":"C:\Users\peter\Desktop\classify\\xiaomi\\xiaomi_dns.txt",
	# "ssdp_xiaomi_dir":"C:\Users\peter\Desktop\classify\\xiaomi\\ssdp.txt",
	# "ssdp_xunjie_dir" : "C:\Users\peter\Desktop\classify\\xunjie\\ssdp.txt",
	# "arp_xunjie_dir" : "C:\Users\peter\\Desktop\\classify\\xunjie\\arp.txt",
	# "arp_xiaomi_dir" : "C:\Users\peter\Desktop\classify\\xiaomi\\arp.txt"
	# }
	# num = int(sys.argv[1])
	# refile_dir = add_dns_100_dir(num,data_dir)
	# feature = []
	# for i in range(len(refile_dir)):
	# 	feature.append(IAT(refile(refile_dir[i])))
	# label_str = []
	# for j in range(len(refile_dir)):
	# 	for i in data_dir:
	# 		if refile_dir[j] == data_dir[i]:
	# 			label_str.append(i)
	# draw_kde(feature,label_str)

	
	data_dir = {}
	num = int(sys.argv[1])
	refile_dir = add_arp_1000_dir(num,data_dir) + add_arp_100_dir(5,data_dir)
	feature = []
	for i in range(len(refile_dir)):
		feature.append(IAT(refile(refile_dir[i])))
	label_str = []
	for j in range(len(refile_dir)):
		for i in data_dir:
			if refile_dir[j] == data_dir[i]:
				label_str.append(i)
	draw_kde(feature,label_str)
