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
	ind = np.arange(0.,6.,0.1)
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
 
if __name__ == '__main__':
	dns_100_dir = "C:\Users\peter\Documents\github_new\CodeLearning\\router_fingerpringting\classify\\xiaomi\dns_100_"
	data_dir = {
	"dns_dir":"C:\Users\peter\Desktop\classify\\xiaomi\\xiaomi_dns.txt",
	"ssdp_xiaomi_dir":"C:\Users\peter\Desktop\classify\\xiaomi\\ssdp.txt",
	"ssdp_xunjie_dir" : "C:\Users\peter\Desktop\classify\\xunjie\\ssdp.txt",
	"arp_xunjie_dir" : "C:\Users\peter\\Desktop\\classify\\xunjie\\arp.txt",
	"arp_xiaomi_dir" : "C:\Users\peter\Desktop\classify\\xiaomi\\arp.txt",
	"dns_100_1_dir" : dns_100_dir+str(1)+".txt",
	"dns_100_2_dir" : dns_100_dir+str(2)+".txt",
	"dns_100_3_dir" : dns_100_dir+str(3)+".txt",
	"dns_100_4_dir" : dns_100_dir+str(4)+".txt",
	"dns_100_5_dir" : dns_100_dir+str(5)+".txt",
	"dns_100_6_dir" : dns_100_dir+str(6)+".txt"
	}
	refile_dir = [data_dir[u"dns_100_1_dir"],
				data_dir[u"dns_100_2_dir"],
				data_dir[u"dns_100_3_dir"],
				data_dir[u"dns_100_4_dir"],
				data_dir[u"dns_100_5_dir"],
				data_dir[u"dns_100_6_dir"]
				]	
	feature = []
	for i in range(len(refile_dir)):
		feature.append(IAT(refile(refile_dir[i])))
	label_str = []
	for j in range(len(refile_dir)):
		for i in data_dir:
			if refile_dir[j] == data_dir[i]:
				label_str.append(i)
	draw_kde(feature,label_str)