#coding:utf-8
from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE
import sys,re,os
import numpy as np
from scipy import stats 
import matplotlib.pylab as plt
import random
from sklearn.datasets import make_blobs
from matplotlib import pyplot
import random
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import RFE, f_regression
from decimal import getcontext, Decimal 



win_path = "F:\github_workspace\CodeLearning\\router_fingerpringting\classify\\"
ubuntu_path = "/home/wendell/Documents/github/CodeLearning/router_fingerpringting/classify/"
win2_path = "C:\Users\peter\Documents\\github_new\\CodeLearning\\router_fingerpringting\\classify\\"


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



filepath_arp_set = filepath_after(win2_path, filepath_arp_set_before)
filepath_icmp_set = filepath_after(win2_path, filepath_icmp_set_before)
# filepath_arp_set = filepath_after(win_path, filepath_arp_set_before)
# filepath_icmp_set = filepath_after(win_path, filepath_icmp_set_before)

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
	return float(time)


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
	'''
	win2 is 81:90
	'''
	num_fo = 81
	num_be = 90
	data_time_IAT_list = []
	data_time_delta_list = []
	data_time_clock_list = []
	data_time_test_list = []
	flag = 0
	global label_str
	label_str.append(filepath[0][num_fo:num_be])
	for i in range(len(filepath)):
		if i != 0:
			name =  filepath[i][num_fo:num_be]
			name_pre = filepath[i-1][num_fo:num_be]
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
			data_time_clock = [((data_time[num+3] - data_time[num+1])/(data_time[num+2] - data_time[num])) for num in range(0,len(data_time)-2,2)]
			data_time_test = [(data_time[num+3]-data_time[num+1])-(data_time[num+2]-data_time[num+0]) for num in range(0, len(data_time)-2,2)]
			data_time_delta_list += data_time_delta
			data_time_IAT_list += data_time_IAT
			data_time_clock_list += data_time_clock
			data_time_test_list += data_time_test
	return [data_time_delta_list,data_time_IAT_list, data_time_clock_list,data_time_test_list]



def readfile_set(filepath):
	data_time_IAT_list = []
	data_time_delta_list = []
	data_time_clock_list = []
	data_time_test_list = []
	flag = 0
	'''
	win2 is 81:90
	'''
	num_fo = 81
	num_be = 90

	global label_str
	label_str.append(filepath[0][num_fo:num_be])
	for i in range(len(filepath)):
		if i != 0:
			name =  filepath[i][num_fo:num_be]
			name_pre = filepath[i-1][num_fo:num_be]
			if name == name_pre:
				flag = 1
			else:
				label_str.append(name)
				flag = 0
		i = filepath[i]
		if flag == 1:
			rf = readfile(i)
			data_time_delta_list[-1]+rf[0]
			data_time_IAT_list[-1]+rf[1]
			data_time_clock_list[-1]+rf[2]
			data_time_test_list[-1]+rf[3]
		else:
			rf = readfile(i)
			data_time_delta_list.append(rf[0])
			data_time_IAT_list.append(rf[1])
			data_time_clock_list.append(rf[2])
			data_time_test_list.append(rf[3])

	return [data_time_delta_list,data_time_IAT_list,data_time_clock_list,data_time_test_list]



def readfile_set_xor(filepath,name_give):
	data_time_IAT_list = []
	data_time_delta_list = []
	data_time_clock_list = []
	data_time_test_list = []
	flag = 0
	flag_name = 0
	'''
	win2 is 81:90
	'''
	num_fo = 81
	num_be = 90

	global label_str
	label_str.append(filepath[0][num_fo:num_be])
	name_choose = ["xiaomi","xjtuwlan","xunjie","tp4f","tpfe"]
	for i in range(5):
		if name_give == name_choose[i]:
			break 
	name_list = ["xiaomi/ar","XJTUWlan/","xunjie/ar","tplink_4f","tplink_fe","xiaomi/ic","XJTUWlan/","xunjie/ic","tplink_4f","tplink_fe"] #"win2"
	name_num_of_xor = i
	for i in range(len(filepath)):
		name =  filepath[i][num_fo:num_be]
		if name == name_list[name_num_of_xor] or name == name_list[name_num_of_xor+5]:
			if i != 0:
				name =  filepath[i][num_fo:num_be]
				name_pre = filepath[i-1][num_fo:num_be]
				if name == name_pre:
					flag = 1
				else:
					label_str.append(name)
					flag = 0
			i = filepath[i]
			if flag == 1:
				rf = readfile(i)
				data_time_delta_list[-1]+rf[0]
				data_time_IAT_list[-1]+rf[1]
				data_time_clock_list[-1]+rf[2]
				data_time_test_list[-1]+rf[3]
			else:
				rf = readfile(i)
				data_time_delta_list.append(rf[0])
				data_time_IAT_list.append(rf[1])
				data_time_clock_list.append(rf[2])
				data_time_test_list.append(rf[3])
			flag_name = 1
		else:
			i = filepath[i]
			if flag_name == 0:
				rf = readfile(i)
				data_time_delta_list[-1]+rf[0]
				data_time_IAT_list[-1]+rf[1]
				data_time_clock_list[-1]+rf[2]
				data_time_test_list[-1]+rf[3]
			else:
				print 
				rf = readfile(i)
				data_time_delta_list.append(rf[0])
				data_time_IAT_list.append(rf[1])
				data_time_clock_list.append(rf[2])
				data_time_test_list.append(rf[3])
				flag_name = 0	


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
		for i in range(min(len(grade[0][j]),len(grade[1][j]))):
			plt.plot(grade[0][j][i],grade[1][j][i],'+',color = co[j])
	plt.title('Kernel Density Estimation')
	plt.legend()
	plt.show()

def learning_feature(feature,inum =1):
	min_len = 99999999
	for i in feature:
		for j in i :
			if len(j) < min_len:
				min_len = len(j)
	new_set = []
	new_fe = []
	min_len = min_len -inum+1 # inum is times of the feature_num
	sample_num = len(feature[0])
	for sample in range(sample_num):
		one_set = []
		for num in range(0,min_len,inum):
			one_sample = []
			for feature_num in feature:
				for i in range(inum):
					one_sample.append(feature_num[sample][num+i])
			one_set.append(one_sample)
			new_fe.append(one_sample)
		new_set.append(one_set)
	new_set = np.array(new_set)
	return new_set,new_fe


def learning_feature_sliding(feature,inum =1):
	min_len = 99999999
	for i in feature:
		for j in i :
			if len(j) < min_len:
				min_len = len(j)
	new_set = []
	new_fe = []
	min_len = min_len -inum+1 # inum is times of the feature_num
	sample_num = len(feature[0])
	for sample in range(sample_num):
		one_set = []
		for num in range(0,min_len,inum):
			one_sample = []
			for feature_num in feature:
				for i in range(inum):
					one_sample.append(feature_num[sample][num+i])
			one_set.append(one_sample)
			new_fe.append(one_sample)
		new_set.append(one_set)
	new_set = np.array(new_set)
	return new_set,new_fe

def random_re(p,p_l):
	pp = []
	pp_l = []
	len_num = len(p)
	for i in range(len_num):
		num = random.randint(0,len_num-i-1)
		pp.append(p.pop(num))
		pp_l.append(p_l.pop(num))
	return pp,pp_l

def learning_label(new_set):
	label_all = []
	for i in range(len(new_set)):
		label = [i for num in range(len(new_set[0]))]
		label_all = label_all+label
		
	return  label_all


def pre_fe_la(filepath_icmp_set,filepath_arp_set,num=1):
	reshape_file_set = []
	icmp_set = readfile_set(filepath_icmp_set)  #二分类或者其它多分类用这个
	arp_set = readfile_set(filepath_arp_set)
	for i in range(4):   # 4 feature number
		reshape_file_set.append(icmp_set[i])
		reshape_file_set.append(arp_set[i])
	p,pp = learning_feature(reshape_file_set,num)
	p_l = learning_label(p)
	return random_re(pp,p_l)

def pre_fe_la_xor(filepath_icmp_set,filepath_arp_set,name_give,num=1):
	
	reshape_file_set = []
	icmp_set = readfile_set_xor(filepath_icmp_set,name_give)  #指定设备识别用这个
	arp_set = readfile_set_xor(filepath_arp_set,name_give)
	for i in range(4):   # 4 feature number
		reshape_file_set.append(icmp_set[i])
		reshape_file_set.append(arp_set[i])
	p,pp = learning_feature(reshape_file_set,num)
	p_l = learning_label(p)
	return random_re(pp,p_l)

def RF(X,y):
	len_x = len(X)
	train = int(0.8 * len_x)
	valid = int(0.1 *len_x)
	train_valid = train + valid

	X_train, y_train = X[:train], y[:train]
	X_valid, y_valid = X[train:train_valid], y[train:train_valid]
	X_train_valid, y_train_valid = X[:train_valid], y[:train_valid]
	X_test, y_test = X[train_valid:], y[train_valid:]

	rfc = RandomForestClassifier(n_estimators=100)
	lr = LogisticRegression()
	gnb = GaussianNB()
	svc = LinearSVC(C=1.0)
	for clf, name in [(lr, 'Logistic'),
	                  (gnb, 'Naive Bayes'),
	                  (svc, 'Support Vector Classification'),
	                  (rfc, 'Random Forest')]:
		

		# clf.fit(X_train, y_train)
		# print name
		# print "train_valid_acc:",clf.score(X_train_valid, y_train_valid)
		# print "test_acc:",clf.score(X_test,y_test)
	# return rfc.score(X_train_valid, y_train_valid),rfc.score(X_test,y_test)


		print name
		clf.fit(X_train_valid, y_train_valid)
		# clf_probs = clf.predict_proba(X_test)
		# score = log_loss(y_test, clf_probs)
		print  "train_valid_acc:",clf.score(X_train_valid, y_train_valid)
		
		print  "test_valid_acc:",clf.score(X_test,y_test)
		with open("result.txt",'a') as file:
			# file.write(str(name)+"\n")
			# file.write("train_valid_acc:\t"+str(clf.score(X_train_valid, y_train_valid))+"\n")
			file.write("test_valid_acc:\t"+str(clf.score(X_test,y_test))+"\n")
		if(name == "Random Forest"):

			RF_TRAIN = clf.score(X_train_valid, y_train_valid)
			RF_TEST  = clf.score(X_test,y_test)
	return RF_TRAIN,RF_TEST

	


def change_file_re(filepath_arp_set,filepath_icmp_set):
	dic_num_route = [[[0,1,2],[0,1]],[[3,4],[4]],[[5,6],[2,3]],[[7],[5,6]],[[8],[7]]]
	icmp_set_set = []
	arp_set_set = []
	for i in range(random.randint(4,4)):
		num = random.randint(0,4-i)
		list_num = dic_num_route.pop(num)
		for ii in list_num[0]:
			arp_set_set.append(filepath_arp_set[ii])	
		for jj in list_num[1]:
			icmp_set_set.append(filepath_icmp_set[jj])
	filepath_icmp_set = icmp_set_set
	filepath_arp_set = arp_set_set
	return filepath_icmp_set,filepath_arp_set


def choice_file_re(filepath_arp_set,filepath_icmp_set):
	dic_num_route = [[[0,1,2],[0,1]],[[3,4],[4]],[[5,6],[2,3]],[[7],[5,6]],[[8],[7]]]
	for i in range(5):
		for j in range(i+1,5):
			icmp_set_set = []
			arp_set_set = []
			list_num = [dic_num_route[j],dic_num_route[i]]
			print list_num
			for k in range(2):
				for ii in list_num[k][0]:
					arp_set_set.append(filepath_arp_set[ii])
					print filepath_arp_set[ii][81:92]	# 显示路由器型号
				for jj in list_num[k][1]:	
					icmp_set_set.append(filepath_icmp_set[jj])
			data_test = []
			print "*****************"
			for k in range(1):  # the ir of test to get mean  
				# for dd in range(3,10):
				# 	X,Y = pre_fe_la(icmp_set_set,arp_set_set,dd)  #dd 是窗口长度
				# 	d1,d2 = RF(X,Y)
				# 	feature_coff(X,Y) # 训练和测试

				X,Y = pre_fe_la(icmp_set_set,arp_set_set,1)  #5 是窗口长度
				d1,d2 = RF(X,Y)# 训练和测试
				# feature_coff(X,Y) # 特征选择
				# X,Y = pre_fe_la(icmp_set_set,arp_set_set,4)
				# d1,d2 = RF(X,Y)
				# feature_coff(X,Y)
				# X,Y = pre_fe_la(icmp_set_set,arp_set_set,3)
				# d1,d2 = RF(X,Y)
				# feature_coff(X,Y)
			data_test = np.array(data_test)
			# print "train_valid_acc:",sum(data_test[:,0])/len(data_test[:,0])
			# print "test_acc:",sum(data_test[:,1])/len(data_test[:,1])
			print "*****************"

def choice_file_re_all(filepath_arp_set,filepath_icmp_set):
	dic_num_route = [[[0,1,2],[0,1]],[[3,4],[4]],[[5,6],[2,3]],[[7],[5,6]],[[8],[7]]]
	icmp_set_set = []
	arp_set_set = []
	list_num = [dic_num_route[0],dic_num_route[1],dic_num_route[2],dic_num_route[3],dic_num_route[4]]
	for k in range(5):
		for ii in list_num[k][0]:
			arp_set_set.append(filepath_arp_set[ii])
			print filepath_arp_set[ii][81:92]	# 显示路由器型号
		for jj in list_num[k][1]:	
			icmp_set_set.append(filepath_icmp_set[jj])
	data_test = []
	print "*****************"
	for k in range(1): 
		X,Y = pre_fe_la(icmp_set_set,arp_set_set,1)  #5 是窗口长度
		d1,d2 = RF(X,Y)
	data_test = np.array(data_test)
	print "*****************"

def choice_file_re_oxr(filepath_arp_set,filepath_icmp_set):
	a =[[[0,1,2],[0,1]],[[3,4],[4]],[[5,6],[2,3]],[[7],[5,6]],[[8],[7]]]	
	for i in range(5):
		icmp_set_set = []
		arp_set_set = []
		p = range(5)
		p.pop(i)
		c = [[],[]]
		for j in p:
			for k in range(2):
	#             print j
				c[k]= c[k] + a[j][k]
		list_num = [a[i],c]
		for k in range(2):
			for ii in list_num[k][0]:
				arp_set_set.append(filepath_arp_set[ii])
				# print filepath_arp_set[ii][81:92]	# 显示路由器型号
			for jj in list_num[k][1]:	
				icmp_set_set.append(filepath_icmp_set[jj])
		data_test = []
		print "*****************"
		name_choose = ["xiaomi","xjtuwlan","xunjie","tp4f","tpfe"]
		name_give = name_choose[i]
		
		for k in range(1): 
			X,Y = pre_fe_la_xor(icmp_set_set,arp_set_set,name_give,1)  #5 是窗口长度
			d1,d2 = RF(X,Y)
		data_test = np.array(data_test)
		print "*****************"


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

def decimal2(x):
	return str("%.2f"%x)



def feature_coff(X,Y):
	len_x_feature = len(X[1])
	print len_x_feature
	names = ["x%s" % i for i in range(len_x_feature)]
	ranks = {}
	lr = LinearRegression(normalize=True)
	lr.fit(X, Y)
	ranks["Lin"] = rank_to_dict(np.abs(lr.coef_), names)

	ridge = Ridge(alpha=7)
	ridge.fit(X, Y)
	ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)


	lasso = Lasso(alpha=.05)
	lasso.fit(X, Y)
	ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)


	#stop the search when 5 features are left (they will get equal scores)
	rfe = RFE(lr, n_features_to_select=5)
	rfe.fit(X,Y)
	ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)

	rf = RandomForestRegressor()
	rf.fit(X,Y)
	ranks["RF"] = rank_to_dict(rf.feature_importances_, names)


	f, pval  = f_regression(X, Y, center=True)
	# ranks["Corr."] = rank_to_dict(f, names)
	ranks["p_val"] = rank_to_dict(pval, names)

	ranks_no = ranks.copy()
	ranks_no.pop("p_val")
	r = {}
	for name in names:
	    r[name] = round(np.mean([ranks[method][name] 
	                             for method in ranks_no.keys()]), 2)

	methods = sorted(ranks.keys())
	ranks["Mean"] = r
	methods.append("Mean")
	

	print "\t%s" % "\t".join(methods)
	

	mean_list_count = []
	for name in names:
	    print "%s\t%s" % (name, "\t".join(map(str, 
	                         [ranks[method][name] for method in methods])))
	    mean_list_count.append(ranks['Mean'][name])

	with open("feature_select_means"+str(len_x_feature)+".txt",'a') as filename:
		filename.write("%s\n" %("\t".join(map(decimal2,mean_list_count))))

	with open("feature_select"+str(len_x_feature)+".txt",'a') as filename:
		filename.write(str(len_x_feature))
		filename.write("\n\t%s\n" % "\t\t".join(methods))
		for name in names:
			filename.write("%s\t%s\n" % (name, "\t".join(map(decimal2, 
	                         [ranks[method][name] for method in methods]))))


# num of feature use this
# f1,f2 = change_file_re(filepath_arp_set,filepath_icmp_set)
# X,Y = pre_fe_la(f1,f2,1)

# print "rf........."
# RF(X,Y)

# choice_file_re(filepath_arp_set,filepath_icmp_set)
choice_file_re_oxr(filepath_arp_set,filepath_icmp_set) #全分类





# x = np.array(readfile_set(filepath_icmp_set))
# draw_kde_icmp_single(readfile_set(filepath_icmp_set)[0])
# draw_sort(readfile_sort_set(filepath_icmp_set)[0])

#print readfile_set(filepath_icmp_set)[2]
# draw_kde_icmp_single(readfile_set(filepath_arp_set)[2])
# draw_kde_icmp_single(readfile_set(filepath_icmp_set)[0])
# draw_single(readfile_set(filepath_arp_set)[1])


#draw_single(readfile_set(filepath_icmp_set)[1])
