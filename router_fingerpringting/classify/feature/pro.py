import numpy as np


for i in range(3,10):
	ele_array = np.array(range(i*8))*0.0
	with open("feature_select_means"+str(i*8)+".txt") as filename:
		p = filename.readlines()
		for ele in p:
			ele_array += np.array(map(float,ele.split("\t")))
		re = ele_array/len(p)
		with open("feature_means.txt",'a') as file:
			file.write("%s\n" %("\t".join(map(str,re))))
