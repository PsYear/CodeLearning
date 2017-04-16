import numpy as np
import matplotlib.pylab as plt

Support_list = [[],[],[],[],[]]
Naive_list = [[],[],[],[],[]]
Random = [[],[],[],[],[]]
Logistic = [[],[],[],[],[]]

for i in range(1,11):
	num_line = 0
	with open("result_ele_"+str(i)+".txt","r") as filename:
		p = filename.readlines()
		print p
		for ele in p:
			if ele[0] == "S":
				Support_list[num_line].append(float(ele.split(":")[1]))
			if ele[0] == "N":
				Naive_list[num_line].append(float(ele.split(":")[1]))
			if ele[0] == "R":
				Random[num_line].append(float(ele.split(":")[1]))
			if ele[0] == "L":
				Logistic[num_line].append(float(ele.split(":")[1]))
				num_line += 1
with open("iajf.txt","a") as filename:
	for i in range(5):
		filename.write(str(Support_list[i][0])+","+str(Naive_list[i][0])+","+str(Random[i][0])+","+str(Logistic[i][0])+"\n")
		filename.write(str(Support_list[i][7])+","+str(Naive_list[i][7])+","+str(Random[i][7])+","+str(Logistic[i][7])+"\n")



# for j in range(5):
# 	co = ['b','k','g','c','m','y','r','w']
# 	plt.plot(Random[j],color=co[j])

# plt.title('sliding window')
# plt.legend()
# plt.show()

