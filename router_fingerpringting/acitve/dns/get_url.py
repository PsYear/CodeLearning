#coding:utf-8
#在 http://gate.guokr.com/ 上获取所有网页的列表
import re
file_w = open("url_write.txt",'w+')
file = open("gate_guokr_com.txt").readlines()
for line in file:
	url = re.findall(r"(?<=k\" href=\"http://).+?(?=\")",line)
	if(len(url)>0):
		file_w.write(url.pop().split("/")[0]+'\n')



# url = "http://gate.guokr.com/"