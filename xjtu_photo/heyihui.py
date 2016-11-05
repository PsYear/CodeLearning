#coding:utf-8

import requests
from xtls.util import BeautifulSoup
import re
import thread
from time import sleep, ctime
headers2 = { "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding":"gzip, deflate, sdch",
    "Accept-Language":"h-CN,zh;q=0.8,en-US;q=0.6,en;q=0.4",
    "Referer":"http://xkfw.xjtu.edu.cn/xsxk/login.xk",
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36",
    "Upgrade-Insecure-Requests":"1",
    "Cache-Control":"max-age=0",
    "Connection":"keep-alive",
        }


class get_uid(object):
	"""docstring for get_uid"""
	def __init__(self):
		super(get_uid, self).__init__()
		self.sample = open("sample.jpg","rb").read()
		self.record = "uid.txt"
		self.flag = [1,1,1]
		self.count = 0
		self.flag2 = [1,1,1]
		self.count2 = 0
	def change(self,num):
		if num<10:
			return "00"+str(num)
		elif num<=99:
			return "0"+str(num)
		else :
			return str(num)

	def change2(self,num):
		if num<10:
			return "0"+str(num)
		elif num<=99:
			return str(num)



	def login(self,uid):
	    session = requests.session()
	    url_portal = 'http://202.117.1.152:8080/Common/GetPhotoByBH?xh='+uid
	    cont = session.get(url_portal,headers=headers2).content
	    if cont == self.sample:
			self.flag[self.count] = 0
			self.count+=1
			return
	    else:
	    	with open(self.record,"a") as record_file:
	    		record_file.write(uid+"\n")
	    	with open(uid+".jpg","wb") as pic:
	            pic.write(cont)
	    
	def start(self):
		for a in range(216,217):
			for i in range(1,15):
				for j in range(1,15):
					if self.flag2[2] == 0:
							self.flag2 = [1,1,1]
							self.count2 = 0
							break
					for k in range(1,300):
						if self.flag[2] == 0:
							self.flag = [1,1,1]
							self.count = 0
							self.flag2[self.count2] = 0
							self.count2 += 1
							break 
						uid = self.change(a)+self.change2(i)+self.change2(j)+self.change(k)
						self.login(uid)
						print uid



y = get_uid().start()