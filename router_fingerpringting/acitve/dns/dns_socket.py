import sys,socket
import os
import random
import threading,time
flushdns = os.popen("ipconfig /flushdns")
print flushdns.read()


# file_copy = file[:-1]
# for i in range(int(sys.argv[1])):
# 	num = random.randint(0,len(file)-2-i)
# 	line = file_copy.pop(num)
# 	url = line.split("\n")[0]
# 	try:
# 		result = socket.getaddrinfo(str(url),None)
# 		print result[0][4]
# 	except:
# 		continue
def dns_send(url, num):
	try:
		result = socket.getaddrinfo(url,None)
		print result[0][4]
		print "done is " + str(num)
	except:
		return

file = open("url_write.txt").readlines()
threadpool=[]
for i in range(int(sys.argv[1])):
	line = file[i]
	url = str(line.split("\n")[0])
	print url
	th = threading.Thread(target= dns_send,args= (url,i))
	threadpool.append(th)

for th in threadpool:
	th.start()
for th in threadpool:
	threading.Thread.join(th)
print "all done"


