import sys,socket
file = open("url_write.txt").readlines()

for i in range(100):
	line = file[i]
	url = line.split("\n")[0]
	try:
		result = socket.getaddrinfo(str(url),None)
		print result[0][4]
	except:
		continue