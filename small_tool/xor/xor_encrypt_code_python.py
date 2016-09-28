def encrypt_code(string, len):
	xorkeys="BB2FA36AAA9541F0"
	code = range(len)
	for i in range(len):
#		print(string[i]," ",xorkeys[i % 16])
		code[i] = chr(ord(string[i]) ^ ord(xorkeys[i % 16]))
	return code

print "m7A4nQ_/nA\t\t",''.join(encrypt_code("m7A4nQ_/nA", 9))
print "m [(n3\t\t\t",''.join(encrypt_code("m [(n3", 5))
print "m6_6n3\t\t\t",''.join(encrypt_code("m6_6n3", 5))
print "m4S4nAC/n&ZV\x1aA/T\t",''.join(encrypt_code("m4S4nAC/n&ZV\x1aA/T",16))
print "m.[$n__#4%\C\x1aB)0\t", ''.join(encrypt_code("m.[$n__#4%\C\x1aB)0", len("m.[$n__#4%\C\x1aB)0")-1))
print "m.[$n3\t\t\t",''.join(encrypt_code("m.[$n3",5))
print "55EwoTQ& 5XA\x00\x04p\x1e!-_i%W\x183 3\t",''.join(encrypt_code("55EwoTQ& 5XA\x00\x04p\x1e!-_i%W\x183 3",26))
print "m4S4nAC/nA\t\t",''.join(encrypt_code("m4S4nAC/nA",9))

"""
fileName = "dd.rar"
fileOpen = open(fileName,'rb')
decodeSave = open("decodeSave.txt",'w')
decode=[]
line = fileOpen.read(16)
try:
	while line:
		decode.append(''.join(encrypt_code(line, len(line))))
		line = fileOpen.read(16)
	decode = ''.join(decode)
	decodeSave.write(decode)
finally:
	fileOpen.close()
	decodeSave.close()
"""
