import matplotlib.pyplot as plt

with open("xjtuwlan_ssdp.txt",'r') as file:
	line = file.readlines()
	line.pop(0)
	len_champ = []
	for line_num in line:
		len_champ.append(int(line_num.split(",")[5].split("\"")[1]))
plt.plot(len_champ[0:200])
plt.show()