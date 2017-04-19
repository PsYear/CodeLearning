#coding:utf-8
import requests
import bs4
from bs4 import BeautifulSoup
import requests
import json
import bs4
import codecs
import time
import tushare as ts


def comment(socks,page):
	headers1 = { "User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36"
	            }
	session = requests.session()
	url = "https://xueqiu.com/s/"+socks
	cont = session.get(url,headers=headers1).content
	unicode_str = unicode(cont, encoding='utf-8')
	headers2 = {
	"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
	"Accept-Encoding":"gzip, deflate, sdch, br",
	"Accept-Language":"zh-CN,zh;q=0.8,en-US;q=0.6,en;q=0.4",
	"Cache-Control":"max-age=0",
	"Connection":"keep-alive",
	"Cookie":"bid=e2201b22d6ecc5fd4dd7994cfe7b5c6c_imznpt8g; s=6g1g0ghdwc; u=221490945723339; xq_a_token=ca292f8d934efc28f3fd052b7dcf46f14a20a0d3; xq_a_token.sig=OfkwBmKBwnYETfT5NOuElwVwhBY; xq_r_token=d1accc7b0cafd743be1b975a863a146e514d9c80; xq_r_token.sig=LV5APomGXuF1PJGnH9SmAPdxYHc; aliyungf_tc=AQAAAAP7YVVJngcA4thzfHHCckmnWGU6; Hm_lvt_1db88642e346389874251b5a1eded6e3=1490945724,1491549758; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1491550521; __utmt=1; __utma=1.1616022943.1457004958.1488871387.1491549769.13; __utmb=1.3.10.1491549769; __utmc=1; __utmz=1.1488871387.12.4.utmcsr=xueqiu.com|utmccn=(referral)|utmcmd=referral|utmcct=/",
	"Host":"xueqiu.com",
	"Upgrade-Insecure-Requests":"1",
	"User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36"
	}
	session = requests.session()
	page = 1
	url2 = "https://xueqiu.com/statuses/search.json?count=10&comment=0&symbol=%s&hl=0&source=user&sort=time&page=%s&_=1491549811492" %(socks,str(page))
	cont = session.get(url2,headers=headers2)
	unicode_str = unicode(cont.content, encoding='utf-8')
	content = unicode_str.encode('utf-8')
	text = ""
	decode_json = json.loads(content)
	cont = decode_json["list"][5]["text"]
	soup = BeautifulSoup(cont)
	    
	import time
	ISOTIMEFORMAT="%Y-%m-%d %X"
	with codecs.open("xueqiu_comment_"+str(socks)+".txt", 'a', 'utf-8') as filename:
		for i in range(len(decode_json["list"])):
		    cont = decode_json["list"][i]["text"]
		    soup = BeautifulSoup(cont)
		    time_text = decode_json["list"][i]["created_at"]
		    filename.write(str(time_text)+'[,,,,,,]')
		    for j in range(len(soup.body.contents)):
		        if isinstance(soup.body.contents[j], bs4.element.NavigableString) or isinstance(soup.body.contents[j], bs4.element.Tag) :
		            if  isinstance(soup.body.contents[j], bs4.element.Tag):
		                if soup.body.contents[j].string != None:
		                    sc = soup.body.contents[j].string
		                    # sentence.append(soup.body.contents[j].string)
		                    filename.write(soup.body.contents[j].string)
		                else:
		                    for child in soup.body.contents[j]:
		                        if isinstance(child,bs4.element.NavigableString):
		                            # print child
		                            # sentence.append(child)                
									filename.write(child)
		            else:
		                if soup.body.contents[j]==" ":
		                    continue
		                # sentence.append(soup.body.contents[j])
		                filename.write(soup.body.contents[j])
		    filename.write("\n")
		     #生成时间
		    # print time.strftime( ISOTIMEFORMAT, time.localtime((float(time_text)/1000)) )

code_300 = ts.get_hs300s()["code"]
code_300_name = []
for i in code_300:
	if int(i) >= 600000:
		name = "SH"+i
	else:
		name = "SZ"+i
	code_300_name.append(name)

for socks in code_300_name[0:50]:
	for page in range(1,101):
		comment(socks,page)
		print socks,"   ",str(page)
		time.sleep(1)

# socks = "SZ000002"
# for page in range(1,3):
# 	comment(socks,page)
# 	print socks,"   ",str(page)
# 	time.sleep(1)