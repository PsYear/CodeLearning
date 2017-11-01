#coding:utf-8
import requests



socks = "SZ000002"
headers = { "User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36"
            }

session = requests.session()
url = "https://xueqiu.com/s/"+socks
cont = session.get(url,headers=headers).content
with open("test_paging.txt",'a') as writing_file:
    writing_file.write(cont)
    