import cookielib
import urllib2

file = open("out_html.txt",'w+')
cookie = cookielib.MozillaCookieJar()

cookie.load('cookie.txt', ignore_discard=True, ignore_expires=True)

req = urllib2.Request("http://xkfw.xjtu.edu.cn/xsxk/login.xk")

opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookie))
response = opener.open(req)
file.write(response.read())