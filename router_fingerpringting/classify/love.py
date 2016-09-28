#coding:utf-8
print'\n'.join([''.join([(u'矣文玲'[(x-y)%3]if x>-10 and x<10 and y<10 and y>-10 else' ')for x in range(-30,30)])for y in range(15,-15,-1)])