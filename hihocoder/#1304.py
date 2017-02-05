used = [False for i in range(4)]
nownum = [0 for i in range(4)]
ops = [0 for i in range(3)]
opType = ["+","-","*","/","f-","f/"]
num = []


def calc(num1, num2, op):
    if op == "+":
        return num1 + num2
    elif op == "-":
        return num1 - num2
    elif op == "*":
        return num1 * num2
    elif op == "/":
        if num2 != 0:
            return num1 *1.0/ num2
        else:
            return False
    elif op == "f-":
        return num2 - num1
    elif op == "f/":
        if num1 != 0:
            return num2 *1.0/ num1*1.0
        else:
            return False

def makenumber(depth):
    if depth >= 4:
        return makeop(0)
    for i in range(4):
        if not used[i]:
            nownum[depth] = num_list[i]
            used[i] = True
            if makenumber(depth+1):
                return True
            used[i] = False
    return False

def makeop(depth):
    if depth >= 3:
        if calculate1(nownum,ops) == 24:
            return True
        if calculate1(nownum,ops) == 24:
            return True
        return False
    for i in range(6):
        ops[depth] = opType[i]
        if makeop(depth+1):
            return True
    return False


def calculate1(nownum,ops):
    tmp = calc(nownum[0], nownum[1], ops[0])
    for i in range(2):
        if not tmp :
            return False
        tmp = calc(tmp, nownum[i+2], ops[i+1])
    return tmp

def calculate2(nownum,ops):
    tmp1 = calc(nownum[0], nownum[1], ops[0])
    tmp2 = calc(nownum[2], nownum[3], ops[2])
    if tmp1 and tmp2:
        return calc(tmp1, tmp2, ops[1])
    else:
        return False



line_num = int(raw_input())
for i in range(line_num):
    num_list = (raw_input()).split(" ")
    num_list = [int(num_list[i]) for i in range(4)]
    num.append(num_list)

for i in range(line_num):
    num_list = num[i]
    if makenumber(0):
        print "Yes"
        print nownum
        print ops
    else:
        print "No"


