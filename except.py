def ary(list):
    top=0
    for x in list:
        top+=x
    avg=top/len(list)
    return top,avg
try:
    t,a=ary([1,7,'mathi'])
    print('total={},avegage={}'.format(t,a))
except TypeError:
    print('Type error.plz provide numbers')
except ZeroDivisionError:
    print('Zero division error, plz dont give empty list')
