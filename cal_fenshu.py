

p1 = '/home/wangxiang/Documents/ocr_train/testr.txt'
p2 = '/home/wangxiang/Documents/full_multi_gpu/recognition_result_1078.txt'
d1 = {}
d2 = {}
f1 = open(p1, 'r')
f2 = open(p2, 'r')
l1 = f1.readlines()
l2 = f2.readlines()
for i in range(len(l1)):
    x = l1[i].split()
    d1[x[0]] = x[1]
for i in range(len(l2)):
    x = l2[i].split()
    d2[x[0]] = x[1]

s = 0
for key in d1:
    if d1[key] == d2[key]:
        s+=1
        print(s)
    print(d1[key])
    print(d2[key])
print(s*1.0000/20000)
print('ok')

