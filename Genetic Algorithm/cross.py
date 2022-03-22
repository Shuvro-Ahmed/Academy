from random import Random

x = [24,23,20,11,1,2,3,4]
y = [20,13,10,21,0,9,8,7]

r = Random()
idx = r.randint(3,5)
print(idx)
child = x[:idx] + y[idx:]
print(child)
idx = r.randint(0, 7) #random index
child[idx] = r.randint(1,8)
print(child)
