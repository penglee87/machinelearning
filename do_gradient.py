#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
线性回归的梯度下降法示例
具体数学推导可见  https://www.cnblogs.com/pinard/p/5970503.html
'''
import numpy as np
import matplotlib.pyplot as plt
#y=2 * (x1) + (x2) + 3 
rate = 0.001
x_train = np.array([[1, 2],[2, 1],[2, 3],[3, 5],[1, 3],[4, 2],[7, 3],[4, 5],[11, 3],[8, 7]])
y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26])
x_test  = np.array([[1, 4],[2, 2],[2, 5],[5, 3],[1, 5],[4, 1]])

a = np.random.normal()  #返回均值为0、标准差为1的正态分布,每次随机返回一个值
b = np.random.normal()
c = np.random.normal()

#定义线性回归函数
def h(x):
    return a*x[0]+b*x[1]+c

#损失函数为  1/2*(a*x[0]+b*x[1]+c-y)**2    即 1/2*(h(x)-y)**2
#对系数a求偏导数  e = (a*x[0]+b*x[1]+c-y)*x[0]    即 (h(x)-y)*x[0]
#调整系数a  a=a-e
for i in range(1000):
    sum_a=0
    sum_b=0
    sum_c=0
    for x, y in zip(x_train, y_train):
        sum_a = sum_a + rate*(y-h(x))*x[0]  #调整系数a
        sum_b = sum_b + rate*(y-h(x))*x[1]
        sum_c = sum_c + rate*(y-h(x))
    a = a + sum_a
    b = b + sum_b
    c = c + sum_c
    plt.plot([h(xi) for xi in x_test])

print(a)
print(b)
print(c)

result=[h(xi) for xi in x_train]
print(result)

result=[h(xi) for xi in x_test]
print(result)

plt.show()


