import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5.0, 5.0, 0.01)

y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

plt.figure(1)

#创建3行1列的子图，画其中第1个图
plt.subplot(311)
plt.plot(x, y1)

#创建3行1列的子图，画其中第2个图
plt.subplot(312)
plt.plot(x, y2)

#创建3行1列的子图，画其中第3个图
plt.subplot(313)
plt.xlim((-5, 5))
plt.ylim((-10, 10))  #更改坐标轴取值范围
plt.plot(x, y3)

plt.show()

'''
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.plot(x, y1)

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
ax.plot(x, y2)

fig3 = plt.figure(3)
ax = fig3.add_subplot(211)
ax.plot(x, y1)
ax = fig3.add_subplot(212)
ax.plot(x, y2)

plt.show()
'''