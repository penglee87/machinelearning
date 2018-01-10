import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0.01, 1.0, 0.01)
y = 1/2*np.log((1-x)/x)
plt.plot(x, y)
plt.show()