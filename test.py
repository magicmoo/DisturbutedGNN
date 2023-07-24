import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1) * 2
y2 = np.cos(2 * np.pi * x2)

plt.subplot(1, 2, 1)
plt.plot(x1, y1)

plt.subplot(1, 2, 2)
plt.plot(x2, y2)
plt.ylabel("hello")



plt.savefig("./image/test.jpg")