import matplotlib.pyplot as plt
import numpy as np

print("generating plot data...")

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi*t)
plt.plot(t, s)

print("turning on grid\nand showing plot")
plt.grid(True)
plt.show()

print("plot has been shown")

