import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

x, y = np.meshgrid(x, y)
eq = 0.12 * x + 0.01 * y + 1.09

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, eq)

plt.show()