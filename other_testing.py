import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function you want to plot in terms of x and y
def my_function(x, y):
    return 2*(x**3)+6*x*(y**2)-3*y**3-150*x

# Create a grid of x and y values
x = np.linspace(-5, 5, 100)  # Define the range of x values
y = np.linspace(-5, 5, 100)  # Define the range of y values
X, Y = np.meshgrid(x, y)    # Create a grid of (x, y) pairs

# Calculate the corresponding z values using your function
Z = my_function(X, Y)
print(my_function(-3,-4))
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis')

# Add labels and a colorbar
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.colorbar(ax.plot_surface(X, Y, Z, cmap='viridis'), ax=ax, pad=0.1)
print((20**1/3)**2+40/(20**1/3))
# Show the plot
plt.show()