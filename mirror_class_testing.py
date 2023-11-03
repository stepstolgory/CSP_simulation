import numpy as np
import matplotlib.pyplot as plt
from random import random
#Set up for the plot 
fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim(-52.5, 52.5)
ax.set_ylim(-52.5, 52.5)
ax.set_zlim(-52.5, 52.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

class Grid():
    def __init__(self, eq, size_x, size_y, margin_x, margin_y):
        self.eq = eq
        self.size_x = size_x
        self.size_y = size_y
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.mirrors_used = 0

    def create_grid_space(self, x_len, y_len): #Returns a grid of possible locations for the mirror centers
        x_coords = np.arange(-self.size_x, self.size_x+1, x_len+2*self.margin_x)
        y_coords = np.arange(-self.size_y, self.size_y+1, y_len+2*self.margin_y)
        grid = np.meshgrid(x_coords, y_coords)
        positions = zip(*(x.flat for x in grid)) #Returns the coordinate (x,y) value
        x_coords = []
        y_coords = []
        for (x,y) in positions: 
            if eval(self.eq): #If the coordinate satisfies a boolean equation i.e. it is in side a defined shape, the coordinate is kept
                x_coords.append(x)
                y_coords.append(y)
        self.mirrors_used = len(list(zip(x_coords, y_coords))) #Calculates the number of mirrors used by taking the number of allowed coordinates
        return zip(x_coords,y_coords)

    def create_mirrors(self, z, x_len, y_len, theta_x, theta_y, reflectivity): #Returns a list of mirrors
        mirrors = []
        grid = self.create_grid_space(x_len, y_len)
        for (x,y) in grid:
            # theta_x = np.pi * random() 
            # theta_y = np.pi * random()
            mirrors.append(Mirror(x, y, z, x_len, y_len, theta_x, theta_y, reflectivity)) #Creates a mirror object for every allowed position within the grid
        return mirrors

    def __str__(self):
        return f"A grid of mirror locations limited by equation {self.eq}, size {self.size_x, self.size_y} and {self.mirrors_used} mirrors used."

class Mirror():
    def __init__(self, x, y, z, x_len, y_len, theta_x, theta_y, reflectivity):
        #x, y, z define the position of the mirror
        self.x = x
        self.y = y
        self.z = z

        self.x_len = x_len
        self.y_len = y_len

        self.theta_x = theta_x
        self.theta_y = theta_y
        
        self.R = np.array([[np.cos(self.theta_y), 0, np.sin(self.theta_y)],
                  [np.sin(self.theta_x)*np.sin(self.theta_y), np.cos(self.theta_x), -np.sin(self.theta_x)*np.cos(self.theta_y)],
                  [-np.sin(self.theta_y)*np.cos(self.theta_x), np.sin(self.theta_x),  np.cos(self.theta_x)*np.cos(self.theta_y)]])

        self.points = np.array([[x,y,z],[x-x_len/2,y-y_len/2,z],[x-x_len/2, y+y_len/2, z],[x+x_len/2, y-y_len/2, z],[x+x_len/2, y+y_len/2, z]]) #Verteces of the mirror
        self.rotated_points = [np.dot(self.R, point) for point in self.points] #Verteces of the mirror after the rotation is applied
        self.vector1 = self.rotated_points[0] - np.array([x,y,z]) #Two vectors used for finding the normal
        self.vector2 = self.rotated_points[1] - np.array([x,y,z])
        self.reflectivity = reflectivity
    @property
    def verteces(self): #Returns a list of final vertices
        return self.rotated_points 
    @property
    def normal_vector(self): #Returns the normalised normal vector
        normal = np.cross(self.vector1, self.vector2)
        return normal/np.linalg.norm(normal)
    @property
    def plot_values(self): #Returns the values needed to plot the mirror as a square
        x_vals = [point[0] for point in self.rotated_points]
        y_vals = [point[1] for point in self.rotated_points]
        z_vals = [point[2] for point in self.rotated_points]
        return x_vals, y_vals, z_vals
    def __str__(self):
        return(f"Mirror at position ({self.x}, {self.y}, {self.z}), dimension {self.x_len, self.y_len}, rotation about x axis of {self.theta_x} radians, and rotation about y of {self.theta_y} radians")


#Equations for the grid shape
all_mirrors = 'True'
circle_eq = '52.5**2 >= x**2+y**2 >= 20**2'
square_eq = 'np.abs(x) > 45 or np.abs(y) > 45'

#Test mirrors
mirror1 = Mirror(0,0,0,10,10,0,0,0.2)
mirror2 = Mirror(0,0,5,10,10,0,0,0.3)
# ax.plot_trisurf(mirror1.plot_values[0], mirror1.plot_values[1], mirror1.plot_values[2], color = 'blue', alpha = mirror1.reflectivity)
# ax.plot_trisurf(mirror2.plot_values[0], mirror2.plot_values[1], mirror2.plot_values[2], color = 'blue', alpha = mirror2.reflectivity)

#Creating the grid and list of mirrors
grid = Grid(circle_eq, 50, 50, 1.25, 1.25)
mirrors = grid.create_mirrors(0, 10, 10, 0, 0, 0.5)


for mirror in mirrors:
    #Plotting each mirror on the same axis
    print(mirror)
    ax.plot_trisurf(mirror.plot_values[0], mirror.plot_values[1], mirror.plot_values[2], color = 'blue', alpha = mirror.reflectivity)

plt.show()
print(grid)
