import numpy as np
import matplotlib.pyplot as plt
from random import random
#Set up for the plot 
fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x_basis = np.array([1,0,0,0])
y_basis = np.array([0,1,0,0])
z_basis = np.array([0,0,1,0])

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

    def create_mirrors(self, z, x_len, y_len, theta_x, theta_y, theta_z, reflectivity): #Returns a list of mirrors
        mirrors = []
        grid = self.create_grid_space(x_len, y_len)
        for (x,y) in grid:
            mirrors.append(Mirror(x, y, z, x_len, y_len, theta_x, theta_y, theta_z, reflectivity)) #Creates a mirror object for every allowed position within the grid
        return mirrors

    def __str__(self):
        return f"A grid of mirror locations limited by equation '{self.eq}', size {self.size_x, self.size_y} and {self.mirrors_used} mirrors used."

class Mirror():
    def __init__(self, x, y, z, x_len, y_len, theta_x, theta_y, theta_z, reflectivity):
        #x, y, z define the position of the mirror
    
        self.x = x
        self.y = y
        self.z = z

        self.x_len = x_len
        self.y_len = y_len

        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z

        self.points = np.array([[x,y,z,1],[x-x_len/2,y-y_len/2,z,1],[x-x_len/2, y+y_len/2, z,1],[x+x_len/2, y-y_len/2, z,1],[x+x_len/2, y+y_len/2, z,1]]) #Verteces of the mirror
        self.reflectivity = reflectivity
    
    @property
    def vectors(self):
        vector1 = self.rotated_points[1]-self.pos  #Two vectors used for finding the normal
        vector2 = self.rotated_points[2]-self.pos
        return vector1, vector2

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z, 1])
    
    @property
    def rotated_points(self): #Returns the list of rotated points
        T1 = self.T(-self.x, -self.y, -self.z)
        T2 = self.T(self.x, self.y, self.z)
        # R = np.dot(self.Rz,np.dot(self.Rx, self.Ry))
        M = np.dot(T2, np.dot(self.R, T1)) #Translates the point to the origin, rotates it, translates it back
        rotated_points = [np.dot(M, point) for point in self.points] 
        return rotated_points
    
    @property
    def verteces(self): #Returns a list of final vertices
        return self.rotated_points

    @property
    def normal_vector(self): #Returns the normalised normal vector
        normal = np.cross(np.delete(self.vectors[1],3), np.delete(self.vectors[0],3))
        return normal/np.linalg.norm(normal)

    @property
    def plot_values(self): #Returns the values needed to plot the mirror as a square
        x_vals = [point[0] for point in self.rotated_points]
        y_vals = [point[1] for point in self.rotated_points]
        z_vals = [point[2] for point in self.rotated_points]
        return np.array([x_vals, y_vals, z_vals])

    def axis_rotation(self, axis, ang):
        axis = axis/np.linalg.norm(axis)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        matrix = np.array([[x*x*(1-np.cos(ang))+np.cos(ang), y*x*(1-np.cos(ang))-z*np.sin(ang), z*x*(1-np.cos(ang))+y*np.sin(ang),0],
                            [x*y*(1-np.cos(ang))+z*np.sin(ang), y*y*(1-np.cos(ang))+np.cos(ang), z*y*(1-np.cos(ang))-x*np.sin(ang), 0],
                            [x*z*(1-np.cos(ang)) - y*np.sin(ang),  y*z*(1 - np.cos(ang)) + x*np.sin(ang), z*z*(1 - np.cos(ang)) + np.cos(ang), 0],
                            [0, 0, 0, 1]])
        return matrix

    @property
    def Rx(self):
        return np.array([[1,0,0,0],  
                [0, np.cos(self.theta_x), -np.sin(self.theta_x), 0],
                [0, np.sin(self.theta_x), np.sin(self.theta_x), 0],
                [0,0,0,1]])

    @property
    def Ry(self):
        return np.array([[np.cos(self.theta_y), 0, np.sin(self.theta_y), 0],
                [0, 1, 0, 0],
                [-np.sin(self.theta_y), 0, np.cos(self.theta_y), 0],
                [0,0,0,1]])

    @property
    def Rz(self):
        return np.array([[np.cos(self.theta_z), -np.sin(self.theta_z), 0, 0],
                [np.sin(self.theta_z), np.cos(self.theta_z), 0, 0],
                [0,0,1,0],
                [0,0,0,1]])
    @property
    def R(self):
        return self._R
    
    @R.setter
    def R(self, value):
        self._R = value

    def T(self, x, y, z):
        return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z], [0,0,0,1]])
    
    #FIXME: Does something weird at negative z values (might not be an issue)
    def point_to_tower(self, tower):
        tower = tower - np.array([self.x, self.y, self.z, 0])
        norm_tower = np.delete(tower/np.linalg.norm(tower), 3)
        axis = np.cross(self.normal_vector, norm_tower)
        ang = np.arccos(np.dot(norm_tower, self.normal_vector))
        value = self.axis_rotation(axis, ang)
        self.R = value
        

    def __str__(self):
        return(f"Mirror at position ({self.x}, {self.y}, {self.z}), dimension {self.x_len, self.y_len}, rotation about x axis of {self.theta_x} radians, and rotation about y of {self.theta_y} radians")


#Equations for the grid shape
all_mirrors = 'True'
no_mirrors = 'False'
circle_eq = '52.5**2 >= x**2+y**2 >= 20**2'
square_eq = 'np.abs(x) > 45 or np.abs(y) > 45'

#Test mirrors
mirror1 = Mirror(5,5,0,10,10,0,0,0,0.5)
mirror1.R = np.dot(mirror1.Rz,np.dot(mirror1.Rx, mirror1.Ry))
mirror2 = Mirror(0,0,20,10,10,0,0,0,0.3)
# ax.plot_trisurf(mirror2.plot_values[0], mirror2.plot_values[1], mirror2.plot_values[2], color = 'blue', alpha = mirror2.reflectivity)

#Creating the grid and list of mirrors
grid = Grid(all_mirrors, 50, 50, 1.25, 1.25)
mirrors = grid.create_mirrors(0, 10, 10, 0, 0, 0, 0.5)

for mirror in mirrors:
    #Plotting each mirror on the same axis
    mirror.R = np.dot(mirror.Rz,np.dot(mirror.Rx, mirror.Ry))
    mirror.point_to_tower([0,0,-70,0])
    try:
        ax.plot_trisurf(mirror.plot_values[0], mirror.plot_values[1], mirror.plot_values[2], color = 'blue', alpha = mirror.reflectivity)
        # ax.quiver(mirror.pos[0], mirror.pos[1], mirror.pos[2], mirror.normal_vector[0], mirror.normal_vector[1], mirror.normal_vector[2], color = 'r', arrow_length_ratio = 0.1)   
    except RuntimeError:
        print(f"Run time error for {mirror}")

#Normal vector
# ax.plot_trisurf(mirror1.plot_values[0], mirror1.plot_values[1], mirror1.plot_values[2], color = 'black', alpha = mirror1.reflectivity)
#Tower vector
# ax.quiver(mirror1.pos[0], mirror1.pos[1], mirror1.pos[2], -10, -5, 20, color = 'r', arrow_length_ratio = 0.1)

ax.scatter(0,0,-70, color = 'g', s=50)
plt.show()
