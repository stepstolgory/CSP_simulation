#Playing around with classes and light physics
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#Set up for the plot 
fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 

class Mirror():
    def __init__(self, x, y, z, x_len, y_len, theta_x, theta_y, theta_z, reflectivity, id):
        """
        x,y,z dictate the position of the mirror
        x_len, y_len dictate the dimensions of the mirror
        theta_x, theta_y, theta_z dictate the rotation of the mirror
        relectivity dictates how reflective the mirror is (currently the alpha value)
        """
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

        self.id = id
    
    @property
    def vectors(self):
        vector1 = self.rotated_points[1]-self.pos  #Two vectors used for finding the normal
        vector1 = vector1/np.linalg.norm(vector1)
        vector2 = self.rotated_points[2]-self.pos
        vector2 = vector2/np.linalg.norm(vector2)
        return vector1, vector2

    @property
    def pos(self): #position of the centre of the mirror
        return np.array([self.x, self.y, self.z, 1])
    
    @property
    def rotated_points(self): #Returns the list of rotated points
        T1 = self.T(-self.x, -self.y, -self.z)
        T2 = self.T(self.x, self.y, self.z)
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
    def plot_values(self): #Returns the values needed to plot the mirror as a square using trisurface
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
        """
        Makes the mirrors face a point.
        Will need to be changed to redirect light towards the tower instead.
        """
        tower = tower - np.array([self.x, self.y, self.z, 0])
        norm_tower = np.delete(tower/np.linalg.norm(tower), 3)
        axis = np.cross(self.normal_vector, norm_tower)
        ang = np.arccos(np.dot(norm_tower, self.normal_vector))
        value = self.axis_rotation(axis, ang)
        self.R = value
        

    def __str__(self):
        return(f"Mirror at position ({self.x}, {self.y}, {self.z}), dimension {self.x_len, self.y_len}, rotation about x axis of {self.theta_x} radians, and rotation about y of {self.theta_y} radians")

class Grid():
    def __init__(self, eq, size_x, size_y, margin_x, margin_y):
        self.eq = eq
        self.size_x = size_x
        self.size_y = size_y
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.mirrors_used = 0

    def create_grid_space(self, x_len, y_len): #Returns a grid of possible locations for the mirror centers
        """
        Creates a grid of possible x and y coordinates by generating arrays of them given the size of the grid, and the margin around each mirror.
        These coordinates are then made into a grid of possible points for the mirror centres.
        The coordinates are then filtered by a boolean equation to clarify the arrangement of the mirror
        Returns a zipped object of all the allowed coordinates
        """
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
        """
        Creates a mirror object for every space in the grid
        Returns a list of the objects
        """
        mirrors = []
        grid = self.create_grid_space(x_len, y_len)
        i=0
        for (x,y) in grid:
            i+=1
            mirrors.append(Mirror(x, y, z, x_len, y_len, theta_x, theta_y, theta_z, reflectivity, i)) #Creates a mirror object for every allowed position within the grid
        return mirrors

    def __str__(self): #Dictates how print(mirror) works
        return f"A grid of mirror locations limited by equation '{self.eq}', size {self.size_x, self.size_y} and {self.mirrors_used} mirrors used."

class Sun(Mirror):
    """
    Takes a slightly eliptical shaped path across the sky
    Calculate position in 15 minute intervals
    Position in 2 dimensions
    Assume longest day of summer for the rise and set times
    """
    rays = []

    def __init__(self, rise_time, set_time, intensity, x, y, z, x_len, y_len, theta_x, theta_y, theta_z, reflectivity, id):
        super().__init__(x, y, z, x_len, y_len, theta_x, theta_y, theta_z, reflectivity, id)
        self.rise_time = rise_time
        self.set_time = set_time
        self.intensity = intensity
        self.grid_space = Grid('True', 100, 100, 2.5, 2.5).create_grid_space(0,0)

    def move(self,t):
        #Takes a time t, to determine the position of the sun and change it
        self.x, self.z = 100*np.cos((np.pi*t/(self.day_time))), 70*np.sin(np.pi*t/(self.day_time))+1
        return self.x, self.z

    @property
    def day_time(self):
        return self.set_time-self.rise_time
    
    @property
    def ray_coordinates(self):
        ray_coordinates = []
        for (x,y) in self.grid_space:
            coord = self.pos + x*self.vectors[0]+y*self.vectors[1]
            ray_coordinates.append(coord)
        print(f"There are {len(ray_coordinates)} rays.")
        return ray_coordinates

    def create_rays(self):
        j = 0
        for coordinate in self.ray_coordinates:
            j+=1
            self.rays.append(Ray(coordinate, self.normal_vector, 100, j))
        return self.rays

    def __str__(self):
        return (f"Sun({self.rise_time}, {self.set_time}, {self.intensity}, {self.x, self.y})")
    

class Ray():
    def __init__(self, origin, direction, magnitude, id):
        self.origin = origin
        self.direction = direction
        self.magnitude = magnitude
        self.id = id

def animate(i):
    # Get the point from the points list at index i
    # Plot that point using the x and y coordinates
    ax.clear()
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    sun = Mirror(x[i], 0, z[i], 150, 150, 0, 0, 0, 1, 0)
    sun.R = np.dot(sun.Rz,np.dot(sun.Rx, sun.Ry))

    sun.point_to_tower([0,0,0,0])
    ax.plot_trisurf(sun.plot_values[0], sun.plot_values[1], sun.plot_values[2], color="orange", alpha = sun.reflectivity)

    for mirror in mirrors:
        #Plotting each mirror on the same axis
        mirror.R = np.dot(mirror.Rz,np.dot(mirror.Rx, mirror.Ry))
        mirror.point_to_tower([x[i],0, z[i],0])
        try:
            ax.plot_trisurf(mirror.plot_values[0], mirror.plot_values[1], mirror.plot_values[2], color = 'blue', alpha = mirror.reflectivity)
            # ax.quiver(mirror.pos[0], mirror.pos[1], mirror.pos[2], mirror.normal_vector[0], mirror.normal_vector[1], mirror.normal_vector[2], color = 'r', arrow_length_ratio = 0.1)   
        except RuntimeError:
            print(f"Run time error for {mirror}")


if __name__ == "__main__":
    #Arrays for animation
    x = []
    z = []  
    #Equations for the grid shape
    all_mirrors = 'True'
    no_mirrors = 'False'
    circle_eq = '52.5**2 >= x**2+y**2 >= 20**2'
    square_eq = 'np.abs(x) > 45 or np.abs(y) > 45'

    sun = Sun(360, 1220, 10, 100, 0, 1, 150, 150, 0, 0, 0, 1, 0)
    sun.R = np.dot(sun.Rz,np.dot(sun.Rx, sun.Ry))
    sun.point_to_tower([0,0,0,0])
    grid = Grid(circle_eq, 50, 50, 1.25, 1.25)
    mirrors = grid.create_mirrors(0, 10, 10, 0, 0, 0, 0.5)

    for mirror in mirrors:
        #Plotting each mirror on the same axis
        mirror.R = np.dot(mirror.Rz,np.dot(mirror.Rx, mirror.Ry))
        mirror.point_to_tower([0,0,70,0])
        try:
            ax.plot_trisurf(mirror.plot_values[0], mirror.plot_values[1], mirror.plot_values[2], color = 'blue', alpha = mirror.reflectivity)
            # ax.quiver(mirror.pos[0], mirror.pos[1], mirror.pos[2], mirror.normal_vector[0], mirror.normal_vector[1], mirror.normal_vector[2], color = 'r', arrow_length_ratio = 0.1)   
        except RuntimeError:
            print(f"Run time error for {mirror}")

    for ray in sun.create_rays():
        ax.scatter(ray.origin[0], ray.origin[1], ray.origin[2], color = 'b', s=1)
    ax.scatter(0,0,70, color = 'g', s=50)
    plt.show()
    print(grid)
    
    # create_animation("simple_animation.gif")

