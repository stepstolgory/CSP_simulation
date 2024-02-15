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
        return(f"Mirror {self.id} at position ({self.x}, {self.y}, {self.z}), dimension {self.x_len, self.y_len}, rotation about x axis of {self.theta_x} radians, and rotation about y of {self.theta_y} radians")

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

class Tower():
    def __init__(self, height, g_height):
        self.height = height
        self.g_height = g_height
        self.grid_space = Grid('True', 75/7, 75/7, 0, 0).create_grid_space(0.01,0.01)
    
    rays = []
        
    @property
    def ray_coordinates(self):
        ray_coordinates = []
        for (x,y) in self.grid_space:
            coord = np.array([x,y,(self.height-self.g_height)])
            ray_coordinates.append(coord)
        return ray_coordinates

    def create_rays(self): 
        """
        Creates a python list of instances of the ray class
        The direction is between the tower and a point on the ray grid, and is normalised
        The magnitude is that which is required to reach z = 0 with the given direction
        """
        j = 0
        for coordinate in self.ray_coordinates:
            j+=1
            direction = coordinate - np.array([0,0,self.height])
            direction = direction/np.linalg.norm(direction)
            magnitude = np.sqrt(self.height**2+(49*(coordinate[0]**2+coordinate[1]**2)))
            self.rays.append(Ray(np.array([0,0,self.height]), direction, magnitude, j))
        print(f"{j} rays were created")
        return self.rays

class Sun(Mirror):
    """
    Takes a slightly eliptical shaped path across the sky
    Calculate position in 15 minute intervals
    Position in 2 dimensions
    Assume longest day of summer for the rise and set times
    """
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
    

    def __str__(self):
        return (f"Sun({self.rise_time}, {self.set_time}, {self.intensity}, {self.x, self.y})")

class Ray():
    """
    The controls for how many rays to create is located in the initialisation of the Tower class
    The lower the two values in "create grid" the more rays are created. However, they cannot equal to 0
    """
    def __init__(self, origin, direction, magnitude, id):
        self.origin = origin
        self.direction = direction
        self.magnitude = magnitude
        self.mirror_hit = None
        self.id = id

    @property
    
    def intersection_point(self):
        """
        Returns the point of intersection with the flat plane at z = 0
        Used to locate the closest mirror centre to the ray
        """
        point = np.array([self.origin[0]-((self.origin[2]*self.direction[0])/self.direction[2]),self.origin[1]-((self.origin[2]*self.direction[1])/self.direction[2]), 0])
        return point
    
    
    def rotated_intersection(self, mirror):
        """
        If the ray is not parallel to the rotated plane of the closest mirror, then returns the intercept of that ray with the plane. 
        Used to check whether the ray hits the mirror
        """
        if np.dot(self.direction, mirror.normal_vector) != 0:
            d = np.dot(np.array([mirror.x, mirror.y, mirror.z])-self.origin, mirror.normal_vector)/(np.dot(self.direction, mirror.normal_vector))
            point = self.origin + d*self.direction
            return point

    def closest_mirror(self, possible_mirrors):
        """
        Finds the closest mirror to the point of interception with the z=0 plane
        """
        closest_mirror = None
        min_distance = float('inf')
        int_pnt = self.intersection_point
        for mirror in possible_mirrors:
            distance = np.linalg.norm(int_pnt - np.array([mirror.x, mirror.y, mirror.z]))
            if distance <= min_distance:
                min_distance = distance
                closest_mirror = mirror
        return closest_mirror

    
    def is_point_in_mirror(self, mirror, intersection):
        """
        Determines whether the intersection of the ray with the infinite plane of the mirror is within the finite size
        Returns false if it is parallel , or if it is outside of the mirror
        Otherwise returns True
        A, B are the vectors for two perpendicular sides of the mirror
        p is the point in question
        """
        A = np.delete(mirror.rotated_points[2],3)-np.delete(mirror.rotated_points[1],3)
        B = np.delete(mirror.rotated_points[3],3)-np.delete(mirror.rotated_points[1],3)
        p = self.rotated_intersection(mirror) - np.delete(mirror.rotated_points[0],3)

        if not np.isclose(np.dot(mirror.normal_vector, p), 0):
            return False

        matrix = np.column_stack((A, B))
        u,v= np.linalg.lstsq(matrix, p, rcond=None)[0]
        inside = 0<=u<=1 and 0<=v<=1
        return inside

    def is_mirror_hit(self, mirrors):
        """
        Returns True if a mirror is hit
        Sets an attribute of the ray object to the mirror object that was hit
        This is to be used in reflection, and tracking the rays
        Returns False if the ray misses all of the mirrors
        """
        closest_mirror = self.closest_mirror(mirrors)
        if self.is_point_in_mirror(closest_mirror, self.rotated_intersection(closest_mirror)):
            self.mirror_hit = closest_mirror
            return True
        return False

    def __str__(self):
        return f"Ray number {self.id} with the starting point {self.origin}, direction {self.direction} and magnitude {self.magnitude}"

def animate(i):
    # Get the point from the points list at indemirrors i
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


 #Arrays for animation
    x = []
    z = []

def create_animation(path):
        for t in range(0, sun.day_time+1, 15):
            x_pos, z_pos = sun.move(t)
            x.append(x_pos)
            z.append(z_pos)
        ani = FuncAnimation(fig, animate, frames=len(x),
                            interval=500, repeat=False)
        ani.save(path, dpi = 300, writer=PillowWriter(fps=5))
        plt.close()

if __name__ == "__main__":
     
    #Equations for the grid shape
    all_mirrors = 'True'
    no_mirrors = 'False'
    circle_eq = '52.5**2 >= x**2+y**2 >= 20**2'
    square_eq = 'np.abs(x) > 45 or np.abs(y) > 45'

    sun = Sun(360, 1220, 10, 100, 0, 1, 150, 150, 0, 0, 0, 1, 0)
    sun.R = np.dot(sun.Rz,np.dot(sun.Rx, sun.Ry))
    tower = Tower(70, 10)
    sun.point_to_tower([0,0,0,0])
    grid = Grid(circle_eq, 75, 75, 2.5, 2.5)
    mirrors = grid.create_mirrors(0, 10, 10, 0, 0, 0, 0.5)
    for mirror in mirrors:
        #Plotting each mirror on the same axis
            mirror.R = np.dot(mirror.Rz,np.dot(mirror.Rx, mirror.Ry))
            """
            Uncomment the following line to point all of the mirrors to any point, leave the 4th value at 0
            """
            # mirror.point_to_tower([0,0,70,0])
            try:
                ax.plot_trisurf(mirror.plot_values[0], mirror.plot_values[1], mirror.plot_values[2], color = 'blue', alpha = mirror.reflectivity)  
            except RuntimeError:
                """
                The mirror straight under the tower vector wants to be rotated 180 degrees
                But that gives a RuntimeError
                As this mirror is technically inside of the tower, the mirror is removed.
                """
                mirrors = np.delete(mirrors, mirror.id - 1)
                print(f"Run time error for {mirror}")
    rays = tower.create_rays()
    useable_rays = np.array([])
    counter = 0
    for ray in rays:
        if ray.is_mirror_hit(mirrors):
            """
            In order of commented lines:
            1. Print out which ray hit which mirror, useful for debugging
            2. Plot each of the rays that hits a mirror (really slow and cluttered for anything below (1,1) in the setting)
            3. Creates a new arrays only with the rays that hit mirrors
            """
            # print(f"Ray {ray.id} hit mirror {ray.mirror_hit}")
            #ax.quiver(ray.origin[0], ray.origin[1], ray.origin[2], ray.direction[0], ray.direction[1], ray.direction[2], color = 'b', length = 1.5*ray.magnitude)
            #useable_rays = np.append(useable_rays, [ray])
            counter +=1    
            
            

    ax.scatter(0,0,70, color = 'g', s=50)
    print(f"{counter} rays were plotted")
    print(f"{(counter/len(rays))*100}% of rays hit the mirrors")
    plt.show()
    #Line below creates and animation of the sun moving across, that is currently saved under simple_animation.gif
    #create_animation("simple_animation.gif")
