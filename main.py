import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from alive_progress import alive_bar
import random

"""TODO: Create a function to reflect the light rays from the mirrors.
         This function needs to calculate the direction of the ray and whether it intersects with the Sun mirror 
         Test the created function thoroughly
         Test Seans idea   
         """
        

"""Set up for the plot """
fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.view_init(elev=90, azim=0, roll=0)
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

        #Non rotated verteces of the mirror
        self.points = np.array([[x,y,z,1],[x-x_len/2,y-y_len/2,z,1],[x-x_len/2, y+y_len/2, z,1],[x+x_len/2, y-y_len/2, z,1],[x+x_len/2, y+y_len/2, z,1]])
        self.reflectivity = reflectivity

        self.id = id
    
    @property
    def vectors(self):
        """
        The cross product is found with these vectors
        The vectors are from bottom left corner to centre and top left corner to centre
        """
        vector1 = self.rotated_points[1]-self.pos
        vector1 = vector1/np.linalg.norm(vector1)
        vector2 = self.rotated_points[2]-self.pos
        vector2 = vector2/np.linalg.norm(vector2)
        return vector1, vector2

    @property
    def pos(self): 
        """Returns the centre of the mirror as a quaternion"""
        return np.array([self.x, self.y, self.z, 1])
    
    @property
    def rotated_points(self): 
        """Returns the rotated points
           Rotation done by translating the point to the origin, rotating it and then translating it back"""
        T1 = self.T(-self.x, -self.y, -self.z)
        T2 = self.T(self.x, self.y, self.z)
        M = np.dot(T2, np.dot(self.R, T1))
        rotated_points = [np.dot(M, point) for point in self.points] 
        return rotated_points
    
    @property
    def normal_vector(self):
        """Finds and normalises the normal vector"""
        normal = np.cross(np.delete(self.vectors[1],3), np.delete(self.vectors[0],3))
        return normal/np.linalg.norm(normal)

    @property
    def plot_values(self): 
        """Returns the values needed to plot the mirror as a square using trisurface as three separate lists of x, y, z values"""
        x_vals = [point[0] for point in self.rotated_points]
        y_vals = [point[1] for point in self.rotated_points]
        z_vals = [point[2] for point in self.rotated_points]
        return np.array([x_vals, y_vals, z_vals])

    def axis_rotation(self, axis, ang):
        """Performs a rotation about any given axis by a given angle"""
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
        """Rotation matrix about x axis"""
        return np.array([[1,0,0,0],  
                [0, np.cos(self.theta_x), -np.sin(self.theta_x), 0],
                [0, np.sin(self.theta_x), np.sin(self.theta_x), 0],
                [0,0,0,1]])

    @property
    def Ry(self):
        """Rotation matrix about y axis"""
        return np.array([[np.cos(self.theta_y), 0, np.sin(self.theta_y), 0],
                [0, 1, 0, 0],
                [-np.sin(self.theta_y), 0, np.cos(self.theta_y), 0],
                [0,0,0,1]])

    @property
    def Rz(self):
        """Rotation matrix about z axis"""
        return np.array([[np.cos(self.theta_z), -np.sin(self.theta_z), 0, 0],
                [np.sin(self.theta_z), np.cos(self.theta_z), 0, 0],
                [0,0,1,0],
                [0,0,0,1]])
   
    @property
    def R(self):
        """Total rotation matrix"""
        return self._R
    
    @R.setter
    def R(self, value):
        self._R = value

    def T(self, x, y, z):
        """Translation matrix"""
        return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z], [0,0,0,1]])
    
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

    """WIP - This is a test of what Sean wrote ages ago for pointing a ray to tower"""
    def ray_to_tower(self, tower, sun):
        theta = np.arctan((tower.y+sun.y-2*self.y)/(tower.x+sun.x-2*self.x))
        phi = np.arctan(np.sqrt((tower.x+sun.x-2*self.x)**2+(tower.y+sun.y-2*self.y)**2)/(tower.z+sun.z-2*self.z))
        
        y_axis = np.array([0,1,0])
        z_axis = np.array([0,0,1])
        
        y_rotation = self.axis_rotation(y_axis, theta)
        z_rotation = self.axis_rotation(z_axis, phi)

        self.R = np.dot(y_rotation, z_rotation)

    def __str__(self):
        return(f"Mirror {self.id} at position ({self.x}, {self.y}, {self.z}), dimension {self.x_len, self.y_len}, rotation  of {self.theta_x, self.theta_y, self.theta_z} radians about x, y, z axis respectively.")

class Grid():
    def __init__(self, eq, size_x, size_y, margin_x, margin_y):
        self.eq = eq
        self.size_x = size_x
        self.size_y = size_y
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.mirrors_used = 0

    def create_grid_space(self, x_len, y_len):
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

    def create_mirrors(self, z, x_len, y_len, theta_x, theta_y, theta_z, reflectivity):
        """
        Creates a mirror object for every space in the grid
        Returns a list of the objects
        """
        mirrors = []
        grid = self.create_grid_space(x_len, y_len)
        i=0
        for (x,y) in grid:
            i+=1
            #Creates a mirror object for every allowed position within the grid
            mirrors.append(Mirror(x, y, z, x_len, y_len, theta_x, theta_y, theta_z, reflectivity, i)) 
        return mirrors

    def __str__(self):
        return f"A grid of mirror locations limited by equation '{self.eq}', size {self.size_x, self.size_y} and {self.mirrors_used} mirrors used."

class Tower():
    def __init__(self, x, y, z, g_height):
        self.x = x
        self.y = y
        self.z = z
        self.g_height = g_height
        """Change the values in 'create_grid_space' to alter density of rays"""
        self.grid_space = Grid('True', np.sqrt(75**2+x**2)/(z/g_height), np.sqrt(75**2+x**2)/(z/g_height), 0, 0).create_grid_space(1,1)
    
    rays = []
        
    @property
    def ray_coordinates(self):
        """Gives the origin coordinates for rays just below the tower"""
        ray_coordinates = []
        for (x,y) in self.grid_space:
            coord = np.array([x+self.x,y+self.y,(self.z-self.g_height)])
            ray_coordinates.append(coord)
        return ray_coordinates

    def create_rays(self): 
        """
        Creates a list of instances of the ray class
        The direction is between the tower and a point on the ray grid, and is normalised
        The magnitude is kept constant at 1 for now
        """
        j = 0
        for coordinate in self.ray_coordinates: 
            j+=1
            direction = coordinate - np.array([self.x, self.y, self.z])
            direction = direction/np.linalg.norm(direction)
            magnitude = 1
            self.rays.append(Ray(np.array([coordinate[0],coordinate[1],self.z]), direction, magnitude, j))
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
        """
        Takes a time t, to determine the position of the sun and change it
        """
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
        By going through all of the possible mirrors and calculating the distance between the centre
        and the point of intersection.
        (This is highly inefficient, pls give ideas on how to improve it)
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
        Does this by seeing if the point is within half of the size away in positive or negative direction, in both y and x
        """
        inside = abs(intersection[0] - mirror.x) <= mirror.x_len/2 and abs(intersection[1]-mirror.y) <= mirror.y_len/2
        return inside

    def is_mirror_hit(self, mirrors):
        """
        Returns True if a mirror is hit
        Sets an attribute of the ray object to the mirror object that was hit
        This is to be used in reflection, and tracking the rays
        Returns False if the ray misses all of the mirrors
        """
        closest_mirror = self.closest_mirror(mirrors)
        int_point = self.rotated_intersection(closest_mirror)
        self.magnitude = np.sqrt((closest_mirror.x-self.origin[0])**2 + (closest_mirror.y-self.origin[1])**2 + (closest_mirror.z-self.origin[2])**2)
        if self.is_point_in_mirror(closest_mirror, int_point):
            self.mirror_hit = closest_mirror
            return True, int_point
        return False, int_point

    @classmethod
    def reflect(cls, ray, mirror):
        """
        Performs a reflection on the mirror
        Uses the same axis rotation matrix as the 'point_to_tower' method of the mirror class
        WIP - needs more testing
        """
        incidence_angle = np.arccos(np.dot(ray.direction, mirror.normal_vector))
        rotation_angle = (np.pi - 2*incidence_angle)
        axis = np.cross(ray.direction, mirror.normal_vector)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        reflection_matrix = np.array([[x*x*(1-np.cos(rotation_angle))+np.cos(rotation_angle), y*x*(1-np.cos(rotation_angle))-z*np.sin(rotation_angle), z*x*(1-np.cos(rotation_angle))+y*np.sin(rotation_angle)],
                            [x*y*(1-np.cos(rotation_angle))+z*np.sin(rotation_angle), y*y*(1-np.cos(rotation_angle))+np.cos(rotation_angle), z*y*(1-np.cos(rotation_angle))-x*np.sin(rotation_angle)],
                            [x*z*(1-np.cos(rotation_angle)) - y*np.sin(rotation_angle),  y*z*(1 - np.cos(rotation_angle)) + x*np.sin(rotation_angle), z*z*(1 - np.cos(rotation_angle)) + np.cos(rotation_angle)]])
        ray.direction = np.dot(ray.direction, reflection_matrix)

    def __str__(self):
        return f"Ray number {self.id} with the starting point {self.origin}, direction {self.direction} and magnitude {self.magnitude}"


"""These two functions are outdated, will update later"""
def animate(i):
    ax.clear()
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    sun = Sun(360, 1220, 10, x[i], 0, z[i], 150, 150, 0, 0, 0, 1, 0)
    tower = Tower(0, 0, 70, 10)
    sun.R = np.dot(sun.Rz,np.dot(sun.Rx, sun.Ry))

    sun.point_to_tower([0,0,0,0])
    ax.plot_trisurf(sun.plot_values[0], sun.plot_values[1], sun.plot_values[2], color="orange", alpha = sun.reflectivity)

    for mirror in mirrors:
        #Plotting each mirror on the same axis
        mirror.R = np.dot(mirror.Rz,np.dot(mirror.Rx, mirror.Ry))
        mirror.ray_to_tower(tower, sun)
        try:
            ax.plot_trisurf(mirror.plot_values[0], mirror.plot_values[1], mirror.plot_values[2], color = 'blue', alpha = mirror.reflectivity)   
        except RuntimeError:
            print(f"Run time error for {mirror}")



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
     
    """Equations for the grid shape"""
    all_mirrors = 'True'
    no_mirrors = 'False'
    circle_eq = '52.5**2 >= x**2+y**2 >= 20**2'
    square_eq = 'np.abs(x) > 45 or np.abs(y) > 45'

    """Sun setup"""
    sun = Sun(360, 1220, 10, 100, 0, 200, 150, 150, 0, 0, 0, 1, 0)
    sun.R = np.dot(sun.Rz,np.dot(sun.Rx, sun.Ry))
    sun.point_to_tower([0,0,0,0])
    #ax.plot_trisurf(sun.plot_values[0], sun.plot_values[1], sun.plot_values[2], color="orange", alpha = sun.reflectivity)

    """Tower setup"""
    tower = Tower(0, 0, 70, 10)
    tower2 = Tower(75, 75, 70, 10)
    ax.scatter(tower.x,tower.y,tower.z, color = 'g', s=50)
    ax.scatter(tower2.x,tower2.y,tower2.z, color = 'r', s=50)

    """ Rays setup"""
    rays = tower.create_rays()
    useable_rays = []

    """Grid of mirrors setup"""
    #Change the first parameter to one of the equations to alter the pattern of the mirror field
    grid = Grid(all_mirrors, 75, 75, 2.5, 2.5)
    mirrors = grid.create_mirrors(0, 10, 10, 0, 0, 0, 0.5)

    """Calculations and plotting"""
    for mirror in mirrors:
            mirror.R = np.dot(mirror.Rz,np.dot(mirror.Rx, mirror.Ry))
            """
            Use the first line to point all of the mirrors to any point, leave the 4th value at 0
            Use the second line to point rays to any point WIP not 100% sure this works yet
            """
            #mirror.point_to_tower([tower2.x, tower2.y, tower2.z, 0])
            mirror.ray_to_tower(tower2, tower)

            try:
                ax.plot_trisurf(mirror.plot_values[0], mirror.plot_values[1], mirror.plot_values[2], color = 'r', alpha = mirror.reflectivity) 
            except RuntimeError:
                """
                The mirror straight under the tower vector wants to be rotated 180 degrees
                But that gives a RuntimeError
                As this mirror is technically inside of the tower, the mirror is removed.
                """
                mirrors = np.delete(mirrors, mirror.id - 1)
                print(f"Run time error for {mirror}")
    
    with alive_bar(len(rays)) as bar:
        for ray in rays:
            hit_check = ray.is_mirror_hit(mirrors)
            if hit_check[0]:
                """
                In order of commented lines:
                1. Print out which ray hit which mirror, useful for debugging
                2. Plot each of the rays that hits a mirror (really slow and cluttered for anything below (0.1,0.1) in the setting)
                3. In tangent with the scatter in the else clause, displays the points at which the rays hit the plane of the mirrors
                   and whether it hit a mirror (green) or not (red)
                """
                # print(f"Ray {ray.id} hit mirror {ray.mirror_hit}")
                #ax.quiver(ray.origin[0], ray.origin[1], ray.origin[2], ray.direction[0], ray.direction[1], ray.direction[2], color = 'b', length = ray.magnitude, arrow_length_ratio = 0.1)
                #ax.scatter(ray.intersection_point[0], ray.intersection_point[1], ray.intersection_point[2], color='g')
                useable_rays.append([ray, hit_check[1]])
            else:
                #ax.quiver(ray.origin[0], ray.origin[1], ray.origin[2], ray.direction[0], ray.direction[1], ray.direction[2], color = 'r', length = ray.magnitude, arrow_length_ratio=0.1)
                #ax.scatter(ray.intersection_point[0], ray.intersection_point[1], ray.intersection_point[2], color='r',s=0.1)
                pass
            bar()

    """Reflection calculations WIP"""
    with alive_bar(len(useable_rays)) as bar:
        for ray in useable_rays:
            Ray.reflect(ray[0], ray[0].mirror_hit)
            #ax.quiver(ray[1][0], ray[1][1], ray[1][2], ray[0].direction[0], ray[0].direction[1], ray[0].direction[2], color = 'g', length = 1.5*ray[0].magnitude, arrow_length_ratio = 0.1)
            bar()

    print(f"{len(useable_rays)} rays were plotted")
    print(f"{(len(useable_rays)/len(rays))*100}% of rays hit the mirrors")
    plt.show()
    
    """Animation creation toggle"""
    #Line below creates and animation of the sun moving across, that is currently saved under simple_animation.gif
    x = []
    z = []
    #create_animation("simple_animation.gif")