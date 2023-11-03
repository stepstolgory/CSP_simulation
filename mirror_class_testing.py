import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')



class Mirror():
    
    theta_x = np.radians(30)
    theta_y = np.radians(30)
    #R is the rotation matrix for the orientation matrix of the mirror
    R = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                  [np.sin(theta_x)*np.sin(theta_y), np.cos(theta_x), -np.sin(theta_x)*np.cos(theta_y)],
                  [-np.sin(theta_y)*np.cos(theta_x), np.sin(theta_x),  np.cos(theta_x)*np.cos(theta_y)]])

    def __init__(self, x, y, z, x_len, y_len, reflectivity):
        #x, y, z define the position of the mirror
        self.x = x
        self.y = y
        self.z = z

        self.x_len = x_len
        self.y_len = y_len
        
        self.points = np.array([[x,y,z],[x-x_len/2,y-y_len/2,z],[x-x_len/2, y+y_len/2, z],[x+x_len/2, y-y_len/2, z],[x+x_len/2, y+y_len/2, z]])
        self.rotated_points = [np.dot(self.R, point) for point in self.points]
        self.vector1 = self.rotated_points[0] - np.array([x,y,z])
        self.vector2 = self.rotated_points[1] - np.array([x,y,z])
        self.reflectivity = reflectivity
    @property
    def verteces(self):
        return self.rotated_points
    @property
    def normal_vector(self):
        normal = np.cross(self.vector1, self.vector2)
        return normal/np.linalg.norm(normal)
    @property
    def plot_values(self):
        x_vals = [point[0] for point in self.rotated_points]
        y_vals = [point[1] for point in self.rotated_points]
        z_vals = [point[2] for point in self.rotated_points]
        return x_vals, y_vals, z_vals
    def __str__(self):
        return(f"Mirror at position ({self.x}, {self.y}, {self.z}), dimension {self.x_len, self.y_len}, rotation about x axis of {Mirror.theta_x} radians, and rotation about y of {Mirror.theta_y} radians")

mirror1 = Mirror(2,3,2,10,7,0.2)
ax.plot_trisurf(mirror1.plot_values[0], mirror1.plot_values[1], mirror1.plot_values[2], color='blue', alpha = mirror1.reflectivity)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
print(mirror1)