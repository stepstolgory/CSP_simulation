import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

class Mirror():
    
    theta_x = 0
    theta_z = 0
    #R is the rotation matrix for the orientation matrix of the mirror
    R = np.array([[np.cos(theta_z), np.sin(theta_z), 0],
                  [np.cos(theta_x)*np.sin(theta_z), np.cos(theta_x)*np.cos(theta_z), -np.sin(theta_x)],
                  [np.sin(theta_x)*np.sin(theta_z), np.cos(theta_x)*np.sin(theta_z),  np.cos(theta_x)]])

    def __init__(self, x, y, z, x_len, y_len, reflectivity):
        #x, y, z define the position of the mirror
        self.x = x
        self.y = y
        self.z = z
        self.x_len = x_len
        self.y_len = y_len
        self.x_ = np.array([x_len,0,0])
        self.y_ = np.array([0,y_len,0])
        self.z_ = np.array([0,0,0])
        self.reflectivity = reflectivity
    @property
    def plane_eq(self):
        return self.x_+self.y_+self.z_

mirror1 = Mirror(0,0,0,10,10,0.95)
plt.show()
