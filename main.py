#Playing around with classes and light physics
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
class Sun():
    #Takes a slightly eliptical shaped path across the sky
    #Calculate position in 15 minute intervals
    #Position in 2 dimensions
    #Assume longest day of summer for the rise and set times
    #x = 50cost, y = sint
    x = 100
    y = 0
    position = np.array([x, y], dtype=float)

    def __init__(self, rise_time, set_time, intensity):
        self.rise_time = rise_time
        self.set_time = set_time
        self.intensity = intensity

    def move(self,t):
        #Takes a time t, to determine the position of the sun and change it
        self.x, self.y = 100*np.cos((np.pi*t/(self.day_time))), 0.5*np.sin(np.pi*t/(self.day_time))
        return self.x, self.y

    @property
    def day_time(self):
        return self.set_time-self.rise_time
    
    def create_ray(self):
        pass

    def __repr__(self):
        return (f"Sun({self.rise_time}, {self.set_time}, {self.intensity}, {self.x, self.y})")
    
class Ray(Sun):
    #Find vector towards given mirror, that is the ray
    #Inherits the position of the sun
    def __init__(self):
        pass 

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
        self.reflectivity = reflectivity

sun = Sun(360, 1220, 10)

fig, ax = plt.subplots(1,1)
fig.set_size_inches(5,5)
x = []
y = []


def animate(i):
    # Get the point from the points list at index i
    # Plot that point using the x and y coordinates
    ax.plot(x[i], y[i], color='orange', 
            label='original', marker='o')
    # Set the x and y axis to display a fixed range
    ax.set_xlim([-100, 100])
    ax.set_ylim([0, 1])

for t in range(0, sun.day_time+1, 15):
    x_pos, y_pos = sun.move(t)
    x.append(x_pos)
    y.append(y_pos)
    print(sun)

ani = FuncAnimation(fig, animate, frames=len(x),
                    interval=500, repeat=False)

plt.close()

ani.save("simple_animation.gif", dpi = 300, writer=PillowWriter(fps=5))
