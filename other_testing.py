import numpy as np

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin  # Starting point of the ray (x, y)
        self.direction = direction / np.linalg.norm(direction)  # Unit vector representing ray's direction

class Mirror:
    def __init__(self, position, normal):
        self.position = position  # Mirror's position (x, y)
        self.normal = normal / np.linalg.norm(normal)  # Unit vector representing the mirror's normal

def reflect(ray, mirror):
    # Calculate the incident angle between the ray and mirror normal
    incident_angle = np.arccos(np.dot(ray.direction, mirror.normal))
    
    # Calculate the reflection angle
    reflection_angle = np.pi - incident_angle
    
    # Calculate the reflected ray's direction using the reflection angle
    reflected_direction = ray.direction - 2 * np.dot(ray.direction, mirror.normal) * mirror.normal
    
    # Create and return a new reflected ray
    reflected_ray = Ray(ray.origin, reflected_direction)
    
    return reflected_ray

# Define the starting point and direction of the incident ray
incident_origin = np.array([0, 0])
incident_direction = np.array([1, 1])

# Create an incident ray
incident_ray = Ray(incident_origin, incident_direction)

# Define the mirror's position and normal vector
mirror_position = np.array([1, 0])
mirror_normal = np.array([-1, 0])

# Create a mirror
mirror = Mirror(mirror_position, mirror_normal)

# Simulate ray reflection
reflected_ray = reflect(incident_ray, mirror)

# Print the results
print("Incident Ray:")
print("Origin:", incident_ray.origin)
print("Direction:", incident_ray.direction)

print("\nMirror:")
print("Position:", mirror.position)
print("Normal:", mirror.normal)

print("\nReflected Ray:")
print("Origin:", reflected_ray.origin)
print("Direction:", reflected_ray.direction)
