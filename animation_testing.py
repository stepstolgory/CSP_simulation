import numpy as np
import matplotlib.pyplot as plt

xs = [i for i in range(-10,10,1)]
ys = [i for i in range(-10,10,1)]
resolution = 100
points1 = []
points2 =[]
for x in xs:
    for y in ys:
        if 6*x**2+6*y**2-150 == 0:
            points1.append((x,y))
for x in xs:
    for y in ys:
        if 12*x*y-9*y**2 == 0:
            points2.append((x,y))

stationary_points = [e for e in points1 if e in points2]
print(stationary_points)

