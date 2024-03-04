import numpy as np
A = np.array([0,10,0])
B = np.array([10,0,0])
p = np.array([-3.42857128, 2.57142754,0])
int_point = np.array([71.57142872, -72.42857246, 0])
point1 = np.array([70, -70, 0])
point2 = np.array([80,-80, 0])
mirror = np.array([75, -75, 0])
matrix = np.column_stack((A, B))
print(np.linalg.lstsq(matrix, p, rcond=None))
print(matrix)
if abs(int_point[0] - mirror[0]) <= 5 and abs(int_point[1]-mirror[1]) <= 5:

    print("hits mirror")
# print(u,v)