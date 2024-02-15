import numpy as np
x = np.arange(1, 122)
x.shape=(int(np.sqrt(len(x))), int(np.sqrt(len(x))))
print(x)

bottom_left = x[np.arange(0,len(x[0])//2+1), 0:len(x[0])//2+1]
bottom_right = x[np.arange(0,len(x[0])//2+1), len(x[0])//2:len(x[0])]
top_left = x[np.arange(len(x[0])//2,len(x[0])), 0:len(x[0])//2+1]
top_right = x[np.arange(len(x[0])//2,len(x[0])), len(x[0])//2:len(x[0])]
print(top_left)
print(top_right)
print(bottom_left)
print(bottom_right)