from test01_sincrono import Screen
import numpy as np

origin, target = np.array((2, 3, 10)), np.array((1,-2, 4))
scr = Screen(origin, target, shape=(5,3), diag=3.)

x, y = scr.x, scr.y
nx, ny = np.linalg.norm(x), np.linalg.norm(y)

print(x, nx)
print(y, ny)
print(np.dot(x, y)/nx/ny)

diag = x*scr.shape[0] + y*scr.shape[1]
print(np.linalg.norm(diag))

for direction in scr:
    print(direction)

