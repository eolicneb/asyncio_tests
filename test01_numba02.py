#!/usr/bin/env python3
import scipy
from numba import jit
import numpy as np

dif = 1e-6
DIST_LIMIT = 1e-3
COUNT_LIMIT = 100
MAX_DIST = 30

deltas = np.eye(3)

@jit(nopython=True)
def normal(p: np.ndarray, DE, de: np.double, dif: np.double=1e-6) -> np.ndarray:
    # normal computation
    n = np.zeros(3)
    for i in range(3):
        d = deltas[i]
        q = p+d*dif
        n[i] = DE(q)
        
    n = (n-de)/dif
    leng = np.linalg.norm(n)
    n = n/leng
    return n
    
@jit(nopython=True)
def shadow(p, normal, light, DE, light_spread=0.001) -> float:
    if (normal*light).sum() < 0.:
        return 0.
    # need to start away from the surfice we are on
    q = p + normal*1*DIST_LIMIT 

    de = DE(q)
    away_dist = 0.
    min_de = 0.
    min_de_achieved = False
    for counter in range(COUNT_LIMIT):
        if not (de > DIST_LIMIT and away_dist < MAX_DIST):
            break

        away_dist += de
        q = q + light*de
        de = DE(q)

        if de < light_spread*away_dist:
            if min_de_achieved:
                min_de = min(min_de, de)
            else:
                min_de = de
                min_de_achieved = True
    
    if de < DIST_LIMIT:
        # when hit an eclipsing light object
        return 0.
        
    if not min_de_achieved:
        return 1.
    # otherwise return oclussion by nearby edges
    return min(1., (min_de/light_spread))

@jit(nopython=True)
def march(origin, direction, DE, light: np.ndarray) -> np.ndarray:
    """
    :params: origin
    :params: direction
    :params DE function: DE  # this function must take a (3,)np.ndarray and return a float
    :params (3,)np.ndarray: light  # the direction from where the ambient light comes
    """
    p = origin
    v = direction
    away_dist = 0.

    de = DE(p)
    dist_limit = False
    for counter in range(MAX_DIST):
        if not (de > DIST_LIMIT and away_dist < MAX_DIST):
            dist_limit = True
            break
        p = p + v*de
        away_dist = away_dist+de
        de = DE(p)
        
    if de > DIST_LIMIT:
        return np.zeros(3)

    n = normal(p, DE, de)
    m = np.absolute(n)

    # light incidence shading
    shade = n*light
    s = min(max(shade.sum(),0.),1.)

    # shadow by eclipsing
    shadowed = shadow(p, n, light, DE, light_spread=.1)

    return m*(.2+.8*(s+shadowed))

    

@jit(nopython=True)
def render(origin, target, DE, light, shape=(80,80), limits=((-1,1),(-1,1)), diag=1., space=None):

    # direction is normalized, comes from origin and points to the target
    direction = target - origin
    dir = direction/np.linalg.norm(direction)

      # march performs the render for every pixel in the screen
    image = np.zeros((len(space),3))
    for i in range(len(space)):
        p = space[i]
        image[i, :] = march(p,dir,DE,light)

    print('render finished')
    image = image.reshape(*shape,3)
    return image

import time
s = time.perf_counter()

@jit(nopython=True)
def fase_DE(p, fase):
    ax1 = np.array((.4, -1., .5))*np.sin(fase)
    ax2 = np.array((.6, .6, .1))*np.cos(fase)
    centre = np.array((.0, .2, .0))
    satelite = centre + ax1 + ax2

    a = np.linalg.norm(p - centre) - .5
    b = np.linalg.norm(p - satelite) - .3
    c = ((p-np.array((0,0,-1)))*np.array((0, .1, 1.))).sum()
    return min((a,b,c))


origin = np.array((0.,0.,5.))
target = np.array((0.,0.,0.))
light = np.array((1.,1.,1.))
eye = np.array((.2,.0,.0))

shape = (400, 400)
limits = ((-2,1.5),(-2,1.5))

# space is a list of points in the space from where the ray-marching starts
x = np.flip(np.linspace(*limits[0], shape[0]))
y = np.linspace(*limits[1], shape[1])
xyz_meshgrid = np.meshgrid(y, x, (10,), indexing='xy')
xyz_vector = [v.reshape(-1,1,1) for v in xyz_meshgrid]
space = [np.array(q).reshape(3) for q in zip(*xyz_vector)]

# light needs to be normalized
light_len = np.linalg.norm(light)
light = light/light_len

def new_image(fase):
    # fase = -.2

    @jit(nopython=True)
    def comp_DE(p):
        return fase_DE(p, fase)

    # comp_DE = fase_DE

    kws = dict(origin=origin+eye, target=target, 
                DE=comp_DE, light=light, 
                shape=shape, limits=limits,
                space=space)
    imageL = render(**kws)

    kws = dict(origin=origin-eye, target=target, 
                DE=comp_DE, light=light, 
                shape=shape, limits=limits,
                space=space)
    imageR = render(**kws)

    image = np.concatenate([imageL,imageR], axis=1)
    return image


f = time.perf_counter()
elapsed = f - s
print(f"{__file__} executed in {elapsed:0.6f} seconds.")

import matplotlib.pyplot as plt    
from matplotlib.animation import FuncAnimation
from os import rename

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
im = ax.imshow(new_image(0.))

def update(i):
    im.set_data(new_image(i))
    return im,

file = str(time.time())+'.gif'
p = time.perf_counter()
fnAn = FuncAnimation(fig, update, frames=np.linspace(np.pi/12,2*np.pi,24), interval=.02, blit=True, repeat=True)
fnAn.save(file, dpi=80, writer='imagemagick')
elapsed = int(time.perf_counter() - p)
# rename(file, f'numbatest_eta{elapsed:d}s.gif')

# plt.imshow(new_image(.6))

# plt.show()
