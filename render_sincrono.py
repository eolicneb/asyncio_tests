#!/usr/bin/env python3
import asyncio
import numpy as np


dif = 1e-8
DIST_LIMIT = 1e-6
COUNT_LIMIT = 100
MAX_DIST = 30

deltas = [np.zeros(3) for _ in range(3)]
for i in range(3):
    deltas[i][i] = 1.0
    
# asyncio.get_current_loop()

class SphereDE():
    def __init__(self, radius=1., pos=np.array((0,0,0))):
        self.radius = radius
        self.pos = pos
    def __call__(self, P: np.ndarray) -> float:
        rod = P - self.pos
        dist = np.linalg.norm(rod)
        de = dist - self.radius
        color = np.array([abs(rod[i])/dist<0.8 for i in range(3)], dtype=np.double)
        normal = rod/dist
        return de, color, normal

class PlaneDE():
    def __init__(self, normal=np.array((0,0,1)), pos=np.array((0,0,0))):
        self.pos = pos
        self.normal = normal/np.linalg.norm(normal)
    def __call__(self, p: np.ndarray) -> float:
        rod = p - self.pos
        de = (self.normal*rod).sum()
        normal = self.normal 
        if de < 0:
            de, normal = -de, -normal
        color = np.array((0,0,1), dtype=np.double)
        return de, color, normal

class ComposeDE():
    def __init__(self, bodys=[]):
        self.bodys = bodys # each element in bodys must by a DE function
    
    def __call__(self, p: np.ndarray) -> float:
        # des = np.array([DE(p) for DE in self.bodys])
        # return des.min()
        des_ = [DE(p) for DE in self.bodys]
        des, colors, normals = zip(*des_)
        min_de = np.argmin(des)
        return des[min_de], colors[min_de], normals[min_de]

def normal(p, DE, de=None, dif=1e-6):
    if not de:
        de, color, nor = DE(p)
        if nor:
            return nor
    v123_ = [DE(p+d*dif) for d in deltas]
    v123, _, _ = zip(*v123_)
    t123 = (np.array(v123)-de)/dif
    n123 = t123/np.linalg.norm(t123)
    return n123

def ambient_light(normal: np.ndarray, light: np.ndarray) -> float:
    shade = (normal*light).sum()
    return max(0.0, shade)

def shadow(p, normal, light, DE, light_spread=0.001) -> float:
    if (normal*light).sum() < 0.:
        return 0.
    # need to start away from the surfice we are on
    q = p + normal*1*DIST_LIMIT 

    de, color, normal = DE(q)
    away_dist = 0.
    min_de = None
    counter = 0
    while de > DIST_LIMIT and counter < COUNT_LIMIT and away_dist < MAX_DIST:
        away_dist += de
        q = q + light*de
        de, color, normal = DE(q)
        if de < light_spread*away_dist:
            if min_de:
                min_de = min(min_de, de)
            else:
                min_de = de
        counter += 1
    
    if de < DIST_LIMIT:
        # when hit an eclipsing light object
        return 0.
    if not min_de:
        return 1.
    # otherwise return oclussion by nearby edges
    return min(1., min_de/light_spread)

def march(origin, direction, DE, light: np.ndarray) -> np.ndarray:
    """
    :params: origin
    :params: direction
    :params DE function: DE  # this function must take a (3,)np.ndarray and return a float
    :params (3,)np.ndarray: light  # the direction from where the ambient light comes
    """
    p, v = origin, direction
    counter, away_dist =  0, 0.
    de, color, nor = DE(p)
    while de > DIST_LIMIT and counter < COUNT_LIMIT and away_dist < MAX_DIST:
        p = p + v*de
        counter += 1
        de, color, nor = DE(p)
        
    if de > DIST_LIMIT:
        return np.zeros(3)

    n = normal(p, DE, de) if nor is None else nor
    shaded = ambient_light(n, light)
    shadowed = shadow(p, n, light, DE, light_spread=.2)

    return color*(shaded*shadowed*.8 + .2)

class Screen():
    def __init__(self, origin, target, shape=(80,80), diag=1.):
        self.origin = origin
        self.target = target
        self.main_axe = target - origin
        self.shape = shape
        a_shape = np.array(shape, dtype=np.double)
        unit = diag/np.linalg.norm(a_shape)
        horiz_ = np.cross(self.main_axe, np.array((0,0,1)))
        verti_ = np.cross(horiz_, self.main_axe)
        self.x = unit*horiz_/np.linalg.norm(horiz_)
        self.y = unit*verti_/np.linalg.norm(verti_)
    def __iter__(self):
        yr, xr = self.shape
        x0, y0 = xr/2, yr/2
        self.iterable = ((-x0+n%xr, y0-n//xr) for n in range(xr*yr))
        return self
    def __next__(self):
        i, j = next(self.iterable)
        v = self.main_axe + i*self.x + j*self.y
        return v/np.linalg.norm(v)

def smooth(image):
    new_image = image[:-1,:-1,:] + image[:-1,1:,:] + image[1:,:-1,:] + image[1:,1:,:]
    return new_image/4.

def render(screen, DE, light):
    # light needs to be normalized
    light = light.copy()/np.linalg.norm(light)
    # march performs the render for every pixel in the screen
    image = [march(screen.origin,dir,DE,light) for dir in screen]
    print('render finished')
    image = np.array(image).reshape(*screen.shape,3)
    return smooth(image)

def stereo(kwargsL, kwargsR):
    return render(**kwargsL), render(**kwargsR)

if __name__ == "__main__":
    import time
    s = time.perf_counter()

    sph_DE = SphereDE(radius=.5, pos=np.array((.0, .4, .0)))
    sph2_DE = SphereDE(radius=.3, pos=np.array((.3, .3, .7)))
    pln_DE = PlaneDE(normal=np.array((0, .1, 1.)), pos=np.array((0,0,-1)))
    comp_DE = ComposeDE([sph_DE, sph2_DE, pln_DE])

    origin = np.array((5.,-2.,2.))
    target = np.array((0.,-.3,0.))
    light = np.array((1.,1.,1.))
    eye = np.array((.6,.0,.0))

    shape = (400, 340)
    diagonal = 4.5

    stereo_kws = (dict(screen=Screen(origin=origin+eye,
                                     target=target,
                                     shape=shape,
                                     diag=diagonal), 
                       DE=comp_DE, 
                       light=light),
                  dict(screen=Screen(origin=origin-eye,
                                     target=target,
                                     shape=shape,
                                     diag=diagonal), 
                       DE=comp_DE, 
                       light=light))
    
    imageL, imageR = stereo(*stereo_kws)

    # image = (render(**stereo_kws[0]))

    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.6f} seconds.")
    # print(f'image size {image.shape}')
    import matplotlib.pyplot as plt
    plt.imshow(np.concatenate([imageL,imageR], axis=1))
    # plt.imshow(image)
    plt.show()