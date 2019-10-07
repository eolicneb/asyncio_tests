#!/usr/bin/env python3


import asyncio
import aiohttp
import numpy as np

dif = 1e-6
DIST_LIMIT = 1e-3
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
    async def __call__(self, P: np.ndarray) -> float:
        return np.linalg.norm(P - self.pos) - self.radius

class PlaneDE():
    def __init__(self, normal=np.array((0,0,1)), pos=np.array((0,0,0))):
        self.pos = pos
        self.normal = normal/np.linalg.norm(normal)
    async def __call__(self, p: np.ndarray) -> float:
        return ((p-self.pos)*self.normal).sum()

class ComposeDE():
    def __init__(self, bodys=[]):
        self.bodys = bodys # each element in bodys must by a DE function
    
    async def __call__(self, p: np.ndarray) -> float:
        des = await asyncio.gather(*[DE(p) for DE in self.bodys])
        return min(des)

async def normal(p, DE, de=None, dif=1e-6):
    if not de:
        de = await DE(p)
    v123 = await asyncio.gather(*[DE(p+d*dif) for d in deltas])
    t123 = (np.array(v123)-de)/dif
    n123 = t123/np.linalg.norm(t123)
    return n123

async def ambient_light(normal: np.ndarray, light: np.ndarray) -> float:
    shade = (normal*light).sum()
    return max(0.0, shade)

async def shadow(p, normal, light, DE, light_spread=0.001) -> float:
    if (normal*light).sum() < 0.:
        return 0.
    # need to start away from the surfice we are on
    q = p + normal*1*DIST_LIMIT 

    de = await DE(q)
    away_dist = 0.
    min_de = None
    counter = 0
    while de > DIST_LIMIT and counter < COUNT_LIMIT/5 and away_dist < MAX_DIST:
        away_dist += de
        q = q + light*de
        de = await DE(q)
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
    return min(1., min_de/away_dist/light_spread)

async def march(origin, direction, DE, light: np.ndarray) -> np.ndarray:
    """
    :params: origin
    :params: direction
    :params DE function: DE  # this function must take a (3,)np.ndarray and return a float
    :params (3,)np.ndarray: light  # the direction from where the ambient light comes
    """
    p, v = origin, direction
    counter, away_dist =  0, 0.
    de = await DE(p)
    while de > DIST_LIMIT and counter < COUNT_LIMIT and away_dist < MAX_DIST:
        p = p + v*de
        counter += 1
        de = await DE(p)
        
    if de > DIST_LIMIT:
        return np.zeros(3)

    n = await normal(p, DE, de)
    shaded = await ambient_light(n, light)
    shadowed = await shadow(p, n, light, DE, light_spread=0.1)
    if shadowed == 0.:
        n = np.array((1.,0.,0.))
    return n*(shaded*shadowed*.8 + .2)

async def render(origin, target, DE, light, shape=(80,80), diag=1.):
    # direction is normalized, comes from origin and points to the target
    direction = target - origin
    dir_versor = direction/np.linalg.norm(direction)
    # light needs to be normalized
    light = light.copy()/np.linalg.norm(light)
    # space is a list of points in the space from where the ray-marching starts
    x = np.linspace(1, -1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    xyz_meshgrid = np.meshgrid(y, x, (10,), indexing='xy')
    xyz_vector = (v.reshape(-1,1,1) for v in xyz_meshgrid)
    space = (np.array(q).reshape(3) for q in zip(*xyz_vector))
    # march performs the render for every pixel in the screen
    image = await asyncio.gather(*[march(p,dir_versor,DE,light) for p in space])
    print('render finished')
    return np.array(image).reshape(*shape,3)

async def stereo(kwargsL, kwargsR):
    return await asyncio.gather(render(**kwargsL), render(**kwargsR))

if __name__ == "__main__":
    import time
    s = time.perf_counter()

    sph_DE = SphereDE(radius=.5, pos=np.array((.0, .4, .0)))
    pln_DE = PlaneDE(normal=np.array((0, .1, 1.)), pos=np.array((0,0,-1)))
    comp_DE = ComposeDE([sph_DE, pln_DE])

    origin = np.array((0.,0.,5.))
    target = np.array((0.,0.,0.))
    light = np.array((1.,1.,1.))
    eye = np.array((.2,.0,.0))

    size = (200, 200)

    stereo_kws = (dict(origin=origin+eye, target=target, 
                        DE=comp_DE, light=light, shape=size),
              dict(origin=origin-eye, target=target, 
                        DE=comp_DE, light=light, shape=size))
    
    # imageL, imageR = asyncio.run(stereo(*stereo_kws))

    image = asyncio.run(render(**stereo_kws[0]))

    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.6f} seconds.")
    # print(f'image size {image.shape}')
    import matplotlib.pyplot as plt
    # plt.imshow(np.concatenate([imageL,imageR], axis=1))
    plt.imshow(image)
    plt.show()