#!/usr/bin/env python3


import asyncio
import aiohttp
import numpy as np

dif = 1e-6
DIST_LIMIT = 1e-3
COUNT_LIMIT = 100

# asyncio.get_current_loop()

class SphereDE():
    def __init__(self, radius=1., pos=np.array((0,0,0))):
        self.radius = radius
        self.pos = pos
    async def __call__(self, P: np.ndarray) -> float:
        return np.linalg.norm(P - self.pos) - self.radius
    
async def march(origin, direction, DE):
    v = direction/np.linalg.norm(direction)
    p, counter = origin, 0
    de = await DE(p)
    while de > DIST_LIMIT and counter < COUNT_LIMIT:
        p = p + v*de
        counter += 1
        de = await DE(p)
    # print(f'for origin {origin} the surface is {p}')
    z = p[2] if de <= DIST_LIMIT else 0
    return z

async def count():
    print("one")
    await asyncio.sleep(1)
    print("two")
    await asyncio.sleep(1)
    print("three")

async def main(DE):
    points = 200
    x = np.linspace(-1, 1, points)
    xyz_meshgrid = np.meshgrid(x, x, (10,))
    xyz_vector = (v.reshape(-1,1,1) for v in xyz_meshgrid)
    space = (np.array(q).reshape(3) for q in zip(*xyz_vector))
    image = await asyncio.gather(*[march(p,np.array((0,0,-1)),DE) for p in space])
    return np.array(image).reshape(points,points)

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    de = SphereDE()
    image = asyncio.run(main(de))
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.6f} seconds.")
    print(image.shape)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()