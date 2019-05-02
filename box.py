import numpy as np


class Box(object):
    def __init__(self, bs, ns, spacing, arr=None):
        self.bx, self.by, self.bz = bs
        self.nx, self.ny, self.nz = ns
        self.sx, self.sy, self.sz = spacing
        self.arr = arr

    @staticmethod
    def _get_alphas(b, s, p1, p2, n):
        if abs(p2-p1) < 1e-10:
            amin, amax = np.array([-np.inf]), np.array([np.inf])
        else:
            amin, amax = (b-p1)/(p2-p1), (b+(n-1)*s-p1)/(p2-p1)
        return amin, amax

    def get_ray_minmax_intersec(self, ray):
        axmin, axmax = self._get_alphas(
            self.bx, self.sx, ray.src[0], ray.dst[0], self.nx)
        aymin, aymax = self._get_alphas(
            self.by, self.sy, ray.src[1], ray.dst[1], self.ny)
        azmin, azmax = self._get_alphas(
            self.bz, self.sz, ray.src[2], ray.dst[2], self.nz)
        amin, amax = max(axmin, aymin, azmin), min(axmax, aymax, azmax)
        if amin < amax and amin > 0:
            pt1 = ray.src + amin*(ray.dst - ray.src)
            # Catch case where amin < amax but the intersections do not lie within the cube
            # In this case, the ray are parallels to one of the axes but still doesn't
            # Intersect the cube
            if (pt1[0] > (self.bx + (self.nx-1)*self.sx) or pt1[0] < self.bx) or \
               (pt1[1] > (self.by + (self.ny-1)*self.sy) or pt1[1] < self.by) or \
               (pt1[2] > (self.bz + (self.nz-1)*self.sz) or pt1[2] < self.bz):
                return None, None
            pt2 = ray.src + amax*(ray.dst - ray.src)
            return pt1, pt2
        else:
            return None, None
