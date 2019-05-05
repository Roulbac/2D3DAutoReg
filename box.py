import math
import numpy as np


class Box(object):
    def __init__(self, bs, ns, spacing, rho=None):
        self.bx, self.by, self.bz = bs
        self.nx, self.ny, self.nz = ns
        self.sx, self.sy, self.sz = spacing
        if rho is None:
            self.rho = np.ones((self.nx-1, self.ny-1, self.nz-1))
        else:
            assert rho.shape == (self.nx-1, self.ny-1, self.nz-1)
            self.rho = rho

    @staticmethod
    def _get_alphas(b, s, p1, p2, n):
        if abs(p2-p1) < 1e-10:
            amin, amax = float("-inf"), float("inf")
        else:
            amin, amax = (b-p1)/(p2-p1), (b+(n-1)*s-p1)/(p2-p1)
        if amin > amax:
            amin, amax = amax, amin
        return amin, amax

    def _get_ray_alphas(self, ray):
        axmin, axmax = self._get_alphas(
            self.bx, self.sx, ray.src[0], ray.dst[0], self.nx)
        aymin, aymax = self._get_alphas(
            self.by, self.sy, ray.src[1], ray.dst[1], self.ny)
        azmin, azmax = self._get_alphas(
            self.bz, self.sz, ray.src[2], ray.dst[2], self.nz)
        amin, amax = max(axmin, aymin, azmin), min(axmax, aymax, azmax)
        alphas = (amin, amax, axmin, axmax, aymin, aymax, azmin, azmax)
        return alphas

    def get_ray_minmax_intersec(self, ray):
        amin, amax, axmin, axmax, aymin, aymax, azmin, azmax = self._get_ray_alphas(ray)
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

    @staticmethod
    def get_iminmax_alpha(p1, p2, n, b, s, axmin, axmax, amin, amax):
        if p1-p2 == 0:
            a = float("inf")
        elif p1 < p2:
            imin = math.ceil((p1 + amin*(p2-p1) - b)/s) if amin != axmin else 1
            imax = math.ceil((p1 + amax*(p2-p1) - b)/s - 1) if amax != axmax else n-1
            a = ((b + imin*s) - p1)/(p2-p1) if imin != imax else float("inf")
        else:
            imin = math.ceil((p1 + amax*(p2-p1) - b)/s) if amax != axmax else 0
            imax = math.ceil((p1 + amin*(p2-p1) - b)/s - 1) if amin != axmin else n-2
            a = ((b + imax*s) - p1)/(p2-p1) if imin != imax else float("inf")
        return a

    def get_radiological_path(self, alphas, ray):
        amin, amax, axmin, axmax, aymin, aymax, azmin, azmax = alphas
        src, dst = ray.src, ray.dst
        # Calculate ijk min/max
        ax = Box.get_iminmax_alpha(
            src[0], dst[0], self.nx, self.bx, self.sx, axmin, axmax, amin, amax)
        ay = Box.get_iminmax_alpha(
            src[1], dst[1], self.ny, self.by, self.sy, aymin, aymax, amin, amax)
        az = Box.get_iminmax_alpha(
            src[2], dst[2], self.nz, self.bz, self.sz, azmin, azmax, amin, amax)
        # np = (imax - imin + 1) + (jmax - jmin + 1) + (kmax - kmin + 1)
        dconv = math.sqrt((dst[0]-src[0])**2 + (dst[1]-src[1])**2 + (dst[2] - src[2])**2)
        d12, ac = 0, amin
        i = math.ceil(
            (src[0] + 0.5*(min(ax, ay, az) + amin)*(dst[0]-src[0]) - self.bx)/self.sx - 1)
        j = math.ceil(
            (src[1] + 0.5*(min(ax, ay, az) + amin)*(dst[1]-src[1]) - self.by)/self.sy - 1)
        k = math.ceil(
            (src[2] + 0.5*(min(ax, ay, az) + amin)*(dst[2]-src[2]) - self.bz)/self.sz - 1)
        while 0 <= i < self.nx - 1 and 0 <= j < self.ny - 1 and 0 <= k < self.nz - 1:
            if ax == min(ax, ay, az):
                d12 = d12 + (ax - ac)*dconv*self.rho[i, j, k]
                i = i + 1 if src[0] < dst[0] else i - 1
                ac = ax
                ax = ax + self.sx/(abs(dst[0] - src[0]))
            elif ay == min(ax, ay, az):
                d12 = d12 + (ay - ac)*dconv*self.rho[i, j, k]
                j = j + 1 if src[1] < dst[1] else j - 1
                ac = ay
                ay = ay + self.sy/(abs(dst[1] - src[1]))
            elif az == min(ax, ay, az):
                d12 = d12 + (az - ac)*dconv*self.rho[i, j, k]
                k = k + 1 if src[2] < dst[2] else k - 1
                ac = az
                az = az + self.sz/(abs(dst[2] - src[2]))
        return d12
