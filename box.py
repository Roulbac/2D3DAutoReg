import math
import numpy as np
from numba import cuda


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

    @cuda.jit(device=True)
    def _get_ijk(src, dst, min_axyz, amin, b, s):
        return math.floor((src + 0.5*(min_axyz + amin)*(dst-src) - b)/s)

    @cuda.jit(device=True)
    def _get_alphas(b, s, p1, p2, n):
        if abs(p2-p1) < 1e-10:
            amin, amax = float("-inf"), float("inf")
        else:
            amin, amax = (b-p1)/(p2-p1), (b+(n-1)*s-p1)/(p2-p1)
        if amin > amax:
            amin, amax = amax, amin
        return amin, amax

    @cuda.jit(device=True)
    def _get_ax(p1, p2, n, b, s, axmin, axmax, amin, amax):
        # IMPORTANT: Replace ceil(x) with floor(x+1) and floor(x) with ceil(x-1)
        if p1 == p2:
            a = float("inf")
        elif p1 < p2:
            imin = math.floor((p1 + amin*(p2-p1) - b)/s + 1) if amin != axmin else 1
            a = ((b + imin*s) - p1)/(p2-p1)
        else:
            imax = math.ceil((p1 + amin*(p2-p1) - b)/s - 1) if amin != axmin else n-2
            a = ((b + imax*s) - p1)/(p2-p1)
        return a

    @cuda.jit
    def trace_ray(src, dsts, raysums, rho, bs, ns, spacing):
        i, j = cuda.grid(2)
        if i < dsts.shape[0] and j < dsts.shape[0]:
            dst = dsts[i, j]
            bx, by, bz = bs
            nx, ny, nz = ns
            sx, sy, sz = spacing
            srcx, srcy, srcz = src[0], src[1], src[2]
            dstx, dsty, dstz = dst[0], dst[1], dst[2]
            axmin, axmax = Box._get_alphas(bx, sx, srcx, dstx, nx)
            aymin, aymax = Box._get_alphas(by, sy, srcy, dsty, ny)
            azmin, azmax = Box._get_alphas(bz, sz, srcz, dstz, nz)
            amin, amax = max(axmin, aymin, azmin), min(axmax, aymax, azmax)
            if amin >= amax or amin <= 0:
                raysums[i, j] = 0
                return
            else:
                ptx = srcx + amin*(dstx - srcx)
                pty = srcy + amin*(dsty - srcy)
                ptz = srcz + amin*(dstz - srcz)
                if (ptx > (bx + (nx-1)*sx) or ptx < bx) or \
                   (pty > (by + (ny-1)*sy) or pty < by) or \
                   (ptz > (bz + (nz-1)*sz) or ptz < bz):
                    raysums[i, j] = 0
                    return
            # Calculate ijk min/max
            ax = Box._get_ax(srcx, dstx, nx, bx, sx, axmin, axmax, amin, amax)
            ay = Box._get_ax(srcy, dsty, ny, by, sy, aymin, aymax, amin, amax)
            az = Box._get_ax(srcz, dstz, nz, bz, sz, azmin, azmax, amin, amax)
            dconv = math.sqrt((dstx-srcx)**2 + (dsty-srcy)**2 + (dstz - srcz)**2)
            d12, ac = 0, amin
            min_axyz = min(ax, ay, az)
            i = Box._get_ijk(srcx, dstx, min_axyz, amin, bx, sx)
            j = Box._get_ijk(srcy, dsty, min_axyz, amin, by, sy)
            k = Box._get_ijk(srcz, dstz, min_axyz, amin, bz, sz)
            while 0 <= i < nx - 1 and 0 <= j < ny - 1 and 0 <= k < nz - 1:
                if ax == min_axyz:
                    d12 = d12 + (ax - ac)*dconv*rho[i, j, k]
                    i = i + 1 if srcx < dstx else i - 1
                    ac = ax
                    ax = ax + sx/(abs(dstx - srcx))
                elif ay == min_axyz:
                    d12 = d12 + (ay - ac)*dconv*rho[i, j, k]
                    j = j + 1 if srcy < dsty else j - 1
                    ac = ay
                    ay = ay + sy/(abs(dsty - srcy))
                elif az == min_axyz:
                    d12 = d12 + (az - ac)*dconv*rho[i, j, k]
                    k = k + 1 if srcz < dstz else k - 1
                    ac = az
                    az = az + sz/(abs(dstz - srcz))
            raysums[i, j] = d12

    def trace_rays(bs, ns, spacing, rho, cam):
        h, w, d_pos, d_dsts, blockspergrid, threadsperblock = cam.backproject_pixels()
        d_rho = cuda.to_device(rho)
        d_raysums = cuda.device_array(shape=(h, w), dtype=np.float32)
        Box.trace_ray[blockspergrid, threadsperblock](
            d_pos, d_dsts, d_raysums, d_rho, bs, ns, spacing)
        return d_raysums.copy_to_host()

    def get_ray_minmax_intersec(self, ray):
        amin, amax, axmin, axmax, aymin, aymax, azmin, azmax = self._get_ray_alphas(
            ray)
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

    def get_radiological_path(self, alphas, ray):
        amin, amax, axmin, axmax, aymin, aymax, azmin, azmax = alphas
        src, dst = ray.src, ray.dst
        # Calculate ijk min/max
        ax = Box._get_ax(
            src[0], dst[0], self.nx, self.bx, self.sx, axmin, axmax, amin, amax)
        ay = Box._get_ax(
            src[1], dst[1], self.ny, self.by, self.sy, aymin, aymax, amin, amax)
        az = Box._get_ax(
            src[2], dst[2], self.nz, self.bz, self.sz, azmin, azmax, amin, amax)
        dconv = math.sqrt((dst[0]-src[0])**2 + (dst[1]-src[1])**2 + (dst[2] - src[2])**2)
        d12, ac = 0, amin
        i = math.floor(
            (src[0] + 0.5*(min(ax, ay, az) + amin)*(dst[0]-src[0]) - self.bx)/self.sx)
        j = math.floor(
            (src[1] + 0.5*(min(ax, ay, az) + amin)*(dst[1]-src[1]) - self.by)/self.sy)
        k = math.floor(
            (src[2] + 0.5*(min(ax, ay, az) + amin)*(dst[2]-src[2]) - self.bz)/self.sz)
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
