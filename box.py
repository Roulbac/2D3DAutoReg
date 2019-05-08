import math
import numpy as np
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    _IMP_PYCUDA = True
except ImportError:
    _IMP_PYCUDA = False
    print('Could not import pycuda')


class Box(object):

    def __init__(self, mode='gpu'):
        self.cuda_kernel = None
        if mode == 'gpu':
            assert _IMP_PYCUDA and cuda.Device.count() > 0
            self.cuda_kernel = SourceModule('kernels.cu')



    # ------------------------- GPU IMPLEMENTATION ---------------------------------


    # ------------------------- CPU IMPLEMENTATION ----------------------------------

    def get_ijk(src, dst, min_axyz, amin, b, s):
        return math.floor((src + 0.5*(min_axyz + amin)*(dst-src) - b)/s)

    def get_alphas(b, s, p1, p2, n):
        if abs(p2-p1) < 1e-10:
            amin, amax = float("-inf"), float("inf")
        else:
            amin, amax = (b-p1)/(p2-p1), (b+(n-1)*s-p1)/(p2-p1)
        if amin > amax:
            amin, amax = amax, amin
        return amin, amax

    def get_ax(p1, p2, n, b, s, axmin, axmax, amin, amax):
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

    def trace_ray(sr, dst, n, s, sp, rho):
        srx, sry, srz = sr[0], sr[1], sr[2]
        dstx, dsty, dstz = dst[0], dst[1], ds[2]
        nx, ny, nz = n[0], n[1], n[2]
        sx, sy, sz = s[0], s[1], s[2]
        # Calculate alphas
        axmin, axmax = Box.get_alphas(bx, sx, srx, dstx, nx)
        aymin, aymax = Box.get_alphas(by, sy, sry, dsty, ny)
        azmin, azmax = Box.get_alphas(bz, sz, srz, dstz, nz)
        amin, amax = max(axmin, aymin, azmin), min(axmax, aymax, azmax)
        # Check intersection
        if amin >= amax or amin < 0:
            return 0
        else:
            ptx = srx + amin*(dstx - srx)
            pty = sry + amin*(dsty - sry)
            ptz = srz + amin*(dstz - srz)
            if (ptx > (bx + (nx-1)*sx) or ptx < bx) or \
               (pty > (by + (ny-1)*sy) or pty < by) or \
               (ptz > (bz + (nz-1)*sz) or ptz < bz):
                return 0
        # Calculate ijk min/max
        ax = Box.get_ax(srx, dstx, nx, bx, sx, axmin, axmax, amin, amax)
        ay = Box.get_ax(sry, dsty, ny, by, sy, aymin, aymax, amin, amax)
        az = Box.get_ax(sry, dstz, nz, bz, sz, azmin, azmax, amin, amax)
        dconv = math.sqrt((dstx-srx)**2 + (dsty-sry)**2 + (dstz - srz)**2)
        d12, ac = 0, amin
        i = math.floor((srx + 0.5*(min(ax, ay, az) + amin)*(dstx-srx) - bx)/sx)
        j = math.floor((sry + 0.5*(min(ax, ay, az) + amin)*(dsty-sry) - by)/sy)
        k = math.floor((srz + 0.5*(min(ax, ay, az) + amin)*(dstz-srz) - bz)/sz)
        # Go forward in the ray
        while 0 <= i < nx - 1 and 0 <= j < ny - 1 and 0 <= k < nz - 1:
            if ax == min(ax, ay, az):
                d12 = d12 + (ax - ac)*dconv*rho[i, j, k]
                i = i + 1 if srx < dstx else i - 1
                ac = ax
                ax = ax + sx/(abs(dstx - srx))
            elif ay == min(ax, ay, az):
                d12 = d12 + (ay - ac)*dconv*rho[i, j, k]
                j = j + 1 if sry < dsty else j - 1
                ac = ay
                ay = ay + sy/(abs(dsty - sry))
            elif az == min(ax, ay, az):
                d12 = d12 + (az - ac)*dconv*rho[i, j, k]
                k = k + 1 if srz < dstz else k - 1
                ac = az
                az = az + sz/(abs(dstz - srz))
        return d12
