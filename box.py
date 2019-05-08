import math
import numpy as np
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    _IMP_PYCUDA = True
except ImportError:
    _IMP_PYCUDA = False
    print('Could not import pycuda')


class Box(object):

    def __init__(self, mode='gpu'):
        if mode == 'gpu':
            assert _IMP_PYCUDA and cuda.Device.count() > 0
            with open('kernels.cu') as f:
                source_str = f.read()
            self.cumod = SourceModule(source_str)
            self.f_backproj = self.cumod.get_function('backprojectPixel')
            self.f_backproj.prepare(['i', 'i', 'P', 'P', 'P', 'i'])
            self.f_trace = self.cumod.get_function('traceRay')
            self.f_trace.prepare(['P', 'P', 'P', 'P', 'P', 'P', 'P', 'i', 'i'])

    def cuInitRho(self, rho, b, n, sp):
        d_rho = cuda.mem_alloc(rho.nbytes)
        d_b = cuda.mem_alloc(b.nbytes)
        d_n = cuda.mem_alloc(n.nbytes)
        d_sp = cuda.mem_alloc(sp.n_bytes)
        cuda.memcpy_htod(d_rho, rho)
        cuda.memcpy_htod(d_b, b)
        cuda.memcpy_htod(d_n, n)
        cuda.memcpy_htod(d_sp, sp)
        self.d_rho = d_rho
        self.d_b = d_b
        self.d_n = d_n
        self.d_sp = d_sp

    def cuInitCams(self, cam1, cam2):
        self.d_kinv1 = cuda.mem_alloc(cam1.kinv.nbytes)
        self.d_kinv2 = cuda.mem_alloc(cam2.kinv.nbytes)
        self.d_minv1 = cuda.mem_alloc(cam1.minv.nbytes)
        self.d_minv2 = cuda.mem_alloc(cam2.minv.nbytes)
        self.d_src1 = cuda.mem_alloc(cam1.pos.nbytes)
        self.d_src2 = cuda.mem_alloc(cam2.pos.nbytes)
        self.d_dsts1 = cuda.mem_alloc(cam1.h*cam1.w*3*np.nbytes[np.float32])
        self.d_dsts2 = cuda.mem_alloc(cam2.h*cam2.w*3*np.nbytes[np.float32])
        self.d_raysums1 = cuda.mem_alloc(cam1.h*cam1.w*np.nbytes[np.float32])
        self.d_raysums2 = cuda.mem_alloc(cam2.h*cam2.w*np.nbytes[np.float32])
        cuda.memcpy_htod(self.d_kinv1, cam1.kinv.flatten())
        cuda.memcpy_htod(self.d_kinv2, cam2.kinv.flatten())
        self.d_kinv1, self.d_minv1, self.d_src1 = d_kinv1, d_minv1, d_src1
        self.d_kinv2, self.d_minv2, self.d_src2 = d_kinv2, d_minv2, d_src2
        self.cam1, self.cam2 = cam1, cam2
        self.h1 = np.array(cam1.h, dtype=np.int32)
        self.w1 = np.array(cam1.w, dtype=np.int32)
        self.h2 = np.array(cam2.h, dtype=np.int32)
        self.w2 = np.array(cam2.w, dtype=np.int32)

    def cuTraceRays(self):
        cuda.memcpy_htod(self.d_src1, self.cam1.pos)
        cuda.memcpy_htod(self.d_src2, self.cam2.pos)
        cuda.memcpy_htod(self.d_minv1, self.cam1.minv.flatten())
        cuda.memcpy_htod(self.d_minv2, self.cam2.minv.flatten())
        h1, w1 = self.h1, self.w1
        h2, w1 = self.h2, self.w2
        z_sign1, z_sign2 = np.int32(self.cam1.z_sign), np.int32(self.cam2.z_sign)
        raysums1 = np.zeros(h1*w1, dtype=np.float32)
        raysums2 = np.zeros(h2*w2, dtype=np.float32)
        block = (16, 16, 1)
        grid1 = (math.ceil(h1/block[0]), math.ceil(w1/block[1]))
        grid2 = (math.ceil(h2/block[0]), math.ceil(w2/block[1]))
        self.f_backproj.prepared_call(
            grid1, block, h1, w1, self.d_dsts1,
            self.d_minv1, self.d_kinv1, z_sign1
        )
        self.f_backproj.prepared_call(
            grid2, block, h2, w2, self.d_dsts2,
            self.d_minv2, self.d_kinv2, z_sign2
        )
        self.f_trace.prepared_call(
            grid1, block, self.d_src1, self.d_dsts1, self.d_raysums1,
            self.d_rho, self.d_b, self.d_sp, h1, w1
        )
        self.f_trace.prepared_call(
            grid2, block, self.d_src2, self.d_dsts2, self.d_raysums1,
            self.d_rho, self.d_b, self.d_sp, h1, w1
        )
        cuda.memcpy_dtoh(raysums1, self.d_raysums1)
        cuda.memcpy_dtoh(raysums2, self.d_raysums2)
        return raysums1, raysums2






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
