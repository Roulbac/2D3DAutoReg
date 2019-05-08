import math
import numpy as np
from numba import jit
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    _IMP_PYCUDA = True
except ImportError:
    _IMP_PYCUDA = False

MU_WATER = 0.85684356
MU_AIR = 0.0007937816
MAX_FLOAT32 = 3.4028e+038
MIN_FLOAT32 = -3.4028e+038
EPS_FLOAT32 = 2.22045e-016


class Box(object):

    def __init__(self, mode='cpu'):
        self.mode = mode
        if mode == 'gpu':
            assert _IMP_PYCUDA and cuda.Device.count() > 0
            with open('kernels.cu') as f:
                source_str = f.read()
            self.cumod = SourceModule(source_str)
            self.f_backproj = self.cumod.get_function('backprojectPixel')
            self.f_trace = self.cumod.get_function('traceRay')

    def init_cams(self, cam1, cam2):
        if self.mode == 'gpu':
            self._cu_init_cams(cam1, cam2)
        else:
            self._cpu_init_cams(cam1, cam2)

    def init_rho(self, rho, b, n, sp):
        if self.mode == 'gpu':
            self._cu_init_rho(rho, b, n, sp)
        else:
            self._cpu_init_rho(rho, b, n, sp)

    def trace_rays(self):
        if self.mode == 'gpu':
            return self._cu_trace_rays()
        else:
            return self._cpu_trace_rays()

    # -------------------- PRIVATE CPU FUNCTIONS ----------------------

    def _cpu_trace_rays(self):
        args = [self.h1, self.w1,
                self.h2, self.w2,
                self.cam1.pos, self.cam2.pos,
                self.minv1, self.minv2,
                self.kinv1, self.kinv2,
                self.sp, self.n, self.b, self.rho,
                self.cam1.z_sign, self.cam2.z_sign]
        return Box._jit_trace_rays(*args)

    def _cpu_init_cams(self, cam1, cam2):
        self.h1, self.w1 = cam1.h, cam1.w
        self.h2, self.w2 = cam2.h, cam2.w
        self.kinv1, self.minv1 = cam1.kinv.flatten(), cam1.minv.flatten()
        self.kinv2, self.minv2 = cam2.kinv.flatten(), cam2.minv.flatten()
        self.cam1, self.cam2 = cam1, cam2

    def _cpu_init_rho(self, rho, b, n, sp):
        self.rho = rho.flatten()
        self.b, self.n, self.sp = b, n, sp

    @jit(nopython=True)
    def _jit_trace_rays(h1, w1,
                         h2, w2,
                         pos1, pos2,
                         minv1, minv2,
                         kinv1, kinv2,
                         sp, n, b, rho,
                         z_sign1, z_sign2):
        dsts1 = np.zeros(h1*w1*3, dtype=np.float32)
        dsts2 = np.zeros(h2*w2*3, dtype=np.float32)
        raysums1 = np.zeros(h1*w1, dtype=np.float32)
        raysums2 = np.zeros(h2*w2, dtype=np.float32)
        for idx1 in range(h1*w1):
            i1, j1 = idx1 // w1, idx1 % w1
            dsts1[3*idx1:3*idx1+3] = _cpu_backproject_pixel(
                h1, w1, minv1, kinv1,
                z_sign1, i1, j1
            )
        for idx2 in range(h2*w2):
            i2, j2 = idx2 // w2, idx2 % w2
            dsts2[3*idx2:3*idx2+3] = _cpu_backproject_pixel(
                h2, w2, minv2, kinv2,
                z_sign2, i2, j2
            )
        for idx1 in range(h1*w1):
            raysums1[idx1] = _cpu_trace_ray(
                pos1[0], pos1[1], pos1[2],
                dsts1[3*idx1], dsts1[3*idx1+1], dsts1[3*idx1+2],
                n[0], n[1], n[2], b[0], b[1], b[2],
                sp[0], sp[1], sp[2], rho
            )
        for idx2 in range(h2*w2):
            raysums2[idx2] = _cpu_trace_ray(
                pos2[0], pos2[1], pos2[2],
                dsts2[3*idx2], dsts2[3*idx2+1], dsts2[3*idx2+2],
                n[0], n[1], n[2], b[0], b[1], b[2],
                sp[0], sp[1], sp[2], rho
            )
        return raysums1.reshape((h1, w1)), raysums2.reshape((h2, w2))

    # -------------------- PRIVATE CUDA FUNCTIONS ----------------------

    def _cu_init_rho(self, rho, b, n, sp):
        # Allocate and copy AABB data
        d_rho = cuda.mem_alloc(rho.size*np.nbytes[np.float32])
        d_b = cuda.mem_alloc(b.size*np.nbytes[np.float32])
        d_n = cuda.mem_alloc(n.size*np.nbytes[np.int32])
        d_sp = cuda.mem_alloc(sp.size*np.nbytes[np.float32])
        cuda.memcpy_htod(d_rho, rho.astype(np.float32))
        cuda.memcpy_htod(d_b, b.astype(np.float32))
        cuda.memcpy_htod(d_n, n.astype(np.int32))
        cuda.memcpy_htod(d_sp, sp.astype(np.float32))
        self.d_rho = d_rho
        self.d_b = d_b
        self.d_n = d_n
        self.d_sp = d_sp

    def _cu_init_cams(self, cam1, cam2):
        # Allocate camera data
        h1, w1 = np.int32(cam1.h), np.int32(cam1.w)
        h2, w2 = np.int32(cam2.h), np.int32(cam2.w)
        self.h1, self.w1 = h1, w1
        self.h2, self.w2 = h2, w2
        self.d_kinv1 = cuda.mem_alloc(cam1.kinv.size * np.nbytes[np.float32])
        self.d_kinv2 = cuda.mem_alloc(cam2.kinv.size * np.nbytes[np.float32])
        self.d_minv1 = cuda.mem_alloc(cam1.minv.size * np.nbytes[np.float32])
        self.d_minv2 = cuda.mem_alloc(cam2.minv.size * np.nbytes[np.float32])
        self.d_src1 = cuda.mem_alloc(cam1.pos.size * np.nbytes[np.float32])
        self.d_src2 = cuda.mem_alloc(cam2.pos.size * np.nbytes[np.float32])
        self.d_dsts1 = cuda.mem_alloc(int(h1*w1*3*np.nbytes[np.float32]))
        self.d_dsts2 = cuda.mem_alloc(int(h2*w2*3*np.nbytes[np.float32]))
        self.d_raysums1 = cuda.mem_alloc(int(h1*w1*3*np.nbytes[np.float32]))
        self.d_raysums2 = cuda.mem_alloc(int(h2*w2*3*np.nbytes[np.float32]))
        # Copy Ks to device
        cuda.memcpy_htod(self.d_kinv1, cam1.kinv.flatten().astype(np.float32))
        cuda.memcpy_htod(self.d_kinv2, cam2.kinv.flatten().astype(np.float32))
        # Save device pointers and constants to object
        self.cam1, self.cam2 = cam1, cam2

    def _cu_trace_rays(self):
        # Copy position and projection camera
        cuda.memcpy_htod(self.d_src1, self.cam1.pos.astype(np.float32))
        cuda.memcpy_htod(self.d_src2, self.cam2.pos.astype(np.float32))
        cuda.memcpy_htod(self.d_minv1, self.cam1.minv.flatten().astype(np.float32))
        cuda.memcpy_htod(self.d_minv2, self.cam2.minv.flatten().astype(np.float32))
        # Init scalar parameters and cuda funs
        h1, w1 = self.h1, self.w1
        h2, w2 = self.h2, self.w2
        z_sign1, z_sign2 = np.int32(
            self.cam1.z_sign), np.int32(self.cam2.z_sign)
        raysums1 = np.zeros(h1*w1, dtype=np.float32)
        raysums2 = np.zeros(h2*w2, dtype=np.float32)
        block = (16, 16, 1)
        grid1 = (math.ceil(h1/block[0]), math.ceil(w1/block[1]))
        grid2 = (math.ceil(h2/block[0]), math.ceil(w2/block[1]))
        # Backproject rays
        self.f_backproj.prepare(['i', 'i', 'P', 'P', 'P', 'i'])
        self.f_backproj.prepared_call(
            grid1, block, h1, w1, self.d_dsts1,
            self.d_minv1, self.d_kinv1, z_sign1
        )
        self.f_backproj.prepare(['i', 'i', 'P', 'P', 'P', 'i'])
        self.f_backproj.prepared_call(
            grid2, block, h2, w2, self.d_dsts2,
            self.d_minv2, self.d_kinv2, z_sign2
        )
        # Do ray tracing
        self.f_trace.prepare(['P', 'P', 'P', 'P', 'P', 'P', 'P', 'i', 'i'])
        self.f_trace.prepared_call(
            grid1, block, self.d_src1, self.d_dsts1, self.d_raysums1,
            self.d_rho, self.d_b, self.d_sp, self.d_n, h1, w1
        )
        self.f_trace.prepare(['P', 'P', 'P', 'P', 'P', 'P', 'P', 'i', 'i'])
        self.f_trace.prepared_call(
            grid2, block, self.d_src2, self.d_dsts2, self.d_raysums2,
            self.d_rho, self.d_b, self.d_sp, self.d_n, h1, w1
        )
        # Copy back to host arrays
        cuda.memcpy_dtoh(raysums1, self.d_raysums1)
        cuda.memcpy_dtoh(raysums2, self.d_raysums2)
        return raysums1.reshape((h1, w1)), raysums2.reshape((h2, w2))

#------------------------JITTED CPU FUNCTIONS-----------------------
# These cannot be class-scoped


@jit(nopython=True)
def _cpu_backproject_pixel(h, w, minv, kinv, z_sign, i, j):
    dotx = z_sign*(kinv[0]*i + kinv[1]*j + kinv[2]*1)
    doty = z_sign*(kinv[3]*i + kinv[4]*j + kinv[5]*1)
    dotz = z_sign*(kinv[6]*i + kinv[7]*j + kinv[8]*1)
    dstx = minv[0]*dotx + minv[1]*doty + minv[2]*dotz + minv[3]*1
    dsty = minv[5]*dotx + minv[5]*doty + minv[6]*dotz + minv[7]*1
    dstz = minv[8]*dotx + minv[9]*doty + minv[10]*dotz + minv[11]*1
    return np.array([dstx, dsty, dstz], dtype=np.float32)

@jit(nopython=True)
def _cpu_trace_ray(srx, sry, srz,
                   dstx, dsty, dstz,
                   nx, ny, nz,
                   bx, by, bz,
                   spx, spy, spz,
                   rho):
    # Calculate alphas
    axmin, axmax = _get_alphas(bx, spx, srx, dstx, nx)
    aymin, aymax = _get_alphas(by, spy, sry, dsty, ny)
    azmin, azmax = _get_alphas(bz, spz, srz, dstz, nz)
    amin, amax = max(axmin, aymin, azmin), min(axmax, aymax, azmax)
    # Check intersection
    if amin >= amax or amin < 0:
        return 1
    else:
        ptx = srx + amin*(dstx - srx)
        pty = sry + amin*(dsty - sry)
        ptz = srz + amin*(dstz - srz)
        if (ptx > (bx + (nx-1)*spx) or ptx < bx) or \
           (pty > (by + (ny-1)*spy) or pty < by) or \
           (ptz > (bz + (nz-1)*spz) or ptz < bz):
            return 1
    # Calculate ijk min/max
    ax = _get_ax(srx, dstx, nx, bx, spx, axmin, axmax, amin, amax)
    ay = _get_ax(sry, dsty, ny, by, spy, aymin, aymax, amin, amax)
    az = _get_ax(srz, dstz, nz, bz, spz, azmin, azmax, amin, amax)
    dconv = math.sqrt((dstx-srx)**2 + (dsty-sry)**2 + (dstz - srz)**2)
    d12, ac = 0, amin
    i = math.floor(
        (srx + 0.5*(min(ax, ay, az) + amin)*(dstx-srx) - bx)/spx)
    j = math.floor(
        (sry + 0.5*(min(ax, ay, az) + amin)*(dsty-sry) - by)/spy)
    k = math.floor(
        (srz + 0.5*(min(ax, ay, az) + amin)*(dstz-srz) - bz)/spz)
    # Go forward in the ray
    while 0 <= i < nx - 1 and 0 <= j < ny - 1 and 0 <= k < nz - 1:
        idx = i + (nx-1)*j + (nx-1)*(ny-1)*k
        mu = (rho[idx]*(MU_WATER-MU_AIR)/1000 + MU_WATER)
        if ax == min(ax, ay, az):
            d12 = d12 + (ax - ac)*dconv*mu
            i = i + 1 if srx < dstx else i - 1
            ac = ax
            ax = ax + spx/(abs(dstx - srx))
        elif ay == min(ax, ay, az):
            d12 = d12 + (ay - ac)*dconv*mu
            j = j + 1 if sry < dsty else j - 1
            ac = ay
            ay = ay + spy/(abs(dsty - sry))
        elif az == min(ax, ay, az):
            d12 = d12 + (az - ac)*dconv*mu
            k = k + 1 if srz < dstz else k - 1
            ac = az
            az = az + spz/(abs(dstz - srz))
    return math.exp(-d12)

@jit(nopython=True)
def _get_alphas(b, s, p1, p2, n):
    if abs(p2-p1) < 1e-10:
        amin, amax = MIN_FLOAT32, MAX_FLOAT32
    else:
        amin, amax = (b-p1)/(p2-p1), (b+(n-1)*s-p1)/(p2-p1)
    if amin > amax:
        amin, amax = amax, amin
    return amin, amax

@jit(nopython=True)
def _get_ax(p1, p2, n, b, s, axmin, axmax, amin, amax):
    # IMPORTANT: Replace ceil(x) with floor(x+1) and floor(x) with ceil(x-1)
    if p1 == p2:
        a = MAX_FLOAT32
    elif p1 < p2:
        imin = math.floor((p1 + amin*(p2-p1) - b)/s + 1) if amin != axmin else 1
        a = ((b + imin*s) - p1)/(p2-p1)
    else:
        imax = math.ceil((p1 + amin*(p2-p1) - b)/s - 1) if amin != axmin else n-2
        a = ((b + imax*s) - p1)/(p2-p1)
    return a

