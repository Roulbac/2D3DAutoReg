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
        self.cams = []
        self.d_cams = []
        if mode == 'gpu':
            assert _IMP_PYCUDA and cuda.Device.count() > 0
            with open('kernels.cu') as f:
                source_str = f.read()
            self.cumod = SourceModule(source_str)
            self.f_backproj = self.cumod.get_function('backprojectPixel')
            self.f_backproj.prepare(['i', 'i', 'P', 'P', 'P', 'i'])
            self.f_trace = self.cumod.get_function('traceRay')
            self.f_trace.prepare(['P', 'P', 'P', 'P', 'P', 'P', 'P', 'i', 'i'])

    def init_cams(self, *cams):
        if self.mode == 'gpu':
            self._cu_init_cams(*cams)
        else:
            self._cpu_init_cams(*cams)

    def init_rho(self, rho, b, n, sp):
        if self.mode == 'gpu':
            self._cu_init_rho(rho, b, n, sp)
        else:
            self._cpu_init_rho(rho, b, n, sp)

    def trace_rays(self):
        if self.mode == 'gpu':
            all_raysums = self._cu_trace_rays()
            hws = [(d_cam[1], d_cam[2]) for d_cam in self.d_cams]
        else:
            all_raysums = self._cpu_trace_rays()
            hws = [(cam.h, cam.w) for cam in self.cams]
        for i in range(len(all_raysums)):
            all_raysums[i] = all_raysums[i].reshape(hws[i], order='F')
        return all_raysums

    # -------------------- PRIVATE CPU FUNCTIONS ----------------------

    def _cpu_trace_rays(self):
        camargs = []
        for cam in self.cams:
            camargs.append(
                (cam.h, cam.w, cam.minv.flatten(),
                 cam.kinv.flatten(), cam.pos, cam.z_sign))
        # Args is a list of tuples for each cam
        return Box._jit_trace_rays(self.n, self.sp, self.b, self.rho, camargs)

    def _cpu_init_cams(self, *cams):
        for cam in cams:
            self.cams.append(cam)

    def _cpu_init_rho(self, rho, b, n, sp):
        self.rho = rho.flatten()
        self.b, self.n, self.sp = b, n, sp

    @jit(nopython=True)
    def _jit_trace_rays(n, sp, b, rho, camargs):
        all_raysums = []
        for arg in camargs:
            h, w, minv, kinv, pos, z_sign = arg
            dsts = np.zeros(h*w*3, dtype=np.float32)
            raysums = np.zeros(h*w, dtype=np.float32)
            for idx in range(h*w):
                i, j = idx // h, idx % h
                dsts[3*idx:3*idx+3] = _cpu_backproject_pixel(
                    h, w, minv, kinv,
                    z_sign, i, j
                )
            for idx in range(h*w):
                raysums[idx] = _cpu_trace_ray(
                    pos[0], pos[1], pos[2],
                    dsts[3*idx], dsts[3*idx+1], dsts[3*idx+2],
                    n[0], n[1], n[2], b[0], b[1], b[2],
                    sp[0], sp[1], sp[2], rho
                )
            all_raysums.append(raysums)
        return all_raysums

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

    def _cu_init_cams(self, *cams):
        # Allocate camera data
        for cam in cams:
            # Convert to device variables and allocate if necessary
            d_h, d_w, d_z_sign = np.int32(cam.h), np.int32(cam.w), np.int32(cam.z_sign)
            d_kinv = cuda.mem_alloc(cam.kinv.size * np.nbytes[np.float32])
            d_minv = cuda.mem_alloc(cam.minv.size * np.nbytes[np.float32])
            d_src = cuda.mem_alloc(cam.pos.size * np.nbytes[np.float32])
            d_dsts = cuda.mem_alloc(int(cam.h*cam.w*3*np.nbytes[np.float32]))
            d_raysums = cuda.mem_alloc(int(cam.h*cam.w*np.nbytes[np.float32]))
            # Copy Kinv to device
            cuda.memcpy_htod(d_kinv, cam.kinv.flatten().astype(np.float32))
            # Save pointers to camera and variables
            self.d_cams.append(
                (cam, d_h, d_w, d_z_sign, d_kinv,
                 d_minv, d_src, d_dsts, d_raysums)
            )

    def _cu_trace_rays(self):
        all_raysums = []
        for d_cam in self.d_cams:
            cam, d_h, d_w, d_z_sign, d_kinv, d_minv, d_src, d_dsts, d_raysums = d_cam
            cuda.memcpy_htod(d_src, cam.pos.astype(np.float32))
            cuda.memcpy_htod(d_minv, cam.minv.flatten().astype(np.float32))
            cuda.memcpy_htod(d_kinv, cam.kinv.flatten().astype(np.float32))
            block = (16, 16, 1)
            grid = (math.ceil(d_h/block[0]), math.ceil(d_w/block[0]))
            self.f_backproj.prepared_call(
                grid, block, d_h, d_w, d_dsts,
                d_minv, d_kinv, d_z_sign
            )
            self.f_trace.prepared_call(
                grid, block, d_src, d_dsts, d_raysums,
                self.d_rho, self.d_b, self.d_sp, self.d_n, d_h, d_w
            )
            raysums = np.zeros(d_h*d_w, dtype=np.float32)
            cuda.memcpy_dtoh(raysums, d_raysums)
            all_raysums.append(raysums)
        return all_raysums

# ------------------------JITTED CPU FUNCTIONS-----------------------
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
