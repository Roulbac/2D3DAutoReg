import math
import numpy as np
from numba import jit
from utils import read_rho, recons_DLT

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    _IMP_PYCUDA = True
except ImportError:
    _IMP_PYCUDA = False

MU_WATER = 0.1*9.187001E-01
MU_AIR = 0.1*8.0859E-04
MAX_FLOAT32 = 3.4028e+038
MIN_FLOAT32 = -3.4028e+038
EPS_FLOAT32 = 2.22045e-016


class RayBox(object):

    def __init__(self, mode='cpu', threshold=500, sid=1001):
        self._mode = mode
        self.cams = []
        self.d_cams = []
        self.threshold = np.float32(threshold)
        self.sid = np.float32(sid)
        if mode == 'gpu':
            self.init_cuda_kernels()

    def init_cuda_kernels(self):
        assert _IMP_PYCUDA and cuda.Device.count() > 0
        with open('kernels.cu') as f:
            source_str = f.read()
        self.cumod = SourceModule(source_str)
        self.f_backproj = self.cumod.get_function('backprojectPixel')
        self.f_backproj.prepare(['i', 'i', 'P', 'P', 'P', 'i', '?', 'f'])
        self.f_trace = self.cumod.get_function('traceRay')
        self.f_trace.prepare(['P', 'P', 'P', 'P', 'P', 'P', 'P', 'i', 'i', 'f'])

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if self.mode == mode:
            return
        else:
            self._mode = mode
            if mode == 'gpu':
                self.init_cuda_kernels()
                if len(self.cams) > 0:
                    print('Set cams')
                    self.set_cams(*self.cams)
                if hasattr(self, 'rho'):
                    print('Set Rho')
                    self.set_rho(self.rho, self.sp)
            else:
                if len(self.d_cams) > 0:
                    cams = list(map(lambda x: x[0], self.d_cams))
                    self.set_cams(*cams)
                if hasattr(self, 'rho'):
                    self.set_rho(self.rho, self.sp)

    def set_threshold(self, threshold):
        self.threshold = np.float32(threshold)

    def set_cams(self, *cams):
        if self.mode == 'gpu':
            self._cu_set_cams(*cams)
        else:
            self._cpu_set_cams(*cams)

    def get_cams(self):
        if self.mode == 'gpu':
            return list(map(lambda x: x[0], self.d_cams))
        else:
            return self.cams

    def set_rho(self, rho, sp):
        if self.mode == 'gpu':
            self._cu_set_rho(rho, sp)
        else:
            self._cpu_set_rho(rho, sp)

    def trace_rays(self):
        if self.mode == 'gpu':
            all_raysums = self._cu_trace_rays()
            hws = [(d_cam[1], d_cam[2]) for d_cam in self.d_cams]
        else:
            all_raysums = self._cpu_trace_rays()
            hws = [(cam.h, cam.w) for cam in self.cams]
        for i in range(len(all_raysums)):
            all_raysums[i] = all_raysums[i].reshape(hws[i], order='C')
        return all_raysums


    # -------------------- PRIVATE CPU FUNCTIONS ----------------------

    def _cpu_trace_rays(self):
        camargs = []
        for cam in self.cams:
            camargs.append(
                (cam.h, cam.w, cam.minv.flatten(),
                 cam.kinv.flatten(), cam.pos, cam.Z_SIGN, cam.DOWN))
        # Args is a list of tuples for each cam
        return RayBox._jit_trace_rays(
            self.n, self.sp, self.b,
            self.rho_c, self.threshold,
            camargs, self.sid)

    def _cpu_set_cams(self, *cams):
        self.cams = cams

    def _cpu_set_rho(self, rho, sp):
        self.n = np.array(rho.shape, dtype=np.int32) + 1
        self.sp = np.array(sp, dtype=np.float32)
        self.b = np.array([0, 0, 0], dtype=np.float32)
        self.rho_c = np.ascontiguousarray(rho.flatten(), dtype=np.float32)
        self.rho = rho

    @jit(nopython=True)
    def _jit_trace_rays(n, sp, b, rho, threshold, camargs, sid):
        all_raysums = []
        for arg in camargs:
            h, w, minv, kinv, pos, z_sign, down = arg
            dsts = np.zeros(h*w*3, dtype=np.float32)
            raysums = np.zeros(h*w, dtype=np.float32)
            for idx in range(h*w):
                if down == 1:
                    j, i = idx // w, idx % w
                else:
                    i, j = idx // w, idx % w
                dsts[3*idx:3*idx+3] = _cpu_backproject_pixel(
                    h, w, minv, kinv,
                    z_sign, i, j, sid
                )
            for idx in range(h*w):
                raysums[idx] = _cpu_trace_ray(
                    pos[0], pos[1], pos[2],
                    dsts[3*idx], dsts[3*idx+1], dsts[3*idx+2],
                    n[0], n[1], n[2], b[0], b[1], b[2],
                    sp[0], sp[1], sp[2], rho, threshold
                )
            all_raysums.append(raysums)
        return all_raysums

    # -------------------- PRIVATE CUDA FUNCTIONS ----------------------

    def _cu_set_rho(self, rho, sp):
        # Allocate and copy AABB data
        self.rho = rho
        self.sp = np.array(sp, dtype=np.float32)
        self.n = np.array(rho.shape, dtype=np.int32) + 1
        self.b = np.array([0, 0, 0], dtype=np.float32)
        d_b = cuda.mem_alloc(3*np.nbytes[np.float32])
        d_n = cuda.mem_alloc(3*np.nbytes[np.int32])
        d_sp = cuda.mem_alloc(3*np.nbytes[np.float32])
        rho = np.ascontiguousarray(rho.flatten(), dtype=np.float32)
        d_rho = cuda.mem_alloc(rho.size*np.nbytes[np.float32])
        cuda.memcpy_htod(
            d_rho,
            np.ascontiguousarray(self.rho.flatten(), dtype=np.float32)
        )
        cuda.memcpy_htod(d_b, self.b)
        cuda.memcpy_htod(d_n, self.n)
        cuda.memcpy_htod(d_sp, self.sp)
        self.d_rho = d_rho
        self.d_b = d_b
        self.d_n = d_n
        self.d_sp = d_sp

    def _cu_set_cams(self, *cams):
        # Allocate camera data
        d_cams = []
        for cam in cams:
            # Convert to device variables and allocate if necessary
            d_h, d_w = np.int32(cam.h), np.int32(cam.w)
            d_z_sign, d_down = np.int32(cam.Z_SIGN), np.bool(cam.DOWN)
            d_kinv = cuda.mem_alloc(cam.kinv.size * np.nbytes[np.float32])
            d_minv = cuda.mem_alloc(cam.minv.size * np.nbytes[np.float32])
            d_src = cuda.mem_alloc(cam.pos.size * np.nbytes[np.float32])
            d_dsts = cuda.mem_alloc(int(cam.h*cam.w*3*np.nbytes[np.float32]))
            d_raysums = cuda.mem_alloc(int(cam.h*cam.w*np.nbytes[np.float32]))
            # Copy Kinv to device
            cuda.memcpy_htod(d_kinv, cam.kinv.flatten().astype(np.float32))
            # Save pointers to camera and variables
            d_cams.append(
                (cam, d_h, d_w, d_z_sign, d_down, d_kinv,
                 d_minv, d_src, d_dsts, d_raysums)
            )
        self.d_cams = d_cams

    def _cu_trace_rays(self):
        all_raysums = []
        for d_cam in self.d_cams:
            cam, d_h, d_w, d_z_sign, d_down, d_kinv, d_minv, d_src, d_dsts, d_raysums = d_cam
            cuda.memcpy_htod(d_src, cam.pos.astype(np.float32))
            cuda.memcpy_htod(d_minv, cam.minv.flatten().astype(np.float32))
            cuda.memcpy_htod(d_kinv, cam.kinv.flatten().astype(np.float32))
            block = (16, 16, 1)
            grid = (math.ceil(d_h/block[0]), math.ceil(d_w/block[0]))
            self.f_backproj.prepared_call(
                grid, block, d_h, d_w, d_dsts,
                d_minv, d_kinv, d_z_sign, d_down, self.sid
            )
            self.f_trace.prepared_call(
                grid, block, d_src, d_dsts, d_raysums,
                self.d_rho, self.d_b, self.d_sp, self.d_n, d_h, d_w, self.threshold
            )
            raysums = np.zeros(d_h*d_w, dtype=np.float32)
            cuda.memcpy_dtoh(raysums, d_raysums)
            all_raysums.append(raysums)
        return all_raysums

# ------------------------JITTED CPU FUNCTIONS-----------------------
# These cannot be class-scoped


@jit(nopython=True)
def _cpu_backproject_pixel(h, w, minv, kinv, z_sign, i, j, sid):
    dotx = sid*z_sign*(kinv[0]*i + kinv[1]*j + kinv[2]*1)
    doty = sid*z_sign*(kinv[3]*i + kinv[4]*j + kinv[5]*1)
    dotz = sid*z_sign*(kinv[6]*i + kinv[7]*j + kinv[8]*1)
    dstx = minv[0]*dotx + minv[1]*doty + minv[2]*dotz + minv[3]*1
    dsty = minv[4]*dotx + minv[5]*doty + minv[6]*dotz + minv[7]*1
    dstz = minv[8]*dotx + minv[9]*doty + minv[10]*dotz + minv[11]*1
    return np.array([dstx, dsty, dstz], dtype=np.float32)


@jit(nopython=True)
def _cpu_trace_ray(srx, sry, srz,
                   dstx, dsty, dstz,
                   nx, ny, nz,
                   bx, by, bz,
                   spx, spy, spz,
                   rho, threshold):
    # Calculate alphas
    axmin, axmax = _get_alphas(bx, spx, srx, dstx, nx)
    aymin, aymax = _get_alphas(by, spy, sry, dsty, ny)
    azmin, azmax = _get_alphas(bz, spz, srz, dstz, nz)
    amin, amax = max(axmin, aymin, azmin), min(axmax, aymax, azmax)
    # Check intersection
    if amin >= amax or amin < 0:
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
        idx = k + (nz-1)*j + (nz-1)*(ny-1)*i
        hu = rho[idx]
        mu = (hu*(MU_WATER-MU_AIR)/1000 + MU_WATER)
        if hu < threshold:
            mu = 0
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
    if abs(p2-p1) < EPS_FLOAT32:
        amin, amax = MIN_FLOAT32, MAX_FLOAT32
    else:
        amin, amax = (b-p1)/(p2-p1), (b+(n-1)*s-p1)/(p2-p1)
    if amin > amax:
        amin, amax = amax, amin
    return amin, amax

@jit(nopython=True)
def _get_ax(p1, p2, n, b, s, axmin, axmax, amin, amax):
    # IMPORTANT: Replace ceil(x) with floor(x+1) and floor(x) with ceil(x-1)
    if abs(p1 - p2) < EPS_FLOAT32:
        a = MAX_FLOAT32
    elif p1 < p2:
        if abs(amin-axmin) > EPS_FLOAT32:
            imin = math.floor((p1 + amin*(p2-p1) - b)/s + 1)
        else:
            imin = 1
        a = ((b + imin*s) - p1)/(p2-p1)
    else:
        if abs(amin-axmin) > EPS_FLOAT32:
            imax = math.ceil((p1 + amin*(p2-p1) - b)/s - 1)
        else:
            imax = n-2
        a = ((b + imax*s) - p1)/(p2-p1)
    return a
