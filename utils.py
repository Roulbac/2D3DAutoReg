import numpy as np
from numba import cuda

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

@cuda.jit(device=True)
def _get_ijk(src, dst, min_axyz, amin, b, s):
    return floor((src + 0.5*(min_axyz + amin)*(dst-src) - b)/s)

@cuda.jit(device=True)
def _get_alphas(b, s, p1, p2, n):
    if abs(p2-p1) < 1e-10:
        amin, amax = -MAX_FLOAT32, MAX_FLOAT32
    else:
        amin, amax = (b-p1)/(p2-p1), (b+(n-1)*s-p1)/(p2-p1)
    if amin > amax:
        amin, amax = amax, amin
    return amin, amax

@cuda.jit(device=True)
def _get_ax(p1, p2, n, b, s, axmin, axmax, amin, amax):
    # IMPORTANT: Replace ceil(x) with floor(x+1) and floor(x) with ceil(x-1)
    if p1 == p2:
        a = MAX_FLOAT32
    elif p1 < p2:
        imin = floor((p1 + amin*(p2-p1) - b)/s + 1) if amin != axmin else 1
        a = ((b + imin*s) - p1)/(p2-p1)
    else:
        imax = ceil((p1 + amin*(p2-p1) - b)/s - 1) if amin != axmin else n-2
        a = ((b + imax*s) - p1)/(p2-p1)
    return a
