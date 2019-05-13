import numpy as np
import SimpleITK as sitk

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

def str_to_mat(x):
    x = x.strip('[]')
    return np.vstack(list(map(lambda x: np.array(x.split(','), dtype=float), x.split(';'))))

def recons_DLT(x1, x2, p1, p2):
    p1, p2 = p1.flatten(), p2.flatten()
    a = np.array([[p1[0] - x1[0]*p1[8], p1[1] - x1[0]*p1[9], p1[2] - x1[0]*p1[10]],
                  [p1[4] - x1[1]*p1[8], p1[5] - x1[1]*p1[9], p1[6] - x1[1]*p1[10]],
                  [p2[0] - x2[0]*p2[8], p2[1] - x2[0]*p2[9], p2[2] - x2[0]*p2[10]],
                  [p2[4] - x2[1]*p2[8], p2[5] - x2[1]*p2[9], p2[6] - x2[1]*p2[10]]])
    b = np.array([[x1[0]*p1[11] - p1[3]],
                  [x1[1]*p1[11] - p1[7]],
                  [x2[0]*p2[11] - p2[3]],
                  [x2[1]*p2[11] - p2[7]]])
    x, _, _, _, = np.linalg.lstsq(a, b, rcond=None)
    return x.flatten()

def read_rho(fpath):
    im = sitk.ReadImage(fpath)
    sp = np.array(im.GetSpacing(), dtype=np.float32)
    n = np.array(im.GetSize(), dtype=np.int32) + 1
    rho = sitk.GetArrayFromImage(im).transpose((2, 1, 0)).astype(np.float32)
    return rho, n, sp


