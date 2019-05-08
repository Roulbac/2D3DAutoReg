import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numba import cuda
import numpy as np
from ray import Ray
import utils


class Camera(object):
    """Camera class
    Convention is xy at upper left corner of 2d image with z pointing towards
    negative view direction.
    NOTE: The focal plane is in positive z direction, i.e BEHIND THE CAMERA
          We work here in homogeneous coords.
    """

    def __init__(self, m=np.eye(4), k=np.eye(3, 4), h=768, w=768):
        self.h, self.w = h, w
        k, m = np.asarray(k), np.asarray(m)
        self.m, self.k = m, k
        self.minv, self.kinv = np.linalg.pinv(m), np.linalg.pinv(k)
        self.z_sign = int(np.sign(m[2, 3]))

    @property
    def r(self):
        """ Rotation matrix from World->Camera"""
        return self.m[:3, :3].copy()

    @property
    def t(self):
        """ Translation vector, also world origin in camera coords """
        return self.m[:3, 3].copy()

    @property
    def p(self):
        return self.k.dot(self.m)

    @property
    def pos(self):
        return -self.r.T.dot(self.t)

    def _make_cam_plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        r, pos = self.r, self.pos
        u, v, w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
        # Camera frame
        ax.text(pos[0], pos[1], pos[2], 'Camera')
        ax.plot3D([pos[0]], [pos[1]], [pos[2]], color='red', marker='*')
        ax.text(0, 0, 0, 'World origin')
        ax.plot3D([0], [0], [0], color='blue', marker='*')
        ax.plot3D([pos[0], pos[0] + u[0]],
                  [pos[1], pos[1] + u[1]],
                  [pos[2], pos[2] + u[2]],
                  color='red')
        ax.plot3D([pos[0], pos[0] + v[0]],
                  [pos[1], pos[1] + v[1]],
                  [pos[2], pos[2] + v[2]],
                  color='green')
        ax.plot3D([pos[0], pos[0] + w[0]],
                  [pos[1], pos[1] + w[1]],
                  [pos[2], pos[2] + w[2]],
                  color='blue')
        # Boundaries
        upperleft = self.get_focalplane_pt([0, 0])
        upperright = self.get_focalplane_pt([0, self.w])
        lowerleft = self.get_focalplane_pt([self.h, 0])
        lowerright = self.get_focalplane_pt([self.h, self.w])
        ax.plot3D([pos[0], upperleft[0] + 300*(upperleft[0] - pos[0])],
                  [pos[1], upperleft[1] + 300*(upperleft[1] - pos[1])],
                  [pos[2], upperleft[2] + 300*(upperleft[2] - pos[2])],
                  color='black')
        ax.plot3D([pos[0], lowerleft[0] + 300*(lowerleft[0] - pos[0])],
                  [pos[1], lowerleft[1] + 300*(lowerleft[1] - pos[1])],
                  [pos[2], lowerleft[2] + 300*(lowerleft[2] - pos[2])],
                  color='black')
        ax.plot3D([pos[0], upperright[0] + 300*(upperright[0] - pos[0])],
                  [pos[1], upperright[1] + 300*(upperright[1] - pos[1])],
                  [pos[2], upperright[2] + 300*(upperright[2] - pos[2])],
                  color='black')
        ax.plot3D([pos[0], lowerright[0] + 300*(lowerright[0] - pos[0])],
                  [pos[1], lowerright[1] + 300*(lowerright[1] - pos[1])],
                  [pos[2], lowerright[2] + 300*(lowerright[2] - pos[2])],
                  color='black')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        utils.set_axes_equal(ax)
        return fig, ax

    def get_focalplane_pt(self, x):
        assert isinstance(x, (list, tuple))
        x = self.kinv.dot(x + [1]) + [0, 0, 0, 1]
        return self.minv.dot(x)[:3]

    @cuda.jit
    def backproject_pixel(h, w, dsts, minv, kinv, z_sign):
        i, j = cuda.grid(2)
        if i < h and j < w:
            dotx = z_sign*(kinv[0, 0]*i + kinv[0, 1]*j + kinv[0, 2]*1)
            doty = z_sign*(kinv[1, 0]*i + kinv[1, 1]*j + kinv[1, 2]*1)
            dotz = z_sign*(kinv[2, 0]*i + kinv[2, 1]*j + kinv[2, 2]*1)
            dsts[i, j, 0] = minv[0, 0]*dotx + minv[0, 1]*doty + minv[0, 2]*dotz + minv[0, 3]*1
            dsts[i, j, 1] = minv[1, 0]*dotx + minv[1, 1]*doty + minv[1, 2]*dotz + minv[1, 3]*1
            dsts[i, j, 2] = minv[2, 0]*dotx + minv[2, 1]*doty + minv[2, 2]*dotz + minv[2, 3]*1

    def backproject_pixels(self):
        d_pos = cuda.to_device(self.pos)
        d_dsts = cuda.device_array((self.h, self.w, 3))
        d_minv = cuda.to_device(self.minv)
        d_kinv = cuda.to_device(self.kinv)
        threadsperblock = (32, 32)
        blockspergrid = (math.ceil(self.h/32), math.ceil(self.w/32))
        Camera.backproject_pixel[blockspergrid, threadsperblock](
            self.h, self.w, d_dsts, d_minv, d_kinv, self.z_sign
        )
        return self.h, self.w, d_pos, d_dsts, blockspergrid, threadsperblock

    def get_scene_ray(self, x):
        src = self.pos
        dest = self.minv.dot(
            self.z_sign*self.kinv.dot(x + [1]) + [0, 0, 0, 1]
        )[:3]
        return Ray(src, dest)

    def plot_camera3d(self):
        fig, ax = self._make_cam_plot()
        plt.show()
