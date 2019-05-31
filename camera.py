import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import utils


class Camera(object):
    """Camera class
    Convention is xy at upper left corner of 2d image with z pointing towards
    negative view direction.
    NOTE: The focal plane is in positive z direction, i.e BEHIND THE CAMERA
          We work here in homogeneous coords.
    """

    # Camera convention
    # -1 if scene is negative z
    # +1 if scene is on positive z
    Z_SIGN = -1
    # 1 if down vector is y, 0 if x
    DOWN = 1

    def __init__(self, m=np.eye(4), k=np.eye(3, 4), h=768, w=768):
        self.h, self.w = h, w
        k, m = np.asarray(k), np.asarray(m)
        if m.shape == (3, 4):
            m = np.vstack([m, [0, 0, 0, 1]])
        if k.shape == (3, 3):
            k = np.hstack([k, np.zeros((3, 1))])
        self._m, self.k = m, k
        self.tfm = np.eye(4)

    @property
    def m(self):
        return np.dot(self._m, self.tfm)

    @property
    def minv(self):
        return np.linalg.pinv(self.m)

    @property
    def kinv(self):
        return np.linalg.pinv(self.k)

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

    def _make_cam_plot_fixed(self, fig, ax, suffix=''):
        r = self._m[:3, :3]
        pos = -r.T.dot(self._m[:3, 3])
        u, v, w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
        # Camera frame
        ax.text(pos[0], pos[1], pos[2], 'Camera_{}'.format(suffix))
        ax.plot3D([pos[0]], [pos[1]], [pos[2]], color='red', marker='*')
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
        utils.set_axes_equal(ax)

    def _make_cam_plot(self, fig, ax, suffix=''):
        r, pos = self.r, self.pos
        u, v, w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
        # Camera frame
        ax.text(pos[0], pos[1], pos[2], 'Camera_{}'.format(suffix))
        ax.plot3D([pos[0]], [pos[1]], [pos[2]], color='red', marker='*')
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

    def get_focalplane_pt(self, x):
        assert isinstance(x, (list, tuple))
        x = self.kinv.dot(x + [1]) + [0, 0, 0, 1]
        return self.minv.dot(x)[:3]

    def plot_camera3d(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self._make_cam_plot(fig, ax)
        plt.show()
