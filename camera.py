import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils

class Camera(object):
    """Camera class
    Convention is xy at upper left corner of 2d image with z pointing towards
    negative view direction
    NOTE: We work here in homogeneous coords.
    """

    def __init__(self, m=np.eye(4), k=np.eye(3, 4), h=768, w=768):
        self.m, self.k = np.asarray(m), np.asarray(k)

    @property
    def r(self):
        """ Rotation matrix from World->Camera"""
        return self.m[:3, :3].copy()

    @property
    def t(self):
        """ Translation vector, also world origin in camera coords """
        return self.m[:3, 3].copy()

    def plot_camera3d(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        r, t = self.r, self.t
        pos = -r.T.dot(t)
        u, v, w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
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
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        utils.set_axes_equal(ax)
        plt.show()
