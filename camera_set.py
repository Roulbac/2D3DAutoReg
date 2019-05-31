import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from camera import Camera
from utils import recons_DLT

class CameraSet(object):
    def __init__(self):
        # tx ty tz rx (x) ry (y) rz (x)
        self.cam1, self.cam2, self.center = None, None, None
        self.params = [0, 0, 0, 0, 0, 0]
        self.tfm = np.eye(4)

    def set_cams(self, cam1, cam2):
        self.cam1 = cam1
        self.cam2 = cam2
        self.center = self._get_center(cam1, cam2)

    def move_to(self, dst):
        tx = self.center[0] - dst[0]
        ty = self.center[1] - dst[1]
        tz = self.center[2] - dst[2]
        rx = self.params[3]
        ry = self.params[4]
        rz = self.params[5]
        self.set_tfm_params(tx, ty, tz, rx, ry, rz)
        print(tx, ty, tz, rx, ry, rz)

    def set_tfm_params(self, *params):
        self.params = list(params)
        p, p_prime = self.make_p_pprime()
        r, t = CameraSet.params_to_mat(p, p_prime, *params)
        self.set_centered_tfm(r, t)

    def set_centered_tfm(self, r, t):
        c = self.center
        t = (-r.dot(c - t) + c).reshape((3, 1))
        tfm = np.block([[r,       t],
                        [0, 0, 0, 1]])
        self.cam1.tfm, self.cam2.tfm = tfm, tfm
        self.tfm = tfm

    @staticmethod
    def _get_center(cam1, cam2):
        x1, x2 = cam1.k[:2, 2], cam2.k[:2, 2]
        p1, p2 = cam1.p, cam2.p
        return recons_DLT(x1, x2, p1, p2)

    def rot_from_euler(angle, axis):
        r = np.eye(3)
        # PHI - X ; THETA - Y ; PSI - Z
        if axis == 'rx':
            r[1:, 1:] = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        elif axis == 'ry':
            r[::2, ::2] = np.array([[np.cos(angle), np.sin(angle)],
                                  [-np.sin(angle), np.cos(angle)]])
        elif axis == 'rz':
            r[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        return r

    def make_p_pprime(self):
        i = self.cam1._m[2, :3]/np.linalg.norm(self.cam1._m[2, :3])
        j = self.cam2._m[2, :3]/np.linalg.norm(self.cam2._m[2, :3])
        k = np.cross(i, j)
        p = np.vstack([i, j, k]).T
        p_prime = np.array([[1, -np.dot(i, j), 0],
                            [0,             1, 0],
                            [0,             0, 1]])
        p_prime[:, 1] /= np.linalg.norm(p_prime[:, 1])
        return p, p_prime

    def make_ijk_rotation(p, p_prime, rx, ry, rz):
        rrx = CameraSet.rot_from_euler(rx, 'rx')
        rry = CameraSet.rot_from_euler(ry, 'ry')
        rrz = CameraSet.rot_from_euler(rz, 'rz')
        rrz = np.linalg.multi_dot([p_prime, rrz, np.linalg.inv(p_prime)])
        rxyz = np.linalg.multi_dot([rrx, rry, rrz])
        rijk = np.linalg.multi_dot([p, rxyz, np.linalg.inv(p)])
        return rijk

    def params_to_mat(p=np.eye(3), p_prime=np.eye(3), *params):
        t = np.array(params[:3])
        rx, ry, rz = params[3:]
        rx, ry, rz = rx*np.pi/180, ry*np.pi/180, rz*np.pi/180
        r = CameraSet.make_ijk_rotation(p, p_prime, rx, ry, rz)
        return r, t

    def plot_camera_set(self, fig, ax):
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        self.cam1._make_cam_plot(fig, ax, suffix='1')
        self.cam2._make_cam_plot(fig, ax, suffix='2')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(False)
        center = self.center - \
                np.array([self.params[0],
                          self.params[1],
                          self.params[2]])
        ax.plot3D([center[0], center[0] + 100],
                  [center[1], center[1] +   0],
                  [center[2], center[2] +   0],
                  color='red')
        ax.plot3D([center[0], center[0] +   0],
                  [center[1], center[1] + 100],
                  [center[2], center[2] +   0],
                  color='green')
        ax.plot3D([center[0], center[0] +   0],
                  [center[1], center[1] +   0],
                  [center[2], center[2] + 100],
                  color='blue')
        ax.plot3D([center[0]],
                  [center[1]],
                  [center[2]],
                  marker='*', color='red')
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        #plt.show(block=False)

