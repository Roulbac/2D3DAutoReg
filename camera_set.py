import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from camera import Camera
from utils import recons_DLT

class CameraSet(object):
    def __init__(self, cam1, cam2):
        self.cam1 = cam1
        self.cam2 = cam2
        self.center = self._get_center(cam1, cam2)
        self.tfm = np.eye(4)

    def displace(self, dst):
        t = dst - self.center
        r = self.tfm[:3, :3]
        self.set_tfm(r, t)

    def set_tfm_params(self, *params):
        r, t = CameraSet.params_to_mat(*params)
        self.set_centered_tfm(r, t)

    def set_centered_tfm(self, r, t):
        c = self.center
        t = (-r.dot(c) + c + t).reshape((3, 1))
        tfm = np.block([[r,       t],
                        [0, 0, 0, 1]])
        self.cam1.tfm, self.cam2.tfm = tfm, tfm
        self.tfm = tfm

    @staticmethod
    def _get_center(cam1, cam2):
        x1, x2 = cam1.k[:2, 3], cam2.k[:2, 3]
        p1, p2 = cam1.p, cam2.p
        return recons_DLT(x1, x2, p1, p2)

    def rot_from_euler(angle, axis):
        r = np.eye(3)
        # PHI - X ; THETA - Y ; PSI - Z
        if axis == 'phi':
            r[1:, 1:] = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        elif axis == 'theta':
            r[::2, ::2] = np.array([[np.cos(angle), np.sin(angle)],
                                  [-np.sin(angle), np.cos(angle)]])
        elif axis == 'psi':
            r[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        return r

    def params_to_mat(*params):
        t = np.array(params[:3])
        phi, theta, psi = params[3:]
        phi, theta, psi = phi*np.pi/180, theta*np.pi/180, psi*np.pi/180
        rphi = CameraSet.rot_from_euler(phi, 'phi')
        rtheta = CameraSet.rot_from_euler(theta, 'theta')
        rpsi = CameraSet.rot_from_euler(psi, 'psi')
        r = np.dot(rpsi, np.dot(rtheta, rphi))
        return r, t

    def plot_camera_set(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.cam1._make_cam_plot(fig, ax)
        self.cam2._make_cam_plot(fig, ax)
        return fig, ax

