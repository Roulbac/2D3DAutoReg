import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
from camera_set import CameraSet
from utils import str_to_mat

def to_grayscale(img):
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        return img[..., :3].mean(2)
    else:
        raise ValueError()

def patch_to_pts(patch):
    if patch is not '':
        patch = str_to_mat(patch).T
        return np.vstack((patch, np.ones((1, patch.shape[1]))))
    else:
        return None

def get_projected_patch(p, x):
    # xs are columns of x
    if x is None:
        return None
    y = p.dot(x)
    y = y[:2, :]/y[2, :]
    return np.hstack((y, y[:, 0].reshape(-1, 1)))

def plot_drr(bg, ol, fids1, fids2, alpha=0.5, patch=None):
    fig, ax = plt.subplots()
    ax.imshow(bg, origin='lower', alpha=1)
    ax.imshow(1-ol, cmap='copper', origin='lower', alpha=alpha)
    ax.set_xticks([])
    ax.set_yticks([])
    # color = list(map(lambda x: to_rgba(x, 1), ['#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FFFF00', '#FF00FF']))
    # ax.scatter(fids1[0, :], fids1[1, :], marker='*', color='red', label='C-Arm markers')
    # ax.scatter(fids2[0, :], fids2[1, :], marker='*', color='blue', label='CT markers')
    if patch is not None:
        ax.plot(patch[0, :], patch[1, :], marker='*', linestyle='-.', color='chartreuse')
    ax.legend(fancybox=True, framealpha=0.25, fontsize=10, borderpad=0.5)
    return fig, ax

def get_rms_err(x):
    return np.sqrt((np.linalg.norm(x, ord=2, axis=0)**2).mean())

def get_rigid_rms(x, y):
    x_c = x - np.mean(x, 1, keepdims=True)
    y_c = y - np.mean(y, 1, keepdims=True)
    s = np.dot(x_c, y_c.T)
    u, _, vh = np.linalg.svd(s)
    r = np.dot(vh.T, u.T)
    t = np.mean(y, 1, keepdims=True) - np.dot(r, np.mean(x, 1, keepdims=True))
    x_tfmd = np.dot(r, x) + t
    return get_rms_err(x_tfmd - y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera1', type=str, required=True,
                        help='Camera 1 file')
    parser.add_argument('--camera2', type=str, required=True,
                        help='Camera 2 file')
    parser.add_argument('--camera3', type=str, required=True,
                        help='Camera 3 file')
    parser.add_argument('--camera4', type=str, required=True,
                        help='Camera 4 file')
    parser.add_argument('--fids1', type=str, required=True,
                        help='First fiducials file')
    parser.add_argument('--fids2', type=str, required=True,
                        help='Second fiducials file')
    parser.add_argument('--params_carm', type=str, default='',
                        help='First params file')
    parser.add_argument('--params_drr', type=str, required=True,
                        help='Second params file')
    parser.add_argument('--drr_dir', type=str, required=True,
                        help='Directory with DRRs')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for images')
    parser.add_argument('--patch', type=str, default='',
                        help='3D Patch to project on 2D')
    parser.add_argument('--carm_dir', type=str, required=True,
                        help='Directory with Carm images')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Results dir')
    args = parser.parse_args()
    camera_set = CameraSet()
    with open(args.camera1, 'r') as f:
        s = f.read()
        m1 = str_to_mat(re.search('[Mm]\s*=\s*\[(.*)\]', s).group(1))
        k1 = str_to_mat(re.search('[Kk]\s*=\s*\[(.*)\]', s).group(1))
        h1 = int(re.search('[Hh]\s*=\s*([0-9]+)', s).group(1))
        w1 = int(re.search('[Ww]\s*=\s*([0-9]+)', s).group(1))
        cam1 = Camera(m=m1, k=k1, h=h1, w=w1)
    with open(args.camera2, 'r') as f:
        s = f.read()
        m2 = str_to_mat(re.search('[Mm]\s*=\s*\[(.*)\]', s).group(1))
        k2 = str_to_mat(re.search('[Kk]\s*=\s*\[(.*)\]', s).group(1))
        h2 = int(re.search('[Hh]\s*=\s*([0-9]+)', s).group(1))
        w2 = int(re.search('[Ww]\s*=\s*([0-9]+)', s).group(1))
        cam2 = Camera(m=m2, k=k2, h=h2, w=w2)
    with open(args.camera3, 'r') as f:
        s = f.read()
        m3 = str_to_mat(re.search('[Mm]\s*=\s*\[(.*)\]', s).group(1))
        k3 = str_to_mat(re.search('[Kk]\s*=\s*\[(.*)\]', s).group(1))
        h3 = int(re.search('[Hh]\s*=\s*([0-9]+)', s).group(1))
        w3 = int(re.search('[Ww]\s*=\s*([0-9]+)', s).group(1))
        cam3 = Camera(m=m3, k=k3, h=h3, w=w3)
    with open(args.camera4, 'r') as f:
        s = f.read()
        m4 = str_to_mat(re.search('[Mm]\s*=\s*\[(.*)\]', s).group(1))
        k4 = str_to_mat(re.search('[Kk]\s*=\s*\[(.*)\]', s).group(1))
        h4 = int(re.search('[Hh]\s*=\s*([0-9]+)', s).group(1))
        w4 = int(re.search('[Ww]\s*=\s*([0-9]+)', s).group(1))
        cam4 = Camera(m=m4, k=k4, h=h4, w=w4)
    camera_set.set_cams(*[cam1, cam2, cam3, cam4])
    if len(args.params_carm) > 0:
        with open(args.params_carm, 'r') as f:
            s = f.read()
            tx = float(re.search('Tx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
            ty = float(re.search('Ty\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
            tz = float(re.search('Tz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
            rx = float(re.search('Rx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
            ry = float(re.search('Ry\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
            rz = float(re.search('Rz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        params_carm = [tx, ty, tz, rx, ry, rz]
    else:
        params_carm = [0, 0, 0, 0, 0, 0]
    camera_set.set_tfm_params(*params_carm)
    tfm1 = camera_set.tfm
    with open(args.params_drr, 'r') as f:
        s = f.read()
        tx = float(re.search('Tx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ty = float(re.search('Ty\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        tz = float(re.search('Tz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rx = float(re.search('Rx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ry = float(re.search('Ry\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rz = float(re.search('Rz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
    params_drr = [tx, ty, tz, rx, ry, rz]
    camera_set.set_tfm_params(*params_drr)
    tfm2 = camera_set.tfm
    with open(args.fids1, 'r') as f:
        s = f.read()
        fids1 = np.array(list(map(lambda x: list(map(float, x.split(','))), s.strip().split('\n')))).T
        fids1 = np.vstack((fids1, np.ones(fids1.shape[1])))
    with open(args.fids2, 'r') as f:
        s = f.read()
        fids2 = np.array(list(map(lambda x: list(map(float, x.split(','))), s.strip().split('\n')))).T
        fids2 = np.vstack((fids2, np.ones(fids2.shape[1])))
    fids1_tfmd = np.dot(tfm1, fids1)
    fids1_tfmd = fids1_tfmd[:3, :]/fids1_tfmd[3, :]
    fids2_tfmd = np.dot(tfm2, fids2)
    fids2_tfmd = fids2_tfmd[:3, :]/fids2_tfmd[3, :]
    errs = fids2_tfmd - fids1_tfmd
    rms_3d = get_rms_err(errs)
    rms_3d_svd = get_rigid_rms(fids1[:3, :], fids2[:3, :])
    # Ground truth fiducials
    camera_set.set_tfm_params(*params_carm)
    fids1_tfmd2d1 = camera_set.cams[0].p.dot(fids1)
    fids1_tfmd2d1 = fids1_tfmd2d1[:2, :]/fids1_tfmd2d1[2, :]
    fids1_tfmd2d2 = camera_set.cams[1].p.dot(fids1)
    fids1_tfmd2d2 = fids1_tfmd2d2[:2, :]/fids1_tfmd2d2[2, :]
    fids1_tfmd2d3 = camera_set.cams[2].p.dot(fids1)
    fids1_tfmd2d3 = fids1_tfmd2d3[:2, :]/fids1_tfmd2d3[2, :]
    fids1_tfmd2d4 = camera_set.cams[3].p.dot(fids1)
    fids1_tfmd2d4 = fids1_tfmd2d4[:2, :]/fids1_tfmd2d4[2, :]
    # Synthetic CT fiducials
    camera_set.set_tfm_params(*params_drr)
    fids2_tfmd2d1 = camera_set.cams[0].p.dot(fids2)
    fids2_tfmd2d1 = fids2_tfmd2d1[:2, :]/fids2_tfmd2d1[2, :]
    fids2_tfmd2d2 = camera_set.cams[1].p.dot(fids2)
    fids2_tfmd2d2 = fids2_tfmd2d2[:2, :]/fids2_tfmd2d2[2, :]
    fids2_tfmd2d3 = camera_set.cams[2].p.dot(fids2)
    fids2_tfmd2d3 = fids2_tfmd2d3[:2, :]/fids2_tfmd2d3[2, :]
    fids2_tfmd2d4 = camera_set.cams[3].p.dot(fids2)
    fids2_tfmd2d4 = fids2_tfmd2d4[:2, :]/fids2_tfmd2d4[2, :]
    rms_2d_1 = get_rms_err(fids1_tfmd2d1 - fids2_tfmd2d1)
    rms_2d_2 = get_rms_err(fids1_tfmd2d2 - fids2_tfmd2d2)
    rms_2d_3 = get_rms_err(fids1_tfmd2d3 - fids2_tfmd2d3)
    rms_2d_4 = get_rms_err(fids1_tfmd2d4 - fids2_tfmd2d4)
    # MRI patch
    patch = patch_to_pts(args.patch)
    patch_proj1 = get_projected_patch(camera_set.cams[0].p, patch)
    patch_proj2 = get_projected_patch(camera_set.cams[1].p, patch)
    patch_proj3 = get_projected_patch(camera_set.cams[2].p, patch)
    patch_proj4 = get_projected_patch(camera_set.cams[3].p, patch)
    # Find cam1 and cam2 images of synthetic CT and real CT, blend, draw markers
    c_arms = sorted([os.path.join(args.carm_dir, x) for x in os.listdir(args.carm_dir) if x.startswith(args.prefix) and x.endswith('.png')])
    drrs = sorted([os.path.join(args.drr_dir, x) for x in os.listdir(args.drr_dir) if x.startswith(args.prefix) and x.endswith('.png')])
    # im11 = to_grayscale(plt.imread(c_arms[0]))
    # im12 = to_grayscale(plt.imread(c_arms[1]))
    # im13 = to_grayscale(plt.imread(c_arms[2]))
    # im14 = to_grayscale(plt.imread(c_arms[3]))
    im11 = plt.imread(c_arms[0])
    im12 = plt.imread(c_arms[1])
    im13 = plt.imread(c_arms[2])
    im14 = plt.imread(c_arms[3])
    im21 = to_grayscale(plt.imread(drrs[0]))
    im22 = to_grayscale(plt.imread(drrs[1]))
    im23 = to_grayscale(plt.imread(drrs[2]))
    im24 = to_grayscale(plt.imread(drrs[3]))
    fig1, ax1 = plot_drr(im11, im21, fids1_tfmd2d1, fids2_tfmd2d1, patch=patch_proj1)
    fig2, ax2 = plot_drr(im12, im22, fids1_tfmd2d2, fids2_tfmd2d2, patch=patch_proj2)
    fig3, ax3 = plot_drr(im13, im23, fids1_tfmd2d3, fids2_tfmd2d3, patch=patch_proj3)
    fig4, ax4 = plot_drr(im14, im24, fids1_tfmd2d4, fids2_tfmd2d4, patch=patch_proj4)
    fig1.savefig(
        os.path.join(args.results_dir, os.path.basename(args.params_drr).split('.')[0] + '_drr1.png'),
        bbox_inches='tight', pad_inches=0, transparent=True, dpi=100
    )
    fig2.savefig(
        os.path.join(args.results_dir, os.path.basename(args.params_drr).split('.')[0] + '_drr2.png'),
        bbox_inches='tight', pad_inches=0, transparent=True, dpi=100
    )
    fig3.savefig(
        os.path.join(args.results_dir, os.path.basename(args.params_drr).split('.')[0] + '_drr3.png'),
        bbox_inches='tight', pad_inches=0, transparent=True, dpi=100
    )
    fig4.savefig(
        os.path.join(args.results_dir, os.path.basename(args.params_drr).split('.')[0] + '_drr4.png'),
        bbox_inches='tight', pad_inches=0, transparent=True, dpi=100
    )
    np.set_printoptions(precision=2)
    print('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(rms_3d_svd, rms_3d, rms_2d_1, rms_2d_2, rms_2d_3, rms_2d_4))
