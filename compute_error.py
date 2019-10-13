import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
from camera_set import CameraSet
from utils import str_to_mat

def plot_drr(bg, ol, fids1, fids2):
    bg = bg.copy()
    ol = 1 - ol
    ol[:, :, 1:] = 0
    fig, ax = plt.subplots()
    ax.imshow(bg, alpha=1, origin='lower')
    ax.imshow(ol, alpha=0.667, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(fids1[0, :], fids1[1, :], '*', color='green', label='Real CT')
    ax.plot(fids2[0, :], fids2[1, :], '*', color='red', label='Synthetic CT')
    ax.legend(fancybox=True, framealpha=0.25, fontsize=10, borderpad=0.25)
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
    parser.add_argument('--fids1', type=str, required=True,
                        help='First fiducials file')
    parser.add_argument('--fids2', type=str, required=True,
                        help='Second fiducials file')
    parser.add_argument('--params1', type=str, required=True,
                        help='First params file')
    parser.add_argument('--params2', type=str, required=True,
                        help='Second params file')
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
    camera_set.set_cams(*[cam1, cam2])
    with open(args.params1, 'r') as f:
        s = f.read()
        tx = float(re.search('Tx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ty = float(re.search('Ty\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        tz = float(re.search('Tz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rx = float(re.search('Rx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ry = float(re.search('Ry\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rz = float(re.search('Rz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        params1 = [tx, ty, tz, rx, ry, rz]
        camera_set.set_tfm_params(*params1)
    tfm1 = camera_set.tfm
    with open(args.params2, 'r') as f:
        s = f.read()
        tx = float(re.search('Tx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ty = float(re.search('Ty\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        tz = float(re.search('Tz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rx = float(re.search('Rx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ry = float(re.search('Ry\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rz = float(re.search('Rz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        params2 = [tx, ty, tz, rx, ry, rz]
    camera_set.set_tfm_params(*params2)
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
    camera_set.set_tfm_params(*params1)
    fids1_tfmd2d1 = camera_set.cams[0].p.dot(fids1)
    fids1_tfmd2d1 = fids1_tfmd2d1[:2, :]/fids1_tfmd2d1[2, :]
    fids1_tfmd2d2 = camera_set.cams[1].p.dot(fids1)
    fids1_tfmd2d2 = fids1_tfmd2d2[:2, :]/fids1_tfmd2d2[2, :]
    # Synthetic CT fiducials
    camera_set.set_tfm_params(*params2)
    fids2_tfmd2d1 = camera_set.cams[0].p.dot(fids2)
    fids2_tfmd2d1 = fids2_tfmd2d1[:2, :]/fids2_tfmd2d1[2, :]
    fids2_tfmd2d2 = camera_set.cams[1].p.dot(fids2)
    fids2_tfmd2d2 = fids2_tfmd2d2[:2, :]/fids2_tfmd2d2[2, :]
    rms_2d_1 = get_rms_err(fids1_tfmd2d1 - fids2_tfmd2d1)
    rms_2d_2 = get_rms_err(fids1_tfmd2d2 - fids2_tfmd2d2)
    # Find cam1 and cam2 images of synthetic CT and real CT, blend, draw markers
    im11 = plt.imread(args.params1.split('.')[0] + '_drr1.png')[..., :3]
    im12 = plt.imread(args.params1.split('.')[0] + '_drr2.png')[..., :3]
    im21 = plt.imread(args.params2.split('.')[0] + '_drr1.png')[..., :3]
    im22 = plt.imread(args.params2.split('.')[0] + '_drr2.png')[..., :3]
    fig1, ax1 = plot_drr(im11, im21, fids1_tfmd2d1, fids2_tfmd2d1)
    fig2, ax2 = plot_drr(im12, im22, fids1_tfmd2d2, fids2_tfmd2d2)
    fig1.savefig(
        os.path.join(args.results_dir, os.path.basename(args.params1).split('.')[0] + '_drr1.png'),
        bbox_inches='tight', pad_inches=0, transparent=True
    )
    fig2.savefig(
        os.path.join(args.results_dir, os.path.basename(args.params1).split('.')[0] + '_drr2.png'),
        bbox_inches='tight', pad_inches=0, transparent=True
    )
    np.set_printoptions(precision=2)
    print('{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(rms_3d_svd, rms_3d, rms_2d_1, rms_2d_2))
