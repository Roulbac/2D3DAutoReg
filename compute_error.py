import re
import argparse
import numpy as np
from camera import Camera
from camera_set import CameraSet
from utils import str_to_mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_1', type=str, required=True,
                        help='Camera 1 file')
    parser.add_argument('--camera_2', type=str, required=True,
                        help='Camera 2 file')
    parser.add_argument('--fiducials', type=str, required=True,
                        help='fiducials file')
    parser.add_argument('--ground_truth_params', type=str, required=True,
                        help='Params file for ground truth pose')
    parser.add_argument('--reg_params', type=str, required=True,
                        help='Params file for registered pose')
    args = parser.parse_args()
    camera_set_gt = CameraSet()
    camera_set_reg = CameraSet()
    with open(args.camera_1, 'r') as f:
        s = f.read()
        m1 = str_to_mat(re.search('[Mm]\s*=\s*\[(.*)\]', s).group(1))
        k1 = str_to_mat(re.search('[Kk]\s*=\s*\[(.*)\]', s).group(1))
        h1 = int(re.search('[Hh]\s*=\s*([0-9]+)', s).group(1))
        w1 = int(re.search('[Ww]\s*=\s*([0-9]+)', s).group(1))
        cam1 = Camera(m=m1, k=k1, h=h1, w=w1)
    with open(args.camera_2, 'r') as f:
        s = f.read()
        m2 = str_to_mat(re.search('[Mm]\s*=\s*\[(.*)\]', s).group(1))
        k2 = str_to_mat(re.search('[Kk]\s*=\s*\[(.*)\]', s).group(1))
        h2 = int(re.search('[Hh]\s*=\s*([0-9]+)', s).group(1))
        w2 = int(re.search('[Ww]\s*=\s*([0-9]+)', s).group(1))
        cam2 = Camera(m=m2, k=k2, h=h2, w=w2)
    camera_set_gt.set_cams(cam1, cam2)
    camera_set_reg.set_cams(cam1, cam2)
    with open(args.ground_truth_params, 'r') as f:
        s = f.read()
        tx = float(re.search('Tx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ty = float(re.search('Ty\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        tz = float(re.search('Tz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rx = float(re.search('Rx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ry = float(re.search('Ry\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rz = float(re.search('Rz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        camera_set_gt.set_tfm_params(tx, ty, tz, rx, ry, rz)
    with open(args.reg_params, 'r') as f:
        s = f.read()
        tx = float(re.search('Tx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ty = float(re.search('Ty\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        tz = float(re.search('Tz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rx = float(re.search('Rx\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        ry = float(re.search('Ry\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        rz = float(re.search('Rz\s*=\s*([-+]?[0-9]*\.?[0-9]*)', s).group(1))
        camera_set_reg.set_tfm_params(tx, ty, tz, rx, ry, rz)
    with open(args.fiducials, 'r') as f:
        s = f.read()
        fiducials = str_to_mat(s).T
        fiducials = np.vstack((fiducials, np.ones(fiducials.shape[1])))
    # Fiducial coords are in gt and registered space
    # Transform them back to original ijk CT space and compute error
    gt_fiducials = np.linalg.inv(camera_set_gt.tfm).dot(fiducials)
    gt_fiducials = gt_fiducials[:3, :] / gt_fiducials[3, :]
    reg_fiducials = np.linalg.inv(camera_set_reg.tfm).dot(fiducials)
    reg_fiducials = reg_fiducials[:3, :] / reg_fiducials[3, :]
    errors = np.linalg.norm(gt_fiducials - reg_fiducials, axis=1)
    print('Errors')
    print(errors)
    print('Mean {}, Std {}'.format(errors.mean(), errors.std()))
