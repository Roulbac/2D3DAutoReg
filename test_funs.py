def test_drr_registration():
    import numpy as np
    from PIL import Image
    from camera import Camera
    from raybox import RayBox
    from drr_set import DrrSet
    from drr_registration import DrrRegistration
    from utils import read_rho, str_to_mat
    xray1 = Image.open('Test_Data/Sawbones_L2L3/0.bmp').convert('L')
    xray2 = Image.open('Test_Data/Sawbones_L2L3/90.bmp').convert('L')
    xray1 = np.array(xray1).astype(np.float32)
    xray2 = np.array(xray2).astype(np.float32)
    xray1 = (xray1-xray1.min())/(xray1.max()-xray1.min())
    xray2 = (xray2-xray2.min())/(xray2.max()-xray2.min())
    m1 = str_to_mat('[-0.785341, -0.068020, -0.615313, -5.901115; 0.559239, 0.348323, -0.752279, -4.000824; 0.265498, -0.934903, -0.235514, -663.099792]')
    m2 = str_to_mat('[-0.214846, 0.964454, 0.153853, 12.792526; 0.557581, 0.250463, -0.791436, -6.176056; -0.801838, -0.084251, -0.591572, -627.625305]')
    k1 = str_to_mat('[3510.918213, 0.000000, 368.718994; 0.000000, 3511.775635, 398.527802; 0.000000, 0.000000, 1.000000]')
    k2 = str_to_mat('[3533.860352, 0.000000, 391.703888; 0.000000, 3534.903809, 395.485229; 0.000000, 0.000000, 1.000000]')
    cam1, cam2 = Camera(m1, k1), Camera(m2, k2)
    raybox = RayBox('cpu')
    rho, sp = read_rho('Test_Data/Sawbones_L2L3/sawbones.nii.gz')
    raybox.set_rho(rho, sp)
    drr_set = DrrSet(cam1, cam2, raybox)
    drr_registration = DrrRegistration(xray1, xray2, drr_set)
    res = drr_registration.register(np.array([-98.92, -106.0, -185.0, -35.0, 25.0, 175]))
    print(res)


def test_camera_set_tfm_fixed():
    import time
    from camera import Camera
    from camera_set import CameraSet
    from utils import str_to_mat
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    m1 = str_to_mat('[-0.785341, -0.068020, -0.615313, -5.901115; 0.559239, 0.348323, -0.752279, -4.000824; 0.265498, -0.934903, -0.235514, -663.099792]')
    m2 = str_to_mat('[-0.214846, 0.964454, 0.153853, 12.792526; 0.557581, 0.250463, -0.791436, -6.176056; -0.801838, -0.084251, -0.591572, -627.625305]')
    k1 = str_to_mat('[3510.918213, 0.000000, 368.718994; 0.000000, 3511.775635, 398.527802; 0.000000, 0.000000, 1.000000]')
    k2 = str_to_mat('[3533.860352, 0.000000, 391.703888; 0.000000, 3534.903809, 395.485229; 0.000000, 0.000000, 1.000000]')
    cam1 = Camera(m=m1, k=k1)
    cam2 = Camera(m=m2, k=k2)
    camset = CameraSet(cam1, cam2)
    center = camset.center
    # Z rotations, i.e PSI from 0 to 360 degrees
    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for rz in range(-180, 180, 5):
        params = [0, 0, 0, 0, 0, rz]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot_fixed(fig, ax, '1')
        cam2._make_cam_plot_fixed(fig, ax, '2')
        ax.text(center[0], center[1], center[2], 'Z rotation')
        r = camset.tfm[:3, :3]
        u, v ,w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
        ax.plot3D([center[0], center[0] + u[0]],
                  [center[1], center[1] + u[1]],
                  [center[2], center[2] + u[2]],
                  color='red')
        ax.plot3D([center[0], center[0] + v[0]],
                  [center[1], center[1] + v[1]],
                  [center[2], center[2] + v[2]],
                  color='green')
        ax.plot3D([center[0], center[0] + w[0]],
                  [center[1], center[1] + w[1]],
                  [center[2], center[2] + w[2]],
                  color='blue')
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        ax.clear()
    for ry in range(-180, 180, 5):
        params = [0, 0, 0, 0, ry, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot_fixed(fig, ax, '1')
        cam2._make_cam_plot_fixed(fig, ax, '2')
        ax.text(center[0], center[1], center[2], 'Y rotation')
        r = camset.tfm[:3, :3]
        u, v ,w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
        ax.plot3D([center[0], center[0] + u[0]],
                  [center[1], center[1] + u[1]],
                  [center[2], center[2] + u[2]],
                  color='red')
        ax.plot3D([center[0], center[0] + v[0]],
                  [center[1], center[1] + v[1]],
                  [center[2], center[2] + v[2]],
                  color='green')
        ax.plot3D([center[0], center[0] + w[0]],
                  [center[1], center[1] + w[1]],
                  [center[2], center[2] + w[2]],
                  color='blue')
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        ax.clear()
    for rx in range(-180, 180, 5):
        params = [0, 0, 0, rx, 0, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot_fixed(fig, ax, '1')
        cam2._make_cam_plot_fixed(fig, ax, '2')
        ax.text(center[0], center[1], center[2], 'X rotation')
        r = camset.tfm[:3, :3]
        u, v ,w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
        ax.plot3D([center[0], center[0] + u[0]],
                  [center[1], center[1] + u[1]],
                  [center[2], center[2] + u[2]],
                  color='red')
        ax.plot3D([center[0], center[0] + v[0]],
                  [center[1], center[1] + v[1]],
                  [center[2], center[2] + v[2]],
                  color='green')
        ax.plot3D([center[0], center[0] + w[0]],
                  [center[1], center[1] + w[1]],
                  [center[2], center[2] + w[2]],
                  color='blue')
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        ax.clear()
    for tx in range(-100, 100, 5):
        params = [tx, 0, 0, 0, 0, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot_fixed(fig, ax, '1')
        cam2._make_cam_plot_fixed(fig, ax, '2')
        r = camset.tfm[:3, :3]
        u, v, w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
        tx, ty, tz = camset.tfm[:3, 3]
        ax.text(center[0] + tx, center[1] + ty, center[2] + tz, 'X translation')
        ax.plot3D([center[0] + tx, center[0] + tx + u[0]],
                  [center[1] + ty, center[1] + ty + u[1]],
                  [center[2] + tz, center[2] + tz + u[2]],
                  color='red')
        ax.plot3D([center[0] + tx, center[0] + tx + v[0]],
                  [center[1] + ty, center[1] + ty + v[1]],
                  [center[2] + tz, center[2] + tz + v[2]],
                  color='green')
        ax.plot3D([center[0] + tx, center[0] + tx + w[0]],
                  [center[1] + ty, center[1] + ty + w[1]],
                  [center[2] + tz, center[2] + tz + w[2]],
                  color='blue')
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        ax.clear()
    for ty in range(-100, 100, 5):
        params = [0, ty, 0, 0, 0, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot_fixed(fig, ax, '1')
        cam2._make_cam_plot_fixed(fig, ax, '2')
        r = camset.tfm[:3, :3]
        u, v, w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
        tx, ty, tz = camset.tfm[:3, 3]
        ax.text(center[0] + tx, center[1] + ty, center[2] + tz, 'Y translation')
        ax.plot3D([center[0] + tx, center[0] + tx + u[0]],
                  [center[1] + ty, center[1] + ty + u[1]],
                  [center[2] + tz, center[2] + tz + u[2]],
                  color='red')
        ax.plot3D([center[0] + tx, center[0] + tx + v[0]],
                  [center[1] + ty, center[1] + ty + v[1]],
                  [center[2] + tz, center[2] + tz + v[2]],
                  color='green')
        ax.plot3D([center[0] + tx, center[0] + tx + w[0]],
                  [center[1] + ty, center[1] + ty + w[1]],
                  [center[2] + tz, center[2] + tz + w[2]],
                  color='blue')
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        ax.clear()
    for tz in range(-100, 100, 5):
        params = [0, 0, tz, 0, 0, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot_fixed(fig, ax, '1')
        cam2._make_cam_plot_fixed(fig, ax, '2')
        r = camset.tfm[:3, :3]
        u, v, w = 100*r[0, :], 100*r[1, :], 100*r[2, :]
        tx, ty, tz = camset.tfm[:3, 3]
        ax.text(center[0] + tx, center[1] + ty, center[2] + tz, 'Z translation')
        ax.plot3D([center[0] + tx, center[0] + tx + u[0]],
                  [center[1] + ty, center[1] + ty + u[1]],
                  [center[2] + tz, center[2] + tz + u[2]],
                  color='red')
        ax.plot3D([center[0] + tx, center[0] + tx + v[0]],
                  [center[1] + ty, center[1] + ty + v[1]],
                  [center[2] + tz, center[2] + tz + v[2]],
                  color='green')
        ax.plot3D([center[0] + tx, center[0] + tx + w[0]],
                  [center[1] + ty, center[1] + ty + w[1]],
                  [center[2] + tz, center[2] + tz + w[2]],
                  color='blue')
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        ax.clear()

def test_camera_set_tfm():
    import time
    import numpy as np
    from camera import Camera
    from camera_set import CameraSet
    from utils import str_to_mat
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    m1 = str_to_mat('[-0.785341, -0.068020, -0.615313, -5.901115; 0.559239, 0.348323, -0.752279, -4.000824; 0.265498, -0.934903, -0.235514, -663.099792]')
    m2 = str_to_mat('[-0.214846, 0.964454, 0.153853, 12.792526; 0.557581, 0.250463, -0.791436, -6.176056; -0.801838, -0.084251, -0.591572, -627.625305]')
    k1 = str_to_mat('[3510.918213, 0.000000, 368.718994; 0.000000, 3511.775635, 398.527802; 0.000000, 0.000000, 1.000000]')
    k2 = str_to_mat('[3533.860352, 0.000000, 391.703888; 0.000000, 3534.903809, 395.485229; 0.000000, 0.000000, 1.000000]')
    cam1 = Camera(m=m1, k=k1)
    cam2 = Camera(m=m2, k=k2)
    camset = CameraSet(cam1, cam2)
    center = camset.center
    # Z rotations, i.e PSI from 0 to 360 degrees
    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for rz in range(-180, 180, 5):
         params = [0, 0, 0, 0, 0, rz]
         camset.set_tfm_params(*params)
         cam1._make_cam_plot(fig, ax, '1')
         cam2._make_cam_plot(fig, ax, '2')
         ax.text(center[0], center[1], center[2], 'Z rotation')
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
         ax.set_xlim3d(center[0] - 600, center[0] + 600)
         ax.set_ylim3d(center[1] - 600, center[1] + 600)
         ax.set_zlim3d(center[2] - 600, center[2] + 600)
         p1 = cam1.pos
         p2 = cam2.pos
         c = camset.center
         verts = [c-(p1+p2), c-(p1-p2), c+(p1+p2), c+(p1-p2)]
         verts = [list(map(lambda x: x.tolist(), verts))]
         parallelogram = Poly3DCollection(verts)
         ax.add_collection3d(parallelogram, zs='z')
         fig.canvas.draw()
         fig.canvas.flush_events()
         plt.pause(0.001)
         ax.clear()
    for ry in range(-180, 180, 5):
        params = [0, 0, 0, 0, ry, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot(fig, ax, '1')
        cam2._make_cam_plot(fig, ax, '2')
        ax.text(center[0], center[1], center[2], 'Y rotation')
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
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        p1 = cam1.pos
        p2 = cam2.pos
        c = camset.center
        verts = [c-(p1+p2), c-(p1-p2), c+(p1+p2), c+(p1-p2)]
        verts = [list(map(lambda x: x.tolist(), verts))]
        parallelogram = Poly3DCollection(verts)
        ax.add_collection3d(parallelogram, zs='z')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        ax.clear()
    for rx in range(-180, 180, 5):
        params = [0, 0, 0, rx, 0, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot(fig, ax, '1')
        cam2._make_cam_plot(fig, ax, '2')
        ax.text(center[0], center[1], center[2], 'X rotation')
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
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        p1 = cam1.pos
        p2 = cam2.pos
        c = camset.center
        verts = [c-(p1+p2), c-(p1-p2), c+(p1+p2), c+(p1-p2)]
        verts = [list(map(lambda x: x.tolist(), verts))]
        parallelogram = Poly3DCollection(verts)
        ax.add_collection3d(parallelogram, zs='z')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        ax.clear()
    for tx in range(-100, 100, 5):
        params = [tx, 0, 0, 0, 0, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot(fig, ax, '1')
        cam2._make_cam_plot(fig, ax, '2')
        ax.text(center[0], center[1], center[2], 'X translation')
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
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        c = camset.center
        t_c = np.linalg.inv(camset.tfm).dot(c.tolist() + [1])[:3]
        ax.plot3D([t_c[0]], [t_c[1]], [t_c[2]], marker='*', color='red')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
        ax.clear()
    for ty in range(-100, 100, 5):
        params = [0, ty, 0, 0, 0, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot(fig, ax, '1')
        cam2._make_cam_plot(fig, ax, '2')
        ax.text(center[0], center[1], center[2], 'Y translation')
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
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        c = camset.center
        t_c = np.linalg.inv(camset.tfm).dot(c.tolist() + [1])[:3]
        ax.plot3D([t_c[0]], [t_c[1]], [t_c[2]], marker='*', color='red')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
        ax.clear()
    for tz in range(-100, 100, 5):
        params = [0, 0, tz, 0, 0, 0]
        camset.set_tfm_params(*params)
        cam1._make_cam_plot(fig, ax, '1')
        cam2._make_cam_plot(fig, ax, '2')
        ax.text(center[0], center[1], center[2], 'Z translation')
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
        ax.set_xlim3d(center[0] - 600, center[0] + 600)
        ax.set_ylim3d(center[1] - 600, center[1] + 600)
        ax.set_zlim3d(center[2] - 600, center[2] + 600)
        c = camset.center
        t_c = np.linalg.inv(camset.tfm).dot(c.tolist() + [1])[:3]
        ax.plot3D([t_c[0]], [t_c[1]], [t_c[2]], marker='*', color='red')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
        ax.clear()

def test_camera_set_pbyref():
    import numpy as np
    from camera import Camera
    from camera_set import CameraSet
    cam1 = Camera()
    cam2 = Camera()
    camset = CameraSet(cam1, cam2)
    camset.set_tfm(np.eye(3)*0.5, np.arange(3))
    assert np.linalg.norm(cam1.tfm - np.eye(4)) > 0.01
    assert np.linalg.norm(cam2.tfm - np.eye(4)) > 0.01
    print('Test OK')

def test_drr_sawbones():
    import numpy as np
    import SimpleITK as sitk
    from camera import Camera
    from raybox import RayBox
    import matplotlib.pyplot as plt
    h = np.int32(460)
    w = np.int32(460)
    m = np.array([[0, 0, -1, 143],
                   [1,  0, 0, -96],
                   [0,  -1, 0, -770]])
    k = np.array([[1001, 0,       204.5, 0],
                  [0,       1001, 137.3, 0],
                  [0,       0,        1, 0]])
    k[0,0], k[1,1] = 1.5*k[0,0], 1.5*k[1,1]
    # m = np.array([[ 1.        ,  0.        ,  0.        ,  2.        ],
    #    [ 0.        ,  0.39073113, -0.92050485, -0.64241966],
    #    [ 0.        ,  0.92050485,  0.39073113, -4.07275054],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    rho = sitk.GetArrayFromImage(sitk.ReadImage('Test_Data/Sawbones_L2L3/sawbones.nii.gz')).transpose((1, 2, 0)).astype(np.float32)
    sp = np.array([0.375, 0.375, 0.625], dtype=np.float32)[[1, 0, 2]]
    n = np.array([513, 513, 456], dtype=np.int32)[[1, 0, 2]]
    rho = rho[::, ::-1, ::]
    #rho = np.ones((512, 512, 455))
    cam1 = Camera(m, k, h=h, w=w)
    cam2 = Camera(m, k, h=h, w=w)
    raybox = RayBox('cpu')
    raybox.set_cams(cam1, cam2)
    raybox.set_rho(rho, sp)
    raybox.mode = 'gpu'
    raysums1, raysums2 = raybox.trace_rays()
    print(raysums1.max(), raysums2.max())
    plt.imsave('raysums1.png', raysums1, cmap='gray', vmin=0, vmax=1)


def test_raybox_class():
    import numpy as np
    from camera import Camera
    from raybox import RayBox
    import matplotlib.pyplot as plt
    # m = np.array([[1, 0, 0,  2],
    #               [0, 0, -1, 1],
    #               [0, 1, 0,  -4],
    #               [0, 0, 0,  1]])
    m = np.array([[0, -1, 0, -1],
                   [0, 0, -1, 1],
                   [1, 0, 0, -3],
                   [0, 0, 0, 1]])
    # m = np.array([[ 1.        ,  0.        ,  0.        ,  2.        ],
    #    [ 0.        ,  0.39073113, -0.92050485, -0.64241966],
    #    [ 0.        ,  0.92050485,  0.39073113, -4.07275054],
    #    [ 0.        ,  0.        ,  0.      #  ,  1.        ]])
    h = np.int32(768)
    w = np.int32(768)
    n = np.array([8, 8, 8], dtype=np.int32)
    sp = np.array([1, 1, 1], dtype=np.float32)
    k = np.array([[2*(h/2), 0,       1*(h/2), 0],
                  [0,       2*(w/2), 1*(w/2), 0],
                  [0,       0,       1,       0]])
    rho = np.ones((n-1).tolist(), dtype=np.float32)
    cam1 = Camera(m, k, h=h, w=w)
    cam2 = Camera(m, k, h=h, w=w)
    raybox = RayBox('gpu')
    raybox.set_cams(cam1, cam2)
    raybox.set_rho(rho, sp)
    raysums1, raysums2 = raybox.trace_rays()
    print(raysums1.max(), raysums2.max())
    plt.imsave('raysums1.png', raysums1, cmap='gray')

def test_trace_rays_pycuda():
    import math
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import numpy as np
    import matplotlib.pyplot as plt
    # Define data
    with open('kernels.cu', 'r') as f:
        cuda_src = f.read()
    mod = SourceModule(cuda_src)
    m = np.array([[1, 0, 0,  2],
                  [0, 0, -1, 1],
                  [0, 1, 0,  -4],
                  [0, 0, 0,  1]])
    h = np.int32(720)
    w = np.int32(720)
    src = np.array([-2, 4, 1], dtype=np.float32)
    b = np.array([-3, -2, 0], dtype=np.float32)
    n = np.array([3, 3, 3], dtype=np.int32)
    sp = np.array([1, 1, 1], dtype=np.float32)
    z_sign = np.int32(-1)
    k = np.array([[2*(h/2), 0,       1*(h/2), 0],
                  [0,       2*(w/2), 1*(w/2), 0],
                  [0,       0,       1,       0]])
    # Compute transformations of data
    minv = np.linalg.pinv(m).flatten().astype(np.float32)
    kinv = np.linalg.pinv(k).flatten().astype(np.float32)
    dsts = np.zeros(h*w*3, dtype=np.float32)
    raysums = np.zeros(h*w, dtype=np.float32)
    rho = np.ones((n-1).tolist(), dtype=np.float32)
    # Allocate on device
    d_minv = cuda.mem_alloc(minv.nbytes)
    d_kinv = cuda.mem_alloc(kinv.nbytes)
    d_dsts = cuda.mem_alloc(dsts.nbytes)
    d_raysums = cuda.mem_alloc(raysums.nbytes)
    d_rho = cuda.mem_alloc(rho.nbytes)
    d_src = cuda.mem_alloc(src.nbytes)
    d_sp = cuda.mem_alloc(sp.nbytes)
    d_n = cuda.mem_alloc(n.nbytes)
    d_b = cuda.mem_alloc(b.nbytes)
    # Get pointers to consts
    # Copy data to device
    cuda.memcpy_htod(d_minv, minv)
    cuda.memcpy_htod(d_kinv, kinv)
    cuda.memcpy_htod(d_rho, rho)
    cuda.memcpy_htod(d_src, src)
    cuda.memcpy_htod(d_sp, sp)
    cuda.memcpy_htod(d_n, n)
    cuda.memcpy_htod(d_b, b)
    # Initialize kernels
    block = (16, 16, 1)
    grid = (math.ceil(h/block[0]), math.ceil(w/block[1]))
    f_backproj = mod.get_function('backprojectPixel')
    f_backproj.prepare(['i', 'i', 'P', 'P', 'P', 'i'])
    f_backproj.prepared_call(grid, block, h, w, d_dsts, d_minv, d_kinv, z_sign)
    f_trace_rays = mod.get_function('traceRay')
    f_trace_rays.prepare(['P', 'P', 'P', 'P', 'P', 'P', 'P', 'i', 'i'])
    f_trace_rays.prepared_call(grid, block, d_src, d_dsts, d_raysums, d_rho,
                               d_b, d_sp, d_n, h, w)
    # Copy results and free memory
    cuda.memcpy_dtoh(raysums, d_raysums)
    print(raysums.max())
    plt.imsave('raysums.png', raysums.reshape((h, w)), cmap='gray')
    print('Write raysums.png')


def test_trace_rays():
    from camera import Camera
    from raybox import RayBox
    import numpy as np
    import matplotlib.pyplot as plt
    m = np.array([[1, 0, 0,  2],
                  [0, 0, -1, 1],
                  [0, 1, 0,  -4],
                  [0, 0, 0,  1]])
    h = 8
    w = 8
    k = np.array([[2*(h/2), 0,       1*(h/2), 0],
                  [0,       2*(w/2), 1*(w/2), 0],
                  [0,       0,       1,       0]])
    cam = Camera(m=m, k=k, h=h, w=w)
    bs = np.array([-3, -2, 0])
    ns = [3, 3, 3]
    spacing = [1, 1, 1]
    rho = np.ones((ns[0] - 1, ns[1] - 1, ns[2] - 1))
    image = RayBox.trace_rays(bs, ns, spacing, rho, cam)
    image = (image - image.min())/(image.max() - image.min())


def test_get_radiological_path():
    import numpy as np
    from raybox import RayBox
    from ray import Ray
    c = np.array([-3, -2, 0])
    i = np.array([0, 4, 0])
    j = np.array([-2, 0, 1])
    k = np.array([3, 0, 0])
    l = np.array([-1, -1, 1])
    m = np.array([-2, 4, 1])
    n = np.array([-3, -1.5, 1.5])
    o = np.array([-3, -2, 1.5])
    p = np.array([-2, -2, 1])
    q = np.array([0, 0, 0])
    r = np.array([-2, -2, 2])
    s = np.array([-1.2, -2, 1.2])
    t = np.array([-1.2, 0, 1])
    ln = 2.12
    jp = 2
    jo = 2.29
    lr = 1.73
    tl = 1.02
    sl = 1.04
    rij = Ray(i, j)
    rkl = Ray(k, l)
    rmj = Ray(m, j)
    rql = Ray(q, l)
    ril = Ray(i, l)
    rml = Ray(m, l)
    raybox = RayBox(c, [3, 3, 3], [1, 1, 1])
    alphas_rij = (1.0, 1.5, 0.5, 1.5, 1.0, 1.5, 0.0, 2.0)
    alphas_rkl = (1.0, 1.5, 1.0, 1.5, -0.0, 2.0, 0.0, 2.0)
    alphas_rmj = (1.0, 1.5, float("-inf"), float("inf"),
                  1.0, 1.5, float("-inf"), float("inf"))
    alphas_rql = (1.0, 2.0, 1.0, 3.0, -0.0, 2.0, 0.0, 2.0)
    alphas_ril = (1.0, 1.2, 1.0, 3.0, 0.8, 1.2, 0.0, 2.0)
    alphas_rml = (0.8, 1.0, -1.0, 1.0, 0.8, 1.2, float("-inf"), float("inf"))
    np.testing.assert_almost_equal(k + (alphas_rkl[0])*(l-k), l)
    np.testing.assert_almost_equal(k + (alphas_rkl[1])*(l-k), n)
    np.testing.assert_almost_equal(
        raybox.get_radiological_path(alphas_rkl, rkl), ln, decimal=2)
    print('Test 1 OK')
    np.testing.assert_almost_equal(i + (alphas_rij[0])*(j-i), j)
    np.testing.assert_almost_equal(i + (alphas_rij[1])*(j-i), o)
    np.testing.assert_almost_equal(
        raybox.get_radiological_path(alphas_rij, rij), jo, decimal=2)
    print('Test 2 OK')
    np.testing.assert_almost_equal(m + (alphas_rmj[0])*(j-m), j)
    np.testing.assert_almost_equal(m + (alphas_rmj[1])*(j-m), p)
    np.testing.assert_almost_equal(
        raybox.get_radiological_path(alphas_rmj, rmj), jp, decimal=2)
    print('Test 3 OK')
    np.testing.assert_almost_equal(q + (alphas_rql[0])*(l-q), l)
    np.testing.assert_almost_equal(q + (alphas_rql[1])*(l-q), r)
    np.testing.assert_almost_equal(
        raybox.get_radiological_path(alphas_rql, rql), lr, decimal=2)
    print('Test 4 OK')
    np.testing.assert_almost_equal(i + (alphas_ril[0])*(l-i), l)
    np.testing.assert_almost_equal(i + (alphas_ril[1])*(l-i), s)
    np.testing.assert_almost_equal(
        raybox.get_radiological_path(alphas_ril, ril), sl, decimal=2)
    print('Test 5 OK')
    np.testing.assert_almost_equal(m + (alphas_rml[0])*(l-m), t)
    np.testing.assert_almost_equal(m + (alphas_rml[1])*(l-m), l)
    np.testing.assert_almost_equal(
        raybox.get_radiological_path(alphas_rml, rml), tl, decimal=2)
    print('Test 6 OK')


def test_ray_minmax_intersec():
    import numpy as np
    from raybox import RayBox
    from ray import Ray
    raybox = RayBox([2, 0, 0], [3, 3, 3], [1, 1, 1])
    i = np.array([0, -4, 0])
    j = np.array([3, 0, 1])
    k = np.array([4, 4/3, 4/3])
    l = np.array([3, -4, 1])
    m = np.array([3, 2, 1])
    n = np.array([0, 0, 0])
    o = np.array([1, 0, 0])
    # Test 1
    ray = Ray(i, j)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    np.testing.assert_almost_equal(pt1, j)
    np.testing.assert_almost_equal(pt2, k)
    print('Test 1 OK')
    # Test 2
    ray = Ray(l, j)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    np.testing.assert_almost_equal(pt1, j)
    np.testing.assert_almost_equal(pt2, m)
    print('Test 2 OK')
    # Test 3
    ray = Ray(l, n)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 3 OK')
    # Test 4
    ray = Ray(l, o)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 4 OK')
    # Test 5
    ray = Ray(i, n)
    pt1, _pt2 = raybox.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 5 OK')
    # Test 6
    ray = Ray(i, o)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 6 OK')
    # Non unit spacing
    raybox = RayBox([2, 0, 0], [5, 5, 5], [0.5, 0.5, 0.5])
    # Test 7
    ray = Ray(i, j)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    np.testing.assert_almost_equal(pt1, j)
    np.testing.assert_almost_equal(pt2, k)
    print('Test 7 OK')
    # Test 8
    ray = Ray(l, j)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    np.testing.assert_almost_equal(pt1, j)
    np.testing.assert_almost_equal(pt2, m)
    print('Test 8 OK')
    # Test 9
    ray = Ray(l, n)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 9 OK')
    # Test 10
    ray = Ray(l, o)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 10 OK')
    # Test 11
    ray = Ray(i, n)
    pt1, _pt2 = raybox.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 11 OK')
    # Test 12
    ray = Ray(i, o)
    pt1, pt2 = raybox.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 12 OK')


def test_camera_plot():
    import numpy as np
    from camera import Camera
    m = np.matrix('[-0.785341, -0.068020, -0.615313, -5.901115;'
                  '0.559239, 0.348323, -0.752279, -4.000824;'
                  '0.265498, -0.934903, -0.235514, -663.099792;'
                  '0,        0,         0,                   1]')
    k = np.matrix('[3510.918213, 0.000000, 368.718994, 0;'
                  '0.000000, 3511.775635, 398.527802,  0;'
                  '0.000000, 0.000000, 1.000000,       0]')
    cam = Camera(m=m, k=k)
    cam.plot_camera3d()


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]]()
