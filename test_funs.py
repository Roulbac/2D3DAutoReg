def test_camera_plot():
    import numpy as np
    from camera import Camera
    m = np.matrix('[-0.785341, -0.068020, -0.615313, -5.901115;'
                  '0.559239, 0.348323, -0.752279, -4.000824;'
                  '0.265498, -0.934903, -0.235514, -663.099792;'
                  '0,        0,         0,                   0]')
    k = np.matrix('[3510.918213, 0.000000, 368.718994, 0;'
                  '0.000000, 3511.775635, 398.527802,  0;'
                  '0.000000, 0.000000, 1.000000,       0]')
    cam = Camera(m=m, k=k)
    cam.plot_camera3d()


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]]()
