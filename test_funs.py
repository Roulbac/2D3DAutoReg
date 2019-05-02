def test_ray_minmax_intersec():
    import numpy as np
    from box import Box
    from ray import Ray
    box = Box([2, 0, 0], [3, 3, 3], [1, 1, 1])
    i = np.array([0, -4, 0])
    j = np.array([3, 0, 1])
    k = np.array([4, 4/3, 4/3])
    l = np.array([3, -4, 1])
    m = np.array([3, 2, 1])
    n = np.array([0, 0, 0])
    o = np.array([1, 0, 0])
    # Test 1
    ray = Ray(i, j)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
    np.testing.assert_almost_equal(pt1, j)
    np.testing.assert_almost_equal(pt2, k)
    print('Test 1 OK')
    # Test 2
    ray = Ray(l, j)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
    np.testing.assert_almost_equal(pt1, j)
    np.testing.assert_almost_equal(pt2, m)
    print('Test 2 OK')
    # Test 3
    ray = Ray(l, n)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 3 OK')
    # Test 4
    ray = Ray(l, o)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 4 OK')
    # Test 5
    ray = Ray(i, n)
    pt1, _pt2 = box.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 5 OK')
    # Test 6
    ray = Ray(i, o)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 6 OK')
    # Non unit spacing
    box = Box([2, 0, 0], [5, 5, 5], [0.5, 0.5, 0.5])
    # Test 7
    ray = Ray(i, j)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
    np.testing.assert_almost_equal(pt1, j)
    np.testing.assert_almost_equal(pt2, k)
    print('Test 7 OK')
    # Test 8
    ray = Ray(l, j)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
    np.testing.assert_almost_equal(pt1, j)
    np.testing.assert_almost_equal(pt2, m)
    print('Test 8 OK')
    # Test 9
    ray = Ray(l, n)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 9 OK')
    # Test 10
    ray = Ray(l, o)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 10 OK')
    # Test 11
    ray = Ray(i, n)
    pt1, _pt2 = box.get_ray_minmax_intersec(ray)
    assert pt1 is None and pt2 is None
    print('Test 11 OK')
    # Test 12
    ray = Ray(i, o)
    pt1, pt2 = box.get_ray_minmax_intersec(ray)
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
