def test_trace_rays_pycuda():
    import math
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import numpy as np
    import matplotlib.pyplot as plt
    with open('cuda_funs.cu', 'r') as f:
        s = f.read()
    mod = SourceModule(s)
    m = np.array([[1, 0, 0,  2],
                  [0, 0, -1, 1],
                  [0, 1, 0,  -4],
                  [0, 0, 0,  1]])
    minv = np.linalg.pinv(m).flatten().astype(np.float32)
    h = np.int32(720)
    w = np.int32(720)
    src = np.array([-2, 4, 1], dtype=np.float32)
    bs = np.array([-3, -2, 0], dtype=np.float32)
    ns = np.array([3, 3, 3], dtype=np.int32)
    spacing = np.array([1, 1, 1], dtype=np.float32)
    z_sign = np.int32(-1)
    k = np.array([[2*(h/2), 0,       1*(h/2), 0],
                  [0,       2*(w/2), 1*(w/2), 0],
                  [0,       0,       1,       0]])
    kinv = np.linalg.pinv(k).flatten().astype(np.float32)
    d_minv = cuda.mem_alloc(minv.nbytes)
    d_kinv = cuda.mem_alloc(kinv.nbytes)
    dsts = np.zeros(h*w*3, dtype=np.float32)
    raysums = np.zeros(h*w, dtype=np.float32)
    rho = np.ones((ns-1).tolist(), dtype=np.float32)
    d_dsts = cuda.mem_alloc(dsts.nbytes)
    d_raysums = cuda.mem_alloc(raysums.nbytes)
    d_rho = cuda.mem_alloc(rho.nbytes)
    d_bs = cuda.mem_alloc(bs.nbytes)
    d_ns = cuda.mem_alloc(ns.nbytes)
    d_spacing = cuda.mem_alloc(spacing.nbytes)
    d_src = cuda.mem_alloc(src.nbytes)
    cuda.memcpy_htod(d_minv, minv)
    cuda.memcpy_htod(d_kinv, kinv)
    cuda.memcpy_htod(d_bs, bs)
    cuda.memcpy_htod(d_ns, ns)
    cuda.memcpy_htod(d_spacing, spacing)
    cuda.memcpy_htod(d_rho, rho)
    cuda.memcpy_htod(d_src, src)
    f_backproj = mod.get_function('backprojectPixel')
    block = (16, 16, 1)
    grid = (math.ceil(h/block[0]), math.ceil(w/block[1]))
    f_backproj.prepare(['i', 'i', 'P', 'P', 'P', 'i'])
    f_backproj.prepared_call(
        grid, block, h, w, d_dsts, d_minv, d_kinv, z_sign
    )
    f_trace_rays = mod.get_function('traceRay')
    f_trace_rays.prepare(['P', 'P', 'P', 'P', 'P', 'P', 'P', 'i', 'i'])
    f_trace_rays.prepared_call(
        grid, block, d_src, d_dsts, d_raysums, d_rho, d_bs, d_ns, d_spacing, h, w)
    cuda.memcpy_dtoh(raysums, d_raysums)
    raysums = (raysums - raysums.min())/(raysums.max() - raysums.min())
    plt.imsave('raysums.png', raysums.reshape((h, w)), cmap='gray')


def test_trace_rays():
    from camera import Camera
    from box import Box
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
    image = Box.trace_rays(bs, ns, spacing, rho, cam)
    image = (image - image.min())/(image.max() - image.min())


def test_get_radiological_path():
    import numpy as np
    from box import Box
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
    box = Box(c, [3, 3, 3], [1, 1, 1])
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
        box.get_radiological_path(alphas_rkl, rkl), ln, decimal=2)
    print('Test 1 OK')
    np.testing.assert_almost_equal(i + (alphas_rij[0])*(j-i), j)
    np.testing.assert_almost_equal(i + (alphas_rij[1])*(j-i), o)
    np.testing.assert_almost_equal(
        box.get_radiological_path(alphas_rij, rij), jo, decimal=2)
    print('Test 2 OK')
    np.testing.assert_almost_equal(m + (alphas_rmj[0])*(j-m), j)
    np.testing.assert_almost_equal(m + (alphas_rmj[1])*(j-m), p)
    np.testing.assert_almost_equal(
        box.get_radiological_path(alphas_rmj, rmj), jp, decimal=2)
    print('Test 3 OK')
    np.testing.assert_almost_equal(q + (alphas_rql[0])*(l-q), l)
    np.testing.assert_almost_equal(q + (alphas_rql[1])*(l-q), r)
    np.testing.assert_almost_equal(
        box.get_radiological_path(alphas_rql, rql), lr, decimal=2)
    print('Test 4 OK')
    np.testing.assert_almost_equal(i + (alphas_ril[0])*(l-i), l)
    np.testing.assert_almost_equal(i + (alphas_ril[1])*(l-i), s)
    np.testing.assert_almost_equal(
        box.get_radiological_path(alphas_ril, ril), sl, decimal=2)
    print('Test 5 OK')
    np.testing.assert_almost_equal(m + (alphas_rml[0])*(l-m), t)
    np.testing.assert_almost_equal(m + (alphas_rml[1])*(l-m), l)
    np.testing.assert_almost_equal(
        box.get_radiological_path(alphas_rml, rml), tl, decimal=2)
    print('Test 6 OK')


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
