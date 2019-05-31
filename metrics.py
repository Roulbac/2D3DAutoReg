import numpy as np

def neg_normalized_cross_correlation(x, y):
    u = x - x.mean()
    v = y - y.mean()
    denom = np.sqrt(np.sum(u**2))*np.sqrt(np.sum(v**2))
    if denom == 0:
        return 0
    else:
        return -np.sum(u*v)/denom

def neg_gradient_corr(x, y):
    dx1, dx2 = np.gradient(x)
    dy1, dy2 = np.gradient(y)
    return 0.5*(
        neg_normalized_cross_correlation(dx1, dy1) + neg_normalized_cross_correlation(dx2, dy2)
    )

def mean_recipr_sqdiff(x, y):
    x_min_y_sq = (x-y)**2
    return np.mean(x_min_y_sq/(1+x_min_y_sq))

def neg_mutual_information(x, y, bins=32):
    x, y = x.flatten(), y.flatten()
    m, M = min(x.min(), y.min()), max(x.max(), y.max())
    px, _ = np.histogram(x, bins=bins, range=(m, M), density=True)
    py, _ = np.histogram(y, bins=bins, range=(m, M), density=True)
    pxpy = np.outer(px, py)
    pxy, _, _ = np.histogram2d(
        x, y, bins=bins, range=((m, M), (m, M)), density=True
    )
    nzero_ids = pxy != 0
    return -np.sum(pxy[nzero_ids]*np.log(pxy[nzero_ids]/pxpy[nzero_ids]))