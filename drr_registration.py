import numpy as np
from scipy.optimize import minimize
import metrics

class DrrRegistration(object):
    def __init__(self,
                 drr_set,
                 optimizer='Powell',
                 metric='neg_normalized_cross_correlation'):
        self.drr_set = drr_set
        self.optimizer = optimizer
        self._metric = getattr(metrics, metric)
        self._mask1, self._mask2 = (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)

    def set_xrays(self, xray1, xray2):
        self.xray1, self.xray2 = xray1, xray2

    def optimizer_callback(self, x):
        print('Metric value: {}, Params: {}'.format(self.metric_cache, x))

    def objective_function(self, x, *args):
        self.drr_set.set_tfm_params(*x.tolist())
        drr1, drr2 = self.drr_set.raybox.trace_rays()
        assert drr1.shape == self.xray1.shape and drr2.shape == self.xray2.shape
        h1, w1 = drr1.shape
        h2, w2 = drr2.shape
        a1, b1, c1, d1 = self.mask1
        a2, b2, c2, d2 = self.mask2
        mask1 = np.zeros(drr1.shape).astype(np.bool)
        mask2 = np.zeros(drr2.shape).astype(np.bool)
        start_dim11, end_dim11 = min(int(b1*h1), int(d1*h1)), max(int(b1*h1), int(d1*h1))
        start_dim21, end_dim21 = min(int(a1*w1), int(c1*w1)), max(int(a1*w1), int(c1*w1))
        start_dim12, end_dim12 = min(int(b2*h2), int(d2*h2)), max(int(b2*h2), int(d2*h2))
        start_dim22, end_dim22 = min(int(a2*w2), int(c2*w2)), max(int(a2*w2), int(c2*w2))
        mask1[start_dim11:end_dim11, start_dim21:end_dim21] = True
        mask2[start_dim12:end_dim12, start_dim22:end_dim22] = True
        self.metric_cache = float(self.metric(self.xray1, drr1, mask1) + self.metric(self.xray2, drr2, mask2))
        return self.metric_cache

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, metric):
        self._metric = getattr(metrics, metric)

    @property
    def mask1(self):
        return self._mask1

    @mask1.setter
    def mask1(self, mask):
        assert isinstance(mask, (tuple, list)) and len(mask) == 4
        self._mask1 = tuple(mask)

    @property
    def mask2(self):
        return self._mask2

    @mask2.setter
    def mask2(self, mask):
        assert isinstance(mask, (tuple, list)) and len(mask) == 4
        self._mask2 = tuple(mask)

    def register(self,
                 x0,
                 xtol=1e-6,
                 ftol=1e-6):
        return minimize(
            self.objective_function,
            x0,
            method=self.optimizer,
            callback=self.optimizer_callback,
            options=dict(xtol=xtol, ftol=ftol, disp=True)
        )
