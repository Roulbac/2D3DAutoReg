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
        self._mask1 = (0.0, 0.0, 1.0, 1.0)
        self._mask2 = (0.0, 0.0, 1.0, 1.0)
        self._mask3 = (0.0, 0.0, 1.0, 1.0)
        self._mask4 = (0.0, 0.0, 1.0, 1.0)
        self.xrays = []

    def set_xrays(self, *xrays):
        self.xrays = xrays

    def optimizer_callback(self, x):
        print('Metric value: {}, Params: {}'.format(self.metric_cache, x))

    @staticmethod
    def array_from_mask(mask, shape):
        h, w = shape
        arr = np.zeros((h, w)).astype(np.bool)
        a, b, c, d = mask
        start_dim1, end_dim1 = min(int(b*h), int(d*h)), max(int(b*h), int(d*h))
        start_dim2, end_dim2 = min(int(a*w), int(c*w)), max(int(a*w), int(c*w))
        arr[start_dim1:end_dim1, start_dim2:end_dim2] = True
        return arr

    def objective_function(self, x, *args):
        self.drr_set.set_tfm_params(*x.tolist())
        drrs = self.drr_set.raybox.trace_rays()
        assert len(drrs) == len(self.xrays)
        n = len(drrs)
        metric_value = 0
        for idx in range(n):
            mask = self.array_from_mask(
                getattr(self, 'mask{:d}'.format(idx+1)),
                drrs[idx].shape
            )
            metric_value += float(self.metric(self.xrays[idx], drrs[idx], mask))
        self.metric_cache = metric_value
        return metric_value

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

    @property
    def mask3(self):
        return self._mask3

    @mask3.setter
    def mask3(self, mask):
        assert isinstance(mask, (tuple, list)) and len(mask) == 4
        self._mask3 = tuple(mask)

    @property
    def mask4(self):
        return self._mask4

    @mask4.setter
    def mask4(self, mask):
        assert isinstance(mask, (tuple, list)) and len(mask) == 4
        self._mask4 = tuple(mask)

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
