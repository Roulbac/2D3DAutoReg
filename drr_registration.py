from scipy.optimize import minimize, basinhopping
import metrics
from utils import make_xray_mask


class DrrRegistration(object):
    def __init__(self,
                 drr_set,
                 optimizer='Powell',
                 metric='neg_normalized_cross_correlation'):
        self.drr_set = drr_set
        self.optimizer = optimizer
        self._metric = getattr(metrics, metric)

    def set_xrays(self, xray1, xray2):
        self.xray1_mask, self.xray2_mask = make_xray_mask(xray1), make_xray_mask(xray2)
        self.xray1, self.xray2 = xray1*self.xray1_mask, xray2*self.xray2_mask


    def optimizer_callback(self, x):
        print('Metric value: {}, Params: {}'.format(self.metric_cache, x))

    def objective_function(self, x, *args):
        self.drr_set.set_tfm_params(*x.tolist())
        drr1, drr2 = self.drr_set.raybox.trace_rays()
        # TODO: Do mask multiplication in raybox
        drr1, drr2 = self.xray1_mask*drr1, self.xray2_mask*drr2
        self.metric_cache = float(self.metric(self.xray1, drr1) + self.metric(self.xray2, drr2))
        return self.metric_cache

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, metric):
        self._metric = getattr(metrics, metric)

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
