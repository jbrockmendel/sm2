import numpy as np
from scipy.stats import rv_discrete
from scipy.special import gammaln


class genpoisson_p_gen(rv_discrete):
    """Generalized Poisson distribution"""
    def _argcheck(self, mu, alpha, p):
        return (mu >= 0) & (alpha == alpha) & (p > 0)

    def _logpmf(self, x, mu, alpha, p):
        mu_p = mu ** (p - 1.)
        a1 = np.maximum(np.nextafter(0, 1), 1 + alpha * mu_p)
        a2 = np.maximum(np.nextafter(0, 1), mu + (a1 - 1.) * x)
        logpmf_ = np.log(mu) + (x - 1.) * np.log(a2)
        logpmf_ -=  x * np.log(a1) + gammaln(x + 1.) + a2 / a1
        return logpmf_

    def _pmf(self, x, mu, alpha, p):
        return np.exp(self._logpmf(x, mu, alpha, p))


genpoisson_p = genpoisson_p_gen(name='genpoisson_p',
                                longname='Generalized Poisson')


def zipoisson_gen(*args, **kwargss):
    raise NotImplementedError('Not ported from upstream')


def zipoisson(*args, **kwargss):
    raise NotImplementedError('Not ported from upstream')


def zigeneralizedpoisson_gen(*args, **kwargss):
    raise NotImplementedError('Not ported from upstream')


def zigenpoisson(*args, **kwargss):
    raise NotImplementedError('Not ported from upstream')


def zinegativebinomial_gen(*args, **kwargss):
    raise NotImplementedError('Not ported from upstream')


def zinegbin(*args, **kwargss):
    raise NotImplementedError('Not ported from upstream')

