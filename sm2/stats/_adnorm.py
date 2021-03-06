# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:23:38 2011

Author: Josef Perktold and Scipy developers
License : BSD-3
"""
import numpy as np
from scipy import stats


def anderson_statistic(x, dist='norm', fit=True, params=(), axis=0):
    """calculate anderson-darling A2 statistic

    Parameters
    ----------
    x : array_like
        data
    dist : 'norm' or callable
        null distribution for the test statistic
    fit : bool
        If True, then the distribution parameters are estimated.
        Currently only for 1d data x, except in case dist='norm'
    params : tuple
        optional distribution parameters if fit is False
    axis : integer
        If dist is 'norm' or fit is False, then data can be an n-dimensional
        and axis specifies the axis of a variable

    Returns
    -------
    ad2 : float or ndarray
        Anderson-Darling statistic
    """
    x = np.asarray(x)
    y = np.sort(x, axis=axis)
    N = y.shape[axis]
    if fit:
        if dist == 'norm':
            xbar = np.expand_dims(np.mean(x, axis=axis), axis)
            s = np.expand_dims(np.std(x, ddof=1, axis=axis), axis)
            w = (y - xbar) / s
            z = stats.norm.cdf(w)
        elif callable(dist):
            params = dist.fit(x)
            z = dist.cdf(y, *params)
    else:
        if callable(dist):
            z = dist.cdf(y, *params)
        else:
            raise ValueError('If `fit` is False, then `dist` must be callable')

    i = np.arange(1, N + 1)

    sl1 = [None] * x.ndim
    sl1[axis] = slice(None)
    sl1 = tuple(sl1)

    sl2 = [slice(None)] * x.ndim
    sl2[axis] = slice(None, None, -1)
    sl2 = tuple(sl2)

    S = np.sum((2 * i[sl1] - 1.0) / N * (np.log(z) + np.log(1 - z[sl2])),
               axis=axis)
    A2 = -N - S
    return A2


def normal_ad(x, axis=0):
    """Anderson-Darling test for normal distribution unknown mean and variance

    Parameters
    ----------
    x : array_like
        data array, currently only 1d

    Returns
    -------
    ad2 : float
        Anderson Darling test statistic
    pval : float
        pvalue for hypothesis that the data comes from a normal distribution
        with unknown mean and variance
    """
    # ad2 = stats.anderson(x)[0]
    ad2 = anderson_statistic(x, dist='norm', fit=True, axis=axis)
    n = x.shape[axis]

    ad2a = ad2 * (1 + 0.75 / n + 2.25 / n**2)

    if np.size(ad2a) == 1:
        if (ad2a >= 0.00 and ad2a < 0.200):
            pval = 1 - np.exp(-13.436 + 101.14 * ad2a - 223.73 * ad2a**2)
        elif ad2a < 0.340:
            pval = 1 - np.exp(-8.318 + 42.796 * ad2a - 59.938 * ad2a**2)
        elif ad2a < 0.600:
            pval = np.exp(0.9177 - 4.279 * ad2a - 1.38 * ad2a**2)
        elif ad2a <= 13:
            pval = np.exp(1.2937 - 5.709 * ad2a + 0.0186 * ad2a**2)
        else:
            pval = 0.0  # is < 4.9542108058458799e-31

    else:
        bounds = np.array([0.0, 0.200, 0.340, 0.600])
        idx = np.searchsorted(bounds, ad2a, side='right')
        pval = np.nan * np.ones_like(ad2a)

        mask = (idx == 0)
        ad2m = ad2a[mask]
        pval[mask] = np.nan * np.ones_like(ad2m)

        mask = (idx == 1)
        ad2m = ad2a[mask]
        pval[mask] = 1 - np.exp(-13.436 + 101.14 * ad2m - 223.73 * ad2m**2)

        mask = (idx == 2)
        ad2m = ad2a[mask]
        pval[mask] = 1 - np.exp(-8.318 + 42.796 * ad2m - 59.938 * ad2m**2)

        mask = (idx == 3)
        ad2m = ad2a[mask]
        pval[mask] = np.exp(0.9177 - 4.279 * ad2m - 1.38 * ad2m**2)

        mask = (idx == 4)
        ad2m = ad2a[mask]
        pval[mask] = np.exp(1.2937 - 5.709 * ad2m + 0.0186 * ad2m**2)

    return ad2, pval
