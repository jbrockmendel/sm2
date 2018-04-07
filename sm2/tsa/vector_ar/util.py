# -*- coding: utf-8 -*-
"""
Miscellaneous utility code for VAR estimation
"""
from __future__ import division

from six import string_types, integer_types
from six.moves import range

import numpy as np
from scipy import stats
from scipy.linalg import decomp

import sm2.tsa.tsatools as tsa

from sm2.tsa.autocov import acf_to_acorr  # noqa:F841


# ---------------------------------------------------------------
# Auxiliary functions for estimation

def get_var_endog(y, lags, trend='c', has_constant='skip'):
    """
    Make predictor matrix for VAR(p) process

    Z := (Z_0, ..., Z_T).T (T x Kp)
    Z_t = [1 y_t y_{t-1} ... y_{t - p + 1}] (Kp x 1)

    Ref: Lütkepohl p.70 (transposed)

    has_constant can be 'raise', 'add', or 'skip'. See add_constant.
    """
    nobs = len(y)
    # Ravel C order, need to put in descending order
    Z = np.array([y[t - lags: t][::-1].ravel() for t in range(lags, nobs)])

    # Add constant, trend, etc.
    if trend != 'nc':
        Z = tsa.add_trend(Z, prepend=True, trend=trend,
                          has_constant=has_constant)
    return Z


def get_trendorder(trend='c'):
    # Handle constant, etc.
    if trend == 'c':
        trendorder = 1
    elif trend == 'nc':
        trendorder = 0
    elif trend == 'ct':
        trendorder = 2
    elif trend == 'ctt':
        trendorder = 3
    return trendorder


def make_lag_names(names, lag_order, trendorder=1, exog=None):
    """
    Produce list of lag-variable names. Constant / trends go at the beginning

    Examples
    --------
    >>> make_lag_names(['foo', 'bar'], 2, 1)
    ['const', 'L1.foo', 'L1.bar', 'L2.foo', 'L2.bar']

    """
    lag_names = []
    if isinstance(names, string_types):
        names = [names]

    # take care of lagged endogenous names
    for i in range(1, lag_order + 1):
        for name in names:
            if not isinstance(name, string_types):
                name = str(name)  # will need consistent unicode handling
            lag_names.append('L' + str(i) + '.' + name)

    # handle the constant name
    # Note: unicode literals are relevant for py2 tests,
    # see test_arima.test_arima_wrapper
    if trendorder != 0:
        lag_names.insert(0, u'const')
    if trendorder > 1:
        lag_names.insert(1, u'trend')
    if trendorder > 2:
        lag_names.insert(2, u'trend**2')
    if exog is not None:
        for i in range(exog.shape[1]):
            lag_names.insert(trendorder + i, u"exog" + str(i))
    return lag_names


def comp_matrix(coefs):
    """
    Return compansion matrix for the VAR(1) representation for a VAR(p)
    process (companion form)

    A = [A_1 A_2 ... A_p-1 A_p
         I_K 0       0     0
         0   I_K ... 0     0
         0 ...       I_K   0]
    """
    p, k, k2 = coefs.shape
    assert k == k2

    kp = k * p

    result = np.zeros((kp, kp))
    result[:k] = np.concatenate(coefs, axis=1)

    # Set I_K matrices
    if p > 1:
        result[np.arange(k, kp), np.arange(kp - k)] = 1

    return result


# ---------------------------------------------------------------
# Miscellaneous stuff


def parse_lutkepohl_data(path):  # pragma: no cover
    raise NotImplementedError("parse_lutkepohl_data not ported from upstream")


def get_logdet(m):  # pragma: no cover
    raise NotImplementedError("get_logdet nor ported from upstream, "
                              "as it is neither used nor tested there.")


get_logdet = np.deprecate(get_logdet,
                          "sm2.tsa.vector_ar.util.get_logdet",
                          "sm2.tools.linalg.logdet_symm",
                          "get_logdet is deprecated and will be removed in "
                          "0.8.0")


def norm_signif_level(alpha=0.05):
    return stats.norm.ppf(1 - alpha / 2)


def varsim(coefs, intercept, sig_u, steps=100, initvalues=None, seed=None):
    """
    Simulate simple VAR(p) process with known coefficients, intercept, white
    noise covariance, etc.
    """
    rs = np.random.RandomState(seed=seed)
    rmvnorm = rs.multivariate_normal
    p, k, k = coefs.shape
    ugen = rmvnorm(np.zeros(len(sig_u)), sig_u, steps)
    result = np.zeros((steps, k))
    if intercept is not None:
        result[p:] = intercept + ugen[p:]
    else:
        result[p:] = ugen[p:]

    # add in AR terms
    for t in range(p, steps):
        ygen = result[t]
        for j in range(p):
            ygen += np.dot(coefs[j], result[t - j - 1])

    return result


def get_index(lst, name):
    try:
        result = lst.index(name)
    except Exception:
        if not isinstance(name, integer_types):
            raise
        result = name
    return result


# method used repeatedly in Sims-Zha error bands
# TODO: Does the above comment refer to get_index or eigval_decomp?

def eigval_decomp(sym_array):
    """
    Returns
    -------
    W: array of eigenvectors
    eigva: list of eigenvalues
    k: largest eigenvector
    """
    # check if symmetric, do not include shock period
    eigva, W = decomp.eig(sym_array, left=True, right=False)
    k = np.argmax(eigva)
    return W, eigva, k


def vech(A):
    """
    Simple vech operator
    Returns
    -------
    vechvec: vector of all elements on and below diagonal
    """
    length = A.shape[1]
    vechvec = []
    for i in range(length):
        b = i
        while b < length:
            vechvec.append(A[b, i])
            b = b + 1
    vechvec = np.asarray(vechvec)
    return vechvec


def seasonal_dummies(n_seasons, len_endog, first_period=0, centered=False):
    raise NotImplementedError("seasonal_dummies not ported from upstream "
                              "(until vecm is ported)")
