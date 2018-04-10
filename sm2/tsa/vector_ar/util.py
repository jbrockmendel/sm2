#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Miscellaneous utility code for VAR estimation
"""
from __future__ import division

from six import integer_types
from six.moves import range

import numpy as np
from scipy import stats
from scipy.linalg import decomp

from sm2.tsa import tsatools
from sm2.tsa.autocov import acf_to_acorr  # noqa:F841
from sm2.base.naming import make_lag_names  # noqa:F841


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
        Z = tsatools.add_trend(Z, prepend=True, trend=trend,
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


# used repeatedly in Sims-Zha error bands
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


def vech(A):  # TODO: why not just use A[np.triu_indices(A.shape[0])]?
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
                              "(until vecm is ported)")  # pragma: no cover
