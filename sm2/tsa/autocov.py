#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions relating to autocovariance
"""
import numpy as np
from scipy import stats
import scipy.linalg
from scipy.linalg import toeplitz

from sm2.tools.sm_exceptions import MissingDataError
from sm2.compat.scipy import _next_regular


# NOTE: Changed unbiased to False
# see for example
# http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
def acf(x, unbiased=False, nlags=40, qstat=True, fft=False, alpha=True,
        missing='none'):
    """
    Autocorrelation function for 1d arrays.

    Parameters
    ----------
    x : array
       Time series data
    unbiased : bool
       If True, then denominators for autocovariance are n-k, otherwise n
    nlags: int, optional
        Number of lags to return autocorrelation for.
    qstat : bool, optional
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, optional
        If True, computes the ACF via FFT.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett's formula.
    missing : str, optional
        A string in ['none', 'raise', 'conservative', 'drop'] specifying how
        any NaNs are to be treated.

    Returns
    -------
    acf : array
        autocorrelation function
    confint : array, optional
        Confidence intervals for the ACF. Returned if confint is not None.
    qstat : array, optional
        The Ljung-Box Q-Statistic.  Returned if q_stat is True.
    pvalues : array, optional
        The p-values associated with the Q-statistics.  Returned if q_stat is
        True.

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    This is based np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted
    but the autocorrelation is not an unbiased estimtor.

    References
    ----------
    .. [*] Parzen, E., 1963. On spectral analysis with missing observations
       and amplitude modulation. Sankhya: The Indian Journal of
       Statistics, Series A, pp.383-392.
    """
    from sm2.tsa.unit_root import q_stat  # avoid circular import

    if not qstat or not alpha:  # pragma: no cover
        raise NotImplementedError("Options `qstat` and `alpha` from upstream "
                                  "are not supported in sm2.  `acf` always "
                                  "returns a tuple "
                                  "(acf, confint, qstat, pvalue)")
    nobs = len(x)  # should this shrink for missing='drop' and NaNs in x?
    avf = acovf(x, unbiased=unbiased, demean=True, fft=fft, missing=missing)
    acf = avf[:nlags + 1] / avf[0]

    varacf = np.ones(nlags + 1) / nobs
    varacf[0] = 0
    varacf[1] = 1. / nobs
    varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1]**2)
    interval = stats.norm.ppf(1 - alpha / 2.) * np.sqrt(varacf)
    confint = np.array(list(zip(acf - interval, acf + interval)))

    qstat, pvalue = q_stat(acf[1:], nobs=nobs)  # drop lag 0

    return acf, confint, qstat, pvalue


def acovf(x, unbiased=False, demean=True, fft=False, missing='none'):
    """
    Autocovariance for 1D

    Parameters
    ----------
    x : array
        Time series data. Must be 1d.
    unbiased : bool
        If True, then denominators is n-k, otherwise n
    demean : bool
        If True, then subtract the mean x from each element of x
    fft : bool
        If True, use FFT convolution.  This method should be preferred
        for long time series.
    missing : str
        A string in ['none', 'raise', 'conservative', 'drop'] specifying
        how any NaNs are to be treated.

    Returns
    -------
    acovf : array
        autocovariance function

    References
    -----------
    .. [*] Parzen, E., 1963. On spectral analysis with missing observations
           and amplitude modulation. Sankhya: The Indian Journal of
           Statistics, Series A, pp.383-392.
    """
    x = np.squeeze(np.asarray(x))
    if x.ndim > 1:
        raise ValueError("x must be 1d. Got %d dims." % x.ndim)

    missing = missing.lower()
    if missing not in ['none', 'raise', 'conservative', 'drop']:
        raise ValueError("`missing` option %s not understood"
                         % missing)  # pragma: no cover
    if missing == 'none':
        deal_with_masked = False
    else:
        deal_with_masked = np.isnan(x).any()
    if deal_with_masked:
        if missing == 'raise':
            raise MissingDataError("NaNs were encountered in the data")
        notmask_bool = ~np.isnan(x)
        if missing == 'conservative':
            x[~notmask_bool] = 0
        else:
            # 'drop'
            x = x[notmask_bool]  # copies non-missing
        notmask_int = notmask_bool.astype(int)

    if demean and deal_with_masked:
        # whether 'drop' or 'conservative':
        xo = x - x.sum() / notmask_int.sum()
        if missing == 'conservative':
            xo[~notmask_bool] = 0
    elif demean:
        xo = x - x.mean()
    else:
        xo = x

    n = len(x)
    if unbiased and deal_with_masked and missing == 'conservative':
        d = np.correlate(notmask_int, notmask_int, 'full')
    elif unbiased:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    elif deal_with_masked:
        # biased and NaNs given and ('drop' or 'conservative')
        d = notmask_int.sum() * np.ones(2 * n - 1)
    else:
        # biased and no NaNs or missing=='none'
        d = n * np.ones(2 * n - 1)

    if fft:
        nobs = len(xo)
        n = _next_regular(2 * nobs + 1)
        Frf = np.fft.fft(xo, n=n)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[nobs - 1:]
        acov = acov.real
    else:
        acov = (np.correlate(xo, xo, 'full') / d)[n - 1:]

    if deal_with_masked and missing == 'conservative':
        # restore data for the user
        x[~notmask_bool] = np.nan

    return acov


# Upstream this is var_model._var_acf
# TODO: update test location
def var_acf(coefs, sig_u):
    """
    Compute autocovariance function ACF_y(h) for h=1,...,p

    Notes
    -----
    Lütkepohl (2005) p.29
    """
    p, k, k2 = coefs.shape
    assert k == k2

    from sm2.tsa.vector_ar import util
    from sm2.tsa import tsatools

    A = util.comp_matrix(coefs)
    # construct VAR(1) noise covariance
    SigU = np.zeros((k * p, k * p))
    SigU[:k, :k] = sig_u

    # vec(ACF) = (I_(kp)^2 - kron(A, A))^-1 vec(Sigma_U)
    vecACF = scipy.linalg.solve(np.eye((k * p)**2) - np.kron(A, A),
                                SigU.ravel('F'))
    # TODO: Does the 'F' matter?

    acf = tsatools.unvec(vecACF)
    acf = acf[:k].T.reshape((p, k, k))

    return acf


# Upstream this is var_model._compute_acov
# TODO: Is there a better implementation of this that we can use instead?
def compute_acov(x, nlags=1):
    x = x - x.mean(0)

    result = []
    for lag in range(nlags + 1):
        if lag > 0:
            r = np.dot(x[lag:].T, x[:-lag])
        else:
            r = np.dot(x.T, x)

        result.append(r)

    return np.array(result) / len(x)


def acf_to_acorr(acf):
    diag = np.diag(acf[0])
    # numpy broadcasting sufficient
    return acf / np.sqrt(np.outer(diag, diag))


def ccovf(x, y, unbiased=True, demean=True):
    """crosscovariance for 1D

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators is n-k, otherwise n

    Returns
    -------
    ccovf : array
        autocovariance function

    Notes
    -----
    This uses np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.
    """
    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if unbiased:
        xi = np.ones(n)
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
    return (np.correlate(xo, yo, 'full') / d)[n - 1:]


def ccf(x, y, unbiased=True):
    """cross-correlation function for 1d

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators for autocovariance is n-k, otherwise n

    Returns
    -------
    ccf : array
        cross-correlation function of x and y

    Notes
    -----
    This is based np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted
    but the autocorrelation is not an unbiased estimtor.
    """
    cvf = ccovf(x, y, unbiased=unbiased, demean=True)
    return cvf / (np.std(x) * np.std(y))


# Upstream this is in regression.linear_model
def yule_walker(X, order=1, method="unbiased", df=None, inv=False, demean=True):
    """
    Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    Unbiased or maximum-likelihood estimator (mle)

    See, for example:

    http://en.wikipedia.org/wiki/Autoregressive_moving_average_model

    Parameters
    ----------
    X : array-like
        1d array
    order : integer, optional
        The order of the autoregressive process.  Default is 1.
    method : string, optional
       Method can be 'unbiased' or 'mle' and this determines
       denominator in estimate of autocorrelation function (ACF) at
       lag k. If 'mle', the denominator is n=X.shape[0], if 'unbiased'
       the denominator is n-k.  The default is unbiased.
    df : integer, optional
       Specifies the degrees of freedom. If `df` is supplied, then it
       is assumed the X has `df` degrees of freedom rather than `n`.
       Default is None.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is
        False.
    demean : bool
        True, the mean is subtracted from `X` before estimation.

    Returns
    -------
    rho
        The autoregressive coefficients
    sigma
        TODO

    Examples
    --------
    >>> import sm2.api as sm
    >>> from sm2.datasets.sunspots import load
    >>> data = load()
    >>> rho, sigma = sm.regression.yule_walker(data.endog,
    ...                                        order=4, method="mle")

    >>> rho
    array([ 1.28310031, -0.45240924, -0.20770299,  0.04794365])
    >>> sigma
    16.808022730464351
    """
    if inv:  # pragma: no cover
        raise NotImplementedError("option `inv` not ported from upstream, "
                                  "since it is not used or tested there.")
    # TODO: define R better, look back at notes and technical notes on YW.
    # First link here is useful
    # http://www-stat.wharton.upenn.edu/~steele/Courses/956/ResourceDetails/YuleWalkerAndMore.htm
    method = str(method).lower()
    if method not in ["unbiased", "mle"]:  # pragma: no cover
        raise ValueError("ACF estimation method must be 'unbiased' or 'MLE'")

    X = np.array(X, dtype=np.float64)
    if demean:
        # automatically demean's X
        X -= X.mean()
    n = df or X.shape[0]

    if method == "unbiased":
        # this is df_resid ie., n - p
        denom = lambda k: n - k
    else:
        denom = lambda k: n
    if X.ndim > 1 and X.shape[1] != 1:  # pragma: no cover
        raise ValueError("expecting a vector to estimate AR parameters")

    r = np.zeros(order + 1, np.float64)
    r[0] = (X**2).sum() / denom(0)
    for k in range(1, order + 1):
        r[k] = (X[0:-k] * X[k:]).sum() / denom(k)
    R = toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    sigmasq = r[0] - (r[1:] * rho).sum()
    return rho, np.sqrt(sigmasq)


def pacf_yw(x, nlags=40, method='unbiased'):
    """Partial autocorrelation estimated with non-recursive yule_walker

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    nlags : int
        largest lag for which pacf is returned
    method : 'unbiased' (default) or 'mle'
        method for the autocovariance calculations in yule walker

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, maxlag+1 elements

    Notes
    -----
    This solves yule_walker for each desired lag and contains
    currently duplicate calculations.
    """
    pacf = [1.]
    for k in range(1, nlags + 1):
        pacf.append(yule_walker(x, k, method=method)[0][-1])
    return np.array(pacf)
