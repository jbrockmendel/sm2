#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical tools for time series analysis
"""
from six.moves import range, zip

import numpy as np
import pandas as pd
from scipy import stats

from sm2.tools.tools import add_constant

from sm2.regression.linear_model import OLS

from sm2.tsa.tsatools import lagmat, lagmat2ds

# upstream these autocov functions are implemented here in stattools
from sm2.tsa.autocov import acf, ccovf, ccf, acovf, pacf_yw

from sm2.tsa.adfvalues import mackinnonp, mackinnoncrit  # noqa:F841
from sm2.tsa._bds import bds
from sm2.tsa.unit_root import (kpss, _sigma_est_kpss, coint,  # noqa:F841
                               adfuller, _autolag, q_stat)

# TODO: bds is not used outside of tests; un-port?
__all__ = ['acovf', 'acf', 'pacf', 'pacf_yw', 'pacf_ols', 'ccovf', 'ccf',
           'periodogram', 'q_stat', 'coint', 'arma_order_select_ic',
           'adfuller', 'kpss', 'bds',
           'innovations_algo', 'innovations_filter',
           'levinson_durbin']


# FIXME: this is incorrect.
def pacf_ols(x, nlags=40):
    """Calculate partial autocorrelations

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    nlags : int
        Number of lags for which pacf is returned.  Lag 0 is not returned.

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, maxlag+1 elements

    Notes
    -----
    This solves a separate OLS estimation for each desired lag.
    """
    # TODO: add warnings for Yule-Walker
    # NOTE: demeaning and not using a constant gave incorrect answers?
    # JP: demeaning should have a better estimate of the constant
    # maybe we can compare small sample properties with a MonteCarlo
    xlags, x0 = lagmat(x, nlags, original='sep')
    # xlags = sm.add_constant(lagmat(x, nlags), prepend=True)
    xlags = add_constant(xlags)
    pacf = [1.]
    for k in range(1, nlags + 1):
        res = OLS(x0[k:], xlags[k:, :k + 1]).fit()
        # np.take(xlags[k:], range(1,k+1)+[-1],

        pacf.append(res.params[-1])
    return np.array(pacf)


def pacf(x, nlags=40, method='ywunbiased', alpha=None):
    """
    Partial autocorrelation estimated

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    nlags : int
        largest lag for which pacf is returned
    method : {'ywunbiased', 'ywmle', 'ols'}
        specifies which method for the calculations to use:

        - yw or ywunbiased : yule walker with bias correction in denominator
          for acovf. Default.
        - ywm or ywmle : yule walker without bias correction
        - ols - regression of time series on lags of it and on constant
        - ld or ldunbiased : Levinson-Durbin recursion with bias correction
        - ldb or ldbiased : Levinson-Durbin recursion without bias correction
    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x))

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, nlags elements, including lag zero
    confint : array, optional
        Confidence intervals for the PACF. Returned if confint is not None.

    Notes
    -----
    This solves yule_walker equations or ols for each desired lag
    and contains currently duplicate calculations.
    """
    if method == 'ols':
        ret = pacf_ols(x, nlags=nlags)
    elif method in ['yw', 'ywu', 'ywunbiased', 'yw_unbiased']:
        ret = pacf_yw(x, nlags=nlags, method='unbiased')
    elif method in ['ywm', 'ywmle', 'yw_mle']:
        ret = pacf_yw(x, nlags=nlags, method='mle')
    elif method in ['ld', 'ldu', 'ldunbiased', 'ld_unbiased']:
        acv = acovf(x, unbiased=True, fft=False)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        ret = ld_[2]

    # FIXME: inconsistent naming with ywmle
    elif method in ['ldb', 'ldbiased', 'ld_biased']:
        acv = acovf(x, unbiased=False, fft=False)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        ret = ld_[2]
    else:  # pragma: no cover
        raise ValueError('method not available')

    if alpha is not None:
        varacf = 1. / len(x)  # for all lags >=1
        interval = stats.norm.ppf(1. - alpha / 2.) * np.sqrt(varacf)
        confint = np.array(list(zip(ret - interval, ret + interval)))
        confint[0] = ret[0]  # fix confidence interval for lag 0 to varpacf=0
        return ret, confint
    else:
        return ret
        # TODO: Get rid of multiple-return


# TODO: not tested; consider un-porting, as it isn't _really_ used upstream
def periodogram(X):
    """
    Returns the periodogram for the natural frequency of X

    Parameters
    ----------
    X : array-like
        Array for which the periodogram is desired.

    Returns
    -------
    pgram : array
        1./len(X) * np.abs(np.fft.fft(X))**2

    References
    ----------
    Brockwell and Davis.
    """
    X = np.asarray(X)
    # if kernel == "bartlett":
    #    w = 1 - np.arange(M + 1.) / M   # JP removed integer division

    pergr = 1. / len(X) * np.abs(np.fft.fft(X))**2
    pergr[0] = 0.  # what are the implications of this?
    return pergr


# TODO: belongs in autocov?
# copied from nitime and sandbox\tsa\examples\try_ld_nitime.py
# TODO: check what to return, for testing and trying out returns everything
def levinson_durbin(s, nlags=10, isacov=False):
    """
    Levinson-Durbin recursion for autoregressive processes

    Parameters
    ----------
    s : array_like
        If isacov is False, then this is the time series. If iasacov is true
        then this is interpreted as autocovariance starting with lag 0
    nlags : integer
        largest lag to include in recursion or order of the autoregressive
        process
    isacov : boolean
        flag to indicate whether the first argument, s, contains the
        autocovariances or the data series.

    Returns
    -------
    sigma_v : float
        estimate of the error variance ?
    arcoefs : ndarray
        estimate of the autoregressive coefficients for a model including nlags
    pacf : ndarray
        partial autocorrelation function
    sigma : ndarray
        entire sigma array from intermediate result, last value is sigma_v
    phi : ndarray
        entire phi array from intermediate result, last column contains
        autoregressive coefficients for AR(nlags)

    Notes
    -----
    This function returns currently all results, but maybe we drop sigma and
    phi from the returns.

    If this function is called with the time series (isacov=False), then the
    sample autocovariance function is calculated with the default options
    (biased, no fft).
    """
    s = np.asarray(s)
    order = nlags

    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s, fft=False)[:order + 1]  # TODO: not tested

    phi = np.zeros((order + 1, order + 1), 'd')
    sig = np.zeros(order + 1)
    # initial points for the recursion
    phi[1, 1] = sxx_m[1] / sxx_m[0]
    sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
    for k in range(2, order + 1):
        phi[k, k] = (sxx_m[k] - np.dot(phi[1:k, k - 1],
                                       sxx_m[1:k][::-1])) / sig[k - 1]
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k]**2)

    sigma_v = sig[-1]
    arcoefs = phi[1:, -1]
    pacf_ = np.diag(phi).copy()
    pacf_[0] = 1.
    return sigma_v, arcoefs, pacf_, sig, phi  # return everything


# TODO: belongs in autocov?
# GH#5042 upstream
def innovations_algo(acov, nobs=None, rtol=None):
    """
    Innovations algorithm to convert autocovariances to MA parameters

    Parameters
    ----------
    acov : array-like
        Array containing autocovariances including lag 0
    nobs : int, optional
        Number of periods to run the algorithm.  If not provided, nobs is
        equal to the length of acovf
    rtol : float, optional
        Tolerance used to check for convergence. Default value is 0 which will
        never prematurely end the algorithm. Checks after 10 iterations and
        stops if sigma2[i] - sigma2[i - 10] < rtol * sigma2[0]. When the
        stopping condition is met, the remaining values in theta and sigma2
        are forward filled using the value of the final iteration.

    Returns
    -------
    theta : ndarray
        Innovation coefficients of MA representation. Array is (nobs, q) where
        q is the largest index of a non-zero autocovariance. theta
        corresponds to the first q columns of the coefficient matrix in the
        common description of the innovation algorithm.
    sigma2 : ndarray
        The prediction error variance (nobs,).

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load_pandas()
    >>> rgdpg = data.data['realgdp'].pct_change().dropna()
    >>> acov = sm.tsa.acovf(rgdpg)
    >>> nobs = activity.shape[0]
    >>> theta, sigma2  = innovations_algo(acov[:4], nobs=nobs)

    See also
    --------
    innovations_filter

    References
    ----------
    Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series and
        forecasting. Springer.
    """
    acov = np.squeeze(np.asarray(acov))
    if acov.ndim != 1:
        raise ValueError('acov must be 1-d or squeezable to 1-d.')

    rtol = 0.0 if rtol is None else rtol
    if not isinstance(rtol, float):
        raise ValueError('rtol must be a non-negative float or None.')

    n = acov.shape[0] if nobs is None else int(nobs)
    if n != nobs or nobs < 1:
        raise ValueError('nobs must be a positive integer')

    max_lag = int(np.max(np.argwhere(acov != 0)))
    v = np.zeros(n + 1)
    v[0] = acov[0]

    # Retain only the relevant columns of theta
    theta = np.zeros((n + 1, max_lag + 1))
    for i in range(1, n):
        for k in range(max(i - max_lag, 0), i):
            sub = 0
            for j in range(max(i - max_lag, 0), k):
                sub += theta[k, k - j] * theta[i, i - j] * v[j]
            theta[i, i - k] = 1. / v[k] * (acov[i - k] - sub)
            v[i] = acov[0]
        for j in range(max(i - max_lag, 0), i):
            v[i] -= theta[i, i - j] ** 2 * v[j]

        # Break if v has converged
        if i >= 10:
            if v[i - 10] - v[i] < v[0] * rtol:
                # Forward fill all remaining values
                v[i + 1:] = v[i]
                theta[i + 1:] = theta[i]
                break

    theta = theta[:-1, 1:]
    v = v[:-1]
    return theta, v


# TODO: belongs in autocov?
# GH#5042 upstream
def innovations_filter(endog, theta):
    """
    Filter observations using the innovations algorithm

    Parameters
    ----------
    endog : array-like
        The time series to filter (nobs,). Should be demeaned if not mean 0.
    theta : ndarray
        Innovation coefficients of MA representation. Array must be (nobs, q)
        where q order of the MA.

    Returns
    -------
    resid : ndarray
        Array of filtered innovations

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load_pandas()
    >>> rgdpg = data.data['realgdp'].pct_change().dropna()
    >>> acov = sm.tsa.acovf(rgdpg)
    >>> nobs = activity.shape[0]
    >>> theta, sigma2  = innovations_algo(acov[:4], nobs=nobs)
    >>> resid = innovations_filter(rgdpg, theta)

    See also
    --------
    innovations_algo

    References
    ----------
    Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series and
        forecasting. Springer.
    """
    orig_endog = endog
    endog = np.squeeze(np.asarray(endog))
    if endog.ndim != 1:
        raise ValueError('endog must be 1-d or squeezable to 1-d.')

    nobs = endog.shape[0]
    n_theta, k = theta.shape
    if nobs != n_theta:
        raise ValueError('theta must be (nobs, q) where q is the moder order')

    is_pandas = isinstance(orig_endog, (pd.DataFrame, pd.Series))
    if is_pandas:
        if len(orig_endog.index) != nobs:
            raise ValueError('If endog is a Series or DataFrame, the index '
                             'must correspond to the number of time series '
                             'observations.')

    u = np.empty(nobs)
    u[0] = endog[0]
    for i in range(1, nobs):
        if i < k:
            hat = (theta[i, :i] * u[:i][::-1]).sum()
        else:
            hat = (theta[i] * u[i - k:i][::-1]).sum()
        u[i] = endog[i] + hat

    if is_pandas:
        u = pd.Series(u, index=orig_endog.index.copy())
    return u


def grangercausalitytests(x, maxlag, addconst=True, verbose=True):
    """four tests for granger non causality of 2 timeseries

    all four tests give similar results
    `params_ftest` and `ssr_ftest` are equivalent based on F test which is
    identical to lmtest:grangertest in R

    Parameters
    ----------
    x : array, 2d
        data for test whether the time series in the second column Granger
        causes the time series in the first column
    maxlag : integer
        the Granger causality test results are calculated for all lags up to
        maxlag
    verbose : bool
        print results if true

    Returns
    -------
    results : dictionary
        all test results, dictionary keys are the number of lags. For each
        lag the values are a tuple, with the first element a dictionary with
        teststatistic, pvalues, degrees of freedom, the second element are
        the OLS estimation results for the restricted model, the unrestricted
        model and the restriction (contrast) matrix for the parameter f_test.

    Notes
    -----
    TODO: convert to class and attach results properly

    The Null hypothesis for grangercausalitytests is that the time series in
    the second column, x2, does NOT Granger cause the time series in the first
    column, x1. Grange causality means that past values of x2 have a
    statistically significant effect on the current value of x1, taking past
    values of x1 into account as regressors. We reject the null hypothesis
    that x2 does not Granger cause x1 if the pvalues are below a desired size
    of the test.

    The null hypothesis for all four test is that the coefficients
    corresponding to past values of the second time series are zero.

    'params_ftest', 'ssr_ftest' are based on F distribution

    'ssr_chi2test', 'lrtest' are based on chi-square distribution

    References
    ----------
    http://en.wikipedia.org/wiki/Granger_causality
    Greene: Econometric Analysis
    """
    if verbose:  # pragma: no cover
        raise NotImplementedError("Option `verbose` from upstream is "
                                  "not supported")
    x = np.asarray(x)

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError("Insufficient observations. Maximum allowable "
                         "lag is {0}"
                         .format(int((x.shape[0] - int(addconst)) / 3) - 1))

    resli = {}

    for mlg in range(1, maxlag + 1):
        result = {}
        mxlg = mlg

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim='both', dropex=1)

        if addconst:
            dtaown = add_constant(dta[:, 1:(mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
        else:
            # TODO: Whats intended here?
            raise NotImplementedError
            # dtaown = dta[:, 1:mxlg]
            # dtajoint = dta[:, 1:]

        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        # for ssr based tests see:
        # http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        # the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        fgc1 = ((res2down.ssr - res2djoint.ssr) /
                res2djoint.ssr / mxlg * res2djoint.df_resid)

        result['ssr_ftest'] = (fgc1,
                               stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                               res2djoint.df_resid, mxlg)

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        result['ssr_chi2test'] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        # likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        result['lrtest'] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack((np.zeros((mxlg, mxlg)),
                                   np.eye(mxlg, mxlg),
                                   np.zeros((mxlg, 1))))
        ftres = res2djoint.f_test(rconstr)
        result['params_ftest'] = (np.squeeze(ftres.fvalue)[()],
                                  np.squeeze(ftres.pvalue)[()],
                                  ftres.df_denom, ftres.df_num)

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])

    return resli


def _safe_arma_fit(y, order, model_kw, trend, fit_kw, start_params=None):
    raise NotImplementedError("_safe_arma_fit not ported from "
                              "upstream")  # pragma: no cover


def arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c',
                         model_kw={}, fit_kw={}):  # pragma: no cover
    raise NotImplementedError("arma_order_select_ic not ported from upstream, "
                              "as it is only used in tests")


def has_missing(data):  # pragma: no cover
    raise NotImplementedError("has_missing not ported from upstream; "
                              "use `np.isnan(data).any()` instead.")
