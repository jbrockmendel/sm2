#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical tools for time series analysis
"""

from six import string_types, integer_types
from six.moves import range, zip

import numpy as np
from numpy.linalg import LinAlgError
from scipy import stats

from sm2.compat.scipy import _next_regular

from sm2.tools.sm_exceptions import InterpolationWarning, MissingDataError
from sm2.tools.tools import add_constant, Bunch

from sm2.regression.linear_model import OLS, yule_walker

from sm2.tsa.tsatools import lagmat, lagmat2ds, add_trend

from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
from statsmodels.tsa._bds import bds
from statsmodels.tsa.arima_model import ARMA


__all__ = ['acovf', 'acf', 'pacf', 'pacf_yw', 'pacf_ols', 'ccovf', 'ccf',
           'periodogram', 'q_stat', 'coint', 'arma_order_select_ic',
           'adfuller', 'kpss', 'bds']


#NOTE: now in two places to avoid circular import
#TODO: I like the bunch pattern for this too.
class ResultsStore(object):
    def __str__(self):
        return self._str  # pylint: disable=E1101


def _autolag(mod, endog, exog, startlag, maxlag, method, modargs=(),
             fitargs=(), regresults=False):
    """
    Returns the results for the lag length that maximizes the info criterion.

    Parameters
    ----------
    mod : Model class
        Model estimator class
    endog : array-like
        nobs array containing endogenous variable
    exog : array-like
        nobs by (startlag + maxlag) array containing lags and possibly other
        variables
    startlag : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : {'aic', 'bic', 't-stat'}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag
    modargs : tuple, optional
        args to pass to model.  See notes.
    fitargs : tuple, optional
        args to pass to fit.  See notes.
    regresults : bool, optional
        Flag indicating to return optional return results

    Returns
    -------
    icbest : float
        Best information criteria.
    bestlag : int
        The lag length that maximizes the information criterion.
    results : dict, optional
        Dictionary containing all estimation results

    Notes
    -----
    Does estimation like mod(endog, exog[:,:i], *modargs).fit(*fitargs)
    where i goes from lagstart to lagstart+maxlag+1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
    #TODO: can tcol be replaced by maxlag + 2?
    #TODO: This could be changed to laggedRHS and exog keyword arguments if
    #    this will be more general.

    results = {}
    method = method.lower()
    for lag in range(startlag, startlag + maxlag + 1):
        mod_instance = mod(endog, exog[:, :lag], *modargs)
        results[lag] = mod_instance.fit()

    if method == "aic":
        icbest, bestlag = min((v.aic, k) for k, v in results.items())
    elif method == "bic":
        icbest, bestlag = min((v.bic, k) for k, v in results.items())
    elif method == "t-stat":
        #stop = stats.norm.ppf(.95)
        stop = 1.6448536269514722
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            if np.abs(icbest) >= stop:
                bestlag = lag
                icbest = icbest
                break
    else:
        raise ValueError("Information Criterion %s not understood.") % method

    if not regresults:
        return icbest, bestlag
    else:
        return icbest, bestlag, results


#this needs to be converted to a class like HetGoldfeldQuandt,
# 3 different returns are a mess
# See:
#Ng and Perron(2001), Lag length selection and the construction of unit root
#tests with good size and power, Econometrica, Vol 69 (6) pp 1519-1554
#TODO: include drift keyword, only valid with regression == "c"
# just changes the distribution of the test statistic to a t distribution
#TODO: autolag is untested
def adfuller(x, maxlag=None, regression="c", autolag='AIC',
             store=False, regresults=False):
    """
    Augmented Dickey-Fuller unit root test

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        data series
    maxlag : int
        Maximum lag which is included in test, default 12*(nobs/100)^{1/4}
    regression : {'c','ct','ctt','nc'}
        Constant and trend order to include in regression

        * 'c' : constant only (default)
        * 'ct' : constant and trend
        * 'ctt' : constant, and linear and quadratic trend
        * 'nc' : no constant, no trend
    autolag : {'AIC', 'BIC', 't-stat', None}
        * if None, then maxlag lags are used
        * if 'AIC' (default) or 'BIC', then the number of lags is chosen
          to minimize the corresponding information criterion
        * 't-stat' based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test
    store : bool
        If True, then a result instance is returned additionally to
        the adf statistic. Default is False
    regresults : bool, optional
        If True, the full regression results are returned. Default is False

    Returns
    -------
    adf : float
        Test statistic
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010)
    usedlag : int
        Number of lags used
    nobs : int
        Number of observations used for the ADF regression and calculation of
        the critical values
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010)
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultStore, optional
        A dummy class with results attached as attributes

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables. If the p-value is close
    to significant, then the critical values should be used to judge whether
    to reject the null.

    The autolag option and maxlag for it are described in Greene.

    Examples
    --------
    See example notebook

    References
    ----------
    .. [*] W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.

    .. [*] Hamilton, J.D.  "Time Series Analysis".  Princeton, 1994.

    .. [*] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  `Journal of Business and Economic
        Statistics` 12, 167-76.

    .. [*] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen's
        University, Dept of Economics, Working Papers.  Available at
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """

    if regresults:
        store = True

    trenddict = {None: 'nc', 0: 'c', 1: 'ct', 2: 'ctt'}
    if regression is None or isinstance(regression, integer_types):
        regression = trenddict[regression]
    regression = regression.lower()
    if regression not in ['c', 'nc', 'ct', 'ctt']:
        raise ValueError("regression option %s not understood") % regression
    x = np.asarray(x)
    nobs = x.shape[0]

    if maxlag is None:
        #from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12. * np.power(nobs / 100., 1 / 4.)))

    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim='both', original='in')
    nobs = xdall.shape[0]  # pylint: disable=E1103

    xdall[:, 0] = x[-nobs - 1:-1]  # replace 0 xdiff with level of x
    xdshort = xdiff[-nobs:]

    if store:
        resstore = ResultsStore()
    if autolag:
        if regression != 'nc':
            fullRHS = add_trend(xdall, regression, prepend=True)
        else:
            fullRHS = xdall
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1  # 1 for level  # pylint: disable=E1103
        #search for lag length with smallest information criteria
        #Note: use the same number of observations to have comparable IC
        #aic and bic: smaller is better

        if not regresults:
            icbest, bestlag = _autolag(OLS, xdshort, fullRHS, startlag,
                                       maxlag, autolag)
        else:
            icbest, bestlag, alres = _autolag(OLS, xdshort, fullRHS, startlag,
                                              maxlag, autolag,
                                              regresults=regresults)
            resstore.autolag_results = alres

        bestlag -= startlag  # convert to lag not column index

        #rerun ols with best autolag
        xdall = lagmat(xdiff[:, None], bestlag, trim='both', original='in')
        nobs = xdall.shape[0]   # pylint: disable=E1103
        xdall[:, 0] = x[-nobs - 1:-1]  # replace 0 xdiff with level of x
        xdshort = xdiff[-nobs:]
        usedlag = bestlag
    else:
        usedlag = maxlag
        icbest = None
    if regression != 'nc':
        resols = OLS(xdshort, add_trend(xdall[:, :usedlag + 1],
                     regression)).fit()
    else:
        resols = OLS(xdshort, xdall[:, :usedlag + 1]).fit()

    adfstat = resols.tvalues[0]
#    adfstat = (resols.params[0]-1.0)/resols.bse[0]
    # the "asymptotically correct" z statistic is obtained as
    # nobs/(1-np.sum(resols.params[1:-(trendorder+1)])) (resols.params[0] - 1)
    # I think this is the statistic that is used for series that are integrated
    # for orders higher than I(1), ie., not ADF but cointegration tests.

    # Get approx p-value and critical values
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {"1%" : critvalues[0], "5%" : critvalues[1],
                  "10%" : critvalues[2]}
    if store:
        resstore.resols = resols
        resstore.maxlag = maxlag
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.nobs = nobs
        resstore.H0 = ("The coefficient on the lagged level equals 1 - "
                       "unit root")
        resstore.HA = "The coefficient on the lagged level < 1 - stationary"
        resstore.icbest = icbest
        resstore._str = 'Augmented Dickey-Fuller Test Results'
        return adfstat, pvalue, critvalues, resstore
    else:
        if not autolag:
            return adfstat, pvalue, usedlag, nobs, critvalues
        else:
            return adfstat, pvalue, usedlag, nobs, critvalues, icbest


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
        A string in ['none', 'raise', 'conservative', 'drop'] specifying how the NaNs
        are to be treated.

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
        raise ValueError("missing option %s not understood" % missing)
    if missing == 'none':
        deal_with_masked = False
    else:
        deal_with_masked = has_missing(x)
    if deal_with_masked:
        if missing == 'raise':
            raise MissingDataError("NaNs were encountered in the data")
        notmask_bool = ~np.isnan(x) #bool
        if missing == 'conservative':
            x[~notmask_bool] = 0
        else: #'drop'
            x = x[notmask_bool] #copies non-missing
        notmask_int = notmask_bool.astype(int) #int

    if demean and deal_with_masked:
        # whether 'drop' or 'conservative':
        xo = x - x.sum()/notmask_int.sum()
        if missing=='conservative':
            xo[~notmask_bool] = 0
    elif demean:
        xo = x - x.mean()
    else:
        xo = x

    n = len(x)
    if unbiased and deal_with_masked and missing=='conservative':
        d = np.correlate(notmask_int, notmask_int, 'full')
    elif unbiased:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    elif deal_with_masked: #biased and NaNs given and ('drop' or 'conservative')
        d = notmask_int.sum() * np.ones(2*n-1)
    else: #biased and no NaNs or missing=='none'
        d = n * np.ones(2 * n - 1)

    if fft:
        nobs = len(xo)
        n = _next_regular(2 * nobs + 1)
        Frf = np.fft.fft(xo, n=n)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[nobs - 1:]
        acov = acov.real
    else:
        acov = (np.correlate(xo, xo, 'full') / d)[n - 1:]

    if deal_with_masked and missing=='conservative':
        # restore data for the user
        x[~notmask_bool] = np.nan

    return acov



def q_stat(x, nobs, type="ljungbox"):
    """
    Return's Ljung-Box Q Statistic

    x : array-like
        Array of autocorrelation coefficients.  Can be obtained from acf.
    nobs : int
        Number of observations in the entire sample (ie., not just the length
        of the autocorrelation function results.

    Returns
    -------
    q-stat : array
        Ljung-Box Q-statistic for autocorrelation parameters
    p-value : array
        P-value of the Q statistic

    Notes
    ------
    Written to be used with acf.
    """
    x = np.asarray(x)
    if type == "ljungbox":
        ret = (nobs * (nobs + 2) *
               np.cumsum((1. / (nobs - np.arange(1, len(x) + 1))) * x**2))
    chi2 = stats.chi2.sf(ret, np.arange(1, len(x) + 1))
    return ret, chi2


#NOTE: Changed unbiased to False
#see for example
# http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
def acf(x, unbiased=False, nlags=40, qstat=False, fft=False, alpha=None,
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
        Bartlett\'s formula.
    missing : str, optional
        A string in ['none', 'raise', 'conservative', 'drop'] specifying how the NaNs
        are to be treated.

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
    nobs = len(x)  # should this shrink for missing='drop' and NaNs in x?
    avf = acovf(x, unbiased=unbiased, demean=True, fft=fft, missing=missing)
    acf = avf[:nlags + 1] / avf[0]
    if not (qstat or alpha):
        return acf
    if alpha is not None:
        varacf = np.ones(nlags + 1) / nobs
        varacf[0] = 0
        varacf[1] = 1. / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1]**2)
        interval = stats.norm.ppf(1 - alpha / 2.) * np.sqrt(varacf)
        confint = np.array(list(zip(acf - interval, acf + interval)))
        if not qstat:
            return acf, confint
    if qstat:
        qstat, pvalue = q_stat(acf[1:], nobs=nobs)  # drop lag 0
        if alpha is not None:
            return acf, confint, qstat, pvalue
        else:
            return acf, qstat, pvalue


def pacf_yw(x, nlags=40, method='unbiased'):
    '''Partial autocorrelation estimated with non-recursive yule_walker

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
    '''
    pacf = [1.]
    for k in range(1, nlags + 1):
        pacf.append(yule_walker(x, k, method=method)[0][-1])
    return np.array(pacf)


#NOTE: this is incorrect.
def pacf_ols(x, nlags=40):
    '''Calculate partial autocorrelations

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
    '''
    #TODO: add warnings for Yule-Walker
    #NOTE: demeaning and not using a constant gave incorrect answers?
    #JP: demeaning should have a better estimate of the constant
    #maybe we can compare small sample properties with a MonteCarlo
    xlags, x0 = lagmat(x, nlags, original='sep')
    #xlags = sm.add_constant(lagmat(x, nlags), prepend=True)
    xlags = add_constant(xlags)
    pacf = [1.]
    for k in range(1, nlags+1):
        res = OLS(x0[k:], xlags[k:, :k+1]).fit()
         #np.take(xlags[k:], range(1,k+1)+[-1],

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
    elif method in ['ld', 'ldu', 'ldunbiase', 'ld_unbiased']:
        acv = acovf(x, unbiased=True)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        #print 'ld', ld_
        ret = ld_[2]
    # inconsistent naming with ywmle
    elif method in ['ldb', 'ldbiased', 'ld_biased']:
        acv = acovf(x, unbiased=False)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        ret = ld_[2]
    else:
        raise ValueError('method not available')
    if alpha is not None:
        varacf = 1. / len(x) # for all lags >=1
        interval = stats.norm.ppf(1. - alpha / 2.) * np.sqrt(varacf)
        confint = np.array(list(zip(ret - interval, ret + interval)))
        confint[0] = ret[0]  # fix confidence interval for lag 0 to varpacf=0
        return ret, confint
    else:
        return ret


def ccovf(x, y, unbiased=True, demean=True):
    ''' crosscovariance for 1D

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
    '''
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
    '''cross-correlation function for 1d

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

    '''
    cvf = ccovf(x, y, unbiased=unbiased, demean=True)
    return cvf / (np.std(x) * np.std(y))


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
    #if kernel == "bartlett":
    #    w = 1 - np.arange(M+1.)/M   #JP removed integer division

    pergr = 1. / len(X) * np.abs(np.fft.fft(X))**2
    pergr[0] = 0.  # what are the implications of this?
    return pergr


#copied from nitime and statsmodels\sandbox\tsa\examples\try_ld_nitime.py
#TODO: check what to return, for testing and trying out returns everything
def levinson_durbin(s, nlags=10, isacov=False):
    '''Levinson-Durbin recursion for autoregressive processes

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
        estimate of the autoregressive coefficients
    pacf : ndarray
        partial autocorrelation function
    sigma : ndarray
        entire sigma array from intermediate result, last value is sigma_v
    phi : ndarray
        entire phi array from intermediate result, last column contains
        autoregressive coefficients for AR(nlags) with a leading 1

    Notes
    -----
    This function returns currently all results, but maybe we drop sigma and
    phi from the returns.

    If this function is called with the time series (isacov=False), then the
    sample autocovariance function is calculated with the default options
    (biased, no fft).
    '''
    s = np.asarray(s)
    order = nlags  # rename compared to nitime
    #from nitime

    ##if sxx is not None and type(sxx) == np.ndarray:
    ##    sxx_m = sxx[:order+1]
    ##else:
    ##    sxx_m = ut.autocov(s)[:order+1]
    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s)[:order + 1]  # not tested

    phi = np.zeros((order + 1, order + 1), 'd')
    sig = np.zeros(order + 1)
    # initial points for the recursion
    phi[1, 1] = sxx_m[1] / sxx_m[0]
    sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
    for k in range(2, order + 1):
        phi[k, k] = (sxx_m[k] - np.dot(phi[1:k, k-1],
                                       sxx_m[1:k][::-1])) / sig[k-1]
        for j in range(1, k):
            phi[j, k] = phi[j, k-1] - phi[k, k] * phi[k-j, k-1]
        sig[k] = sig[k-1] * (1 - phi[k, k]**2)

    sigma_v = sig[-1]
    arcoefs = phi[1:, -1]
    pacf_ = np.diag(phi).copy()
    pacf_[0] = 1.
    return sigma_v, arcoefs, pacf_, sig, phi  # return everything


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
    x = np.asarray(x)

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError("Insufficient observations. Maximum allowable "
                         "lag is {0}".format(int((x.shape[0] - int(addconst)) /
                                                 3) - 1))

    resli = {}

    for mlg in range(1, maxlag + 1):
        result = {}
        if verbose:
            print('\nGranger Causality')
            print('number of lags (no zero)', mlg)
        mxlg = mlg

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim='both', dropex=1)

        #add constant
        if addconst:
            dtaown = add_constant(dta[:, 1:(mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
        else:
            raise NotImplementedError('Not Implemented')
            #dtaown = dta[:, 1:mxlg]
            #dtajoint = dta[:, 1:]

        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        #print results
        #for ssr based tests see:
        #http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        #the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        fgc1 = ((res2down.ssr - res2djoint.ssr) /
                res2djoint.ssr / mxlg * res2djoint.df_resid)
        if verbose:
            print('ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
                   ' df_num=%d' % (fgc1,
                                    stats.f.sf(fgc1, mxlg,
                                               res2djoint.df_resid),
                                    res2djoint.df_resid, mxlg))
        result['ssr_ftest'] = (fgc1,
                               stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                               res2djoint.df_resid, mxlg)

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        if verbose:
            print('ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, '
                   'df=%d' % (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg))
        result['ssr_chi2test'] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        #likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        if verbose:
            print('likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d' %
                   (lr, stats.chi2.sf(lr, mxlg), mxlg))
        result['lrtest'] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack((np.zeros((mxlg, mxlg)),
                                   np.eye(mxlg, mxlg),
                                   np.zeros((mxlg, 1))))
        ftres = res2djoint.f_test(rconstr)
        if verbose:
            print('parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
                   ' df_num=%d' % (ftres.fvalue, ftres.pvalue, ftres.df_denom,
                                    ftres.df_num))
        result['params_ftest'] = (np.squeeze(ftres.fvalue)[()],
                                  np.squeeze(ftres.pvalue)[()],
                                  ftres.df_denom, ftres.df_num)

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])

    return resli


def coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag='aic',
          return_results=None):
    """Test for no-cointegration of a univariate equation

    The null hypothesis is no cointegration. Variables in y0 and y1 are
    assumed to be integrated of order 1, I(1).

    This uses the augmented Engle-Granger two-step cointegration test.
    Constant or trend is included in 1st stage regression, i.e. in
    cointegrating equation.

    Parameters
    ----------
    y1 : array_like, 1d
        first element in cointegrating vector
    y2 : array_like
        remaining elements in cointegrating vector
    trend : str {'c', 'ct'}
        trend term included in regression for cointegrating equation
        * 'c' : constant
        * 'ct' : constant and linear trend
        * also available quadratic trend 'ctt', and no constant 'nc'

    method : string
        currently only 'aeg' for augmented Engle-Granger test is available.
        default might change.
    maxlag : None or int
        keyword for `adfuller`, largest or given number of lags
    autolag : string
        keyword for `adfuller`, lag selection criterion.
    return_results : bool
        for future compatibility, currently only tuple available.
        If True, then a results instance is returned. Otherwise, a tuple
        with the test outcome is returned.
        Set `return_results=False` to avoid future changes in return.


    Returns
    -------
    coint_t : float
        t-statistic of unit-root test on residuals
    pvalue : float
        MacKinnon's approximate, asymptotic p-value based on MacKinnon (1994)
    crit_value : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels based on regression curve. This depends on the number of
        observations.

    Notes
    -----
    The Null hypothesis is that there is no cointegration, the alternative
    hypothesis is that there is cointegrating relationship. If the pvalue is
    small, below a critical size, then we can reject the hypothesis that there
    is no cointegrating relationship.

    P-values and critical values are obtained through regression surface
    approximation from MacKinnon 1994 and 2010.

    TODO: We could handle gaps in data by dropping rows with nans in the
    auxiliary regressions. Not implemented yet, currently assumes no nans
    and no gaps in time series.

    References
    ----------
    MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions for
        Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.
    MacKinnon, J.G. 2010.  "Critical Values for Cointegration Tests."
        Queen's University, Dept of Economics Working Papers 1227.
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """

    trend = trend.lower()
    if trend not in ['c', 'nc', 'ct', 'ctt']:
        raise ValueError("trend option %s not understood" % trend)
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    if y1.ndim < 2:
        y1 = y1[:, None]
    nobs, k_vars = y1.shape
    k_vars += 1   # add 1 for y0

    if trend == 'nc':
        xx = y1
    else:
        xx = add_trend(y1, trend=trend, prepend=False)

    res_co = OLS(y0, xx).fit()

    if res_co.rsquared < 1 - np.sqrt(np.finfo(np.double).eps):
        res_adf = adfuller(res_co.resid, maxlag=maxlag, autolag=None,
                           regression='nc')
    else:
        import warnings
        warnings.warn("y0 and y1 are perfectly colinear.  Cointegration test "
                      "is not reliable in this case.")
        # Edge case where series are too similar
        res_adf = (0,)

    # no constant or trend, see egranger in Stata and MacKinnon
    if trend == 'nc':
        crit = [np.nan] * 3  # 2010 critical values not available
    else:
        crit = mackinnoncrit(N=k_vars, regression=trend, nobs=nobs - 1)
        #  nobs - 1, the -1 is to match egranger in Stata, I don't know why.
        #  TODO: check nobs or df = nobs - k

    pval_asy = mackinnonp(res_adf[0], regression=trend, N=k_vars)
    return res_adf[0], pval_asy, crit


def _safe_arma_fit(y, order, model_kw, trend, fit_kw, start_params=None):
    try:
        return ARMA(y, order=order, **model_kw).fit(disp=0, trend=trend,
                                                    start_params=start_params,
                                                    **fit_kw)
    except LinAlgError:
        # SVD convergence failure on badly misspecified models
        return

    except ValueError as error:
        if start_params is not None:  # don't recurse again
            # user supplied start_params only get one chance
            return
        # try a little harder, should be handled in fit really
        elif ('initial' not in error.args[0] or 'initial' in str(error)):
            start_params = [.1] * sum(order)
            if trend == 'c':
                start_params = [.1] + start_params
            return _safe_arma_fit(y, order, model_kw, trend, fit_kw,
                                  start_params)
        else:
            return
    except:  # no idea what happened
        return


def arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c',
                         model_kw={}, fit_kw={}):
    """
    Returns information criteria for many ARMA models

    Parameters
    ----------
    y : array-like
        Time-series data
    max_ar : int
        Maximum number of AR lags to use. Default 4.
    max_ma : int
        Maximum number of MA lags to use. Default 2.
    ic : str, list
        Information criteria to report. Either a single string or a list
        of different criteria is possible.
    trend : str
        The trend to use when fitting the ARMA models.
    model_kw : dict
        Keyword arguments to be passed to the ``ARMA`` model
    fit_kw : dict
        Keyword arguments to be passed to ``ARMA.fit``.

    Returns
    -------
    obj : Results object
        Each ic is an attribute with a DataFrame for the results. The AR order
        used is the row index. The ma order used is the column index. The
        minimum orders are available as ``ic_min_order``.

    Examples
    --------

    >>> from statsmodels.tsa.arima_process import arma_generate_sample
    >>> import statsmodels.api as sm
    >>> import numpy as np

    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> arparams = np.r_[1, -arparams]
    >>> maparam = np.r_[1, maparams]
    >>> nobs = 250
    >>> np.random.seed(2014)
    >>> y = arma_generate_sample(arparams, maparams, nobs)
    >>> res = sm.tsa.arma_order_select_ic(y, ic=['aic', 'bic'], trend='nc')
    >>> res.aic_min_order
    >>> res.bic_min_order

    Notes
    -----
    This method can be used to tentatively identify the order of an ARMA
    process, provided that the time series is stationary and invertible. This
    function computes the full exact MLE estimate of each model and can be,
    therefore a little slow. An implementation using approximate estimates
    will be provided in the future. In the meantime, consider passing
    {method : 'css'} to fit_kw.
    """
    from pandas import DataFrame

    ar_range = list(range(0, max_ar + 1))
    ma_range = list(range(0, max_ma + 1))
    if isinstance(ic, string_types):
        ic = [ic]
    elif not isinstance(ic, (list, tuple)):
        raise ValueError("Need a list or a tuple for ic if not a string.")

    results = np.zeros((len(ic), max_ar + 1, max_ma + 1))

    for ar in ar_range:
        for ma in ma_range:
            if ar == 0 and ma == 0 and trend == 'nc':
                results[:, ar, ma] = np.nan
                continue

            mod = _safe_arma_fit(y, (ar, ma), model_kw, trend, fit_kw)
            if mod is None:
                results[:, ar, ma] = np.nan
                continue

            for i, criteria in enumerate(ic):
                results[i, ar, ma] = getattr(mod, criteria)

    dfs = [DataFrame(res, columns=ma_range, index=ar_range) for res in results]

    res = dict(zip(ic, dfs))

    # add the minimums to the results dict
    min_res = {}
    for i, result in res.items():
        mins = np.where(result.min().min() == result)
        min_res.update({i + '_min_order' : (mins[0][0], mins[1][0])})
    res.update(min_res)

    return Bunch(**res)

def has_missing(data):
    """
    Returns True if 'data' contains missing entries, otherwise False
    """
    return np.isnan(np.sum(data))


def kpss(x, regression='c', lags=None, store=False):
    """
    Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

    Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null
    hypothesis that x is level or trend stationary.

    Parameters
    ----------
    x : array_like, 1d
        Data series
    regression : str{'c', 'ct'}
        Indicates the null hypothesis for the KPSS test
        * 'c' : The data is stationary around a constant (default)
        * 'ct' : The data is stationary around a trend
    lags : int
        Indicates the number of lags to be used. If None (default),
        lags is set to int(12 * (n / 100)**(1 / 4)), as outlined in
        Schwert (1989).
    store : bool
        If True, then a result instance is returned additionally to
        the KPSS statistic (default is False).

    Returns
    -------
    kpss_stat : float
        The KPSS test statistic
    p_value : float
        The p-value of the test. The p-value is interpolated from
        Table 1 in Kwiatkowski et al. (1992), and a boundary point
        is returned if the test statistic is outside the table of
        critical values, that is, if the p-value is outside the
        interval (0.01, 0.1).
    lags : int
        The truncation lag parameter
    crit : dict
        The critical values at 10%, 5%, 2.5% and 1%. Based on
        Kwiatkowski et al. (1992).
    resstore : (optional) instance of ResultStore
        An instance of a dummy class with results attached as attributes

    Notes
    -----
    To estimate sigma^2 the Newey-West estimator is used. If lags is None,
    the truncation lag parameter is set to int(12 * (n / 100) ** (1 / 4)),
    as outlined in Schwert (1989). The p-values are interpolated from
    Table 1 of Kwiatkowski et al. (1992). If the computed statistic is
    outside the table of critical values, then a warning message is
    generated.

    Missing values are not handled.

    References
    ----------
    D. Kwiatkowski, P. C. B. Phillips, P. Schmidt, and Y. Shin (1992): Testing
    the Null Hypothesis of Stationarity against the Alternative of a Unit Root.
    `Journal of Econometrics` 54, 159-178.
    """
    from warnings import warn

    nobs = len(x)
    x = np.asarray(x)
    hypo = regression.lower()

    # if m is not one, n != m * n
    if nobs != x.size:
        raise ValueError("x of shape {0} not understood".format(x.shape))

    if hypo == 'ct':
        # p. 162 Kwiatkowski et al. (1992): y_t = beta * t + r_t + e_t,
        # where beta is the trend, r_t a random walk and e_t a stationary
        # error term.
        resids = OLS(x, add_constant(np.arange(1, nobs + 1))).fit().resid
        crit = [0.119, 0.146, 0.176, 0.216]
    elif hypo == 'c':
        # special case of the model above, where beta = 0 (so the null
        # hypothesis is that the data is stationary around r_0).
        resids = x - x.mean()
        crit = [0.347, 0.463, 0.574, 0.739]
    else:
        raise ValueError("hypothesis '{0}' not understood".format(hypo))

    if lags is None:
        # from Kwiatkowski et al. referencing Schwert (1989)
        lags = int(np.ceil(12. * np.power(nobs / 100., 1 / 4.)))

    pvals = [0.10, 0.05, 0.025, 0.01]

    eta = sum(resids.cumsum()**2) / (nobs**2)  # eq. 11, p. 165
    s_hat = _sigma_est_kpss(resids, nobs, lags)

    kpss_stat = eta / s_hat
    p_value = np.interp(kpss_stat, crit, pvals)

    if p_value == pvals[-1]:
        warn("p-value is smaller than the indicated p-value", InterpolationWarning)
    elif p_value == pvals[0]:
        warn("p-value is greater than the indicated p-value", InterpolationWarning)

    crit_dict = {'10%': crit[0], '5%': crit[1], '2.5%': crit[2], '1%': crit[3]}

    if store:
        rstore = ResultsStore()
        rstore.lags = lags
        rstore.nobs = nobs

        stationary_type = "level" if hypo == 'c' else "trend"
        rstore.H0 = "The series is {0} stationary".format(stationary_type)
        rstore.HA = "The series is not {0} stationary".format(stationary_type)

        return kpss_stat, p_value, crit_dict, rstore
    else:
        return kpss_stat, p_value, lags, crit_dict


def _sigma_est_kpss(resids, nobs, lags):
    """
    Computes equation 10, p. 164 of Kwiatkowski et al. (1992). This is the
    consistent estimator for the variance.
    """
    s_hat = sum(resids**2)
    for i in range(1, lags + 1):
        resids_prod = np.dot(resids[i:], resids[:nobs - i])
        s_hat += 2 * resids_prod * (1. - (i / (lags + 1.)))
    return s_hat / nobs
