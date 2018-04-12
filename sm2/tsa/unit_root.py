#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for time series being stationarity or integration

Upstream most of this is in tsa.stattools
"""
import warnings

from six import integer_types
import numpy as np
from pandas.util._decorators import deprecate_kwarg
from scipy import stats

from sm2.tools.sm_exceptions import InterpolationWarning
from sm2.tools.tools import add_constant

from sm2.regression.linear_model import OLS

from sm2.tsa.tsatools import add_trend, lagmat
from sm2.tsa.adfvalues import mackinnonp, mackinnoncrit


class ResultsStore(object):
    def __str__(self):
        return self._str


@deprecate_kwarg("type", "method")
def q_stat(x, nobs, method="ljungbox"):
    """
    Return's Ljung-Box Q Statistic

    x : array-like
        Array of autocorrelation coefficients.  Can be obtained from acf.
    nobs : int
        Number of observations in the entire sample (ie., not just the length
        of the autocorrelation function results.

    TODO: The parameters section here is missing `method`

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
    if method == "ljungbox":  # TODO: Is this the only supported method?
        ret = (nobs * (nobs + 2) *
               np.cumsum((1. / (nobs - np.arange(1, len(x) + 1))) * x**2))
    chi2 = stats.chi2.sf(ret, np.arange(1, len(x) + 1))
    return ret, chi2


# upstream this is in tsa.stattools
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

    eta = (resids.cumsum()**2).sum() / nobs**2  # eq. 11, p. 165
    s_hat = _sigma_est_kpss(resids, nobs, lags)

    kpss_stat = eta / s_hat
    p_value = np.interp(kpss_stat, crit, pvals)

    if p_value == pvals[-1]:
        warnings.warn("p-value is smaller than the indicated p-value",
                      InterpolationWarning)
    elif p_value == pvals[0]:
        warnings.warn("p-value is greater than the indicated p-value",
                      InterpolationWarning)

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


# upstream this is in tsa.stattools
def _sigma_est_kpss(resids, nobs, lags):
    """
    Computes equation 10, p. 164 of Kwiatkowski et al. (1992). This is the
    consistent estimator for the variance.
    """
    s_hat = (resids**2).sum()
    for i in range(1, lags + 1):
        resids_prod = np.dot(resids[i:], resids[:nobs - i])
        s_hat += 2 * resids_prod * (1. - (i / (lags + 1.)))
    return s_hat / nobs


# upstream this is in tsa.stattools
# this needs to be converted to a class like HetGoldfeldQuandt,
# 3 different returns are a mess
# See:
# Ng and Perron(2001), Lag length selection and the construction of unit root
# tests with good size and power, Econometrica, Vol 69 (6) pp 1519-1554
# TODO: include drift keyword, only valid with regression == "c"
# just changes the distribution of the test statistic to a t distribution
# TODO: autolag is untested
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

    .. [*] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution
        functions for unit-root and cointegration tests.  `Journal of Business
        and Economic Statistics` 12, 167-76.

    .. [*] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."
        Queen's University, Dept of Economics, Working Papers.  Available at
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """

    if regresults:
        store = True

    trenddict = {None: 'nc', 0: 'c', 1: 'ct', 2: 'ctt'}
    if regression is None or isinstance(regression, integer_types):
        regression = trenddict[regression]
    regression = regression.lower()
    if regression not in ['c', 'nc', 'ct', 'ctt']:  # pragma: no cover
        raise ValueError("regression option %s not understood" % regression)
    x = np.asarray(x)
    nobs = x.shape[0]

    if maxlag is None:
        # from Greene referencing Schwert 1989
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
        # add +1 for level
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1
        # search for lag length with smallest information criteria
        # Note: use the same number of observations to have comparable IC
        # aic and bic: smaller is better

        icbest, bestlag, alres = _autolag(OLS, xdshort, fullRHS, startlag,
                                          maxlag, autolag, regresults=True)
        if regresults:
            resstore.autolag_results = alres

        bestlag -= startlag  # convert to lag not column index

        # rerun ols with best autolag
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
    # adfstat = (resols.params[0]-1.0)/resols.bse[0]
    # the "asymptotically correct" z statistic is obtained as
    # nobs/(1-np.sum(resols.params[1:-(trendorder+1)])) (resols.params[0] - 1)
    # I think this is the statistic that is used for series that are integrated
    # for orders higher than I(1), ie., not ADF but cointegration tests.

    # Get approx p-value and critical values
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {"1%": critvalues[0],
                  "5%": critvalues[1],
                  "10%": critvalues[2]}
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
        # TODO: remove multiple-return


def _autolag(mod, endog, exog, startlag, maxlag, method, modargs=(),
             fitargs=(), regresults=True):
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
    if not regresults:  # pragma: no cover
        # TODO: update docstring
        raise NotImplementedError("option `regresults=False` not ported "
                                  "from upstream.  _autolag always returns "
                                  "a tuple (icbest, bestlag, results)")

    # TODO: can tcol be replaced by maxlag + 2?
    # TODO: This could be changed to laggedRHS and exog keyword arguments if
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
        # stop = stats.norm.ppf(.95)
        stop = 1.6448536269514722
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            if np.abs(icbest) >= stop:
                bestlag = lag
                icbest = icbest
                break
    else:  # pragma: no cover
        raise ValueError("Information Criterion %s not understood." % method)

    return icbest, bestlag, results


# upstream this is in tsa.stattools
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
    if trend not in ['c', 'nc', 'ct', 'ctt']:  # pragma: no cover
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
