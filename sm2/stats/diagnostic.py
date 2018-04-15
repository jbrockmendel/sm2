"""
Upstream these are imported into stats.diagnostic from
sandbox.stats.diagnostic.  The remainder of this docstring is copied from that
file's docstring.

Author: josef-pktd
License: BSD-3

Notes
-----
Almost fully verified against R or Gretl, not all options are the same.
In many cases of Lagrange multiplier tests both the LM test and the F test is
returned. In some but not all cases, R has the option to choose the test
statistic. Some alternative test statistic results have not been verified.


TODO
* refactor to store intermediate results
* how easy is it to attach a test that is a class to a result instance,
  for example CompareCox as a method compare_cox(self, other) ?
* StatTestMC has been moved and should be deleted

missing:

* pvalues for breaks_hansen
* additional options, compare with R, check where ddof is appropriate
* new tests:
  - breaks_ap, more recent breaks tests
  - specification tests against nonparametric alternatives
"""
from __future__ import print_function

from six import integer_types
import numpy as np
from scipy import stats

from ._lilliefors import (kstest_fit, lilliefors, lillifors,  # noqa: F401
                          kstest_normal, kstest_exponential)
# lillifors is deprecated, misspelling
from ._adnorm import normal_ad  # noqa: F401


# TODO: The comment below was from upstream.  I _dislike_ this pattern
# TODO: I like the bunch pattern for this too.
class ResultsStore(object):
    # Upstream ResultsStore.__str__ will raise, see GH#4446
    pass


def linear_lm(resid, exog, func=None):
    """Lagrange multiplier test for linearity against functional alternative

    limitations: Assumes currently that the first column is integer.
    Currently it doesn't check whether the transformed variables contain NaNs,
    for example log of negative number.

    Parameters
    ----------
    resid : ndarray
        residuals of a regression
    exog : ndarray
        exogenous variables for which linearity is tested
    func : callable
        If func is None, then squares are used. func needs to take an array
        of exog and return an array of transformed variables.

    Returns
    -------
    lm : float
       Lagrange multiplier test statistic
    lm_pval : float
       p-value of Lagrange multiplier tes
    ftest : ContrastResult instance
       the results from the F test variant of this test

    Notes
    -----
    written to match Gretl's linearity test.
    The test runs an auxilliary regression of the residuals on the combined
    original and transformed regressors.
    The Null hypothesis is that the linear specification is correct.
    """
    from sm2.regression.linear_model import OLS

    if func is None:
        # default to f(x) == x**2
        func = np.square

    exog_aux = np.column_stack((exog, func(exog[:, 1:])))

    nobs, k_vars = exog.shape
    ls = OLS(resid, exog_aux).fit()
    ftest = ls.f_test(np.eye(k_vars - 1, k_vars * 2 - 1, k_vars))
    lm = nobs * ls.rsquared
    lm_pval = stats.chi2.sf(lm, k_vars - 1)
    return lm, lm_pval, ftest


def acorr_ljungbox(x, lags=None, boxpierce=True):
    """
    Ljung-Box test for no autocorrelation

    Parameters
    ----------
    x : array_like, 1d
        data series, regression residuals when used as diagnostic test
    lags : None, int or array_like
        If lags is an integer then this is taken to be the largest lag
        that is included, the test result is reported for all smaller lag
        length.
        If lags is a list or array, then all lags are included up to the
        largest lag in the list, however only the tests for the lags in the
        list are reported.
        If lags is None, then the default maxlag is 'min((nobs // 2 - 2), 40)'
    boxpierce : {False, True}
        If true, then additional to the results of the Ljung-Box test also the
        Box-Pierce test results are returned

    Returns
    -------
    lbvalue : float or array
        test statistic
    pvalue : float or array
        p-value based on chi-square distribution
    bpvalue : (optionsal), float or array
        test statistic for Box-Pierce test
    bppvalue : (optional), float or array
        p-value based for Box-Pierce test on chi-square distribution

    Notes
    -----
    Ljung-Box and Box-Pierce statistic differ in their scaling of the
    autocorrelation function. Ljung-Box test is reported to have better
    small sample properties.

    TODO: could be extended to work with more than one series
    1d or nd ? axis ? ravel ?
    needs more testing

    *Verification*

    Looks correctly sized in Monte Carlo studies.
    not yet compared to verified values

    Examples
    --------
    see example script  # TODO: either port this script or get new example

    References
    ----------
    Greene
    Wikipedia
    """
    if not boxpierce:  # pragma: no cover  # TODO: Update docstring
        raise NotImplementedError("`boxpierce=False` option is not ported from "
                                  "upstream.  In sm2, acorr_ljungbox always "
                                  "returns a tuple "
                                  "(qljungbox, pval, qboxpierce, pvalbp)")
    x = np.asarray(x)
    nobs = x.shape[0]
    if lags is None:
        lags = np.arange(1, min((nobs // 2 - 2), 40) + 1)
    elif isinstance(lags, integer_types):
        lags = np.arange(1, lags + 1)

    lags = np.asarray(lags)
    maxlag = max(lags)

    from sm2.tsa.stattools import acf
    acfx = acf(x, nlags=maxlag)[0]  # normalize by nobs not (nobs - nlags)
    # SS: unbiased=False is default now

    acf2norm = acfx[1:maxlag + 1]**2 / (nobs - np.arange(1, maxlag + 1))
    qljungbox = nobs * (nobs + 2) * np.cumsum(acf2norm)[lags - 1]
    # TODO: Is the above identical to tsa.stattools.q_stat?
    pval = stats.chi2.sf(qljungbox, lags)

    qboxpierce = nobs * np.cumsum(acfx[1:maxlag + 1]**2)[lags - 1]
    pvalbp = stats.chi2.sf(qboxpierce, lags)
    return qljungbox, pval, qboxpierce, pvalbp


def acorr_lm(x, maxlag=None, autolag='AIC', store=False, regresults=False):
    """Lagrange Multiplier tests for autocorrelation

    This is a generic Lagrange Multiplier test for autocorrelation. I don't
    have a reference for it, but it returns Engle's ARCH test if x is the
    squared residual array. A variation on it with additional exogenous
    variables is the Breusch-Godfrey autocorrelation test.

    Parameters
    ----------
    resid : ndarray, (nobs,)
        residuals from an estimation, or time series
    maxlag : int
        highest lag to use
    autolag : None or string
        If None, then a fixed number of lags given by maxlag is used.
    store : bool
        If true then the intermediate results are also returned

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    fval : float
        fstatistic for F test, alternative version of the same test based on
        F test for the parameter restriction
    fpval : float
        pvalue for F test
    resstore : instance (optional)
        a class instance that holds intermediate results. Only returned if
        store=True

    See Also
    --------
    het_arch
    acorr_breusch_godfrey
    """
    from sm2.regression.linear_model import OLS
    from sm2.tsa.tsatools import lagmat

    if regresults:
        store = True

    x = np.asarray(x)
    nobs = x.shape[0]
    if maxlag is None:
        # for adf from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12. * np.power(nobs / 100., 1 / 4.)))
        # nobs//4  # TODO: check default, or do AIC/BIC

    xdiff = np.diff(x)  # TODO: This is unused.  whats it for?

    xdall = lagmat(x[:, None], maxlag, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = x[-nobs:]

    if store:
        resstore = ResultsStore()

    if autolag:
        # search for lag length with highest information criteria
        # Note: I use the same number of observations to have comparable IC
        results = {}
        for mlag in range(1, maxlag + 1):
            results[mlag] = OLS(xshort, xdall[:, :mlag + 1]).fit()

        if autolag.lower() == 'aic':
            bestic, icbestlag = min((v.aic, k) for k, v in results.items())
        elif autolag.lower() == 'bic':
            icbest, icbestlag = min((v.bic, k) for k, v in results.items())
        else:  # pragma: no cover
            raise ValueError("autolag can only be None, 'AIC' or 'BIC'")

        # rerun ols with best ic
        xdall = lagmat(x[:, None], icbestlag, trim='both')
        nobs = xdall.shape[0]
        xdall = np.c_[np.ones((nobs, 1)), xdall]
        xshort = x[-nobs:]
        usedlag = icbestlag
        if regresults:
            resstore.results = results
    else:
        usedlag = maxlag

    resols = OLS(xshort, xdall[:, :usedlag + 1]).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, usedlag)
    # Note: degrees of freedom for LM test is nvars minus constant = usedlags

    if store:
        resstore.resols = resols
        resstore.usedlag = usedlag
        return lm, lmpval, fval, fpval, resstore
    else:
        return lm, lmpval, fval, fpval
    # TODO: remove multiple-return


def het_arch(resid, maxlag=None, autolag=None, store=False, regresults=False,
             ddof=0):
    """Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH)

    Parameters
    ----------
    resid : ndarray
        residuals from an estimation, or time series
    maxlag : int
        highest lag to use
    autolag : None or string
        If None, then a fixed number of lags given by maxlag is used.
    store : bool
        If true then the intermediate results are also returned
    ddof : int
        Not Implemented Yet
        If the residuals are from a regression, or ARMA estimation, then there
        are recommendations to correct the degrees of freedom by the number
        of parameters that have been estimated, for example ddof=p+a for an
        ARMA(p, q) (TODO: need reference, based on discussion on R finance
        mailinglist)

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    fval : float
        fstatistic for F test, alternative version of the same test based on
        F test for the parameter restriction
    fpval : float
        pvalue for F test
    resstore : instance (optional)
        a class instance that holds intermediate results. Only returned if
        store=True

    Notes
    -----
    verified agains R:FinTS::ArchTest
    """
    return acorr_lm(resid**2, maxlag=maxlag, autolag=autolag, store=store,
                    regresults=regresults)


def het_breuschpagan(resid, exog_het):
    """Breusch-Pagan Lagrange Multiplier test for heteroscedasticity

    The tests the hypothesis that the residual variance does not depend on
    the variables in x in the form

    :math: \sigma_i = \\sigma * f(\\alpha_0 + \\alpha z_i)

    Homoscedasticity implies that $\\alpha=0$

    Parameters
    ----------
    resid : array-like
        For the Breusch-Pagan test, this should be the residual of a
        regression.  If an array is given in exog, then the residuals are
        calculated by the an OLS regression or resid on exog. In this case
        resid should contain the dependent variable. Exog can be the same
        as x.
        TODO: I dropped the exog option, should I add it back?
    exog_het : array_like
        This contains variables that might create data dependent
        heteroscedasticity.

    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue :float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x
    f_pvalue : float
        p-value for the f-statistic

    Notes
    -----
    Assumes x contains constant (for counting dof and calculation of R^2).
    In the general description of LM test, Greene mentions that this test
    exaggerates the significance of results in small or moderately large
    samples. In this case the F-statistic is preferrable.

    *Verification*

    Chisquare test statistic is exactly (<1e-13) the same result as bptest
    in R-stats with defaults (studentize=True).

    Implementation
    This is calculated using the generic formula for LM test using $R^2$
    (Greene, section 17.6) and not with the explicit formula
    (Greene, section 11.4.3).
    The degrees of freedom for the p-value assume x is full rank.

    References
    ----------
    http://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test
    Greene 5th edition
    Breusch, Pagan article
    """
    from sm2.regression.linear_model import OLS

    x = np.asarray(exog_het)
    y = np.asarray(resid)**2
    nobs, nvars = x.shape
    resols = OLS(y, x).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    return lm, stats.chi2.sf(lm, nvars - 1), fval, fpval


def het_white(resid, exog, retres=False):
    """White's Lagrange Multiplier Test for Heteroscedasticity

    Parameters
    ----------
    resid : array_like
        residuals, square of it is used as endogenous variable
    exog : array_like
        possible explanatory variables for variance, squares and interaction
        terms are included in the auxilliary regression.
    resstore : instance (optional)
        a class instance that holds intermediate results. Only returned if
        store=True

    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue :float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x. This is an alternative test variant not the original LM test.
    f_pvalue : float
        p-value for the f-statistic

    Notes
    -----
    assumes x contains constant (for counting dof)

    question: does f-statistic make sense? constant ?
    TODO: answer the question above copied from upstream

    References
    ----------
    Greene section 11.4.1 5th edition p. 222
    now test statistic reproduces Greene 5th, example 11.3
    """
    from sm2.regression.linear_model import OLS

    x = np.asarray(exog)
    y = np.asarray(resid)
    if x.ndim == 1:  # pragma: no cover
        raise ValueError('x should have constant and at least one '
                         'more variable')

    nobs, nvars0 = x.shape
    i0, i1 = np.triu_indices(nvars0)
    exog = x[:, i0] * x[:, i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0 * (nvars0 - 1) / 2. + nvars0
    resols = OLS(y**2, exog).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    # degrees of freedom take possible reduced rank in exog into account
    # df_model checks the rank to determine df
    # extra calculation that can be removed:
    assert resols.df_model == np.linalg.matrix_rank(exog) - 1
    lmpval = stats.chi2.sf(lm, resols.df_model)
    return lm, lmpval, fval, fpval
