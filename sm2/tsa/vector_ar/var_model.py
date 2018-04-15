#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vector Autoregression (VAR) processes

References
----------
Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
"""
from __future__ import division, print_function

from six import string_types
from six.moves import range, StringIO
from collections import defaultdict

import numpy as np
from scipy import stats
import scipy.linalg

from sm2.tools.decorators import (cache_readonly, cached_value, cached_data,
                                  deprecated_alias, copy_doc)
from sm2.tools.linalg import logdet_symm, chain_dot

import sm2.base.wrapper as wrap

from sm2.iolib.table import SimpleTable

from sm2.tsa.tsatools import vec, duplication_matrix
from sm2.tsa.base import tsa_model
from sm2.tsa import wold, autocov

from . import irf, output, plotting, util
from .hypothesis_test_results import (CausalityTestResults,
                                      NormalityTestResults,
                                      WhitenessTestResults)

# aliases for upstream compat
_compute_acov = autocov.compute_acov
_acovs_to_acorrs = autocov.acf_to_acorr
_var_acf = autocov.var_acf


# --------------------------------------------------------------------
# VAR process routines

def ma_rep(coefs, maxn=10):  # pragma: no cover
    raise NotImplementedError("ma_rep is not ported from upstream, "
                              "is instead implemented directly in VARProcess "
                              "(or more specifically, VARParams, of which "
                              "VARProcess is a subclass).")


def is_stable(coefs, verbose=False):  # pragma: no cover
    raise NotImplementedError("is_stable is not ported from upstream, "
                              "is instead implemented directly in "
                              "VARProcess.is_stable.")


def var_acf(coefs, sig_u, nlags=None):  # pragma: no cover
    raise NotImplementedError("var_acf is not ported from upstream, "
                              "is instead implemented directly in "
                              "var_acf.is_stable.")


def forecast_cov(ma_coefs, sigma_u, steps):  # pragma: no cover
    raise NotImplementedError("forecast_cov not ported from upstream, since "
                              "it is effectively identical to "
                              "VARProcess.forecast_cov (or VARResults.mse)")


mse = forecast_cov


def _forecast_vars(steps, ma_coefs, sig_u):  # pragma: no cover
    raise NotImplementedError("_forecast_vars not ported from upstream, "
                              "as (contrary to its docstring) it is entirely "
                              "redundant with VARREsults methods.  See GH#4459")


def forecast_interval(y, coefs, trend_coefs, sig_u, steps=5, alpha=0.05,
                      exog=1):  # pragma: no cover
    raise NotImplementedError("forecast_interval not ported from upstream "
                              "since it is redundant with "
                              "VARProcess.forecast_interval.  Use that method "
                              "(or VARResults.forecast_interval) instead.")


def var_loglike(resid, omega, nobs):  # pragma: no cover
    raise NotImplementedError("var_loglike not ported from upstream, "
                              "implemented directly in VARResults.llf")


def orth_ma_rep(results, maxn=10, P=None):  # pragma: no cover
    raise NotImplementedError("orth_ma_rep not ported from upstream, "
                              "is instead implemented directly as a "
                              "VARResults method.")


def test_normality(results, signif=0.05):  # pragma: no cover
    raise NotImplementedError("test_normality is not ported from upstream, is"
                              "instead implemented directly in "
                              "VARResults.test_normality")


def forecast(y, coefs, trend_coefs, steps, exog=None):
    """
    Produce linear minimum MSE forecast

    Parameters
    ----------
    y : ndarray (k_ar x neqs)
    coefs : ndarray (k_ar x neqs x neqs)
    trend_coefs : ndarray (1 x neqs) or (neqs)
    steps : int
    exog : ndarray (trend_coefs.shape[1] x neqs)

    Returns
    -------
    forecasts : ndarray (steps x neqs)

    Notes
    -----
    Lütkepohl p. 37
    """
    p = len(coefs)
    k = len(coefs[0])
    # initial value
    forcs = np.zeros((steps, k))
    if exog is not None and trend_coefs is not None:
        forcs += np.dot(exog, trend_coefs)
    # to make existing code (with trend_coefs=intercept and without exog) work:
    elif exog is None and trend_coefs is not None:
        forcs += trend_coefs  # TODO: not hit, also dumb overloading

    # h=0 forecast should be latest observation
    # forcs[0] = y[-1]

    # make indices easier to think about
    for h in range(1, steps + 1):
        # y_t(h) = intercept + sum_1^p A_i y_t_(h-i)
        f = forcs[h - 1]
        for i in range(1, p + 1):
            # slightly hackish
            if h - i <= 0:
                # e.g. when h=1, h-1 = 0, which is y[-1]
                prior_y = y[h - i - 1]
            else:
                # e.g. when h=2, h-1=1, which is forcs[0]
                prior_y = forcs[h - i - 1]

            # i=1 is coefs[0]
            f = f + np.dot(coefs[i - 1], prior_y)

        forcs[h - 1] = f

    return forcs


def _validate_causing(causing, name="causing", allow_none=False):
    allowed_types = (string_types, int)
    if isinstance(causing, allowed_types):
        causing = [causing]
    if not all(isinstance(c, allowed_types) for c in causing):
        msg = ("{name} has to be of type string or int "
               "(or a sequence of these types)")
        if allow_none:
            msg += " or None."
        raise TypeError(msg.format(name=name))
    return causing


def _reordered(self, order):
    # Create new arrays to hold rearranged results from .fit()
    endog = self.endog
    endog_lagged = self.endog_lagged
    params = self.params
    sigma_u = self.sigma_u
    names = self.names
    k_ar = self.k_ar
    endog_new = np.zeros([np.size(endog, 0), np.size(endog, 1)])
    endog_lagged_new = np.zeros([np.size(endog_lagged, 0),
                                 np.size(endog_lagged, 1)])
    params_new_inc, params_new = [np.zeros([np.size(params, 0),
                                            np.size(params, 1)])
                                  for i in range(2)]
    sigma_u_new_inc, sigma_u_new = [np.zeros([np.size(sigma_u, 0),
                                              np.size(sigma_u, 1)])
                                    for i in range(2)]
    num_end = len(self.params[0])
    names_new = []

    # Rearrange elements and fill in new arrays
    k = self.k_trend
    for i, c in enumerate(order):
        endog_new[:, i] = self.endog[:, c]
        if k > 0:
            params_new_inc[0, i] = params[0, i]
            endog_lagged_new[:, 0] = endog_lagged[:, 0]
        for j in range(k_ar):
            jnk = j * num_end + k
            params_new_inc[i + jnk, :] = self.params[c + jnk, :]
            endog_lagged_new[:, i + jnk] = endog_lagged[:, c + jnk]

        sigma_u_new_inc[i, :] = sigma_u[c, :]
        names_new.append(names[c])
    for i, c in enumerate(order):
        params_new[:, i] = params_new_inc[:, c]
        sigma_u_new[:, i] = sigma_u_new_inc[:, c]

    return VARResults(endog=endog_new, endog_lagged=endog_lagged_new,
                      params=params_new, sigma_u=sigma_u_new,
                      lag_order=self.k_ar, model=self.model,
                      trend='c', names=names_new, dates=self.dates)


class LagOrderResults:
    """
    Results class for choosing a model's lag order.

    Parameters
    ----------
    ics : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. A corresponding value is a list of information criteria for
        various numbers of lags.
    selected_orders: dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. The corresponding value is an integer specifying the number
        of lags chosen according to a given criterion (key).
    vecm : bool, default: `False`
        `True` indicates that the model is a VECM. In case of a VAR model
        this argument must be `False`.

    Notes
    -----
    In case of a VECM the shown lags are lagged differences.
    """
    def __init__(self, ics, selected_orders, vecm=False):
        self.title = ("VECM" if vecm else "VAR") + " Order Selection"
        self.title += " (* highlights the minimums)"
        self.ics = ics
        self.selected_orders = selected_orders
        self.vecm = vecm
        self.aic = selected_orders["aic"]
        self.bic = selected_orders["bic"]
        self.hqic = selected_orders["hqic"]
        self.fpe = selected_orders["fpe"]

    def summary(self):  # basically copied from (now deleted) print_ic_table()
        cols = sorted(self.ics)  # ["aic", "bic", "hqic", "fpe"]
        str_data = np.array([["%#10.4g" % v for v in self.ics[c]]
                             for c in cols],
                            dtype=object).T
        # mark minimum with an asterisk
        for i, col in enumerate(cols):
            idx = int(self.selected_orders[col]), i
            str_data[idx] += '*'
        return SimpleTable(str_data, [col.upper() for col in cols],
                           list(range(len(str_data))), title=self.title)

    def __str__(self):
        return "<" + self.__module__ + "." + self.__class__.__name__ \
                   + " object. Selected orders are: AIC -> " + str(self.aic) \
                   + ", BIC -> " + str(self.bic)  \
                   + ", FPE -> " + str(self.fpe) \
                   + ", HQIC -> " + str(self.hqic) + ">"

# -------------------------------------------------------------------------------
# VARProcess class: for known or unknown VAR process


class VAR(tsa_model.TimeSeriesModel):
    r"""
    Fit VAR(p) process and do lag order selection

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    Parameters
    ----------
    endog : array-like
        2-d endogenous response variable. The independent variable.
    exog : array-like
        2-d exogenous variable.
    dates : array-like
        must match number of rows of endog

    References
    ----------
    Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
    """

    y = deprecated_alias('y', 'endog')

    def __init__(self, endog, exog=None, dates=None, freq=None, missing='none'):
        super(VAR, self).__init__(endog, exog, dates, freq, missing=missing)
        if self.endog.ndim == 1:  # pragma: no cover
            raise ValueError("Only gave one variable to VAR")
        self.neqs = self.endog.shape[1]
        self.n_totobs = len(endog)

    def _get_predict_start(self, start, k_ar):
        if start is None:
            start = k_ar
        return super(VAR, self)._get_predict_start(start)

    def predict(self, params, start=None, end=None, lags=1, trend='c'):
        """
        Returns in-sample predictions or forecasts
        """
        start = self._get_predict_start(start, lags)
        end, out_of_sample = self._get_predict_end(end)

        if end < start:
            raise ValueError("end is before start")
        if end == start + out_of_sample:
            return np.array([])

        k_trend = util.get_trendorder(trend)
        k = self.neqs
        k_ar = lags

        predictedvalues = np.zeros((end + 1 - start + out_of_sample, k))
        if k_trend != 0:
            intercept = params[:k_trend]
            predictedvalues += intercept

        y = self.endog
        X = util.get_var_endog(y, lags, trend=trend, has_constant='raise')
        fittedvalues = np.dot(X, params)

        fv_start = start - k_ar
        pv_end = min(len(predictedvalues), len(fittedvalues) - fv_start)
        fv_end = min(len(fittedvalues), end - k_ar + 1)
        predictedvalues[:pv_end] = fittedvalues[fv_start:fv_end]

        if not out_of_sample:
            return predictedvalues

        # fit out of sample
        y = y[-k_ar:]
        coefs = params[k_trend:].reshape((k_ar, k, k)).swapaxes(1, 2)
        predictedvalues[pv_end:] = forecast(y, coefs, intercept, out_of_sample)
        return predictedvalues

    def fit(self, maxlags=None, method='ols', ic=None, trend='c',
            verbose=False):
        # TODO: this code is only supporting deterministic terms as exog.
        # This means that all exog-variables have lag 0. If dealing with
        # different exogs is necessary, a `lags_exog`-parameter might make
        # sense (e.g. a sequence of ints specifying lags).
        # Alternatively, leading zeros for exog-variables with smaller number
        # of lags than the maximum number of exog-lags might work.
        """
        Fit the VAR model

        Parameters
        ----------
        maxlags : int
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        method : {'ols'}
            Estimation method to use
        ic : {'aic', 'fpe', 'hqic', 'bic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz
        verbose : bool, default False
            Print order selection output to the screen
        trend : str {"c", "ct", "ctt", "nc"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "nc" - co constant, no trend
            Note that these are prepended to the columns of the dataset.

        Notes
        -----
        Lütkepohl pp. 146-153

        Returns
        -------
        est : VARResultsWrapper
        """
        lags = maxlags

        if trend not in ['c', 'ct', 'ctt', 'nc']:
            raise ValueError("trend '{}' not supported for VAR".format(trend))

        if ic is not None:
            selections = self.select_order(maxlags=maxlags)
            if not hasattr(selections, ic):
                # TODO: upstream this is Exception, fix to ValueError
                raise ValueError("%s not recognized, must be among %s"
                                 % (ic, sorted(selections)))
            lags = getattr(selections, ic)
            if verbose:
                print(selections)
                print('Using %d based on %s criterion' % (lags, ic))
        else:
            if lags is None:
                lags = 1

        k_trend = util.get_trendorder(trend)
        self.exog_names = util.make_lag_names(self.endog_names, lags, k_trend)
        # TODO: Don't set self attrs at this level
        self.nobs = self.n_totobs - lags

        # add exog to data.xnames (necessary because the length of xnames also
        # determines the allowed size of VARResults.params)
        if self.exog is not None:
            x_names_to_add = [("exog%d" % i)
                              for i in range(self.exog.shape[1])]
            self.data.xnames = (self.data.xnames[:k_trend] + x_names_to_add +
                                self.data.xnames[k_trend:])

        return self._estimate_var(lags, trend=trend)

    def _build_rhs(self, lags, offset=0, trend='c'):
        """
        Construct the array of regressors to use as RHS variables in the
        VAR Model.

        Parameters
        ----------
        lags : int
        offset : int, default 0
        trend : {'nc', 'c', 'ct', 'ctt'}, default 'c'

        Returns
        -------
        rhs : np.ndarray
        """
        k_trend = util.get_trendorder(trend)

        if offset < 0:  # pragma: no cover
            raise ValueError('offset must be >= 0')

        nobs = self.n_totobs - lags - offset
        endog = self.endog[offset:]
        exog = None if self.exog is None else self.exog[offset:]
        z = util.get_var_endog(endog, lags, trend=trend,
                               has_constant='raise')
        if exog is not None:
            # TODO: currently only deterministic terms supported (exoglags==0)
            # and since exoglags==0, x will be an array of size 0.
            x = util.get_var_endog(exog[-nobs:], 0, trend="nc",
                                   has_constant="raise")
            x_inst = exog[-nobs:]
            x = np.column_stack((x, x_inst))
            del x_inst  # free memory
            temp_z = z
            z = np.empty((x.shape[0], x.shape[1] + z.shape[1]))
            z[:, :k_trend] = temp_z[:, :k_trend]
            z[:, k_trend:k_trend + x.shape[1]] = x
            z[:, k_trend + x.shape[1]:] = temp_z[:, k_trend:]
            del temp_z, x  # free memory

        # the following modification of z is necessary to get the same results
        # as JMulTi for the constant-term-parameter...
        for i in range(k_trend):
            if (np.diff(z[:, i]) == 1).all():  # modify the trend-column
                z[:, i] += lags
            # make the same adjustment for the quadratic term
            if (np.diff(np.sqrt(z[:, i])) == 1).all():
                z[:, i] = (np.sqrt(z[:, i]) + lags)**2

        return z

    def _estimate_var(self, lags, offset=0, trend='c'):
        """
        lags : int
            Lags of the endogenous variable.
        offset : int
            Periods to drop from beginning-- for order selection so it's an
            apples-to-apples comparison
        trend : string or None
            As per above
        """
        # have to do this again because select_order doesn't call fit
        k_trend = util.get_trendorder(trend)
        # Note: upstream sets self.k_trend, shouldnt

        if offset < 0:  # pragma: no cover
            raise ValueError('offset must be >= 0')

        endog = self.endog[offset:]
        exog = None if self.exog is None else self.exog[offset:]

        rhs = self._build_rhs(lags, offset, trend)
        y_sample = endog[lags:]
        # Lütkepohl p75, about 5x faster than stated formula
        params = np.linalg.lstsq(rhs, y_sample, rcond=-1)[0]
        resid = y_sample - np.dot(rhs, params)

        # Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
        # process $u$
        # equivalent definition
        # .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
        # Z^\prime) Y
        # Ref: Lutkepohl p.75
        # df_resid right now is T - Kp - 1, which is a suggested correction

        avobs = len(y_sample)
        if exog is not None:
            k_trend += exog.shape[1]  # TODO: Just define k_exog?
        df_resid = avobs - (self.neqs * lags + k_trend)

        sse = np.dot(resid.T, resid)
        omega = sse / df_resid

        varfit = VARResults(endog, rhs, params, omega, lags,
                            names=self.endog_names, trend=trend,
                            dates=self.data.dates, model=self, exog=self.exog)
        return VARResultsWrapper(varfit)

    def select_order(self, maxlags=None, trend="c"):
        """
        Compute lag order selections based on each of the available information
        criteria

        Parameters
        ----------
        maxlags : int
            if None, defaults to 12 * (nobs/100.)**(1./4)
        trend : str {"nc", "c", "ct", "ctt"}
            * "nc" - no deterministic terms
            * "c" - constant term
            * "ct" - constant and linear term
            * "ctt" - constant, linear, and quadratic term

        Returns
        -------
        selections : LagOrderResults
        """
        if maxlags is None:
            maxlags = int(round(12 * (len(self.endog) / 100.)**(1 / 4.)))
            # TODO: This expression shows up in a bunch of places, but
            # in some it is `int` and in others `np.ceil`.  Also in some
            # it multiplies by 4 instead of 12.  Let's put these all in
            # one place and document when to use which variant.

        ics = defaultdict(list)
        p_min = 0 if self.exog is not None or trend != "nc" else 1
        for p in range(p_min, maxlags + 1):
            # exclude some periods to same amount of data used for each lag
            # order
            result = self._estimate_var(p, offset=maxlags - p, trend=trend)

            for k, v in result.info_criteria.items():
                ics[k].append(v)

        selected_orders = dict((k, np.array(v).argmin() + p_min)
                               for k, v in ics.items())
        return LagOrderResults(ics, selected_orders, vecm=False)


class VARProcess(wold.VARProcess):
    """
    Class represents a known VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
    intercept : ndarray (length k)
    sigma_u : ndarray (k x k)
    names : sequence (length k)
    """
    # -------------------------------------------------------------
    # Methods requiring `coefs` and `sigma_u`, but not `exog`

    def plot_acorr(self, nlags=10, linewidth=8):  # TODO: belongs in wold?
        "Plot theoretical autocorrelation function"
        plotting.plot_full_acorr(self.acorr(nlags=nlags), linewidth=linewidth)

    def _forecast_vars(self, steps):
        # TODO: Should this go in wold?  Even if forecast_cov is
        # overriden in VARResults?
        covs = self.forecast_cov(steps)

        # Take diagonal for each cov
        inds = np.arange(self.neqs)
        return covs[:, inds, inds]

    # TODO: having this involve `exog` doesn't fit with the
    # "known VAR(p) process" definition in the docstring
    def __init__(self, coefs, intercept, exog, sigma_u, trend, names=None):
        wold.VARProcess.__init__(self, coefs,
                                 intercept=intercept, sigma_u=sigma_u)
        self.exog = exog
        self.trend = trend
        self.names = names
        # TODO: names not used here, move them out of this class?

    def get_eq_index(self, name):  # pragma: no cover
        raise NotImplementedError("get_eq_index is not ported from upstream, "
                                  "as it is neither used nor tested.")

    def __str__(self):
        out = ('VAR(%d) process for %d-dimensional response y_t'
               % (self.k_ar, self.neqs))
        out += '\nstable: %s' % self.is_stable()
        out += '\nmean: %s' % self.mean()
        return out

    def plotsim(self, steps=1000):
        """
        Plot a simulation from the VAR(p) process for the desired number of
        steps
        """
        # TODO: we might need to pass intercept along with /instead of exog?
        Y = util.varsim(self.coefs, self.exog, self.sigma_u, steps=steps)
        plotting.plot_mts(Y)
        # FIXME: passing self.exog here is wrong

    def forecast(self, y, steps, exog_future=None):
        """Produce linear minimum MSE forecasts for desired number of steps
        ahead, using prior values y

        Parameters
        ----------
        y : ndarray (p x k)
        steps : int

        Returns
        -------
        forecasts : ndarray (steps x neqs)

        Notes
        -----
        Lütkepohl pp 37-38
        """
        exog_future, trend_coefs = self._build_exog_future(exog_future, steps)
        return forecast(y, self.coefs, trend_coefs, steps, exog_future)

    def _build_exog_future(self, exog_future, steps):
        if self.exog is None and exog_future is not None:
            raise ValueError("No exog in model, so no exog_future supported "
                             "in forecast method.")  # pragma: no cover
        if self.exog is not None and exog_future is None:
            raise ValueError("Please provide an exog_future argument to "
                             "the forecast method.")  # pragma: no cover
        trend_coefs = None if self.coefs_exog.size == 0 else self.coefs_exog.T
        # TODO: upstream is really, really dumb sometimes.  coefs_exog doesn't
        # exist at this level!
        trend = self.trend

        exogs = []
        if trend.startswith("c"):  # constant term
            exogs.append(np.ones(steps))
        exog_lin_trend = np.arange(self.n_totobs + 1,
                                   self.n_totobs + 1 + steps)
        if "t" in trend:
            exogs.append(exog_lin_trend)
        if "tt" in trend:
            exogs.append(exog_lin_trend**2)
        if exog_future is not None:
            exogs.append(exog_future)

        if exogs == []:
            exog_future = None
        else:
            exog_future = np.column_stack(exogs)
        return exog_future, trend_coefs

    def forecast_interval(self, y, steps, alpha=0.05, exog_future=None):
        """Construct forecast interval estimates assuming the y are Gaussian

        Notes
        -----
        Lütkepohl pp. 39-40

        Returns
        -------
        (mid, lower, upper) : (ndarray, ndarray, ndarray)
        """
        assert 0 < alpha < 1
        q = util.norm_signif_level(alpha)

        point_forecast = self.forecast(y, steps, exog_future=exog_future)
        sigma = np.sqrt(self._forecast_vars(steps))

        forc_lower = point_forecast - q * sigma
        forc_upper = point_forecast + q * sigma

        return point_forecast, forc_lower, forc_upper

    @copy_doc(wold.VARParams.mean.__doc__)
    def mean(self):
        if self.trend != "c" or (self.exog is not None and
                                 self.exog.shape[1] != 1):
            raise NotImplementedError("VAR Process mean is not well-defined "
                                      "when there are exogenous regressors "
                                      "other than a constant, or if there "
                                      "is no constant term")
        return wold.VARParams.mean(self)

    # TODO: move to VARParams?
    def to_vecm(self):
        k = self.coefs.shape[1]
        p = self.coefs.shape[0]
        A = self.coefs
        pi = -(np.identity(k) - np.sum(A, 0))
        gamma = np.zeros((p - 1, k, k))
        for i in range(p - 1):
            gamma[i] = -(np.sum(A[i + 1:], 0))
        gamma = np.concatenate(gamma, 1)
        return {"Gamma": gamma, "Pi": pi}


# -------------------------------------------------------------------
# VARResults class


# TODO: Make this subclass Results?
class VARResults(VARProcess, tsa_model.TimeSeriesModelResults):
    """Estimate VAR(p) process with fixed number of lags

    Parameters
    ----------
    endog : array
    endog_lagged : array
    params : array
    sigma_u : array
    lag_order : int
    model : VAR model instance
    trend : str {'nc', 'c', 'ct'}
    names : array-like
        List of names of the endogenous variables in order of
        appearance in `endog`.
    dates
    exog : array

    Returns
    -------
    **Attributes**
    aic
    bic
    bse
    coefs : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    cov_params
    dates
    detomega
    df_model : int
    df_resid : int
    endog
    endog_lagged
    fittedvalues
    fpe
    intercept
    info_criteria
    k_ar : int
    k_trend : int
    llf
    model
    names
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    params
    k_ar : int
        Order of VAR process
    params : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]
    pvalues
    names : list
        variables names
    resid
    roots : array
        The roots of the VAR process are the solution to
        (I - coefs[0]*z - coefs[1]*z**2 ... - coefs[p-1]*z**k_ar) = 0.
        Note that the inverse roots are returned, and stability requires that
        the roots lie outside the unit circle.
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    sigma_u_mle
    stderr
    trenorder
    tvalues
    y :
    endog_lagged
    """
    _model_type = 'VAR'  # TODO: remove?

    @property
    def df_model(self):
        """Number of estimated parameters, including the intercept / trends"""
        return self.neqs * self.k_ar + self.k_trend
        # TODO: Should neqs be multiplying k_trend?

    @property
    def df_resid(self):
        """Number of observations minus number of estimated parameters"""
        return self.nobs - self.df_model

    @cache_readonly
    def nobs(self):
        return self.n_totobs - self.k_ar

    y = deprecated_alias('y', 'endog')
    ys_lagged = deprecated_alias('ys_lagged', 'endog_lagged')

    def __init__(self, endog, endog_lagged, params, sigma_u, lag_order,
                 model=None, trend='c', names=None, dates=None, exog=None):

        self.model = model
        self.params = params
        self.endog = endog
        # TODO: Most results classes dont have endog; should this?
        self.endog_lagged = endog_lagged
        self.dates = dates
        self.trend = trend

        self.n_totobs, neqs = self.endog.shape
        k_trend = util.get_trendorder(trend)
        self.exog_names = util.make_lag_names(names, lag_order, k_trend, exog)

        # Initialize VARProcess parent class
        # construct coefficient matrices
        # Each matrix needs to be transposed
        endog_start = k_trend
        if exog is not None:
            endog_start += exog.shape[1]
        reshaped = self.params[endog_start:]
        reshaped = reshaped.reshape((lag_order, neqs, neqs))
        # Need to transpose each coefficient matrix
        coefs = reshaped.swapaxes(1, 2).copy()

        self.coefs_exog = params[:endog_start].T
        self.k_trend = self.coefs_exog.shape[1]
        mshape = model.exog.shape if model.exog is not None else (0, 0)
        assert self.k_trend == k_trend + mshape[1], (k_trend, self.k_trend,
                                                     trend, mshape,
                                                     self.coefs_exog)

        if "c" in trend:
            intercept = params[0, :]
        else:
            intercept = np.zeros(neqs)

        VARProcess.__init__(self, coefs, intercept, exog, sigma_u,
                            trend=trend, names=names)

    @cache_readonly
    def sigma_u_mle(self):
        """(Biased) maximum likelihood estimate of noise process covariance"""
        return self.sigma_u * self.df_resid / self.nobs

    @cached_data
    def fittedvalues(self):
        """The predicted insample values of the response variables of
        the model.
        """
        return np.dot(self.endog_lagged, self.params)
        # TODO: Can we use self.model.predict here?  then base class is OK

    @cached_data
    def resid(self):
        """Residuals of response variable resulting from estimated coefficients
        """
        return self.endog[self.k_ar:] - self.fittedvalues

    @cache_readonly
    def llf(self):
        r"""Compute VAR(p) loglikelihood

        Returns
        -------
        llf : float
            The value of the loglikelihood function for a VAR(p) model

        Notes
        -----
        The loglikelihood function for the VAR(p) is

        .. math::

            -\left(\frac{T}{2}\right)
            \left(\ln\left|\Omega\right|-K\ln\left(2\pi\right)-K\right)
        """
        omega = self.sigma_u_mle
        nobs = self.nobs

        logdet = logdet_symm(np.asarray(omega))
        neqs = len(omega)
        part1 = - (nobs * neqs / 2) * np.log(2 * np.pi)
        part2 = - (nobs / 2) * (logdet + neqs)
        return part1 + part2
        # TODO: Can we define a general loglike?

    @cached_value
    def cov_params(self):
        """Estimated variance-covariance of model coefficients

        Notes
        -----
        Covariance of vec(B), where B is the matrix
        [params_for_deterministic_terms, A_1, ..., A_p] with the shape
        (K x (Kp + number_of_deterministic_terms))
        Adjusted to be an unbiased estimator
        Ref: Lütkepohl p.74-75
        """
        z = self.endog_lagged
        return np.kron(scipy.linalg.inv(np.dot(z.T, z)), self.sigma_u)
        # TODO: Why are we using sigma_u here instead of sigma_u_mle?

    @cache_readonly
    def stderr(self):
        """Standard errors of coefficients, reshaped to match in size"""
        stderr = np.sqrt(np.diag(self.cov_params))
        return stderr.reshape((self.df_model, self.neqs), order='C')
        # TODO: why df_model?  could we use self.params.shape?
        # then could we just use the base class version?

    bse = stderr  # sm2 interface?

    # TODO: Just use default version form LikelihodoodModelResults?
    # Only docstring different
    @cache_readonly
    def tvalues(self):
        """Compute t-statistics. Use Student-t(T - Kp - 1) = t(df_resid)
        to test significance.
        """
        return self.params / self.stderr

    # TODO: Just use default version form LikelihodoodModelResults?
    # Only docstring different
    @cache_readonly
    def pvalues(self):
        """Two-sided p-values for model coefficients from
        Student t-distribution
        """
        # return 2 * stats.t.sf(np.abs(self.tvalues), self.df_resid)
        return 2 * stats.norm.sf(np.abs(self.tvalues))
        # TODO: is the docstring inaccurate? this uses stats.norm, not stats.t

    # ------------------------------------------------------------
    # Sample Methods - just require endog (and names, dates, k_ar)

    def plot(self):
        """Plot input time series"""
        plotting.plot_mts(self.endog, names=self.names, index=self.dates)

    def sample_acov(self, nlags=1):
        return _compute_acov(self.endog[self.k_ar:], nlags=nlags)

    def sample_acorr(self, nlags=1):
        acovs = self.sample_acov(nlags=nlags)
        return autocov.acf_to_acorr(acovs)

    def plot_sample_acorr(self, nlags=10, linewidth=8):
        "Plot theoretical autocorrelation function"
        plotting.plot_full_acorr(self.sample_acorr(nlags=nlags),
                                 linewidth=linewidth)

    # ------------------------------------------------------------
    # Resid Methods - require only self.resid

    def resid_acov(self, nlags=1):
        """
        Compute centered sample autocovariance (including lag 0)

        Parameters
        ----------
        nlags : int
        """
        return _compute_acov(self.resid, nlags=nlags)

    def resid_acorr(self, nlags=1):
        """
        Compute sample autocorrelation (including lag 0)

        Parameters
        ----------
        nlags : int
        """
        acovs = self.resid_acov(nlags=nlags)
        return autocov.acf_to_acorr(acovs)

    @cached_value
    def resid_corr(self):
        "Centered residual correlation matrix"
        return self.resid_acorr(0)[0]

    # ------------------------------------------------------------
    # Estimation-related things

    @cache_readonly
    def _zz(self):  # pragma: no cover
        raise NotImplementedError("_zz not ported from upstream, "
                                  "as it is neither used nor tested there.")

    @property
    def _cov_alpha(self):
        """
        Estimated covariance matrix of model coefficients w/o exog
        """
        # drop exog
        return self.cov_params[self.k_trend * self.neqs:,
                               self.k_trend * self.neqs:]

    @cache_readonly
    def stderr_endog_lagged(self):
        start = self.k_trend
        return self.stderr[start:]

    @cache_readonly
    def stderr_dt(self):
        end = self.k_trend
        return self.stderr[:end]

    @cache_readonly
    def tvalues_endog_lagged(self):
        start = self.k_trend
        return self.tvalues[start:]

    @cache_readonly
    def tvalues_dt(self):
        end = self.k_trend
        return self.tvalues[:end]

    @cache_readonly
    def pvalues_endog_lagged(self):
        start = self.k_trend
        return self.pvalues[start:]

    @cache_readonly
    def pvalues_dt(self):
        end = self.k_trend
        return self.pvalues[:end]

    # TODO: -------------------------------------------------------------
    # Forecast error covariance functions

    def forecast_cov(self, steps=1):
        r"""Compute forecast covariance matrices for desired number of steps

        Parameters
        ----------
        steps : int

        Notes
        -----
        .. math:: \Sigma_{\hat y}(h) = \Sigma_y(h) + \Omega(h) / T

        Ref: Lütkepohl pp. 96-97

        Returns
        -------
        covs : ndarray (steps x k x k)
        """
        mse = self.mse(steps)
        # omegas = self._omega_forc_cov(steps)
        # TODO: use omega or don't define it.
        return mse  # + omegas / self.nobs

    @copy_doc(irf.IRAnalysis.irf_errband_mc.__doc__)
    def irf_errband_mc(self, orth=False, repl=1000, T=10, signif=0.05,
                       seed=None, burn=100, cum=False):  # pragma: no cover
        # Monte Carlo irf standard errors
        # Upstream this is implemented directly in VARResults but used only
        # in IRAnalysis
        return self.irf().irf_errband_mc(orth=orth, repl=repl, T=T,
                                         seed=seed, burn=burn, cum=cum)

    @copy_doc(irf.IRAnalysis.irf_resim.__doc__)
    def irf_resim(self, orth=False, repl=1000, T=10,
                  seed=None, burn=100, cum=False):  # pragma: no cover
        # Upstream this is implemented directly in VARResults but used only
        # in IRAnalysis
        return self.irf().irf_resim(orth=orth, repl=repl, T=T,
                                    seed=seed, burn=burn, cum=cum)

    def _omega_forc_cov(self, steps):  # pragma: no cover
        # Approximate MSE matrix \Omega(h) as defined in Lut p97
        raise NotImplementedError("_bmat_forc_cov not ported from upstream, "
                                  "as it is neither used nor tested. "
                                  "See GH#4433")

    def _bmat_forc_cov(self):  # pragma: no cover
        # B as defined on p. 96 of Lut
        raise NotImplementedError("_bmat_forc_cov not ported from upstream, "
                                  "as it is neither used nor tested. "
                                  "See GH#4433")

    def reorder(self, order):
        """Reorder variables for structural specification"""
        if len(order) != len(self.params[0, :]):  # pragma: no cover
            raise ValueError("Reorder specification length should match "
                             "number of endogenous variables")
        # This converts order to list of integers if given as strings
        if isinstance(order[0], string_types):
            order_new = []
            for i, nam in enumerate(order):
                order_new.append(self.names.index(order[i]))
            order = order_new
        return _reordered(self, order)

    @cached_value
    def info_criteria(self):
        "information criteria for lagorder selection"
        nobs = self.nobs
        neqs = self.neqs
        lag_order = self.k_ar
        free_params = lag_order * neqs ** 2 + neqs * self.k_trend

        ld = logdet_symm(self.sigma_u_mle)

        # See Lutkepohl pp. 146-150

        aic = ld + (2. / nobs) * free_params
        bic = ld + (np.log(nobs) / nobs) * free_params
        hqic = ld + (2. * np.log(np.log(nobs)) / nobs) * free_params
        fpe = ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)

        return {'aic': aic,
                'bic': bic,
                'hqic': hqic,
                'fpe': fpe}

    @property
    def aic(self):
        """Akaike information criterion"""
        return self.info_criteria['aic']

    @property
    def fpe(self):
        """Final Prediction Error (FPE)

        Lütkepohl p. 147, see info_criteria
        """
        return self.info_criteria['fpe']

    @property
    def hqic(self):
        """Hannan-Quinn criterion"""
        return self.info_criteria['hqic']

    @property
    def bic(self):
        """Bayesian a.k.a. Schwarz info criterion"""
        return self.info_criteria['bic']

    # -----------------------------------------------------------------
    # Summary, Plotting, IRF, FEVD, ... methods

    def summary(self):
        """Compute console output summary of estimates

        Returns
        -------
        summary : VARSummary
        """
        return output.VARSummary(self)

    def irf(self, periods=10, var_decomp=None, var_order=None):
        """Analyze impulse responses to shocks in system

        Parameters
        ----------
        periods : int
        var_decomp : ndarray (k x k), lower triangular
            Must satisfy Omega = P P', where P is the passed matrix.
            Defaults to Cholesky decomposition of Omega
        var_order : sequence
            Alternate variable order for Cholesky decomposition

        Returns
        -------
        irf : IRAnalysis
        """
        if var_order is not None:
            raise NotImplementedError('alternate variable order not '
                                      'implemented (yet)')

        return irf.IRAnalysis(self, P=var_decomp, periods=periods)
        # TODO: Could this go higher up in the inheritance hierarchy?

    def fevd(self, periods=10, var_decomp=None):
        """
        Compute forecast error variance decomposition ("fevd")

        Returns
        -------
        fevd : FEVD instance
        """
        return FEVD(self, P=var_decomp, periods=periods)

    def plot_forecast(self, steps, alpha=0.05, plot_stderr=True):
        mid, lower, upper = self.forecast_interval(self.endog[-self.k_ar:],
                                                   steps,
                                                   alpha=alpha)
        plotting.plot_var_forc(self.endog, mid, lower, upper, names=self.names,
                               plot_stderr=plot_stderr)

    # -----------------------------------------------------------------
    # VAR Diagnostics: Granger-causality, whiteness of
    # residuals, normality, etc

    # TODO: See if this can be de-duplicated with grangercausalitytests
    def test_causality(self, caused, causing=None, kind='f', signif=0.05):
        """
        Test Granger causality

        Parameters
        ----------
        caused : int or str or sequence of int or str
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-caused by the variable(s) specified
            by `causing`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-caused by the variable(s) specified
            by `causing`.
        causing : int or str or sequence of int or str or None, default: None
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-causing the variable(s) specified by
            `caused`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-causing the variable(s) specified by
            `caused`.
            If None, `causing` is assumed to be the complement of `caused`.
        kind : {'f', 'wald'}
            Perform F-test or Wald (chi-sq) test
        signif : float, default 5%
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level

        Notes
        -----
        Null hypothesis is that there is no Granger-causality for the indicated
        variables. The degrees of freedom in the F-test are based on the
        number of variables in the VAR system, that is, degrees of freedom
        are equal to the number of equations in the VAR times degree of freedom
        of a single equation.

        Test for Granger-causality as described in chapter 7.6.3 of [1]_.
        Test H0: "`causing` does not Granger-cause the remaining variables of
        the system" against  H1: "`causing` is Granger-causal for the
        remaining variables".

        Returns
        -------
        results : CausalityTestResults

        References
        ----------
        .. [1] Lütkepohl, H. 2005.
               *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        if not (0 < signif < 1):  # pragma: no cover
            raise ValueError("signif has to be between 0 and 1")

        caused = _validate_causing(caused, "caused")
        caused = [self.names[c] if type(c) == int else c for c in caused]
        caused_ind = [util.get_index(self.names, c) for c in caused]

        if causing is not None:
            causing = _validate_causing(causing, "causing", allow_none=True)
            causing = [self.names[c] if type(c) == int else c for c in causing]
            causing_ind = [util.get_index(self.names, c) for c in causing]
        else:
            causing_ind = [i for i in range(self.neqs) if i not in caused_ind]
            causing = [self.names[c] for c in caused_ind]

        neqs, k_ar = self.neqs, self.k_ar

        # number of restrictions
        num_restr = len(causing) * len(caused) * k_ar
        num_det_terms = self.k_trend

        # Make restriction matrix
        C = np.zeros((num_restr, neqs * num_det_terms + neqs**2 * k_ar),
                     dtype=float)
        cols_det = neqs * num_det_terms
        row = 0
        for j in range(k_ar):
            for ing_ind in causing_ind:
                for ed_ind in caused_ind:
                    C[row, cols_det + ed_ind + neqs * ing_ind + neqs**2 * j] = 1
                    row += 1

        # Lutkepohl 3.6.5
        Cb = np.dot(C, vec(self.params.T))
        chained = chain_dot(C, self.cov_params, C.T)
        middle = scipy.linalg.inv(chained)

        # wald statistic
        lam_wald = statistic = chain_dot(Cb, middle, Cb)

        if kind.lower() == 'wald':
            df = num_restr
            dist = stats.chi2(df)
        elif kind.lower() == 'f':
            statistic = lam_wald / num_restr
            df = (num_restr, neqs * self.df_resid)
            dist = stats.f(*df)
        else:
            # TODO: this is Exception upstream, fix to ValueError
            raise ValueError('kind %s not recognized' % kind)

        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)
        return CausalityTestResults(causing, caused, statistic,
                                    crit_value, pvalue, df, signif,
                                    test="granger", method=kind)

    def test_inst_causality(self, causing, signif=0.05):
        """
        Test for instantaneous causality

        Parameters
        ----------
        causing :
            If int or str, test whether the corresponding variable is causing
            the variable(s) specified in caused.
            If sequence of int or str, test whether the corresponding variables
            are causing the variable(s) specified in caused.
        signif : float between 0 and 1, default 5 %
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level

        Returns
        -------
        results : dict
            A dict holding the test's results. The dict's keys are:

            "statistic" : float
              The calculated test statistic.

            "crit_value" : float
              The critical value of the Chi^2-distribution.

            "pvalue" : float
              The p-value corresponding to the test statistic.

            "df" : float
              The degrees of freedom of the Chi^2-distribution.

            "conclusion" : str {"reject", "fail to reject"}
              Whether H0 can be rejected or not.

            "signif" : float
              Significance level

        Notes
        -----
        Test for instantaneous causality as described in chapters 3.6.3 and
        7.6.4 of [1]_.
        Test H0: "No instantaneous causality between caused and causing"
        against H1: "Instantaneous causality between caused and causing
        exists".

        Instantaneous causality is a symmetric relation (i.e. if causing is
        "instantaneously causing" caused, then also caused is "instantaneously
        causing" causing), thus the naming of the parameters (which is chosen
        to be in accordance with test_granger_causality()) may be misleading.

        This method is not returning the same result as JMulTi. This is because
        the test is based on a VAR(k_ar) model in sm2 (in accordance to
        pp. 104, 320-321 in [1]_) whereas JMulTi seems to be using a
        VAR(k_ar+1) model.

        References
        ----------
        .. [1] Lütkepohl, H. 2005.
               *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        if not (0 < signif < 1):
            raise ValueError("signif has to be between 0 and 1")

        causing = _validate_causing(causing, "causing")
        causing = [self.names[c] if type(c) == int else c for c in causing]
        causing_ind = [util.get_index(self.names, c) for c in causing]

        caused_ind = [i for i in range(self.neqs) if i not in causing_ind]
        caused = [self.names[c] for c in caused_ind]

        # Note: JMulTi seems to be using k_ar+1 instead of k_ar
        nobs = self.nobs
        neqs = self.neqs

        num_restr = len(causing) * len(caused)  # called N in Lutkepohl

        sigma_u = self.sigma_u
        vech_sigma_u = util.vech(sigma_u)
        sig_mask = np.zeros(sigma_u.shape)
        # set =1 twice to ensure, that all the ones needed are below the main
        # diagonal:
        sig_mask[causing_ind, caused_ind] = 1
        sig_mask[caused_ind, causing_ind] = 1
        vech_sig_mask = util.vech(sig_mask)
        inds = np.nonzero(vech_sig_mask)[0]

        # Make restriction matrix
        C = np.zeros((num_restr, len(vech_sigma_u)), dtype=float)
        for row in range(num_restr):
            C[row, inds[row]] = 1

        Cs = np.dot(C, vech_sigma_u)
        d = np.linalg.pinv(duplication_matrix(neqs))
        Cd = np.dot(C, d)
        chained = chain_dot(Cd, np.kron(sigma_u, sigma_u), Cd.T)
        middle = scipy.linalg.inv(chained) / 2

        wald_statistic = nobs * chain_dot(Cs.T, middle, Cs)
        df = num_restr
        dist = stats.chi2(df)

        pvalue = dist.sf(wald_statistic)
        crit_value = dist.ppf(1 - signif)
        return CausalityTestResults(causing, caused, wald_statistic,
                                    crit_value, pvalue, df, signif,
                                    test="inst", method="wald")

    # TODO: Is this test implemented elsewhere?
    def test_whiteness(self, nlags=10, signif=0.05, adjusted=False):
        """
        Residual whiteness tests using Portmanteau

        Parameters
        ----------
        nlags : int > 0
        signif : float, between 0 and 1
        adjusted : bool, default False

        Returns
        -------
        results : WhitenessTestResults

        Notes
        -----
        Test the whiteness of the residuals using the Portmanteau test as
        described in [1]_, chapter 4.4.3.

        References
        ----------
        .. [1] Lütkepohl, H. 2005.
               *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        statistic = 0
        u = np.asarray(self.resid)
        acov_list = _compute_acov(u, nlags)
        cov0_inv = scipy.linalg.inv(acov_list[0])
        for t in range(1, nlags + 1):
            ct = acov_list[t]
            to_add = np.trace(chain_dot(ct.T, cov0_inv, ct, cov0_inv))
            if adjusted:
                to_add /= (self.nobs - t)
            statistic += to_add
        statistic *= self.nobs**2 if adjusted else self.nobs
        df = self.neqs**2 * (nlags - self.k_ar)
        dist = stats.chi2(df)
        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)

        return WhitenessTestResults(statistic, crit_value, pvalue, df, signif,
                                    nlags, adjusted)

    def test_whiteness_new(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use `test_whiteness` instead.  "
                                  "statsmodels' version of test_whiteness "
                                  "is not an actual hypothesis test.  "
                                  "sm2 gets rid of the older version and "
                                  "retains only the correct version.  "
                                  "See GH#4036 upstream")

    # TODO: Can we use any of the jarque_bera code in stats.stattools?
    def test_normality(self, signif=0.05):
        """
        Test assumption of normal-distributed errors using Jarque-Bera-style
        omnibus Chi^2 test.

        Parameters
        ----------
        signif : float
            Test significance level.

        Returns
        -------
        result : NormalityTestResults

        Notes
        -----
        H0 (null) : data are generated by a Gaussian-distributed process

        References
        ----------
        .. [1] Lütkepohl, H. 2005.*New Introduction to Multiple Time
           Series Analysis*. Springer.

        .. [2] Kilian, L. & Demiroglu, U. (2000). "Residual-Based Tests for
           Normality in Autoregressions: Asymptotic Theory and Simulation
           Evidence." Journal of Business & Economic Statistics
        """
        resid_c = self.resid - self.resid.mean(0)
        sig = np.dot(resid_c.T, resid_c) / self.nobs
        Pinv = np.linalg.inv(np.linalg.cholesky(sig))

        w = np.dot(Pinv, resid_c.T)
        b1 = (w**3).sum(1)[:, None] / self.nobs
        b2 = (w**4).sum(1)[:, None] / self.nobs - 3

        lam_skew = self.nobs * np.dot(b1.T, b1) / 6
        lam_kurt = self.nobs * np.dot(b2.T, b2) / 24

        lam_omni = float(lam_skew + lam_kurt)
        omni_dist = stats.chi2(self.neqs * 2)
        omni_pvalue = float(omni_dist.sf(lam_omni))
        crit_omni = float(omni_dist.ppf(1 - signif))

        return NormalityTestResults(lam_omni, crit_omni, omni_pvalue,
                                    self.neqs * 2, signif)


# TODO: wrapping for endog_lagged?
class VARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'bse': 'columns_eq',
              'cov_params': 'cov',
              'params': 'columns_eq',
              'pvalues': 'columns_eq',
              'tvalues': 'columns_eq',
              'sigma_u': 'cov_eq',
              'sigma_u_mle': 'cov_eq',
              'stderr': 'columns_eq'}
    _wrap_attrs = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_methods, _methods)
    _wrap_methods.pop('cov_params')  # not yet a method in VARResults
wrap.populate_wrapper(VARResultsWrapper, VARResults)  # noqa:E305


class FEVD(object):
    """
    Compute and plot Forecast error variance decomposition and asymptotic
    standard errors
    """
    def __init__(self, model, P=None, periods=None):
        self.periods = periods

        # TODO: Does self.model need to exist?
        # TODO: model is a misnomer; this is a Results object
        self.model = model
        self.neqs = model.neqs
        self.names = model.model.endog_names

        self.irfobj = model.irf(var_decomp=P, periods=periods)
        self.orth_irfs = self.irfobj.orth_irfs

        # cumulative impulse responses
        irfs = (self.orth_irfs[:periods]**2).cumsum(axis=0)

        rng = list(range(self.neqs))
        mse = model.mse(periods)[:, rng, rng]

        # lag x equation x component
        fevd = np.empty_like(irfs)

        for i in range(periods):
            fevd[i] = (irfs[i].T / mse[i]).T

        # switch to equation x lag x component
        self.decomp = fevd.swapaxes(0, 1)

    def summary(self):
        buf = StringIO()

        rng = list(range(self.periods))
        for i in range(self.neqs):
            ppm = output.pprint_matrix(self.decomp[i], rng, self.names)

            buf.write('FEVD for %s\n' % self.names[i])
            buf.write(ppm + '\n')

        print(buf.getvalue())

    def cov(self):
        """Compute asymptotic standard errors"""
        raise NotImplementedError

    def plot(self, periods=None, figsize=(10, 10), **plot_kwds):
        """Plot graphical display of FEVD

        Parameters
        ----------
        periods : int, default None
            Defaults to number originally specified. Can be at most that number
        """
        import matplotlib.pyplot as plt

        k = self.neqs
        periods = periods or self.periods

        fig, axes = plt.subplots(nrows=k, figsize=figsize)

        fig.suptitle('Forecast error variance decomposition (FEVD)')

        colors = [str(c) for c in np.arange(k, dtype=float) / k]
        ticks = np.arange(periods)

        limits = self.decomp.cumsum(2)

        for i in range(k):
            ax = axes[i]

            this_limits = limits[i].T

            handles = []

            for j in range(k):
                lower = this_limits[j - 1] if j > 0 else 0
                upper = this_limits[j]
                handle = ax.bar(ticks, upper - lower, bottom=lower,
                                color=colors[j], label=self.names[j],
                                **plot_kwds)

                handles.append(handle)

            ax.set_title(self.names[i])

        # just use the last axis to get handles for plotting
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plotting.adjust_subplots(right=0.85)
