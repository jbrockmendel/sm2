"""
This module implements standard regression models:

Generalized Least Squares (GLS)
Ordinary Least Squares (OLS)
Weighted Least Squares (WLS)
Generalized Least Squares with autoregressive error terms GLSAR(p)

Models are specified with an endogenous response variable and an
exogenous design matrix and are fit using their `fit` method.

Subclasses that have more complicated covariance matrices
should write over the 'whiten' method as the fit method
prewhitens the response by calling 'whiten'.

General reference for regression models:

D. C. Montgomery and E.A. Peck. "Introduction to Linear Regression
    Analysis." 2nd. Ed., Wiley, 1992.

Econometrics references for regression models:

R. Davidson and J.G. MacKinnon.  "Econometric Theory and Methods," Oxford,
    2004.

W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.
"""
from __future__ import print_function
# TODO: Determine which tests are valid for GLSAR, and under what conditions
# TODO: Fix issue with constant and GLS
# TODO: GLS: add options Iterative GLS, for iterative fgls if sigma is None
# TODO: GLS: default if sigma is none should be two-step GLS
# TODO: Check nesting when performing model based tests, lr, wald, lm
import warnings

from six.moves import range

import numpy as np
from scipy import stats

from sm2.tools.sm_exceptions import InvalidTestWarning, MissingDataError
from sm2.tools.tools import chain_dot, pinv_extended
from sm2.tools.decorators import (resettable_cache,
                                  cache_readonly, cached_data, cached_value,
                                  cache_writable, copy_doc)
import sm2.stats.sandwich_covariance as sw

import sm2.base.model as base
import sm2.base.wrapper as wrap
from sm2.base import covtype

# upstream yule_walker is implemented directly in linear_model
from sm2.tsa.autocov import yule_walker

# need import in module instead of lazily to copy `__doc__`
from ._prediction import PredictionResults
from . import _prediction as pred

__docformat__ = 'restructuredtext en'

__all__ = ['GLS', 'WLS', 'OLS', 'GLSAR', 'PredictionResults']


_fit_regularized_doc = r"""
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : string
            Only the 'elastic_net' approach is currently implemented.
        alpha : scalar or array-like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        L1_wt: scalar
            The fraction of the penalty given to the L1 penalty term.
            Must be between 0 and 1 (inclusive).  If 0, the fit is a
            ridge fit, if 1 it is a lasso fit.
        start_params : array-like
            Starting values for ``params``.
        profile_scale : bool
            If True the penalized fit is computed using the profile
            (concentrated) log-likelihood for the Gaussian model.
            Otherwise the fit uses the residual sum of squares.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        distributed : bool
            If True, the model uses distributed methods for fitting,
            will raise an error if True and partitions is None.
        generator : function
            generator used to partition the model, allows for handling
            of out of memory/parallel computing.
        partitions : scalar
            The number of partitions desired for the distributed
            estimation.
        threshold : scalar or array-like
            The threshold below which coefficients are zeroed out,
            only used for distributed estimation

        Returns
        -------
        A RegularizedResults instance.

        Notes
        -----
        The elastic net approach closely follows that implemented in
        the glmnet package in R.  The penalty is a combination of L1
        and L2 penalties.

        The function that is minimized is:

        .. math::

            0.5*RSS/n + alpha*((1-L1\_wt)*|params|_2^2/2 + L1\_wt*|params|_1)

        where RSS is the usual regression sum of squares, n is the
        sample size, and :math:`|*|_1` and :math:`|*|_2` are the L1 and L2
        norms.

        For WLS and GLS, the RSS is calculated using the whitened endog and
        exog data.

        Post-estimation results are based on the same data used to
        select variables, hence may be subject to overfitting biases.

        The elastic_net method uses the following keyword arguments:

        maxiter : int
            Maximum number of iterations
        cnvrg_tol : float
            Convergence threshold for line searches
        zero_tol : float
            Coefficients below this threshold are treated as zero.

        References
        ----------
        Friedman, Hastie, Tibshirani (2008).  Regularization paths for
        generalized linear models via coordinate descent.  Journal of
        Statistical Software 33(1), 1-22 Feb 2010.
        """


def _get_sigma(sigma, nobs):
    """
    Returns sigma (matrix, nobs by nobs) for GLS and the inverse of its
    Cholesky decomposition.  Handles dimensions and checks integrity.
    If sigma is None, returns None, None. Otherwise returns sigma,
    cholsigmainv.
    """
    if sigma is None:
        return None, None
    sigma = np.asarray(sigma).squeeze()
    if sigma.ndim == 0:
        sigma = np.repeat(sigma, nobs)
    if sigma.ndim == 1:
        if sigma.shape != (nobs,):
            raise ValueError("Sigma must be a scalar, 1d of length %s or a 2d "
                             "array of shape %s x %s" % (nobs, nobs, nobs))
        cholsigmainv = 1 / np.sqrt(sigma)
    else:
        if sigma.shape != (nobs, nobs):
            raise ValueError("Sigma must be a scalar, 1d of length %s or a 2d "
                             "array of shape %s x %s" % (nobs, nobs, nobs))
        cholsigmainv = np.linalg.cholesky(np.linalg.inv(sigma)).T
    return sigma, cholsigmainv


class RegressionModel(base.LikelihoodModel):
    """
    Base class for linear regression models. Should not be directly called.

    Intended for subclassing.
    """
    @property
    def _res_classes(self):
        from sm2.base.elastic_net import (
            RegularizedResults, RegularizedResultsWrapper)

        return {"fit": (RegressionResults, RegressionResultsWrapper),
                "fit_regularized": (RegularizedResults,
                                    RegularizedResultsWrapper)}

    @cached_value
    def nobs(self):
        return float(self.wexog.shape[0])

    def __init__(self, endog, exog, **kwargs):
        self._df_model = None
        self._df_resid = None
        self.rank = None

        super(RegressionModel, self).__init__(endog, exog, **kwargs)
        self._data_attr.extend(['pinv_wexog', 'wendog', 'wexog', 'weights'])
        # TODO: can we get rid of _data_attr?

        self._check_finite(list(kwargs.keys()))

    # TODO: merge this into __init__
    def initialize(self):
        # Note: in GLSAR these get re-evaluated repeatedly,
        # makes merging this into __init__ troublesome
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)

    def _check_finite(self, additional_vars):
        # GH#4969
        checks = ['endog', 'exog'] + additional_vars
        msg = '{0} contains Inf/NaN values. Model cannot be estimated.'
        for check in checks:
            if not hasattr(self, check):
                continue
            val = getattr(self, check)
            if isinstance(val, np.ndarray) and not np.all(np.isfinite(val)):
                raise MissingDataError(msg.format(check))

    @property
    def df_model(self):
        """
        The model degree of freedom, defined as the rank of the regressor
        matrix minus 1 if a constant is included.
        """
        if self._df_model is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_model = float(self.rank - self.k_constant)
        return self._df_model

    @df_model.setter
    def df_model(self, value):
        self._df_model = value

    @property
    def df_resid(self):
        """
        The residual degree of freedom, defined as the number of observations
        minus the rank of the regressor matrix.
        """
        if self._df_resid is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_resid = self.nobs - self.rank
        return self._df_resid

    @df_resid.setter
    def df_resid(self, value):
        self._df_resid = value

    def whiten(self, X):  # pragma: no cover
        raise NotImplementedError("Subclasses should implement.")

    def fit(self, method="pinv", cov_type='nonrobust', cov_kwds=None,
            use_t=None, **kwargs):
        """
        Full fit of the model.

        The results include an estimate of covariance matrix, (whitened)
        residuals and an estimate of scale.

        Parameters
        ----------
        method : str, optional
            Can be "pinv", "qr".  "pinv" uses the Moore-Penrose pseudoinverse
            to solve the least squares problem. "qr" uses the QR
            factorization.
        cov_type : str, optional
            See `regression.linear_model.RegressionResults` for a description
            of the available covariance estimators
        cov_kwds : list or None, optional
            See `linear_model.RegressionResults.get_robustcov_results` for a
            description required keywords for alternative covariance estimators
        use_t : bool, optional
            Flag indicating to use the Student's t distribution when computing
            p-values.  Default behavior depends on cov_type. See
            `linear_model.RegressionResults.get_robustcov_results` for
            implementation details.

        Returns
        -------
        A RegressionResults class instance.

        See Also
        ---------
        regression.linear_model.RegressionResults
        regression.linear_model.RegressionResults.get_robustcov_results

        Notes
        -----
        The fit method uses the pseudoinverse of the design/exogenous variables
        to solve the least squares minimization.
        """
        if method == "pinv":
            if not (hasattr(self, 'pinv_wexog') and
                    hasattr(self, 'normalized_cov_params') and
                    hasattr(self, 'rank')):

                self.pinv_wexog, singular_values = pinv_extended(self.wexog)
                self.normalized_cov_params = np.dot(self.pinv_wexog,
                                                    self.pinv_wexog.T)
                # TODO: above is similar to RLM._initialize and GLM.initialize
                # Cache these singular values for use later.
                self.wexog_singular_values = singular_values
                self.rank = np.linalg.matrix_rank(np.diag(singular_values))

            beta = np.dot(self.pinv_wexog, self.wendog)

        elif method == "qr":
            if not (hasattr(self, 'exog_Q') and
                    hasattr(self, 'exog_R') and
                    hasattr(self, 'normalized_cov_params') and
                    hasattr(self, 'rank')):
                Q, R = np.linalg.qr(self.wexog)
                self.exog_Q, self.exog_R = Q, R
                self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))

                # Cache singular values from R.
                self.wexog_singular_values = np.linalg.svd(R, 0, 0)
                self.rank = np.linalg.matrix_rank(R)
            else:
                Q, R = self.exog_Q, self.exog_R

            # used in ANOVA
            self.effects = effects = np.dot(Q.T, self.wendog)
            beta = np.linalg.solve(R, effects)
        else:
            raise ValueError('`method` has to be "pinv" or "qr"')

        # TODO: Is this necessary given the existing properties?
        if self._df_model is None:
            self._df_model = float(self.rank - self.k_constant)
        if self._df_resid is None:
            self.df_resid = self.nobs - self.rank

        res_cls, wrap_cls = self._res_classes["fit"]
        if isinstance(self, OLS):
            lfit = res_cls(
                self, beta,
                normalized_cov_params=self.normalized_cov_params,
                cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
            # TODO: Why aren't we passing **kwargs here like we do in the
            # other case?  If we can pass them anyway, we can get rid of this
            # if/else block
        else:
            lfit = res_cls(
                self, beta,
                normalized_cov_params=self.normalized_cov_params,
                cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
                **kwargs)
        return wrap_cls(lfit)

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array-like
            Parameters of a linear model
        exog : array-like, optional.
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model has not yet been fit, params is not optional.
        """
        # JP: this doesn't look correct for GLMAR
        # SS: it needs its own predict method
        if exog is None:
            exog = self.exog

        return np.dot(exog, params)

    def get_distribution(self, params, scale, exog=None, dist_class=None):
        raise NotImplementedError("get_distribution is not ported from "
                                  "upstream, since it is neither used nor "
                                  "tested there.")  # pragma: no cover


class GLS(RegressionModel):
    __doc__ = r"""
    Generalized least squares model with a general covariance structure.

    %(params)s
    sigma : scalar or array
        `sigma` is the weighting matrix of the covariance.
        The default is None for no scaling.  If `sigma` is a scalar, it is
        assumed that `sigma` is an n x n diagonal matrix with the given
        scalar, `sigma` as the value of each diagonal element.  If `sigma`
        is an n-length vector, then `sigma` is assumed to be a diagonal
        matrix with the given `sigma` on the diagonal.  This should be the
        same as WLS.
    %(extra_params)s

    **Attributes**

    pinv_wexog : array
        `pinv_wexog` is the p x n Moore-Penrose pseudoinverse of `wexog`.
    cholsimgainv : array
        The transpose of the Cholesky decomposition of the pseudoinverse.
    df_model : float
        p - 1, where p is the number of regressors including the intercept.
        of freedom.
    df_resid : float
        Number of observations n less the number of parameters p.
    llf : float
        The value of the likelihood function of the fitted model.
    nobs : float
        The number of observations n.
    normalized_cov_params : array
        p x p array :math:`(X^{T}\Sigma^{-1}X)^{-1}`
    results : RegressionResults instance
        A property that returns the RegressionResults class if fit.
    sigma : array
        `sigma` is the n x n covariance structure of the error terms.
    wexog : array
        Design matrix whitened by `cholsigmainv`
    wendog : array
        Response variable whitened by `cholsigmainv`

    Notes
    -----
    If sigma is a function of the data making one of the regressors
    a constant, then the current postestimation statistics will not be correct.

    Examples
    --------
    >>> import numpy as np
    >>> import sm2.api as sm
    >>> data = sm.datasets.longley.load(as_pandas=False)
    >>> data.exog = sm.add_constant(data.exog)
    >>> ols_resid = sm.OLS(data.endog, data.exog).fit().resid
    >>> res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()
    >>> rho = res_fit.params

    `rho` is a consistent estimator of the correlation of the residuals from
    an OLS fit of the longley data.  It is assumed that this is the true rho
    of the AR process data.

    >>> from scipy.linalg import toeplitz
    >>> order = toeplitz(np.arange(16))
    >>> sigma = rho**order

    `sigma` is an n x n matrix of the autocorrelation structure of the
    data.

    >>> gls_model = sm.GLS(data.endog, data.exog, sigma=sigma)
    >>> gls_results = gls_model.fit()
    >>> print(gls_results.summary())

    """ % {'params': base._model_params_doc,
           'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog, sigma=None, missing='none', hasconst=None,
                 **kwargs):
        # TODO: add options igls, for iterative fgls if sigma is None
        # TODO: default if sigma is none should be two-step GLS
        sigma, cholsigmainv = _get_sigma(sigma, len(endog))

        super(GLS, self).__init__(endog, exog, missing=missing,
                                  hasconst=hasconst, sigma=sigma,
                                  cholsigmainv=cholsigmainv, **kwargs)

        # store attribute names for data arrays
        self._data_attr.extend(['sigma', 'cholsigmainv'])

    def whiten(self, X):
        """
        GLS whiten method.

        Parameters
        -----------
        X : array-like
            Data to be whitened.

        Returns
        -------
        np.dot(cholsigmainv, X)

        See Also
        --------
        regression.GLS
        """
        X = np.asarray(X)
        if self.sigma is None or self.sigma.shape == ():
            return X
        elif self.sigma.ndim == 1:
            if X.ndim == 1:
                return X * self.cholsigmainv
            else:
                return X * self.cholsigmainv[:, None]
        else:
            return np.dot(self.cholsigmainv, X)

    def loglike(self, params):
        r"""
        Returns the value of the Gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `endog`.

        Parameters
        ----------
        params : array-like
            The parameter estimates

        Returns
        -------
        loglike : float
            The value of the log-likelihood function for a GLS Model.

        Notes
        -----
        The log-likelihood function for the normal distribution is

        .. math:: -\frac{n}{2}\log\left((Y-\hat{Y})^{\prime}(Y-\hat{Y})\right)
            -\frac{n}{2}\left(1+\log\left(\frac{2\pi}{n}\right)\right)
            -\frac{1}{2}\log\left(\left|\Sigma\right|\right)

        Y and Y-hat are whitened.
        """
        # TODO: combine this with OLS/WLS loglike and add _det_sigma argument
        nobs2 = self.nobs / 2.0
        wresid = self.wendog - np.dot(self.wexog, params)
        SSR = np.sum(wresid**2, axis=0)
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2  # with likelihood constant
        if np.any(self.sigma):
            # FIXME: robust-enough check? unneeded if _det_sigma gets defined
            if self.sigma.ndim == 2:
                det = np.linalg.slogdet(self.sigma)
                llf -= .5 * det[1]
            else:
                llf -= 0.5 * np.sum(np.log(self.sigma))
            # with error covariance matrix
        return llf

    @copy_doc(base.LikelihoodModel.hessian_factor.__doc__)
    def hessian_factor(self, params, scale=None, observed=True):
        if self.sigma is None or self.sigma.shape == ():
            return np.ones(self.exog.shape[0])
        elif self.sigma.ndim == 1:
            return self.cholsigmainv
        else:
            return np.diag(self.cholsigmainv)

    @copy_doc(_fit_regularized_doc)
    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):

        # Need to adjust since RSS/n term in elastic net uses nominal
        # n in denominator
        if self.sigma is not None:
            alpha = alpha * np.sum(1 / np.diag(self.sigma)) / len(self.endog)
            # TODO: Make this into _set_alpha

        rslt = OLS(self.wendog, self.wexog).fit_regularized(
            method=method, alpha=alpha,
            L1_wt=L1_wt,
            start_params=start_params,
            profile_scale=profile_scale,
            refit=refit, **kwargs)

        res_cls, wrap_cls = self._res_classes["fit_regularized"]
        rrslt = res_cls(self, rslt.params)
        return wrap_cls(rrslt)


class WLS(RegressionModel):
    __doc__ = """
    A regression model with diagonal but non-identity covariance structure.

    The weights are presumed to be (proportional to) the inverse of
    the variance of the observations.  That is, if the variables are
    to be transformed by 1/sqrt(W) you must supply weights = 1/W.

    %(params)s
    weights : array-like, optional
        1d array of weights.  If you supply 1/W then the variables are
        pre- multiplied by 1/sqrt(W).  If no weights are supplied the
        default value is 1 and WLS results are the same as OLS.
    %(extra_params)s

    Attributes
    ----------
    weights : array
        The stored weights supplied as an argument.

    See regression.GLS

    Examples
    ---------
    >>> import numpy as np
    >>> import sm2.api as sm
    >>> Y = [1,3,4,5,2,3,4]
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>> wls_model = sm.WLS(Y,X, weights=list(range(1,8)))
    >>> results = wls_model.fit()
    >>> results.params
    array([ 2.91666667,  0.0952381 ])
    >>> results.tvalues
    array([ 2.0652652 ,  0.35684428])
    >>> print(results.t_test([1, 0]))
    <T test: effect=array([ 2.91666667]), sd=array([[ 1.41224801]]), t=array([[ 2.0652652]]), p=array([[ 0.04690139]]), df_denom=5>
    >>> print(results.f_test([0, 1]))
    <F test: F=array([[ 0.12733784]]), p=[[ 0.73577409]], df_denom=5, df_num=1>

    Notes
    -----
    If the weights are a function of the data, then the post estimation
    statistics such as fvalue and mse_model might not be correct, as the
    package does not yet support no-constant regression.
    """ % {'params': base._model_params_doc,  # noqa:E501
           'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog, weights=1., missing='none', hasconst=None,
                 **kwargs):
        weights = np.array(weights)
        if weights.shape == ():
            if (missing == 'drop' and 'missing_idx' in kwargs and
                    kwargs['missing_idx'] is not None):
                # patsy may have truncated endog
                weights = np.repeat(weights, len(kwargs['missing_idx']))
            else:
                weights = np.repeat(weights, len(endog))
        # handle case that endog might be of len == 1
        if len(weights) == 1:
            weights = np.array([weights.squeeze()])
        else:
            weights = weights.squeeze()
        super(WLS, self).__init__(endog, exog, missing=missing,
                                  weights=weights, hasconst=hasconst, **kwargs)
        nobs = self.exog.shape[0]
        weights = self.weights
        # Experimental normalization of weights
        weights = weights / np.sum(weights) * nobs
        if weights.size != nobs and weights.shape[0] != nobs:
            raise ValueError('Weights must be scalar or same length as design')

    def whiten(self, X):
        """
        Whitener for WLS model, multiplies each column by sqrt(self.weights)

        Parameters
        ----------
        X : array-like
            Data to be whitened

        Returns
        -------
        whitened : array-like
            sqrt(weights)*X
        """
        X = np.asarray(X)
        if X.ndim == 1:
            return X * np.sqrt(self.weights)
        elif X.ndim == 2:
            return np.sqrt(self.weights)[:, None] * X

    def loglike(self, params):
        r"""
        Returns the value of the gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `Y`.

        Parameters
        ----------
        params : array-like
            The parameter estimates.

        Returns
        -------
        llf : float
            The value of the log-likelihood function for a WLS Model.

        Notes
        --------
        .. math:: -\frac{n}{2}\log\left(Y-\hat{Y}\right)
            -\frac{n}{2}\left(1+\log\left(\frac{2\pi}{n}\right)\right)
            -\frac{1}{2}log\left(\left|W\right|\right)

        where :math:`W` is a diagonal matrix
        """
        nobs2 = self.nobs / 2.0
        wresid = self.wendog - np.dot(self.wexog, params)
        SSR = np.sum(wresid**2, axis=0)
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2  # with constant
        llf += 0.5 * np.sum(np.log(self.weights))
        return llf

    @copy_doc(GLS.hessian_factor.__doc__)
    def hessian_factor(self, params, scale=None, observed=True):
        return self.weights

    @copy_doc(_fit_regularized_doc)
    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):
        # Need to adjust since RSS/n in elastic net uses nominal n in
        # denominator
        alpha = alpha * np.sum(self.weights) / len(self.weights)
        # TODO: make this into `_set_alpha`?

        rslt = OLS(self.wendog, self.wexog).fit_regularized(
            method=method, alpha=alpha,
            L1_wt=L1_wt,
            start_params=start_params,
            profile_scale=profile_scale,
            refit=refit, **kwargs)

        res_cls, wrap_cls = self._res_classes["fit_regularized"]
        rrslt = res_cls(self, rslt.params)
        return wrap_cls(rrslt)


class OLS(WLS):
    __doc__ = """
    A simple ordinary least squares model.

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    weights : scalar
        Has an attribute weights = array(1.0) due to inheritance from WLS.

    See Also
    --------
    GLS

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> import sm2.api as sm
    >>>
    >>> Y = [1,3,4,5,2,3,4]
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>>
    >>> model = sm.OLS(Y,X)
    >>> results = model.fit()
    >>> results.params
    array([ 2.14285714,  0.25      ])
    >>> results.tvalues
    array([ 1.87867287,  0.98019606])
    >>> print(results.t_test([1, 0]))
    <T test: effect=array([ 2.14285714]), sd=array([[ 1.14062282]]), t=array([[ 1.87867287]]), p=array([[ 0.05953974]]), df_denom=5>
    >>> print(results.f_test(np.identity(2)))
    <F test: F=array([[ 19.46078431]]), p=[[ 0.00437251]], df_denom=5, df_num=2>

    Notes
    -----
    No constant is added by the model unless you are using formulas.
    """ % {'params': base._model_params_doc,  # noqa:E501
           'extra_params': base._missing_param_doc + base._extra_param_doc}

    @property
    def _res_classes(self):
        cls_dict = RegressionModel._res_classes.fget(self)
        cls_dict["fit"] = (OLSResults, RegressionResultsWrapper)
        return cls_dict

    # TODO: change example to use datasets.  This was the point of datasets!
    def __init__(self, endog, exog=None, missing='none', hasconst=None,
                 **kwargs):
        super(OLS, self).__init__(endog, exog, missing=missing,
                                  hasconst=hasconst, **kwargs)
        if "weights" in self._init_keys:
            self._init_keys.remove("weights")

    def loglike(self, params, scale=None):
        """
        The likelihood function for the OLS model.

        Parameters
        ----------
        params : array-like
            The coefficients with which to estimate the log-likelihood.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        The likelihood function evaluated at params.
        """
        nobs = float(self.nobs)
        resid = self.endog - np.dot(self.exog, params)
        if hasattr(self, 'offset'):
            resid -= self.offset
        ssr = np.sum(resid**2)
        if scale is None:
            # profile log likelihood
            llf = -.5 * nobs * (np.log(2 * np.pi) + np.log(ssr / nobs) + 1)
        else:
            # log-likelihood
            llf = -.5 * (nobs * np.log(2 * np.pi * scale) + ssr / scale)
        return llf

    def whiten(self, Y):
        """
        OLS model whitener does nothing: returns Y.
        """
        return Y

    def score(self, params, scale=None):
        """
        Evaluate the score function at a given point.

        The score corresponds to the profile (concentrated)
        log-likelihood in which the scale parameter has been profiled
        out.

        Parameters
        ----------
        params : array-like
            The parameter vector at which the score function is
            computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        The score vector.
        """
        self._setup_score_hess()

        xtxb = np.dot(self._wexog_xprod, params)
        sdr = -self._wexog_x_wendog + xtxb

        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T,
                                                  params)
            ssr += np.dot(params, xtxb)
            # scale-esque
            scale = ssr / self.nobs
        return -sdr / scale

    def _setup_score_hess(self):
        if hasattr(self, "_wexog_xprod"):
            # TODO: Don't do this with unpredictable attributes
            return

        y = self.wendog
        if hasattr(self, 'offset'):
            y = y - self.offset
        self._wendog_xprod = np.sum(y * y)
        self._wexog_xprod = np.dot(self.wexog.T, self.wexog)
        self._wexog_x_wendog = np.dot(self.wexog.T, y)

    def hessian(self, params, scale=None):
        """
        Evaluate the Hessian function at a given point.

        Parameters
        ----------
        params : array-like
            The parameter vector at which the Hessian is computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        The Hessian matrix.
        """
        self._setup_score_hess()

        xtxb = np.dot(self._wexog_xprod, params)

        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T,
                                                  params)
            ssr += np.dot(params, xtxb)
            ssrp = -2 * self._wexog_x_wendog + 2 * xtxb
            hm = self._wexog_xprod / ssr - np.outer(ssrp, ssrp) / ssr**2
            return -self.nobs * hm / 2
        else:
            return -self._wexog_xprod / scale

    @copy_doc(GLS.hessian_factor.__doc__)
    def hessian_factor(self, params, scale=None, observed=True):
        return np.ones(self.exog.shape[0])

    @copy_doc(_fit_regularized_doc)
    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):

        from sm2.base.elastic_net import fit_elasticnet

        if L1_wt == 0:
            return self._fit_ridge(alpha)

        # In the future we could add support for other penalties, e.g. SCAD.
        if method != "elastic_net":  # pragma: no cover
            raise ValueError("method for fit_regularized must be elastic_net")

        # Set default parameters.
        defaults = {"maxiter": 50, "cnvrg_tol": 1e-10,
                    "zero_tol": 1e-10}
        defaults.update(kwargs)

        # If a scale parameter is passed in, the non-profile
        # likelihood (residual sum of squares divided by -2) is used,
        # otherwise the profile likelihood is used.
        if profile_scale:
            loglike_kwds = {}
            score_kwds = {}
            hess_kwds = {}
        else:
            loglike_kwds = {"scale": 1}
            score_kwds = {"scale": 1}
            hess_kwds = {"scale": 1}

        return fit_elasticnet(self, method=method,
                              alpha=alpha,
                              L1_wt=L1_wt,
                              start_params=start_params,
                              loglike_kwds=loglike_kwds,
                              score_kwds=score_kwds,
                              hess_kwds=hess_kwds,
                              refit=refit,
                              check_step=False,
                              **defaults)

    def _fit_ridge(self, alpha):
        """
        Fit a linear model using ridge regression.

        Parameters
        ----------
        alpha : scalar or array-like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.

        Notes
        -----
        Equivalent to fit_regularized with L1_wt = 0 (but implemented
        more efficiently).
        """
        u, s, vt = np.linalg.svd(self.exog, 0)
        v = vt.T
        q = np.dot(u.T, self.endog) * s
        s2 = s * s
        if np.isscalar(alpha):
            sd = s2 + alpha * self.nobs
            params = q / sd
            params = np.dot(v, params)
        else:
            vtav = self.nobs * np.dot(vt, alpha[:, None] * v)
            d = np.diag(vtav) + s2
            np.fill_diagonal(vtav, d)
            r = np.linalg.solve(vtav, q)
            params = np.dot(v, r)

        from sm2.base.elastic_net import RegularizedResults
        return RegularizedResults(self, params)


class GLSAR(GLS):
    __doc__ = """
    A regression model with an AR(p) covariance structure.

    %(params)s
    rho : int
        Order of the autoregressive covariance
    %(extra_params)s

    Examples
    --------
    >>> import sm2.api as sm
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>> Y = [1,3,4,5,8,10,9]
    >>> model = sm.GLSAR(Y, X, rho=2)
    >>> for i in range(6):
    ...     results = model.fit()
    ...     print("AR coefficients: {0}".format(model.rho))
    ...     rho, sigma = sm.regression.yule_walker(results.resid,
    ...                                            order=model.order)
    ...     model = sm.GLSAR(Y, X, rho)
    ...
    AR coefficients: [ 0.  0.]
    AR coefficients: [-0.52571491 -0.84496178]
    AR coefficients: [-0.6104153  -0.86656458]
    AR coefficients: [-0.60439494 -0.857867  ]
    AR coefficients: [-0.6048218  -0.85846157]
    AR coefficients: [-0.60479146 -0.85841922]
    >>> results.params
    array([-0.66661205,  1.60850853])
    >>> results.tvalues
    array([ -2.10304127,  21.8047269 ])
    >>> print(results.t_test([1, 0]))
    <T test: effect=array([-0.66661205]), sd=array([[ 0.31697526]]), t=array([[-2.10304127]]), p=array([[ 0.06309969]]), df_denom=3>
    >>> print(results.f_test(np.identity(2)))
    <F test: F=array([[ 1815.23061844]]), p=[[ 0.00002372]], df_denom=3, df_num=2>

    Or, equivalently

    >>> model2 = sm.GLSAR(Y, X, rho=2)
    >>> res = model2.iterative_fit(maxiter=6)
    >>> model2.rho
    array([-0.60479146, -0.85841922])

    Notes
    -----
    GLSAR is considered to be experimental.
    The linear autoregressive process of order p--AR(p)--is defined as:
    TODO
    """ % {'params': base._model_params_doc,  # noqa:E501
           'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog=None, rho=1, missing='none', **kwargs):
        # this looks strange, interpreting rho as order if it is int
        if isinstance(rho, np.int):
            self.order = rho
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0, 1]:  # pragma: no cover
                raise ValueError("AR parameters must be a scalar or a vector")
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        if exog is None:
            # TODO: JP this looks wrong, should be a regression on constant
            # results for rho estimate now identical to yule-walker on y
            # super(AR, self).__init__(endog, add_constant(endog))
            super(GLSAR, self).__init__(endog, np.ones((endog.shape[0], 1)),
                                        missing=missing, **kwargs)
        else:
            super(GLSAR, self).__init__(endog, exog, missing=missing,
                                        **kwargs)

    def iterative_fit(self, maxiter=3, rtol=1e-4, **kwds):
        """
        Perform an iterative two-stage procedure to estimate a GLS model.

        The model is assumed to have AR(p) errors, AR(p) parameters and
        regression coefficients are estimated iteratively.

        Parameters
        ----------
        maxiter : integer, optional
            the number of iterations
        rtol : float, optional
            Relative tolerance between estimated coefficients to stop the
            estimation.  Stops if

            max(abs(last - current) / abs(last)) < rtol
        """
        # TODO: update this after going through example.
        converged = False
        i = -1  # need to initialize for maxiter < 1 (skip loop)
        history = {'params': [], 'rho': [self.rho]}
        for i in range(maxiter - 1):
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog
                # TODO: What does this accomplish?
            self.initialize()
            results = self.fit()
            history['params'].append(results.params)
            if i == 0:
                last = results.params
            else:
                diff = np.max(np.abs(last - results.params) / np.abs(last))
                if diff < rtol:
                    converged = True
                    break
                last = results.params
            self.rho, _ = yule_walker(results.resid,
                                      order=self.order, df=None)
            history['rho'].append(self.rho)

        # why not another call to self.initialize
        # Use kwarg to insert history
        if not converged and maxiter > 0:
            # maxiter <= 0 just does OLS
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog
            self.initialize()

        # if converged then this is a duplicate fit, because we didn't
        # update rho
        results = self.fit(history=history, **kwds)
        results.iter = i + 1
        # add last fit to history, not if duplicate fit
        if not converged:
            results.history['params'].append(results.params)
            results.iter += 1

        results.converged = converged
        return results

    def whiten(self, X):
        """
        Whiten a series of columns according to an AR(p)
        covariance structure. This drops initial p observations.

        Parameters
        ----------
        X : array-like
            The data to be whitened,

        Returns
        -------
        whitened array
        """
        # TODO: notation for AR process
        X = np.asarray(X, np.float64)
        _X = X.copy()

        # the following loops over the first axis,  works for 1d and nd
        for i in range(self.order):
            _X[(i + 1):] = _X[(i + 1):] - self.rho[i] * X[0:-(i + 1)]
        return _X[self.order:]


class RegressionResults(base.LikelihoodModelResults):
    r"""
    This class summarizes the fit of a linear regression model.

    It handles the output of contrasts, estimates of covariance, etc.

    Returns
    -------
    **Attributes**

    aic
        Akaike's information criteria. For a model with a constant
        :math:`-2llf + 2(df\_model + 1)`. For a model without a constant
        :math:`-2llf + 2(df\_model)`.
    bic
        Bayes' information criteria. For a model with a constant
        :math:`-2llf + \log(n)(df\_model+1)`. For a model without a constant
        :math:`-2llf + \log(n)(df\_model)`
    bse
        The standard errors of the parameter estimates.
    pinv_wexog
        See specific model class docstring
    centered_tss
        The total (weighted) sum of squares centered about the mean.
    cov_HC0
        Heteroscedasticity robust covariance matrix. See HC0_se below.
    cov_HC1
        Heteroscedasticity robust covariance matrix. See HC1_se below.
    cov_HC2
        Heteroscedasticity robust covariance matrix. See HC2_se below.
    cov_HC3
        Heteroscedasticity robust covariance matrix. See HC3_se below.
    cov_type
        Parameter covariance estimator used for standard errors and t-stats
    df_model
        Model degrees of freedom. The number of regressors `p`. Does not
        include the constant if one is present
    df_resid
        Residual degrees of freedom. `n - p - 1`, if a constant is present.
        `n - p` if a constant is not included.
    ess
        Explained sum of squares.  If a constant is present, the centered
        total sum of squares minus the sum of squared residuals. If there is
        no constant, the uncentered total sum of squares is used.
    fvalue
        F-statistic of the fully specified model.  Calculated as the mean
        squared error of the model divided by the mean squared error of the
        residuals.
    f_pvalue
        p-value of the F-statistic
    fittedvalues
        The predicted values for the original (unwhitened) design.
    het_scale
        adjusted squared residuals for heteroscedasticity robust standard
        errors. Is only available after `HC#_se` or `cov_HC#` is called.
        See HC#_se for more information.
    history
        Estimation history for iterative estimators
    HC0_se
        White's (1980) heteroskedasticity robust standard errors.
        Defined as sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1)
        where e_i = resid[i]
        HC0_se is a cached property.
        When HC0_se or cov_HC0 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is just
        resid**2.
    HC1_se
        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as sqrt(diag(n/(n-p)*HC_0)
        HC1_see is a cached property.
        When HC1_se or cov_HC1 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        n/(n-p)*resid**2.
    HC2_se
        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T
        HC2_see is a cached property.
        When HC2_se or cov_HC2 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        resid^(2)/(1-h_ii).
    HC3_se
        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)^(2)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T
        HC3_see is a cached property.
        When HC3_se or cov_HC3 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        resid^(2)/(1-h_ii)^(2).
    model
        A pointer to the model instance that called fit() or results.
    mse_model
        Mean squared error the model. This is the explained sum of
        squares divided by the model degrees of freedom.
    mse_resid
        Mean squared error of the residuals.  The sum of squared
        residuals divided by the residual degrees of freedom.
    mse_total
        Total mean squared error.  Defined as the uncentered total sum
        of squares divided by n the number of observations.
    nobs
        Number of observations n.
    normalized_cov_params
        See specific model class docstring
    params
        The linear coefficients that minimize the least squares
        criterion.  This is usually called Beta for the classical
        linear model.
    pvalues
        The two-tailed p values for the t-stats of the params.
    resid
        The residuals of the model.
    resid_pearson
        `wresid` normalized to have unit variance.
    rsquared
        R-squared of a model with an intercept.  This is defined here
        as 1 - `ssr`/`centered_tss` if the constant is included in the
        model and 1 - `ssr`/`uncentered_tss` if the constant is
        omitted.
    rsquared_adj
        Adjusted R-squared.  This is defined here as 1 -
        (`nobs`-1)/`df_resid` * (1-`rsquared`) if a constant is
        included and 1 - `nobs`/`df_resid` * (1-`rsquared`) if no
        constant is included.
    scale
        A scale factor for the covariance matrix.  Default value is
        ssr/(n-p).  Note that the square root of `scale` is often
        called the standard error of the regression.
    ssr
        Sum of squared (whitened) residuals.
    uncentered_tss
        Uncentered sum of squares.  Sum of the squared values of the
        (whitened) endogenous response variable.
    wresid
        The residuals of the transformed/whitened regressand and
        regressor(s)
    """
    _cache = {}

    @cached_value
    def nobs(self):
        # TODO: make this not-depend on wexog in case data has been removed
        return float(self.model.wexog.shape[0])

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
                 cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        super(RegressionResults, self).__init__(
            model, params, normalized_cov_params, scale)

        self._cache = resettable_cache()
        if hasattr(model, 'wexog_singular_values'):
            self._wexog_singular_values = model.wexog_singular_values
        else:
            self._wexog_singular_values = None

        self.df_model = model.df_model
        self.df_resid = model.df_resid

        if cov_type == 'nonrobust':
            # TODO: roll this in to get_robustcov_results?
            self.cov_type = 'nonrobust'
            self.cov_kwds = {
                'description': 'Standard Errors assume that the '
                               'covariance matrix of the errors is correctly '
                               'specified.'}
            if use_t is None:
                self.use_t = True  # TODO: class default
        else:
            if cov_kwds is None:
                cov_kwds = {}
            if 'use_t' in cov_kwds:
                # TODO: we want to get rid of 'use_t' in cov_kwds
                use_t_2 = cov_kwds.pop('use_t')
                if use_t is None:
                    use_t = use_t_2
                # TODO: warn or not?
            self.get_robustcov_results(cov_type=cov_type, use_self=True,
                                       use_t=use_t, **cov_kwds)
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __str__(self):
        # FIXME: This should _return_ something right?
        self.summary()

    @cached_data
    def wresid(self):
        return self.model.wendog - self.model.predict(self.params,
                                                      self.model.wexog)

    # TODO: fix writable example
    @cache_writable
    def scale(self):
        wresid = self.wresid
        return np.dot(wresid, wresid) / self.df_resid

    @cache_readonly
    def ssr(self):
        wresid = self.wresid
        return np.dot(wresid, wresid)

    @cache_readonly
    def centered_tss(self):
        model = self.model
        weights = getattr(model, 'weights', None)
        sigma = getattr(model, 'sigma', None)

        if weights is not None:
            mean = np.average(model.endog, weights=weights)
            return np.sum(weights * (model.endog - mean)**2)
        elif sigma is not None:
            # Exactly matches WLS when sigma is diagonal
            iota = np.ones_like(model.endog)
            iota = model.whiten(iota)
            mean = model.wendog.dot(iota) / iota.dot(iota)
            err = model.endog - mean
            err = model.whiten(err)
            return np.sum(err**2)
        else:
            centered_endog = model.wendog - model.wendog.mean()
            return np.dot(centered_endog, centered_endog)

    @cache_readonly
    def uncentered_tss(self):
        wendog = self.model.wendog
        return np.dot(wendog, wendog)

    @cache_readonly
    def ess(self):
        if self.k_constant:
            return self.centered_tss - self.ssr
        else:
            return self.uncentered_tss - self.ssr

    @cache_readonly
    def rsquared(self):
        if self.k_constant:
            return 1 - self.ssr / self.centered_tss
        else:
            return 1 - self.ssr / self.uncentered_tss

    @cache_readonly
    def rsquared_adj(self):
        return 1 - (np.divide(self.nobs - self.k_constant, self.df_resid) *
                    (1 - self.rsquared))

    @cache_readonly
    def mse_model(self):
        return self.ess / self.df_model

    @cache_readonly
    def mse_resid(self):
        return self.ssr / self.df_resid

    @cache_readonly
    def mse_total(self):
        if self.k_constant:
            return self.centered_tss / (self.df_resid + self.df_model)
        else:
            return self.uncentered_tss / (self.df_resid + self.df_model)

    @cache_readonly
    def fvalue(self):
        if hasattr(self, 'cov_type') and self.cov_type != 'nonrobust':
            # with heteroscedasticity or correlation robustness
            k_params = self.normalized_cov_params.shape[0]
            mat = np.eye(k_params)
            const_idx = self.model.data.const_idx
            # TODO: What if model includes implicit constant, e.g. all
            #       dummies but no constant regressor?
            # TODO: Restats as LM test by projecting orthogonalizing
            #       to constant?
            if self.model.data.k_constant == 1:
                # if constant is implicit, return nan see GH#2444
                if const_idx is None:
                    return np.nan

                idx = list(range(k_params))
                idx.pop(const_idx)
                mat = mat[idx]  # remove constant
                if mat.size == 0:  # see  GH#3642
                    return np.nan
            ft = self.f_test(mat)
            # using backdoor to set another attribute that we already have
            self._cache['f_pvalue'] = ft.pvalue
            return ft.fvalue
        else:
            # for standard homoscedastic case
            return self.mse_model / self.mse_resid

    @cache_readonly
    def f_pvalue(self):
        return stats.f.sf(self.fvalue, self.df_model, self.df_resid)

    # TODO: default implementation?
    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.cov_params()))

    @cached_value
    def aic(self):
        return -2 * self.llf + 2 * (self.df_model + self.k_constant)

    @cached_value
    def bic(self):
        return (-2 * self.llf + np.log(self.nobs) * (self.df_model +
                                                     self.k_constant))

    @cache_readonly
    def eigenvals(self):
        """
        Return eigenvalues sorted in decreasing order.
        """
        if self._wexog_singular_values is not None:
            eigvals = self._wexog_singular_values ** 2
        else:
            eigvals = np.linalg.linalg.eigvalsh(np.dot(self.model.wexog.T,
                                                       self.model.wexog))
        return np.sort(eigvals)[::-1]

    @cache_readonly
    def condition_number(self):
        """
        Return condition number of exogenous matrix.

        Calculated as ratio of largest to smallest eigenvalue.
        """
        eigvals = self.eigenvals
        return np.sqrt(eigvals[0] / eigvals[-1])

    # TODO: make these properties reset bse
    def _HCCM(self, scale):
        H = np.dot(self.model.pinv_wexog,
                   scale[:, None] * self.model.pinv_wexog.T)
        return H

    @cache_readonly
    def cov_HC0(self):
        """See sm2.RegressionResults"""
        self.het_scale = self.wresid**2
        cov_HC0 = self._HCCM(self.het_scale)
        return cov_HC0

    @cache_readonly
    def cov_HC1(self):  # TODO: I'm not wild about setting het_scale attr
        """See sm2.RegressionResults"""
        self.het_scale = self.nobs / (self.df_resid) * (self.wresid**2)
        cov_HC1 = self._HCCM(self.het_scale)
        return cov_HC1

    @cache_readonly
    def cov_HC2(self):
        """See sm2.RegressionResults"""
        # probably could be optimized
        h = np.diag(chain_dot(self.model.wexog,
                              self.normalized_cov_params,
                              self.model.wexog.T))
        self.het_scale = self.wresid**2 / (1 - h)
        cov_HC2 = self._HCCM(self.het_scale)
        return cov_HC2

    @cache_readonly
    def cov_HC3(self):
        """See sm2.RegressionResults"""
        h = np.diag(chain_dot(self.model.wexog,
                              self.normalized_cov_params,
                              self.model.wexog.T))
        self.het_scale = (self.wresid / (1 - h))**2
        cov_HC3 = self._HCCM(self.het_scale)
        return cov_HC3

    @cache_readonly
    def HC0_se(self):
        """See sm2.RegressionResults"""
        return np.sqrt(np.diag(self.cov_HC0))

    @cache_readonly
    def HC1_se(self):
        """See sm2.RegressionResults"""
        return np.sqrt(np.diag(self.cov_HC1))

    @cache_readonly
    def HC2_se(self):
        """See sm2.RegressionResults"""
        return np.sqrt(np.diag(self.cov_HC2))

    @cache_readonly
    def HC3_se(self):
        """See sm2.RegressionResults"""
        return np.sqrt(np.diag(self.cov_HC3))

    @cached_data
    def resid_pearson(self):
        """
        Residuals, normalized to have unit variance.

        Returns
        -------
        An array wresid standardized by the sqrt if scale
        """
        if not hasattr(self, 'resid'):  # pragma: no cover
            raise ValueError('Method requires residuals.')
        eps = np.finfo(self.wresid.dtype).eps
        if np.sqrt(self.scale) < 10 * eps * self.model.endog.mean():
            # don't divide if scale is zero close to numerical precision
            warnings.warn("All residuals are 0, cannot compute normed "
                          "residuals.", RuntimeWarning)
            return self.wresid
        else:
            return self.wresid / np.sqrt(self.scale)

    def _is_nested(self, restricted):
        """
        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the current
            model. The result instance of the restricted model is required to
            have two attributes, residual sum of squares, `ssr`, residual
            degrees of freedom, `df_resid`.

        Returns
        -------
        nested : bool
            True if nested, otherwise false

        Notes
        -----
        A most nests another model if the regressors in the smaller
        model are spanned by the regressors in the larger model and
        the regressand is identical.
        """
        if self.model.nobs != restricted.model.nobs:
            return False

        full_rank = self.model.rank
        restricted_rank = restricted.model.rank
        if full_rank <= restricted_rank:
            return False

        restricted_exog = restricted.model.wexog
        full_wresid = self.wresid

        scores = restricted_exog * full_wresid[:, None]
        score_l2 = np.sqrt(np.mean(scores.mean(0) ** 2))
        # TODO: Could be improved, and may fail depending on scale of
        # regressors
        return np.allclose(score_l2, 0)

    def compare_lm_test(self, restricted, demean=True, use_lr=False):
        """
        Use Lagrange Multiplier test to test whether restricted
        model is correct

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the
            current model. The result instance of the restricted model
            is required to have two attributes, residual sum of
            squares, `ssr`, residual degrees of freedom, `df_resid`.

        demean : bool
            Flag indicating whether the demean the scores based on the
            residuals from the restricted model.  If True, the
            covariance of the scores are used and the LM test is
            identical to the large sample version of the LR test.

        Returns
        -------
        lm_value : float
            test statistic, chi2 distributed
        p_value : float
            p-value of the test statistic
        df_diff : int
            degrees of freedom of the restriction, i.e. difference in df
            between models

        Notes
        -----
        TODO: explain LM text
        """
        if not self._is_nested(restricted):
            raise ValueError("Restricted model is not nested by full model.")

        wresid = restricted.wresid
        wexog = self.model.wexog
        scores = wexog * wresid[:, None]

        n = self.nobs
        df_full = self.df_resid
        df_restr = restricted.df_resid
        df_diff = (df_restr - df_full)

        s = scores.mean(axis=0)
        if use_lr:
            scores = wexog * self.wresid[:, None]
            demean = False

        if demean:
            scores = scores - scores.mean(0)[None, :]
        # Form matters here.  If homoskedastics can be sigma^2 (X'X)^-1
        # If Heteroskedastic then the form below is fine
        # If HAC then need to use HAC
        # If Cluster, shoudl use cluster

        cov_type = getattr(self, 'cov_type', 'nonrobust')
        if cov_type == 'nonrobust':
            sigma2 = np.mean(wresid**2)
            XpX = np.dot(wexog.T, wexog) / n
            Sinv = np.linalg.inv(sigma2 * XpX)
        elif cov_type in ('HC0', 'HC1', 'HC2', 'HC3'):
            Sinv = np.linalg.inv(np.dot(scores.T, scores) / n)
        elif cov_type == 'HAC':
            maxlags = self.cov_kwds['maxlags']
            Sinv = np.linalg.inv(sw.S_hac_simple(scores, maxlags) / n)
        elif cov_type == 'cluster':
            # cluster robust standard errors
            groups = self.cov_kwds['groups']
            # TODO: Might need demean option in S_crosssection by group?
            Sinv = np.linalg.inv(sw.S_crosssection(scores, groups))
        else:  # pragma: no cover
            raise ValueError('Only nonrobust, HC, HAC and cluster are '
                             'currently connected')

        lm_value = n * chain_dot(s, Sinv, s.T)
        p_value = stats.chi2.sf(lm_value, df_diff)
        return lm_value, p_value, df_diff

    def compare_f_test(self, restricted):
        """use F test to test whether restricted model is correct

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the
            current model. The result instance of the restricted model
            is required to have two attributes, residual sum of
            squares, `ssr`, residual degrees of freedom, `df_resid`.

        Returns
        -------
        f_value : float
            test statistic, F distributed
        p_value : float
            p-value of the test statistic
        df_diff : int
            degrees of freedom of the restriction, i.e. difference in
            df between models

        Notes
        -----
        See mailing list discussion October 17,

        This test compares the residual sum of squares of the two
        models.  This is not a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results under
        the assumption of homoscedasticity and no autocorrelation
        (sphericity).
        """
        has_robust1 = getattr(self, 'cov_type', 'nonrobust') != 'nonrobust'
        has_robust2 = (getattr(restricted, 'cov_type', 'nonrobust') !=
                       'nonrobust')

        if has_robust1 or has_robust2:
            warnings.warn('F test for comparison is likely invalid with '
                          'robust covariance, proceeding anyway',
                          InvalidTestWarning)

        ssr_full = self.ssr
        ssr_restr = restricted.ssr
        df_full = self.df_resid
        df_restr = restricted.df_resid

        df_diff = (df_restr - df_full)
        f_value = (ssr_restr - ssr_full) / df_diff / ssr_full * df_full
        p_value = stats.f.sf(f_value, df_diff, df_full)
        return f_value, p_value, df_diff

    def compare_lr_test(self, restricted, large_sample=False):
        """
        Likelihood ratio test to test whether restricted model is correct

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the current model.
            The result instance of the restricted model is required to have two
            attributes, residual sum of squares, `ssr`, residual degrees of
            freedom, `df_resid`.

        large_sample : bool
            Flag indicating whether to use a heteroskedasticity robust version
            of the LR test, which is a modified LM test.

        Returns
        -------
        lr_stat : float
            likelihood ratio, chisquare distributed with df_diff degrees of
            freedom
        p_value : float
            p-value of the test statistic
        df_diff : int
            degrees of freedom of the restriction, i.e. difference in df
            between models

        Notes
        -----
        The exact likelihood ratio is valid for homoskedastic data,
        and is defined as

        .. math:: D=-2\\log\\left(\\frac{\\mathcal{L}_{null}}
           {\\mathcal{L}_{alternative}}\\right)

        where :math:`\\mathcal{L}` is the likelihood of the
        model. With :math:`D` distributed as chisquare with df equal
        to difference in number of parameters or equivalently
        difference in residual degrees of freedom.

        The large sample version of the likelihood ratio is defined as

        .. math:: D=n s^{\\prime}S^{-1}s

        where :math:`s=n^{-1}\\sum_{i=1}^{n} s_{i}`

        .. math:: s_{i} = x_{i,alternative} \\epsilon_{i,null}

        is the average score of the model evaluated using the
        residuals from null model and the regressors from the
        alternative model and :math:`S` is the covariance of the
        scores, :math:`s_{i}`.  The covariance of the scores is
        estimated using the same estimator as in the alternative
        model.

        This test compares the loglikelihood of the two models.  This
        may not be a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results
        without taking unspecified heteroscedasticity or correlation
        into account.

        This test compares the loglikelihood of the two models.  This
        may not be a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results
        without taking unspecified heteroscedasticity or correlation
        into account.

        is the average score of the model evaluated using the
        residuals from null model and the regressors from the
        alternative model and :math:`S` is the covariance of the
        scores, :math:`s_{i}`.  The covariance of the scores is
        estimated using the same estimator as in the alternative
        model.

        TODO: put into separate function, needs tests
        """
        # See mailing list discussion October 17,
        # TODO: link?  or at least year?
        if large_sample:
            return self.compare_lm_test(restricted, use_lr=True)

        has_robust1 = (getattr(self, 'cov_type', 'nonrobust') != 'nonrobust')
        has_robust2 = (
            getattr(restricted, 'cov_type', 'nonrobust') != 'nonrobust')

        if has_robust1 or has_robust2:
            warnings.warn('Likelihood Ratio test is likely invalid with '
                          'robust covariance, proceeding anyway',
                          InvalidTestWarning)

        llf_full = self.llf
        llf_restr = restricted.llf
        df_full = self.df_resid
        df_restr = restricted.df_resid

        lrdf = (df_restr - df_full)
        lrstat = -2 * (llf_restr - llf_full)
        lr_pvalue = stats.chi2.sf(lrstat, lrdf)

        return lrstat, lr_pvalue, lrdf

    # TODO: De-duplicate with covtype.get_robustcov_results
    @copy_doc(covtype.get_robustcov_results.__doc__)
    def get_robustcov_results(self, cov_type='HC1', use_t=None, **kwds):
        cov_type = covtype.normalize_cov_type(cov_type)

        if 'kernel' in kwds:
            kwds['weights_func'] = kwds.pop('kernel')
        if 'weights_func' in kwds and not callable(kwds['weights_func']):
            kwds['weights_func'] = sw.kernel_dict[kwds['weights_func']]

        # TODO: make separate function that returns a robust cov plus info
        use_self = kwds.pop('use_self', False)
        if use_self:
            res = self
        else:
            res = self.__class__(
                self.model, self.params,
                normalized_cov_params=self.normalized_cov_params,
                scale=self.scale)

        res.cov_type = cov_type
        # use_t might already be defined by the class, and already set
        if use_t is None:
            use_t = self.use_t
        res.cov_kwds = {'use_t': use_t}  # store for information
        res.use_t = use_t

        adjust_df = covtype.set_df_adjustment(kwds, cov_type)
        res.cov_kwds['adjust_df'] = adjust_df

        n_groups = None  # this isn't set upstream

        # verify and set kwds, and calculate cov
        # TODO: this should be outsourced in a function so we can reuse it in
        #       other models
        # TODO: make it DRYer   repeated code for checking kwds
        if cov_type in ['fixed scale', 'fixed_scale']:
            res.cov_kwds['description'] = ('Standard Errors are based on '
                                           'fixed scale')

            res.cov_kwds['scale'] = scale = kwds.get('scale', 1.)
            res.cov_params_default = scale * res.normalized_cov_params
        elif cov_type.upper() in ('HC0', 'HC1', 'HC2', 'HC3'):
            covtype.robust_stderrs(self, res, kwds, cov_type)
        elif cov_type.lower() == 'hac':
            n_groups = covtype.hac_stderrs(self, res, kwds)
        elif cov_type.lower() == 'cluster':
            # cluster robust standard errors, one- or two-way
            n_groups = covtype.clustered_stderrs(self, res, kwds,
                                                 cov_type, adjust_df)
        elif cov_type.lower() == 'hac-panel':
            # cluster robust standard errors
            n_groups = covtype.hac_panel(self, res, kwds, cov_type)
        elif cov_type.lower() == 'hac-groupsum':
            # Driscoll-Kraay standard errors
            n_groups = covtype.hac_groupsum(self, res, kwds,
                                            adjust_df, cov_type)
        else:  # pragma: no cover
            raise ValueError('cov_type not recognized. See docstring for '
                             'available options and spelling')

        if adjust_df:
            # Note: df_resid is used for scale and others, add new attribute
            res.df_resid_inference = n_groups - 1

        return res

    @copy_doc(pred.get_prediction.__doc__)
    def get_prediction(self, exog=None, transform=True, weights=None,
                       row_labels=None, **kwds):
        return pred.get_prediction(self, exog=exog, transform=transform,
                                   weights=weights, row_labels=row_labels,
                                   **kwds)

    @copy_doc(base.Results.summary.__doc__)
    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        # TODO: import where we need it (for now), add as cached attributes
        from sm2.stats.stattools import (jarque_bera, omni_normtest,
                                         durbin_watson)

        jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)

        eigvals = self.eigenvals
        condno = self.condition_number

        # TODO: WTF why is this getting attached to self?
        self.diagn = dict(jb=jb, jbpv=jbpv, skew=skew, kurtosis=kurtosis,
                          omni=omni, omnipv=omnipv, condno=condno,
                          mineigval=eigvals[-1])

        # TODO: not used yet
        # diagn_left_header = ['Models stats']
        # diagn_right_header = ['Residual stats']

        # TODO: requiring list/iterable is a bit annoying
        # need more control over formatting
        # TODO: default don't work if it's not identically spelled

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Least Squares']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Observations:', None),
                    ('Df Residuals:', None),
                    ('Df Model:', None),
                    ]

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        rsquared_type = '' if self.k_constant else ' (uncentered)'
        top_right = [
            ('R-squared' + rsquared_type + ':',
             ["%#8.3f" % self.rsquared]),
            ('Adj. R-squared' + rsquared_type + ':',
             ["%#8.3f" % self.rsquared_adj]),
            ('F-statistic:', ["%#8.4g" % self.fvalue]),
            ('Prob (F-statistic):', ["%#6.3g" % self.f_pvalue]),
            ('Log-Likelihood:', None),  #["%#6.4g" % self.llf]),
            ('AIC:', ["%#8.4g" % self.aic]),
            ('BIC:', ["%#8.4g" % self.bic])
        ]

        diagn_left = [('Omnibus:', ["%#6.3f" % omni]),
                      ('Prob(Omnibus):', ["%#6.3f" % omnipv]),
                      ('Skew:', ["%#6.3f" % skew]),
                      ('Kurtosis:', ["%#6.3f" % kurtosis])]

        diagn_right = [('Durbin-Watson:',
                        ["%#8.3f" % durbin_watson(self.wresid)]),
                       ('Jarque-Bera (JB):', ["%#8.3f" % jb]),
                       ('Prob(JB):', ["%#8.3g" % jbpv]),
                       ('Cond. No.', ["%#8.3g" % condno])]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        # create summary table instance
        from sm2.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                              use_t=self.use_t)

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                             yname=yname, xname=xname,
                             title="")

        # add warnings/notes, added to text format only
        etext = []
        if hasattr(self, 'cov_type'):
            etext.append(self.cov_kwds['description'])
        if self.model.exog.shape[0] < self.model.exog.shape[1]:
            wstr = "The input rank is higher than the number of observations."
            etext.append(wstr)
        if eigvals[-1] < 1e-10:
            wstr = ("The smallest eigenvalue is %6.3g. This might indicate "
                    "that there are\n"
                    "strong multicollinearity problems or that the design "
                    "matrix is singular." % eigvals[-1])
            etext.append(wstr)
        elif condno > 1000:  # TODO: what is recommended?
            wstr = ("The condition number is large, %6.3g. This might "
                    "indicate that there are\n"
                    "strong multicollinearity or other numerical "
                    "problems." % condno)
            etext.append(wstr)

        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Warnings:")
            smry.add_extra_txt(etext)
        return smry


class OLSResults(RegressionResults):
    """
    Results class for for an OLS model.

    As of 2018-03-05 none of the OLSResults-specific methods are ported
    from upstream.
    """
    def get_influence(self):  # pragma: no cover
        raise NotImplementedError("get_influence not ported from upstream")

    def outlier_test(self, method='bonf', alpha=.05):  # pragma: no cover
        raise NotImplementedError("get_influence not ported from upstream")

    def el_test(self, b0_vals, param_nums, return_weights=0,
                ret_params=0, method='nm',
                stochastic_exog=1, return_params=0):  # pragma: no cover
        raise NotImplementedError("el_test not ported from upstream")

    def conf_int_el(self, param_num, sig=.05, upper_bound=None,
                    lower_bound=None, method='nm',
                    stochastic_exog=1):  # pragma: no cover
        raise NotImplementedError("conf_int_el not ported from upstream")


class RegressionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'chisq': 'columns',
        'sresid': 'rows',
        'weights': 'rows',
        'wresid': 'rows',
        'bcov_unscaled': 'cov',
        'bcov_scaled': 'cov',
        'HC0_se': 'columns',
        'HC1_se': 'columns',
        'HC2_se': 'columns',
        'HC3_se': 'columns',
        'norm_resid': 'rows'}
    _wrap_attrs = wrap.union_dicts(base.LikelihoodResultsWrapper._attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        base.LikelihoodResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(RegressionResultsWrapper,  # noqa:E305
                      RegressionResults)
