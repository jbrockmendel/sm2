#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Limited dependent variable and qualitative variables.

Includes binary outcomes, count data, (ordered) ordinal data and limited
dependent variables.

General References
--------------------
A.C. Cameron and P.K. Trivedi.  `Regression Analysis of Count Data`.
    Cambridge, 1998

G.S. Madalla. `Limited-Dependent and Qualitative Variables in Econometrics`.
    Cambridge, 1983.

W. Greene. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
"""
from __future__ import division
import warnings

from six.moves import range
from sm2.compat.scipy import loggamma

import numpy as np
import pandas as pd
from pandas.util._decorators import deprecate_kwarg

from scipy.special import gammaln
from scipy import stats, special

from sm2.tools.decorators import (resettable_cache,
                                  cache_readonly, cached_data, cached_value,
                                  copy_doc)
from sm2.tools.sm_exceptions import PerfectSeparationError
from sm2.tools.numdiff import approx_fprime_cs
from sm2.tools import data as data_tools

from sm2.base import naming
import sm2.base.model as base
from sm2.base.data import handle_data  # for mnlogit
import sm2.base.wrapper as wrap

from sm2.distributions.discrete import genpoisson_p
import sm2.regression.linear_model as lm

try:
    import cvxopt  # noqa:F401
    have_cvxopt = True
except ImportError:
    have_cvxopt = False

# alias for upstream/backwards compat
_numpy_to_dummies = data_tools.numpy_to_dummies
_pandas_to_dummies = data_tools.pandas_to_dummies

__all__ = ["Poisson", "Logit", "Probit", "MNLogit", "NegativeBinomial",
           "GeneralizedPoisson", "NegativeBinomialP"]

# TODO: When we eventually get user-settable precision, we need to change
#      this
FLOAT_EPS = np.finfo(float).eps

# TODO: add options for the parameter covariance/variance
# ie., OIM, EIM, and BHHH see Green 21.4
# Dogit
# Ordered Logit/Probit
# Generalized Ordered Logit

_discrete_results_docs = """
    %(one_line_description)s

    Parameters
    ----------
    model : A DiscreteModel instance
    params : array-like
        The parameters of a fitted model.
    hessian : array-like
        The hessian of the fitted model.
    scale : float
        A scale parameter for the covariance matrix.

    Returns
    -------
    *Attributes*

    aic : float
        Akaike information criterion.  `-2*(llf - p)` where `p` is the number
        of regressors including the intercept.
    bic : float
        Bayesian information criterion. `-2*llf + ln(nobs)*p` where `p` is the
        number of regressors including the intercept.
    bse : array
        The standard errors of the coefficients.
    df_resid : float
        See model definition.
    df_model : float
        See model definition.
    fitted_values : array
        Linear predictor XB.
    llf : float
        Value of the loglikelihood
    llnull : float
        Value of the constant-only loglikelihood
    llr : float
        Likelihood ratio chi-squared statistic; `-2*(llnull - llf)`
    llr_pvalue : float
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
    prsquared : float
        McFadden's pseudo-R-squared. `1 - (llf / llnull)`
    %(extra_attr)s"""

_l1_results_attr = """    nnz_params : Integer
        The number of nonzero parameters in the model.  Train with
        trim_params == True or else numerical error will distort this.
    trimmed : Boolean array
        trimmed[i] == True if the ith parameter was trimmed from the model."""


# ----------------------------------------------------------------
# Helper Functions

def _validate_l1_method(method):
    """
    As of 0.10.0, the supported values for `method` in `fit_regularized`
    are "l1" and "l1_cvxopt_cp".  If an invalid value is passed, raise
    with a helpful error message

    Parameters
    ----------
    method : str

    Raises
    ------
    ValueError
    """
    if method not in ['l1', 'l1_cvxopt_cp']:
        raise ValueError('`method` = {method} is not supported, use either '
                         '"l1" or "l1_cvxopt_cp"'.format(method=method))


# ----------------------------------------------------------------
# Private Model Classes


class DiscreteModel(base.LikelihoodModel):
    """
    Abstract class for discrete choice models.

    This class does not do anything itself but lays out the methods and
    call signature expected of child classes in addition to those of
    sm2.model.LikelihoodModel.
    """

    @cached_value
    def nobs(self):
        return self.exog.shape[0]

    @cached_value
    def df_model(self):
        # assumes constant
        rank = np.linalg.matrix_rank(self.exog)
        return float(rank) - 1

    @cached_value
    def df_resid(self):
        return float(self.nobs) - (self.df_model + 1)

    def __init__(self, endog, exog, **kwargs):
        super(DiscreteModel, self).__init__(endog, exog, **kwargs)
        self.raise_on_perfect_prediction = True

    def cdf(self, X):  # pragma: no cover
        """
        The cumulative distribution function of the model.
        """
        raise NotImplementedError

    def pdf(self, X):  # pragma: no cover
        """
        The probability density (mass) function of the model.
        """
        raise NotImplementedError

    def _check_perfect_pred(self, params, *args):
        endog = self.endog
        Xb = np.dot(self.exog, params[:self.exog.shape[1]])
        fittedvalues = self.cdf(Xb)
        if (self.raise_on_perfect_prediction and
                np.allclose(fittedvalues - endog, 0)):
            raise PerfectSeparationError("Perfect separation detected, "
                                         "results not available")

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        """
        Fit the model using maximum likelihood.

        The rest of the docstring is from
        sm2.base.model.LikelihoodModel.fit
        """
        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass  # TODO: make a function factory to have multiple call-backs

        mlefit = super(DiscreteModel, self).fit(start_params=start_params,
                                                method=method, maxiter=maxiter,
                                                full_output=full_output,
                                                disp=disp, callback=callback,
                                                **kwargs)

        return mlefit  # it is up to subclasses to wrap results

    fit.__doc__ += base.LikelihoodModel.fit.__doc__

    def _set_alpha(self, alpha):  # TODO: Move higher up?  rename?
        """
        Call to setup parameter transformations at the beginning
        of fit_regularized.
        """
        return alpha

    def fit_regularized(self, start_params=None, method='l1',
                        maxiter='defined_by_method', full_output=1, disp=True,
                        callback=None, alpha=0, trim_mode='auto',
                        auto_trim_tol=0.01, size_trim_tol=1e-4, qc_tol=0.03,
                        qc_verbose=False, **kwargs):
        """
        Fit the model using a regularized maximum likelihood.
        The regularization method AND the solver used is determined by the
        argument method.

        Parameters
        ----------
        start_params : array-like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : 'l1' or 'l1_cvxopt_cp'
            See notes for details.
        maxiter : Integer or 'defined_by_method'
            Maximum number of iterations to perform.
            If 'defined_by_method', then use method defaults (see notes).
        full_output : bool
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool
            Set to True to print convergence messages.
        fargs : tuple
            Extra arguments passed to the likelihood function, i.e.,
            loglike(x, *args)
        callback : callable callback(xk)
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.
        alpha : non-negative scalar or numpy array (same size as parameters)
            The weight multiplying the l1 penalty term
        trim_mode : 'auto, 'size', or 'off'
            If not 'off', trim (set to zero) parameters that would have been
            zero if the solver reached the theoretical minimum.
            If 'auto', trim params using the Theory above.
            If 'size', trim params if they have very small absolute value
        size_trim_tol : float or 'auto' (default = 'auto')
            For use when trim_mode == 'size'
        auto_trim_tol : float
            For sue when trim_mode == 'auto'.  Use
        qc_tol : float
            Print warning and don't allow auto trim when (ii) (above) is
            violated by this much.
        qc_verbose : Boolean
            If true, print out a full QC report upon failure

        Notes
        -----
        Extra parameters are not penalized if alpha is given as a scalar.
        An example is the shape parameter in NegativeBinomial `nb1` and `nb2`.

        Optional arguments for the solvers (available in Results.mle_settings):

            'l1'
                acc : float (default 1e-6)
                    Requested accuracy as used by slsqp
            'l1_cvxopt_cp'
                abstol : float
                    absolute accuracy (default: 1e-7).
                reltol : float
                    relative accuracy (default: 1e-6).
                feastol : float
                    tolerance for feasibility conditions (default: 1e-7).
                refinement : int
                    number of iterative refinement steps when solving KKT
                    equations (default: 1).

        Optimization methodology

        With :math:`L` the negative log likelihood, we solve the convex but
        non-smooth problem

        .. math:: \\min_\\beta L(\\beta) + \\sum_k\\alpha_k |\\beta_k|

        via the transformation to the smooth, convex, constrained problem
        in twice as many variables (adding the "added variables" :math:`u_k`)

        .. math:: \\min_{\\beta,u} L(\\beta) + \\sum_k\\alpha_k u_k,

        subject to

        .. math:: -u_k \\leq \\beta_k \\leq u_k.

        With :math:`\\partial_k L` the derivative of :math:`L` in the
        :math:`k^{th}` parameter direction, theory dictates that, at the
        minimum, exactly one of two conditions holds:

        (i) :math:`|\\partial_k L| = \\alpha_k`  and  :math:`\\beta_k \\neq 0`
        (ii) :math:`|\\partial_k L| \\leq \\alpha_k`  and  :math:`\\beta_k = 0`
        """
        _validate_l1_method(method)
        # Set attributes based on method

        if qc_verbose:  # pragma: no cover
            # TODO: Update docstring to reflect this restriction
            raise NotImplementedError("option `qc_verbose` is available "
                                      "upstream, but is disabled in sm2.")

        alpha = self._set_alpha(alpha)
        start_params = self._get_start_params_l1(
            start_params, method=method, maxiter=maxiter,
            full_output=full_output, disp=0, callback=callback,
            alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
            size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)

        cov_params_func = self.cov_params_func_l1

        # Bundle up extra kwargs for the dictionary kwargs.  These are
        # passed through super(...).fit() as kwargs and unpacked at
        # appropriate times
        alpha = np.array(alpha)
        assert alpha.min() >= 0
        try:
            kwargs['alpha'] = alpha  # TODO: how would this happen?
        except TypeError:
            kwargs = {'alpha': alpha}
        kwargs['alpha_rescaled'] = kwargs['alpha'] / float(self.endog.shape[0])
        kwargs['trim_mode'] = trim_mode
        kwargs['size_trim_tol'] = size_trim_tol
        kwargs['auto_trim_tol'] = auto_trim_tol
        kwargs['qc_tol'] = qc_tol
        kwargs['qc_verbose'] = qc_verbose

        # Define default keyword arguments to be passed to super(...).fit()
        if maxiter == 'defined_by_method':
            if method == 'l1':
                maxiter = 1000
            elif method == 'l1_cvxopt_cp':
                maxiter = 70

        # Parameters to pass to super(...).fit()
        # For the 'extra' parameters, pass all that are available,
        # even if we know (at this point) we will only use one.
        from sm2.base.l1_slsqp import fit_l1_slsqp
        extra_fit_funcs = {'l1': fit_l1_slsqp}
        if have_cvxopt and method == 'l1_cvxopt_cp':
            from sm2.base.l1_cvxopt import fit_l1_cvxopt_cp
            extra_fit_funcs['l1_cvxopt_cp'] = fit_l1_cvxopt_cp
        elif method.lower() == 'l1_cvxopt_cp':
            message = ("Attempt to use l1_cvxopt_cp failed since cvxopt "
                       "could not be imported")
            # FIXME: Um... do something with this message?

        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass  # TODO: make a function factory to have multiple call-backs

        mlefit = super(DiscreteModel, self).fit(
            start_params=start_params, method=method, maxiter=maxiter,
            full_output=full_output, disp=disp, callback=callback,
            extra_fit_funcs=extra_fit_funcs, cov_params_func=cov_params_func,
            **kwargs)

        res_cls, wrap_cls = self._res_classes["fit_regularized"]
        discretefit = res_cls(self, mlefit)
        return wrap_cls(discretefit)

    # TODO: Might this go higher up the hierarchy?
    def cov_params_func_l1(self, likelihood_model, xopt, retvals):
        """
        Computes cov_params on a reduced parameter space
        corresponding to the nonzero parameters resulting from the
        l1 regularized fit.

        Returns a full cov_params matrix, with entries corresponding
        to zero'd values set to np.nan.
        """
        H = likelihood_model.hessian(xopt)
        trimmed = retvals['trimmed']
        nz_idx = np.nonzero(trimmed == False)[0]  # noqa:E712
        nnz_params = (trimmed == False).sum()  # noqa:E712
        if nnz_params > 0:
            H_restricted = H[nz_idx[:, None], nz_idx]
            # Covariance estimate for the nonzero params
            H_restricted_inv = np.linalg.inv(-H_restricted)
        else:
            H_restricted_inv = np.zeros(0)

        cov_params = np.nan * np.ones(H.shape)
        cov_params[nz_idx[:, None], nz_idx] = H_restricted_inv

        return cov_params

    # TODO: Might this go higher up the hierarchy?
    def predict(self, params, exog=None, linear=False):  # pragma: no cover
        """
        Predict response variable of a model given exogenous variables.
        """
        raise NotImplementedError

    def _derivative_exog(self, params, exog=None, dummy_idx=None,
                         count_idx=None):
        """
        This should implement the derivative of the non-linear function
        """
        raise NotImplementedError  # pragma: no cover

    def _wrap_derivative_exog(self, margeff, params, exog, dummy_idx,
                              count_idx, transform):
        """
        Helper for _derivative_exog to wrap results appropriately
        """
        if count_idx is not None:
            from sm2.discrete.discrete_margins import _get_count_effects
            margeff = _get_count_effects(margeff, exog, count_idx, transform,
                                         self, params)
        if dummy_idx is not None:
            from sm2.discrete.discrete_margins import _get_dummy_effects
            margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform,
                                         self, params)
        return margeff


# TODO: Can we just move this all the way up into DiscreteModel?
class FitBase(DiscreteModel):
    """Mixin to wrap DiscreteModel.fit"""

    @copy_doc(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):

        bnryfit = DiscreteModel.fit(self, start_params=start_params,
                                    method=method, maxiter=maxiter,
                                    full_output=full_output,
                                    disp=disp, callback=callback,
                                    **kwargs)

        res_cls, wrap_cls = self._res_classes["fit"]
        discretefit = res_cls(self, bnryfit)
        return wrap_cls(discretefit)


class BinaryModel(FitBase):
    @property
    def _res_classes(self):
        return {"fit": (BinaryResults, BinaryResultsWrapper),
                "fit_regularized": (L1BinaryResults, L1BinaryResultsWrapper)}

    def __init__(self, endog, exog, **kwargs):
        super(BinaryModel, self).__init__(endog, exog, **kwargs)
        if (not issubclass(self.__class__, MultinomialModel) and
                not np.all((self.endog >= 0) & (self.endog <= 1))):
            raise ValueError("endog must be in the unit interval.")
            # TODO: Just do this check in MultinomialModel.__init__?

    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array-like
            Fitted parameters of the model.
        exog : array-like
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used.
        linear : bool, optional
            If True, returns the linear predictor dot(exog, params).  Else,
            returns the value of the cdf at the linear predictor.

        Returns
        -------
        array
            Fitted values at exog.
        """  # TODO: Can we share this docstring?
        if exog is None:
            exog = self.exog
        if not linear:
            return self.cdf(np.dot(exog, params))
        else:
            return np.dot(exog, params)

    # TODO: De-duplicate docstring with the other two in this module.  This
    # and one other look like they might be truncated in the second paragraph
    def _derivative_predict(self, params, exog=None, transform='dydx'):
        """
        For computing marginal effects standard errors.

        This is used only in the case of discrete and count regressors to
        get the variance-covariance of the marginal effects. It returns
        [d F / d params] where F is the predict.

        Transform can be 'dydx' or 'eydx'. Checking is done in margeff
        computations for appropriate transform.
        """
        if exog is None:
            exog = self.exog
        dF = self.pdf(np.dot(exog, params))[:, None] * exog
        if 'ey' in transform:
            dF /= self.predict(params, exog)[:, None]
        return dF

    def _derivative_exog(self, params, exog=None, transform='dydx',
                         dummy_idx=None, count_idx=None):
        """
        For computing marginal effects returns dF(XB) / dX where F(.) is
        the predicted probabilities

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        # note, this form should be appropriate for
        # group 1 probit, logit, logistic, cloglog, heckprob, xtprobit
        if exog is None:
            exog = self.exog
        Xb = np.dot(exog, params)
        margeff = np.dot(self.pdf(Xb)[:, None], params[None, :])

        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None]

        return self._wrap_derivative_exog(margeff, params, exog, dummy_idx,
                                          count_idx, transform)


class MultinomialModel(BinaryModel):
    _check_perfect_pred = None  # placeholder until implemented

    @cached_value
    def J(self):
        return self.wendog.shape[1]

    @cached_value
    def K(self):
        return self.exog.shape[1]

    @cached_value
    def df_model(self):
        rank = np.linalg.matrix_rank(self.exog)
        return (rank - 1) * (self.J - 1)  # for each J - 1 equation.
        # TODO: Does "assumes constant" apply here?

    @cached_value
    def df_resid(self):
        return self.nobs - self.df_model - (self.J - 1)

    @property
    def _res_classes(self):
        return {"fit": (MultinomialResults, MultinomialResultsWrapper),
                "fit_regularized": (L1MultinomialResults,
                                    L1MultinomialResultsWrapper)}

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        # FIXME: I don't think we go through this correctly when
        # using from_formula --> eqn_names doesn't get set
        if data_tools._is_using_ndarray_type(endog, None):
            endog_dummies, ynames = _numpy_to_dummies(endog)
            yname = 'y'
        elif data_tools._is_using_pandas(endog, None):
            endog_dummies, ynames, yname = _pandas_to_dummies(endog)
        else:
            endog = np.asarray(endog)
            endog_dummies, ynames = _numpy_to_dummies(endog)
            yname = 'y'

        if not isinstance(ynames, dict):
            ynames = dict(zip(range(endog_dummies.shape[1]), ynames))

        self._ynames_map = ynames
        data = handle_data(endog_dummies, exog, missing, hasconst, **kwargs)

        eqn_names = [x[1] for x in sorted(list(ynames.items()))]
        data.ynames = pd.Index(eqn_names, name=yname)
        # e.g. if user passes a pandas Series as endog, the index.name here
        # will match the name of that Series.
        # TODO: make this a Data method?
        # TODO: upstram had `data.ynames = yname` and comment to overwrite
        # single endog name; but why?
        data.orig_endog = endog

        self.wendog = data.endog

        # TODO: use super here?
        # repeating from upstream...
        for key in kwargs:
            if key in ['design_info', 'formula']:  # leave attached to data
                continue
            try:
                setattr(self, key, data.__dict__.pop(key))
            except KeyError:
                pass
        return data

    def initialize(self):
        """
        Preprocesses the data for MNLogit.
        """
        super(MultinomialModel, self).initialize()
        # This is also a "whiten" method in other models (eg regression)
        self.endog = self.endog.argmax(1)  # turn it into an array of col idx

    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array-like
            2d array of fitted parameters of the model. Should be in the
            order returned from the model.
        exog : array-like
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used. If a 1d array is given
            it assumed to be 1 row of exogenous variables. If you only have
            one regressor and would like to do prediction, you must provide
            a 2d array with shape[1] == 1.
        linear : bool, optional
            If True, returns the linear predictor dot(exog, params).  Else,
            returns the value of the cdf at the linear predictor.

        Notes
        -----
        Column 0 is the base case, the rest conform to the rows of params
        shifted up one for the base case.
        """
        if exog is None:  # do here to accomodate user-given exog
            exog = self.exog
        if exog.ndim == 1:
            exog = exog[None]
        pred = super(MultinomialModel, self).predict(params, exog, linear)
        if linear:
            pred = np.column_stack((np.zeros(len(exog)), pred))
        return pred

    def _get_start_params(self, start_params):
        if start_params is None:
            start_params = np.zeros((self.K * (self.J - 1)))
        else:
            start_params = np.asarray(start_params)
        return start_params

    def _derivative_predict(self, params, exog=None, transform='dydx'):
        """
        For computing marginal effects standard errors.

        This is used only in the case of discrete and count regressors to
        get the variance-covariance of the marginal effects. It returns
        [d F / d params] where F is the predicted probabilities for each
        choice. dFdparams is of shape nobs x (J*K) x (J-1)*K.
        The zero derivatives for the base category are not included.

        Transform can be 'dydx' or 'eydx'. Checking is done in margeff
        computations for appropriate transform.
        """
        if exog is None:
            exog = self.exog
        if params.ndim == 1:  # will get flatted from approx_fprime
            params = params.reshape(self.K, self.J - 1, order='F')

        eXB = np.exp(np.dot(exog, params))
        sum_eXB = (1 + eXB.sum(1))[:, None]
        J = int(self.J)
        K = int(self.K)
        repeat_eXB = np.repeat(eXB, J, axis=1)
        X = np.tile(exog, J - 1)
        # this is the derivative wrt the base level
        F0 = -repeat_eXB * X / sum_eXB ** 2
        # this is the derivative wrt the other levels when
        # dF_j / dParams_j (ie., own equation)
        # NOTE: this computes too much, any easy way to cut down?
        F1 = eXB.T[:, :, None] * X * (sum_eXB - repeat_eXB) / (sum_eXB**2)
        F1 = F1.transpose((1, 0, 2))  # put the nobs index first

        # other equation index
        other_idx = ~np.kron(np.eye(J - 1), np.ones(K)).astype(bool)
        F1[:, other_idx] = ((-eXB.T[:, :, None] * X * repeat_eXB /
                            (sum_eXB**2)).transpose((1, 0, 2))[:, other_idx])
        dFdX = np.concatenate((F0[:, None, :], F1), axis=1)

        if 'ey' in transform:
            dFdX /= self.predict(params, exog)[:, :, None]
        return dFdX

    def _derivative_exog(self, params, exog=None, transform='dydx',
                         dummy_idx=None, count_idx=None):
        """
        For computing marginal effects returns dF(XB) / dX where F(.) is
        the predicted probabilities

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.

        For Multinomial models the marginal effects are

        P[j] * (params[j] - sum_k P[k]*params[k])

        It is returned unshaped, so that each row contains each of the J
        equations. This makes it easier to take derivatives of this for
        standard errors. If you want average marginal effects you can do
        margeff.reshape(nobs, K, J, order='F).mean(0) and the marginal effects
        for choice J are in column J
        """
        J = int(self.J)  # number of alternative choices
        K = int(self.K)  # number of variables
        # note, this form should be appropriate for
        # group 1 probit, logit, logistic, cloglog, heckprob, xtprobit
        if exog is None:
            exog = self.exog
        if params.ndim == 1:  # will get flatted from approx_fprime
            params = params.reshape(K, J - 1, order='F')
        zeroparams = np.c_[np.zeros(K), params]  # add base in

        cdf = self.cdf(np.dot(exog, params))
        margeff = np.array(
            [cdf[:, [j]] * (zeroparams[:, j] -
                            np.array([cdf[:, [i]] * zeroparams[:, i]
                                      for i in range(int(J))]).sum(0))
             for j in range(J)])
        margeff = np.transpose(margeff, (1, 2, 0))
        # swap the axes to make sure margeff are in order nobs, K, J

        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None, :]
            # TODO: The extra ":" at the end of this line is the main difference
            # between this and the other two versions of _derivative_exog

        margeff = self._wrap_derivative_exog(margeff, params, exog, dummy_idx,
                                             count_idx, transform)
        return margeff.reshape(len(exog), -1, order='F')


class CountModel(FitBase):
    @property
    def _res_classes(self):
        return {"fit": (CountResults, CountResultsWrapper),
                "fit_regularized": (L1CountResults,
                                    L1CountResultsWrapper)}

    def __init__(self, endog, exog, offset=None, exposure=None, missing='none',
                 **kwargs):
        super(CountModel, self).__init__(endog, exog, missing=missing,
                                         offset=offset,
                                         exposure=exposure, **kwargs)
        if exposure is not None:
            self.exposure = np.log(self.exposure)
        self._check_inputs(self.offset, self.exposure, self.endog)
        if offset is None:
            delattr(self, 'offset')
        if exposure is None:
            delattr(self, 'exposure')

        # promote dtype to float64 if needed
        dt = np.promote_types(self.endog.dtype, np.float64)
        self.endog = np.asarray(self.endog, dt)
        dt = np.promote_types(self.exog.dtype, np.float64)
        self.exog = np.asarray(self.exog, dt)

    def _check_inputs(self, offset, exposure, endog):
        if offset is not None and offset.shape[0] != endog.shape[0]:
            raise ValueError("offset is not the same length as endog")

        if exposure is not None and exposure.shape[0] != endog.shape[0]:
            raise ValueError("exposure is not the same length "
                             "as endog")  # pragma: no cover

    def _get_init_kwds(self):
        # this is a temporary fixup because exposure has been transformed
        # see GH#1609
        kwds = super(CountModel, self)._get_init_kwds()
        if 'exposure' in kwds and kwds['exposure'] is not None:
            kwds['exposure'] = np.exp(kwds['exposure'])
        return kwds

    def _get_start_params_null(self):
        """
        Compute one-step moment estimator for null (constant-only) model

        This is a preliminary estimator used as start_params.

        Returns
        -------
        params : ndarray
            parameter estimate based one one-step moment matching
        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)

        const = (self.endog / np.exp(offset + exposure)).mean()
        params = [np.log(const)]

        if hasattr(self, '_estimate_dispersion'):
            # includes NegativeBinomial, GeneralizedPoisson, NegativeBinomialP
            # excludes Poisson
            mu = const * np.exp(offset + exposure)
            resid = self.endog - mu
            a = self._estimate_dispersion(mu, resid,
                                          df_resid=resid.shape[0] - 1)
            params.append(a)
        return np.array(params)

    def predict(self, params, exog=None, exposure=None, offset=None,
                linear=False):
        """
        Predict response variable of a count model given exogenous variables.

        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
        """
        # TODO: add offset tp
        if exog is None:
            exog = self.exog

        if exposure is None:
            # If self.exposure exists, it will already be in logs.
            exposure = getattr(self, 'exposure', 0)
        else:
            exposure = np.log(exposure)

        if offset is None:
            offset = getattr(self, 'offset', 0)

        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset
        if not linear:
            return np.exp(linpred)  # not cdf
        else:
            return linpred

    @copy_doc(BinaryModel._derivative_predict.__doc__)
    def _derivative_predict(self, params, exog=None, transform='dydx'):
        if exog is None:
            exog = self.exog
        # NOTE: this handles offset and exposure
        dF = self.predict(params, exog)[:, None] * exog
        if 'ey' in transform:
            dF /= self.predict(params, exog)[:, None]
        return dF

    def _derivative_exog(self, params, exog=None, transform="dydx",
                         dummy_idx=None, count_idx=None):
        """
        For computing marginal effects. These are the marginal effects
        d F(XB) / dX
        For the Poisson model F(XB) is the predicted counts rather than
        the probabilities.

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        # group 3 poisson, nbreg, zip, zinb
        if exog is None:
            exog = self.exog
        k_extra = getattr(self, 'k_extra', 0)
        params_exog = params if k_extra == 0 else params[:-k_extra]
        margeff = self.predict(params, exog)[:, None] * params_exog[None, :]
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None]

        return self._wrap_derivative_exog(margeff, params, exog, dummy_idx,
                                          count_idx, transform)


class _CountMixin(object):
    """
    Mixin for methods that are common to some but not all CountModel
    subclasses
    """
    @property
    def _should_append(self):  # TODO: Better name?
        # compatibility shim for NegativeBinomial; semi-kludge
        loglike_method = getattr(self, 'loglike_method', None)
        return loglike_method is None or loglike_method.startswith('nb')

    @copy_doc(DiscreteModel._set_alpha.__doc__)
    def _set_alpha(self, alpha):
        self._transparams = False

        if self._should_append and (np.size(alpha) == 1 and alpha != 0):
            # don't penalize alpha if alpha is scalar
            k_params = self.exog.shape[1] + self.k_extra
            alpha = alpha * np.ones(k_params)
            alpha[-1] = 0
        return alpha

    # TODO: This overlaps a lot with count_model version
    def _get_start_params_l1(self, start_params, method, maxiter,
                             full_output, disp, callback, alpha, trim_mode,
                             auto_trim_tol, size_trim_tol, qc_tol,
                             **kwargs):
        if start_params is not None:
            return start_params

        # alpha for regularized poisson to get starting values
        if self.k_extra and np.size(alpha) > 1:
            alpha_p = alpha[:-1]
        else:
            alpha_p = alpha

        # TODO: Warning: this assumes exposure is logged
        offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
        if np.size(offset) == 1 and offset == 0:
            offset = None

        # Use poisson fit as first guess.
        mod_poi = Poisson(self.endog, self.exog, offset=offset)
        res_poi = mod_poi.fit_regularized(
            start_params=start_params, method=method, maxiter=maxiter,
            full_output=full_output, disp=0, callback=callback,
            alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
            size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)

        start_params = res_poi.params

        if self._should_append:
            start_params = np.append(start_params, 0.1)
            # TODO: Document reason for 0.1
        return start_params

    # TODO: Overlap with _get_start_params_l1?
    def _get_start_params(self, start_params, **kwargs):
        if start_params is not None:
            return start_params

        offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
        if np.size(offset) == 1 and offset == 0:
            offset = None

        optim_kwds_prelim = {'disp': 0, 'skip_hessian': True,
                             'warn_convergence': False}
        optim_kwds_prelim.update(kwargs.get('optim_kwds_prelim', {}))
        mod_poi = Poisson(self.endog, self.exog, offset=offset)
        res_poi = mod_poi.fit(**optim_kwds_prelim)
        start_params = res_poi.params

        if self._should_append:
            a = self._estimate_dispersion(res_poi.predict(), res_poi.resid,
                                          df_resid=res_poi.df_resid)
            start_params = np.append(start_params, max(0.05, a))
            # TODO: upstream uses -0.1 for GeneralizedPoisson and 0.05
            # for the others.  GH#4521
            # TODO: reasoning for -0.1?
        return start_params


class OrderedModel(DiscreteModel):  # TODO: Why does this exist?
    pass


# --------------------
# Public Model Classes

class Poisson(CountModel):
    __doc__ = """
    Poisson model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params': base._model_params_doc,
           'extra_params':
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}

    @property
    def _res_classes(self):
        return {"fit": (PoissonResults, PoissonResultsWrapper),
                "fit_regularized": (L1PoissonResults,
                                    L1PoissonResultsWrapper)}

    def cdf(self, X):
        r"""
        Poisson model cumulative distribution function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the model.  See notes.

        Returns
        -------
        The value of the Poisson CDF at each point.

        Notes
        -----
        The CDF is defined as

        .. math:: \exp\left(-\lambda\right)\sum_{i=0}^{y}\frac{\lambda^{i}}{i!}

        where :math:`\lambda` assumes the loglinear model. I.e.,

        .. math:: \ln\lambda_{i}=X\beta

        The parameter `X` is :math:`X\beta` in the above formula.
        """
        return stats.poisson.cdf(self.endog, np.exp(X))

    def pdf(self, X):
        """
        Poisson model probability mass function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the model.  See notes.

        Returns
        -------
        pdf : ndarray
            The value of the Poisson probability mass function, PMF, for each
            point of X.

        Notes
        --------
        The PMF is defined as

        .. math:: \\frac{e^{-\\lambda_{i}}\\lambda_{i}^{y_{i}}}{y_{i}!}

        where :math:`\\lambda` assumes the loglinear model. I.e.,

        .. math:: \\ln\\lambda_{i}=x_{i}\\beta

        The parameter `X` is :math:`x_{i}\\beta` in the above formula.
        """
        return np.exp(stats.poisson.logpmf(self.endog, np.exp(X)))

    def loglikeobs(self, params):
        r"""
        Loglikelihood for observations of Poisson model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : array-like
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        --------
        .. math:: \ln L_{i}
            =\left[-\lambda_{i}+y_{i}x_{i}^{\prime}\beta-\ln y_{i}!\right]

        for observations :math:`i=1, ..., n`
        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        Xb = np.dot(self.exog, params)
        linpred = Xb + offset + exposure
        endog = self.endog
        # np.sum(stats.poisson.logpmf(endog, np.exp(XB)))
        # TODO: cache gammaln(endog + 1)?
        return -np.exp(linpred) + endog * linpred - gammaln(endog + 1)

    def score(self, params):
        """
        Poisson model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}
            =\\sum_{i=1}^{n}\\left(y_{i}-\\lambda_{i}\\right)x_{i}

        where the loglinear model is assumed

        .. math:: \\ln\\lambda_{i}=x_{i}\\beta
        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        Xb = np.dot(self.exog, params)
        linpred = Xb + offset + exposure
        L = np.exp(linpred)
        return np.dot(self.endog - L, self.exog)
        # Note: this is non-trivially more performant than wrapping score_obs
        # TODO: Make a score_factor?

    def score_obs(self, params):
        """
        Poisson model Jacobian of the log-likelihood for each observation

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : array-like
            The score vector (nobs, k_vars) of the model evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}
            =\\left(y_{i}-\\lambda_{i}\\right)x_{i}

        for observations :math:`i=1, ..., n`

        where the loglinear model is assumed

        .. math:: \\ln\\lambda_{i}=x_{i}\\beta
        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        Xb = np.dot(self.exog, params)
        linpred = Xb + offset + exposure
        L = np.exp(linpred)
        return (self.endog - L)[:, None] * self.exog

    def hessian(self, params):
        r"""
        Poisson model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \frac{\partial^{2}\ln L}{\partial\beta\partial\beta^{\prime}}
            =-\sum_{i=1}^{n}\lambda_{i}x_{i}x_{i}^{\prime}

        where the loglinear model is assumed

        .. math:: \ln\lambda_{i}=x_{i}\beta
        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        Xb = np.dot(self.exog, params)
        linpred = Xb + offset + exposure
        L = np.exp(linpred)
        return -np.dot(L * self.exog.T, self.exog)

    def _get_start_params(self, start_params, robust=True):
        if start_params is not None:
            pass
        elif (self.data.const_idx is not None and not robust):
            # TODO: apparent numerical instability if we use these start_params
            # with fit_regularized --> not `robust`
            # TODO: k_params or k_exog not available?
            # TODO: document why 0.001?
            start_params = 0.001 * np.ones(self.exog.shape[1])
            start_params[self.data.const_idx] = self._get_start_params_null()[0]
        else:
            start_params = CountModel._get_start_params(self, start_params)
        return start_params

    # TODO: can we use FitBase?
    @copy_doc(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):

        start_params = self._get_start_params(start_params, robust=False)
        # TODO: can we get rid of robust now?

        cntfit = DiscreteModel.fit(
            self, start_params=start_params, method=method, maxiter=maxiter,
            full_output=full_output, disp=disp, callback=callback, **kwargs)

        # TODO: Can we avoid doing this here?  Do it in res_cls.__init__?
        if 'cov_type' in kwargs:
            cov_kwds = kwargs.get('cov_kwds', {})
            kwds = {'cov_type': kwargs['cov_type'], 'cov_kwds': cov_kwds}
        else:
            kwds = {}
        # TODO: This (along with passing **kwds below) doesnt appear necessary

        res_cls, wrap_cls = self._res_classes["fit"]
        discretefit = res_cls(self, cntfit, **kwds)
        return wrap_cls(discretefit)

    # TODO: Is anything special about this Poisson-specific?
    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        """fit the model subject to linear equality constraints

        The constraints are of the form   `R params = q`
        where R is the constraint_matrix and q is the vector of
        constraint_values.

        The estimation creates a new model with transformed design matrix,
        exog, and converts the results back to the original parameterization.

        Parameters
        ----------
        constraints : formula expression or tuple
            If it is a tuple, then the constraint needs to be given by two
            arrays (constraint_matrix, constraint_value), i.e. (R, q).
            Otherwise, the constraints can be given as strings or list of
            strings.
            see t_test for details
        start_params : None or array_like
            starting values for the optimization. `start_params` needs to be
            given in the original parameter space and are internally
            transformed.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the transformed model.

        Returns
        -------
        results : Results instance
        """
        # constraints = (R, q)
        # TODO: temporary trailing underscore to not overwrite the monkey
        #       patched version
        # TODO: decide whether to move the imports
        from patsy import DesignInfo
        from sm2.base._constraints import fit_constrained

        # same pattern as in base.LikelihoodModel.t_test
        lc = DesignInfo(self.exog_names).linear_constraint(constraints)
        R, q = lc.coefs, lc.constants

        # TODO: add start_params option, need access to tranformation
        #       fit_constrained needs to do the transformation
        params, cov, res_constr = fit_constrained(self, R, q,
                                                  start_params=start_params,
                                                  fit_kwds=fit_kwds)
        # create dummy results Instance, TODO: wire up properly
        res = self.fit(maxiter=0, method='nm', disp=0,
                       warn_convergence=False)  # we get a wrapper back

        constr_retvals = res_constr.mle_retvals
        res.mle_retvals['fcall'] = constr_retvals.get('fcall', np.nan)
        res.mle_retvals['iterations'] = constr_retvals.get('iterations',
                                                           np.nan)
        res.mle_retvals['converged'] = constr_retvals['converged']
        res._results.params = params
        res._results.cov_params_default = cov
        cov_type = fit_kwds.get('cov_type', 'nonrobust')
        if cov_type != 'nonrobust':
            res._results.normalized_cov_params = cov  # assume scale=1
        else:
            res._results.normalized_cov_params = None

        k_constr = len(q)
        res._results.df_resid += k_constr
        res._results.df_model -= k_constr
        # FIXME: don't alter these in-place
        res._results.constraints = lc
        res._results.k_constr = k_constr
        res._results.results_constrained = res_constr
        # TODO: De-duplicate with _constraints.fit_constrained_wrap and genmod
        return res


class GeneralizedPoisson(_CountMixin, CountModel):
    __doc__ = """
    Generalized Poisson model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params': base._model_params_doc,
           'extra_params': """
    p: scalar
        P denotes parameterizations for GP regression. p=1 for GP-1 and
        p=2 for GP-2. Default is p=1.
    offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}
    _check_perfect_pred = None  # placeholder until implemented GH#3895

    @property
    def _res_classes(self):
        return {"fit": (GeneralizedPoissonResults,
                        GeneralizedPoissonResultsWrapper),
                "fit_regularized": (L1GeneralizedPoissonResults,
                                    L1GeneralizedPoissonResultsWrapper)}

    def __init__(self, endog, exog, p=1, offset=None,
                 exposure=None, missing='none', **kwargs):
        super(GeneralizedPoisson, self).__init__(endog, exog, offset=offset,
                                                 exposure=exposure,
                                                 missing=missing, **kwargs)
        self.parameterization = p - 1
        self.exog_names.append('alpha')
        self.k_extra = 1
        self._transparams = False

    def _get_init_kwds(self):
        kwds = super(GeneralizedPoisson, self)._get_init_kwds()
        kwds['p'] = self.parameterization + 1
        return kwds

    def loglikeobs(self, params):
        r"""
        Loglikelihood for observations of Generalized Poisson model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        --------
        .. math:: \ln L=\sum_{i=1}^{n}\left[\mu_{i}+(y_{i}-1)*ln(\mu_{i}+
            \alpha*\mu_{i}^{p-1}*y_{i})-y_{i}*ln(1+\alpha*\mu_{i}^{p-1})-
            ln(y_{i}!)-\frac{\mu_{i}+\alpha*\mu_{i}^{p-1}*y_{i}}{1+\alpha*
            \mu_{i}^{p-1}}\right]

        for observations :math:`i=1, ..., n`
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        endog = self.endog
        mu = self.predict(params)
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + (a1 - 1) * endog
        # TODO: cache gammaln(endog+1)?
        return (np.log(mu) + (endog - 1) * np.log(a2) -
                endog * np.log(a1) - gammaln(endog + 1) - a2 / a1)

    def score_obs(self, params):
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]

        params = params[:-1]
        p = self.parameterization
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        mu_p = np.power(mu, p)
        amp = alpha * mu_p
        a1 = 1 + amp
        a2 = mu + amp * y
        a3 = amp * p / mu
        a4 = a3 * y
        dmudb = mu * exog

        dalpha = mu_p * (y * ((y - 1) / a2 - 2 / a1) + a2 / a1**2)
        dparams = dmudb * (a3 * a2 / a1**2
                           + (1 + a4) * ((y - 1) / a2 - 2 / a1)
                           + 1 / a1
                           + 1 / mu)

        return np.concatenate((dparams, np.atleast_2d(dalpha)),
                              axis=1)

    def score(self, params):
        score = np.sum(self.score_obs(params), axis=0)
        if self._transparams:
            score[-1] == score[-1] ** 2
            return score
        else:
            return score

    def _score_p(self, params):  # pragma: no cover
        raise NotImplementedError("_score_p not ported from upstream, "
                                  "as it is neither used nor tested.")

    def hessian(self, params):
        """
        Generalized Poisson model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]

        params = params[:-1]
        p = self.parameterization
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        mu_p = np.power(mu, p)
        amp = alpha * mu_p
        a1 = 1 + amp
        a2 = mu + amp * y
        a3 = amp * p / mu
        a4 = a3 * y
        dmudb = mu * exog

        # for dl/dparams dparams
        dim = exog.shape[1]
        hess_arr = np.empty((dim + 1, dim + 1))

        for i in range(dim):
            for j in range(i + 1):
                hess_arr[i, j] = np.sum(
                    mu * exog[:, i, None] * exog[:, j, None] *
                    (mu * (3 * y * a3**2 / a1**2
                           - 2 * a2 * a3**2 / a1**3
                           + 2 * a3 / a1**2
                           - (1 - p) * a3 * a2 / (mu * a1**2)
                           + 2 * (1 - p) * a4 / (a1 * mu)
                           - (1 - p) * (y - 1) * a4 / (a2 * mu)
                           - (y - 1) * (1 + a4)**2 / a2**2
                           - 1 / mu**2) +
                     (-2 * y * a3 / a1
                      + a3 * a2 / a1**2
                      + (y - 1) * (1 + a4) / a2
                      - 1 / a1
                      + 1 / mu)), axis=0)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]

        # for dl/dparams dalpha
        dldpda = np.sum((3 * y * a3 * mu_p / a1**2
                         - 2 * a3 * mu_p * a2 / a1**3
                         - mu_p * y * (y - 1) * (1 + a4) / a2**2
                         + mu_p / a1**2
                         + (p / mu) * mu_p * y * (y - 1) / a2
                         - 2 * (p / mu) * mu_p * y / a1
                         + (p / mu) * mu_p * a2 / a1**2) * dmudb,
                        axis=0)

        hess_arr[-1, :-1] = dldpda
        hess_arr[:-1, -1] = dldpda

        # for dl/dalpha dalpha
        dldada = mu_p**2 * (3 * y / a1**2 - (y / a2)**2. * (y - 1) - 2 * a2 /
                            a1**3)

        hess_arr[-1, -1] = dldada.sum()

        return hess_arr

    def _estimate_dispersion(self, mu, resid, df_resid=None):
        q = self.parameterization
        if df_resid is None:
            df_resid = resid.shape[0]
        a = ((np.abs(resid) / np.sqrt(mu) - 1) * mu**(-q)).sum() / df_resid
        return a

    # ----------------------------------------------------------------

    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None, use_transparams=False,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        """
        Parameters
        ----------
        use_transparams : bool
            This parameter enable internal transformation to impose
            non-negativity.  True to enable. Default is False.
            use_transparams=True imposes the no underdispersion (alpha > 0)
            constaint. In case use_transparams=True and method="newton"
            or "ncg" transformation is ignored.
        """
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = True
        else:
            if use_transparams:
                warnings.warn('Parameter "use_transparams" is ignored',
                              RuntimeWarning)
            self._transparams = False

        start_params = self._get_start_params(start_params, **kwargs)

        # TODO: skip CountModel and go straight to DiscreteModel?
        # Yes, just need to change "mlefit._results" --> "mlefit" below
        mlefit = CountModel.fit(
            self, start_params=start_params, maxiter=maxiter, method=method,
            disp=disp, full_output=full_output, callback=callback,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t, **kwargs)

        if self._transparams:
            self._transparams = False
            mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])
            # ensure cov_params are re-evaluated with updated params
            delattr(mlefit._results, "cov_type")
            # TODO: not hit in tests

        res_cls, wrap_cls = self._res_classes["fit"]
        gpfit = res_cls(self, mlefit._results,
                        cov_type=cov_type, use_t=use_t, cov_kwds=cov_kwds)
        return wrap_cls(gpfit)

    fit.__doc__ = DiscreteModel.fit.__doc__ + fit.__doc__

    def predict(self, params, exog=None, exposure=None, offset=None,
                which='mean'):
        """
        Predict response variable of a count model given exogenous variables.

        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
        """
        if exog is None:
            exog = self.exog

        if exposure is None:
            exposure = getattr(self, 'exposure', 0)
        elif exposure != 0:
            exposure = np.log(exposure)

        if offset is None:
            offset = getattr(self, 'offset', 0)

        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset

        if which == 'mean':
            return np.exp(linpred)
        elif which == 'linear':
            return linpred
        elif which == 'prob':
            counts = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
            mu = self.predict(params, exog=exog, exposure=exposure,
                              offset=offset)[:, None]
            return genpoisson_p.pmf(counts, mu, params[-1],
                                    self.parameterization + 1)
        else:  # pragma: no cover
            raise ValueError('keyword "which" not recognized')


class NegativeBinomial(_CountMixin, CountModel):
    __doc__ = """
    Negative Binomial Model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.

    References
    ----------
    Greene, W. 2008. "Functional forms for the negtive binomial model
        for count data". Economics Letters. Volume 99, Number 3, pp.585-590.
    Hilbe, J.M. 2011. "Negative binomial regression". Cambridge University
        Press.
    """ % {'params': base._model_params_doc,
           'extra_params':
           """loglike_method : string
        Log-likelihood type. 'nb2', 'nb1', or 'geometric'.
        Fitted value :math:`\\mu`
        Heterogeneity parameter :math:`\\alpha`

        - nb2: Variance equal to :math:`\\mu + \\alpha\\mu^2` (most common)
        - nb1: Variance equal to :math:`\\mu + \\alpha\\mu`
        - geometric: Variance equal to :math:`\\mu + \\mu^2`
    offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}
    _check_perfect_pred = None  # placeholder until implemented GH#3895

    @property
    def _res_classes(self):
        return {"fit": (NegativeBinomialResults,
                        NegativeBinomialResultsWrapper),
                "fit_regularized": (L1NegativeBinomialResults,
                                    L1NegativeBinomialResultsWrapper)}

    def __init__(self, endog, exog, loglike_method='nb2', offset=None,
                 exposure=None, missing='none', **kwargs):
        super(NegativeBinomial, self).__init__(endog, exog, offset=offset,
                                               exposure=exposure,
                                               missing=missing, **kwargs)
        self.loglike_method = loglike_method
        self._initialize()
        if loglike_method in ['nb2', 'nb1']:
            self.exog_names.append('alpha')
            self.k_extra = 1
        else:
            self.k_extra = 0
        # store keys for extras if we need to recreate model instance
        # we need to append keys that don't go to super
        self._init_keys.append('loglike_method')

    def _initialize(self):
        if self.loglike_method == 'nb2':
            self.hessian = self._hessian_nb2
            self.score = self._score_nbin
            self.loglikeobs = self._ll_nb2
            self._transparams = True  # transform lnalpha -> alpha in fit
        elif self.loglike_method == 'nb1':
            self.hessian = self._hessian_nb1
            self.score = self._score_nb1
            self.loglikeobs = self._ll_nb1
            self._transparams = True  # transform lnalpha -> alpha in fit
        elif self.loglike_method == 'geometric':
            self.hessian = self._hessian_geom
            self.score = self._score_geom
            self.loglikeobs = self._ll_geometric
        else:  # pragma: no cover
            raise ValueError("Likelihood type must nb1, nb2 or geometric")

    # TODO: Can we move this to the base class?
    def __getstate__(self):
        odict = self.__dict__.copy()  # copy the dict since we change it
        # Workaround to pickle instance methods; see GH#4083
        import types
        # TODO: can we just say `callable(odict[key])`?
        methods = [key for key in odict
                   if isinstance(odict[key], types.MethodType)]
        for key in methods:
            # In this case we need to get rid of hessian, score, and
            # loglikeobs.  The implementation here is more general.
            del odict[key]
        return odict

    def __setstate__(self, indict):
        self.__dict__.update(indict)
        self._initialize()

    # ----------------------------------------------------------------
    # Loglike/Score/Hessian Methods

    def _ll_nbin(self, params, alpha, Q=0):
        if np.any(np.iscomplex(params)) or np.iscomplex(alpha):
            gamma_ln = loggamma
        else:
            gamma_ln = gammaln
        endog = self.endog
        mu = self.predict(params)
        size = 1 / alpha * mu**Q
        prob = size / (size + mu)
        coeff = gamma_ln(size + endog) - gamma_ln(endog + 1) - gamma_ln(size)
        # TODO: cache gamma_ln(endog+1)
        llf = coeff + size * np.log(prob) + endog * np.log(1 - prob)
        return llf

    def _ll_nb2(self, params):
        if self._transparams:  # got lnalpha during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        return self._ll_nbin(params[:-1], alpha, Q=0)

    def _ll_nb1(self, params):
        if self._transparams:  # got lnalpha during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        return self._ll_nbin(params[:-1], alpha, Q=1)

    def _ll_geometric(self, params):
        # we give alpha of 1 because it's actually log(alpha) where alpha=0
        return self._ll_nbin(params, 1, 0)

    def _score_geom(self, params):
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        dparams = self.exog * (y - mu) / (mu + 1)
        return dparams.sum(0)

    def _score_nbin(self, params, Q=0):
        """
        Score vector for NB2 model
        """  # TODO: Is docstring accurate?  NB2?
        if self._transparams:  # lnalpha came in during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]

        a1 = 1 / alpha * mu**Q
        prob = a1 / (a1 + mu)
        dgpart = special.digamma(y + a1) - special.digamma(a1)

        if Q:
            # nb1
            assert Q == 1, Q
            # in this case:
            #    a1 = mu / alpha
            #    prob = 1 / (alpha + 1)
            dparams = exog * a1 * (np.log(prob) + dgpart)
            dalphas = (prob * (y - mu) - a1 * (np.log(prob) + dgpart)) / alpha
            dalpha = dalphas.sum()
        else:
            # nb2
            # in this case:
            #   a1 = 1 / alpha
            #   prob = a1 / (a1 + mu)
            dparams = exog * a1 * (y - mu) / (mu + a1)
            da1 = -alpha**-2
            dalpha = (dgpart + np.log(prob) - (y - mu) / (a1 + mu)).sum() * da1

        # multiply above by constant outside sum to reduce rounding error
        if self._transparams:
            return np.r_[dparams.sum(0), dalpha * alpha]
        else:
            return np.r_[dparams.sum(0), dalpha]

    def _score_nb1(self, params):
        return self._score_nbin(params, Q=1)

    def _hessian_geom(self, params):
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]

        # for dl/dparams dparams
        dim = exog.shape[1]
        hess_arr = np.empty((dim, dim))
        const_arr = mu * (1 + y) / (mu + 1)**2
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i, j] = np.sum(-exog[:, i, None] * exog[:, j, None] *
                                        const_arr, axis=0)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]
        return hess_arr

    def _hessian_nb1(self, params):
        """
        Hessian of NB1 model.
        """
        if self._transparams:  # lnalpha came in during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]

        params = params[:-1]
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]

        a1 = mu / alpha
        prob = 1 / (alpha + 1)  # Note: this equals a1 / (a1 + mu)
        dgpart = special.digamma(y + a1) - special.digamma(a1)
        pgpart = special.polygamma(1, a1 + y) - special.polygamma(1, a1)

        # for dl/dparams dparams
        dim = exog.shape[1]
        hess_arr = np.empty((dim + 1, dim + 1))
        # const_arr = a1*mu*(a1+y)/(mu+a1)**2
        # not all of dparams
        dparams = exog / alpha * (np.log(prob) + dgpart)

        dmudb = exog * mu
        xmu_alpha = exog * a1
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hij = ((dparams[:, i, None] * dmudb[:, j, None]) +
                       (xmu_alpha[:, i, None] *
                        xmu_alpha[:, j, None] *
                        pgpart))
                hess_arr[i, j] = hij.sum(axis=0)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]

        dldpda = np.sum(-a1 * dparams + exog * a1 *
                        (-pgpart * a1 / alpha - prob), axis=0)

        hess_arr[-1, :-1] = dldpda
        hess_arr[:-1, -1] = dldpda

        # for dl/dalpha dalpha

        log_alpha = np.log(prob)
        alpha3 = alpha**3
        alpha2 = alpha**2
        mu2 = mu**2
        dada = ((2 * alpha * (alpha + 1)**2 * mu * (log_alpha + dgpart)
                 + (alpha + 1)**2 * mu2 * pgpart
                 + 3 * alpha3 * mu
                 - 2 * alpha3 * y
                 + alpha2 * (2 * mu - y)
                 ) / (alpha**4 * (alpha + 1)**2))
        hess_arr[-1, -1] = dada.sum()

        return hess_arr

    def _hessian_nb2(self, params):
        """
        Hessian of NB2 model.
        """
        if self._transparams:  # lnalpha came in during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]

        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]

        a1 = 1 / alpha
        prob = a1 / (a1 + mu)
        dgpart = special.digamma(a1 + y) - special.digamma(a1)
        pgpart = special.polygamma(1, a1 + y) - special.polygamma(1, a1)

        # for dl/dparams dparams
        dim = exog.shape[1]
        hess_arr = np.empty((dim + 1, dim + 1))
        const_arr = a1 * mu * (a1 + y) / (mu + a1)**2
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i, j] = np.sum(-exog[:, i, None] * exog[:, j, None] *
                                        const_arr, axis=0)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]

        # for dl/dparams dalpha
        da1 = -alpha**-2
        # assert da1 == -a1**2, (da1, -a1**2)
        # this assertion fails only due to floating point error
        dldpda = np.sum(mu * exog * (y - mu) * da1 / (mu + a1)**2, axis=0)
        hess_arr[-1, :-1] = dldpda
        hess_arr[:-1, -1] = dldpda

        # for dl/dalpha dalpha
        # NOTE: polygamma(1, x) is the trigamma function
        da2 = 2 * alpha**-3
        # assert da2 == 2*a1**3, (da2, 2*a1**3)
        # this assertion fails only due to floating point error
        dalpha = dgpart + np.log(prob) - (y - mu) / (a1 + mu)
        dada = (da2 * dalpha + da1**2 * (pgpart + 1 / a1 - 1 / (a1 + mu) +
                (y - mu) / (mu + a1)**2)).sum()
        hess_arr[-1, -1] = dada

        return hess_arr

    # TODO: replace this with analytic where is it used?
    def score_obs(self, params):
        return approx_fprime_cs(params, self.loglikeobs)

    # ----------------------------------------------------------------

    def _estimate_dispersion(self, mu, resid, df_resid=None):
        if df_resid is None:
            df_resid = resid.shape[0]
        if self.loglike_method == 'nb2':
            a = ((resid**2 / mu - 1) / mu).sum() / df_resid
        else:
            # i.e. self.loglike_method == 'nb1':
            a = (resid**2 / mu - 1).sum() / df_resid
        return a

    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):

        # Note: don't let super handle robust covariance because it has
        # transformed params
        self._transparams = False  # always define attribute
        if self.loglike_method.startswith('nb') and method not in ['newton',
                                                                   'ncg']:
            self._transparams = True  # in case same Model instance is refit

        if start_params is not None and self._transparams:
            # Note: we cannot do this in `_get_start_params` or it risks
            # being done repeatedly
            # transform user provided start_params dispersion, see GH#3918
            start_params = np.array(start_params, copy=True)
            start_params[-1] = np.log(start_params[-1])

        start_params = self._get_start_params(start_params, **kwargs)

        # TODO: can we skip CountModel and go straight to DiscreteModel?
        # Yes, just need to change "mlefit._results" --> "mlefit" below
        mlefit = CountModel.fit(
            self, start_params=start_params, maxiter=maxiter, method=method,
            disp=disp, full_output=full_output, callback=callback,
            cov_type=cov_type, use_t=use_t, cov_kwds=cov_kwds, **kwargs)
        # TODO: Fix NBin _check_perfect_pred

        res_cls, wrap_cls = self._res_classes["fit"]
        if self._transparams:
            # mlefit is a wrapped counts results
            self._transparams = False  # don't need to transform anymore now

            # change from lnalpha to alpha
            mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])
            # ensure cov_params are re-evaluated with updated params
            delattr(mlefit._results, "cov_type")

            nbinfit = res_cls(self, mlefit._results,
                              cov_type=cov_type, use_t=use_t, cov_kwds=cov_kwds)
            result = wrap_cls(nbinfit)
        else:
            result = mlefit  # TODO: Shouldn't this be wrapped?
            # FIXME: result.bse -->
            # ValueError: cannot reshape array of size 1 into shape (2,)
            # TODO: can we avoid doing this here?
            cov_kwds = cov_kwds or {}
            # FIXME: What's the point of re-calling _get_robutcov_results?
            #  normalized_cov_params already exists and is unchanged after
            #  re-calling
            result._get_robustcov_results(cov_type=cov_type,
                                          use_self=True, use_t=use_t,
                                          **cov_kwds)
            # FIXME: Should this return CountResultsWrapper?  GH#4529
        return result


class NegativeBinomialP(_CountMixin, CountModel):
    __doc__ = """
    Generalized Negative Binomial (NB-P) model for count data
    %(params)s
    %(extra_params)s
    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    p : scalar
        P denotes parameterizations for NB-P regression. p=1 for NB-1 and
        p=2 for NB-2. Default is p=1.
    """ % {'params': base._model_params_doc,
           'extra_params': """p: scalar
        P denotes parameterizations for NB regression. p=1 for NB-1 and
        p=2 for NB-2. Default is p=2.
    offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.
    """ + base._missing_param_doc}
    _check_perfect_pred = None  # placeholder until implemented GH#3895

    @property
    def _res_classes(self):
        return {"fit": (NegativeBinomialResults,
                        NegativeBinomialResultsWrapper),
                "fit_regularized": (L1NegativeBinomialResults,
                                    L1NegativeBinomialResultsWrapper)}

    def __init__(self, endog, exog, p=2, offset=None,
                 exposure=None, missing='none', **kwargs):
        super(NegativeBinomialP, self).__init__(endog, exog, offset=offset,
                                                exposure=exposure,
                                                missing=missing, **kwargs)
        self.parameterization = p
        self.exog_names.append('alpha')
        self.k_extra = 1
        self._transparams = False

    def _get_init_kwds(self):
        kwds = super(NegativeBinomialP, self)._get_init_kwds()
        kwds['p'] = self.parameterization
        return kwds

    # TODO: This (and score_obs, hessian) is pretty slow.  can it be optimized?
    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generalized Negative
        Binomial (NB-P) model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]

        params = params[:-1]
        p = self.parameterization
        y = self.endog

        mu = self.predict(params)
        mu_p = mu**(2 - p)
        a1 = mu_p / alpha
        a2 = mu + a1

        llf = (gammaln(y + a1) - gammaln(y + 1) - gammaln(a1) +
               a1 * np.log(a1 / a2) + y * np.log(mu / a2))

        return llf

    def score_obs(self, params):
        """
        Generalized Negative Binomial (NB-P) model score (gradient)
        vector of the log-likelihood for each observations.

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]

        params = params[:-1]
        p = 2 - self.parameterization
        y = self.endog

        mu = self.predict(params)
        mu_p = mu**p
        a1 = mu_p / alpha
        a2 = mu + a1
        a3 = y + a1
        a4 = a1 * p / mu

        dgpart = special.digamma(y + a1) - special.digamma(a1)
        dgterm = np.log(a1 / a2) + dgpart + 1 - a3 / a2

        dparams = (a4 * dgterm -
                   a3 / a2 +
                   y / mu)
        dparams = (self.exog.T * mu * dparams).T
        dalpha = -a1 / alpha * dgterm

        return np.concatenate((dparams, np.atleast_2d(dalpha).T),
                              axis=1)

    def score(self, params):
        """
        Generalized Negative Binomial (NB-P) model score (gradient)
        vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        score = np.sum(self.score_obs(params), axis=0)
        if self._transparams:
            score[-1] == score[-1] ** 2
            return score
        else:
            return score

    def hessian(self, params):
        """
        Generalized Negative Binomial (NB-P) model hessian maxtrix
        of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hessian : ndarray, 2-D
            The hessian matrix of the model.
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]

        p = 2 - self.parameterization
        y = self.endog
        exog = self.exog
        mu = self.predict(params)

        mu_p = mu**p
        a1 = mu_p / alpha  # AKA size
        a2 = mu + a1
        a3 = y + a1
        a4 = a1 * p / mu

        dim = exog.shape[1]
        hess_arr = np.zeros((dim + 1, dim + 1))

        dgpart = special.digamma(y + a1) - special.digamma(a1)
        pgpart = special.polygamma(1, a1) - special.polygamma(1, y + a1)
        dgterm = np.log(a1 / a2) + dgpart + 1 - a3 / a2
        # TODO: better name or interpretation for dgterm?  (ditto elsewhere)

        coeff = mu**2 * ((1 + a4)**2 * a3 / a2**2
                         - 2 * (p / mu) * (1 + a4) * a1 / a2
                         + a1 * (p / mu)**2 * (dgterm + 1)
                         - a1**2 * (p / mu)**2 * pgpart
                         - a3 / a2 / mu)

        for i in range(dim):
            hess_arr[i, :-1] = np.sum(exog[:, :].T * exog[:, i] * coeff,
                                      axis=1)

        hess_arr[-1, :-1] = (exog[:, :].T * mu * a1 *
                             ((1 + a4) * (1 - a3 / a2) / a2
                              - (p / mu) * (dgterm + 1)
                              + p * a4 / a2
                              + a4 * pgpart
                              ) / alpha).sum(axis=1)

        da2 = (a1 * (2 * dgterm
                     + 1
                     - a1 * pgpart
                     - 2 * a1 / a2
                     + (a1 / a2) * (a3 / a2)
                     ) / alpha**2)

        hess_arr[-1, -1] = da2.sum()

        tri_idx = np.triu_indices(dim + 1, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]

        return hess_arr

    # --------------------------------------------------------------

    def _estimate_dispersion(self, mu, resid, df_resid=None):
        q = self.parameterization - 1
        if df_resid is None:
            df_resid = resid.shape[0]
        a = ((resid**2 / mu - 1) * mu**(-q)).sum() / df_resid
        return a

    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None, use_transparams=False,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        """
        Parameters
        ----------
        use_transparams : bool
            This parameter enable internal transformation to impose
            non-negativity.  True to enable. Default is False.
            use_transparams=True imposes the no underdispersion (alpha > 0)
            constaint.  In case use_transparams=True and method="newton"
            or "ncg" transformation is ignored.
        """
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = True
        else:
            if use_transparams:
                warnings.warn('Parameter "use_transparams" is ignored',
                              RuntimeWarning)
            self._transparams = False

        start_params = self._get_start_params(start_params, **kwargs)

        # TODO: can we skip CountModel and go straight to DiscreteModel?
        mlefit = CountModel.fit(
            self, start_params=start_params, maxiter=maxiter, method=method,
            disp=disp, full_output=full_output, callback=callback,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
            **kwargs)

        if self._transparams:
            self._transparams = False
            mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])
            # ensure that cov_params is re-evaluated with updated params
            delattr(mlefit._results, "cov_type")

        res_class, wrap_cls = self._res_classes["fit"]
        nbinfit = res_class(self, mlefit._results,
                            cov_type=cov_type, use_t=use_t, cov_kwds=cov_kwds)
        return wrap_cls(nbinfit)

    fit.__doc__ += DiscreteModel.fit.__doc__

    def predict(self, params, exog=None, exposure=None, offset=None,
                which='mean'):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array-like
            2d array of fitted parameters of the model. Should be in the
            order returned from the model.
        exog : array-like, optional
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used. If a 1d array is given
            it assumed to be 1 row of exogenous variables. If you only have
            one regressor and would like to do prediction, you must provide
            a 2d array with shape[1] == 1.
        linear : bool, optional
            If True, returns the linear predictor dot(exog, params).  Else,
            returns the value of the cdf at the linear predictor.
        offset : array_like, optional
            Offset is added to the linear prediction with coefficient
            equal to 1.
        exposure : array_like, optional
            Log(exposure) is added to the linear prediction with coefficient
        equal to 1.
        which : 'mean', 'linear', 'prob', optional.
            'mean' returns the exp of linear predictor exp(dot(exog, params)).
            'linear' returns the linear predictor dot(exog, params).
            'prob' return probabilities for counts from 0 to max(endog).
            Default is 'mean'.
        """
        if exog is None:
            exog = self.exog

        if exposure is None:
            exposure = getattr(self, 'exposure', 0)
        elif exposure != 0:
            exposure = np.log(exposure)

        if offset is None:
            offset = getattr(self, 'offset', 0)

        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset

        if which == 'mean':
            return np.exp(linpred)
        elif which == 'linear':
            return linpred
        elif which == 'prob':
            counts = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
            mu = self.predict(params, exog, exposure, offset)
            size, prob = self.convert_params(params, mu)
            return stats.nbinom.pmf(counts, size[:, None], prob[:, None])
        else:  # pragma: no cover
            raise ValueError('keyword "which" = %s not recognized' % which)

    def convert_params(self, params, mu):  # TODO: use this more?  privatize?
        alpha = params[-1]
        p = 2 - self.parameterization

        size = 1. / alpha * mu**p
        prob = size / (size + mu)
        return (size, prob)


# ----------------------------------------------------------------

class Logit(BinaryModel):
    __doc__ = """
    Binary choice Logit model

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params': base._model_params_doc,
           'extra_params': base._missing_param_doc}

    @property
    def _res_classes(self):
        return {"fit": (LogitResults, BinaryResultsWrapper),
                "fit_regularized": (L1BinaryResults, L1BinaryResultsWrapper)}

    def cdf(self, X):
        r"""
        The logistic cumulative distribution function

        Parameters
        ----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.

        Returns
        -------
        1/(1 + exp(-X))

        Notes
        ------
        In the logit model,

        .. math:: \Lambda\left(x^{\prime}\beta\right)
            =\text{Prob}\left(Y=1|x\right)
            =\frac{e^{x^{\prime}\beta}}{1+e^{x^{\prime}\beta}}
        """
        X = np.asarray(X)
        return 1 / (1 + np.exp(-X))

    def pdf(self, X):
        r"""
        The logistic probability density function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.

        Returns
        -------
        pdf : ndarray
            The value of the Logit probability mass function, PMF, for each
            point of X. ``np.exp(-x)/(1+np.exp(-X))**2``

        Notes
        -----
        In the logit model,

        .. math:: \lambda\left(x^{\prime}\beta\right)
            =\frac{e^{-x^{\prime}\beta}}{\left(1+e^{-x^{\prime}\beta}\right)^{2}}
        """  # noqa:E501
        X = np.asarray(X)
        negexp = np.exp(-X)
        return negexp / (1 + negexp)**2

    def loglikeobs(self, params):
        """
        Log-likelihood of logit model for each observation.

        Parameters
        -----------
        params : array-like
            The parameters of the logit model.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        ------
        .. math:: \\ln L
            =\\sum_{i}\\ln\\Lambda\\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        for observations :math:`i=1, ..., n`

        where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        q = 2 * self.endog - 1
        Xb = np.dot(self.exog, params)
        prob = self.cdf(q * Xb)
        return np.log(prob)

    def score(self, params):
        """
        Logit model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params: array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}
            =\\sum_{i=1}^{n}\\left(y_{i}-\\Lambda_{i}\\right)x_{i}
        """
        Xb = np.dot(self.exog, params)
        prob = self.cdf(Xb)
        return np.dot(self.endog - prob, self.exog)
        # Note: this is non-trivially more performant than wrapping score_obs

    def score_obs(self, params):
        """
        Logit model Jacobian of the log-likelihood for each observation

        Parameters
        ----------
        params: array-like
            The parameters of the model

        Returns
        -------
        jac : array-like
            The derivative of the loglikelihood for each observation evaluated
            at `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}
            =\\left(y_{i}-\\Lambda_{i}\\right)x_{i}

        for observations :math:`i=1, ..., n`
        """
        Xb = np.dot(self.exog, params)
        prob = self.cdf(Xb)
        return (self.endog - prob)[:, None] * self.exog

    def hessian(self, params):
        r"""
        Logit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \frac{\partial^{2}\ln L}{\partial\beta\partial\beta^{\prime}}
            =-\sum_{i}\Lambda_{i}\left(1-\Lambda_{i}\right)x_{i}x_{i}^{\prime}
        """
        Xb = np.dot(self.exog, params)
        prob = self.cdf(Xb)
        return -np.dot(prob * (1 - prob) * self.exog.T, self.exog)


class Probit(BinaryModel):
    __doc__ = Logit.__doc__.replace("Logit", "Probit")

    @property
    def _res_classes(self):
        return {"fit": (ProbitResults, BinaryResultsWrapper),
                "fit_regularized": (L1BinaryResults, L1BinaryResultsWrapper)}

    cdf = stats.norm._cdf
    pdf = stats.norm._pdf

    def loglikeobs(self, params):
        r"""
        Log-likelihood of probit model for each observation

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : array-like
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----
        .. math:: \ln L_{i}=\ln\Phi\left(q_{i}x_{i}^{\prime}\beta\right)

        for observations :math:`i=1, ..., n`

        where :math:`q=2y-1`. This simplification comes from the fact that the
        normal distribution is symmetric.
        """
        q = 2 * self.endog - 1
        Xb = np.dot(self.exog, params)
        prob = self.cdf(q * Xb)
        return np.log(np.clip(prob, FLOAT_EPS, 1))

    def score(self, params):
        r"""
        Probit model score (gradient) vector

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \frac{\partial\ln L}{\partial\beta}
            =\sum_{i=1}^{n}
                \left[\frac{q_{i}\phi\left(q_{i}x_{i}^{\prime}\beta\right)}*
                {\Phi\left(q_{i}x_{i}^{\prime}\beta\right)}\right]x_{i}

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        normal distribution is symmetric.
        """
        q = 2 * self.endog - 1
        Xb = np.dot(self.exog, params)
        prob = self.cdf(q * Xb)
        # clip to get rid of invalid divide complaint
        L = q * self.pdf(q * Xb) / np.clip(prob,
                                           FLOAT_EPS, 1 - FLOAT_EPS)
        return np.dot(L, self.exog)
        # Note: this is non-trivially more performant than wrapping score_obs

    def score_obs(self, params):
        r"""
        Probit model Jacobian for each observation

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        jac : array-like
            The derivative of the loglikelihood for each observation evaluated
            at `params`.

        Notes
        -----
        .. math:: \frac{\partial\ln L_{i}}{\partial\beta}
            =\left[\frac{q_{i}\phi\left(q_{i}x_{i}^{\prime}\beta\right)}*
                {\Phi\left(q_{i}x_{i}^{\prime}\beta\right)}\right]x_{i}

        for observations :math:`i=1, ..., n`

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        normal distribution is symmetric.
        """
        Xb = np.dot(self.exog, params)
        q = 2 * self.endog - 1
        prob = self.cdf(q * Xb)
        # clip to get rid of invalid divide complaint
        L = q * self.pdf(q * Xb) / np.clip(prob,
                                           FLOAT_EPS, 1 - FLOAT_EPS)
        return L[:, None] * self.exog

    def hessian(self, params):
        r"""
        Probit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \frac{\partial^{2}\ln L}{\partial\beta\partial\beta^{\prime}}
            =-\lambda_{i}*
                \left(\lambda_{i}+x_{i}^{\prime}\beta\right)x_{i}x_{i}^{\prime}

        where

        .. math:: \lambda_{i}
            =\frac{q_{i}\phi\left(q_{i}x_{i}^{\prime}\beta\right)}*
                {\Phi\left(q_{i}x_{i}^{\prime}\beta\right)}

        and :math:`q=2y-1`
        """
        Xb = np.dot(self.exog, params)
        q = 2 * self.endog - 1
        prob = self.cdf(q * Xb)
        L = q * self.pdf(q * Xb) / prob
        return np.dot(-L * (L + Xb) * self.exog.T, self.exog)


class MNLogit(MultinomialModel):
    __doc__ = """
    Multinomial logit model

    Parameters
    ----------
    endog : array-like
        `endog` is an 1-d vector of the endogenous response.  `endog` can
        contain strings, ints, or floats.  Note that if it contains strings,
        every distinct string will be a category.  No stripping of whitespace
        is done.
    exog : array-like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See `sm2.tools.add_constant`.
    %(extra_params)s

    Attributes
    ----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    J : float
        The number of choices for the endogenous variable. Note that this
        is zero-indexed.
    K : float
        The actual number of parameters for the exogenous design.  Includes
        the constant if the design has one.
    names : dict
        A dictionary mapping the column number in `wendog` to the variables
        in `endog`.
    wendog : array
        An n x j array where j is the number of unique categories in `endog`.
        Each column of j is a dummy variable indicating the category of
        each observation. See `names` for a dictionary mapping each column to
        its category.

    Notes
    -----
    See developer notes for further information on `MNLogit` internals.
    """ % {'extra_params': base._missing_param_doc}

    def pdf(self, X, dropfirst=False, submax=False):
        r"""
        We take a derivative of `cdf` using the quotient
        rule: (f'g - g'f) / g^2
        Here "g" is `denom` and "f" is `eXB`

        For each row, this derivative will be a nvars x nvars array, with
        the first dimension representing the coordinate of `cdf` being
        differentiated and the second dimension representing the variable
        doing the differentiation [AWK: how to phrase the last sentence?]

        i.e. row[:, i, j] == \frac{\partial cdf[:, i]}{\partial X[:, j]}

        See tests.test_multinomial.mnlogit_pdf for an alternative (slower)
        non-loop implementation.
        """
        drop = int(dropfirst)

        nobs = len(X)
        XB = np.column_stack((np.zeros(nobs), X))
        if submax:
            XB -= XB.max(1)[:, None]
            # TODO: benchmark how this affects speed

        eXB = np.exp(XB)
        denom = eXB.sum(1)
        prob = eXB / denom[:, None]

        J = self.J

        mat_partials = [[None for idx1 in range(J - drop)]
                        for idx2 in range(J - drop)]
        for idx in range(J - drop):
            for jdx in range(idx, J - drop):
                mpartial = prob[:, idx + drop] * prob[:, jdx + drop]
                if idx == jdx:
                    mpartial -= prob[:, idx + drop]
                    # An option is to avoid multiplying by `denom` here
                    # because we will just re-divide by it later, but that
                    # appears to be slower than just doing it here.

                mat_partials[idx][jdx] = mpartial
                mat_partials[jdx][idx] = mpartial
                # Exploit symmetry to cut down on iterations

        deriv = -np.asarray(mat_partials).T
        return deriv

    def cdf(self, X):
        r"""
        Multinomial logit cumulative distribution function.

        Parameters
        ----------
        X : array
            The linear predictor of the model XB.

        Returns
        --------
        cdf : ndarray
            The cdf evaluated at `X`.

        Notes
        -----
        In the multinomial logit model.
        .. math:: \frac{\exp(\beta_{j}^{\prime}x_{i})}{\sum_{k=0}^{J}\exp(\beta_{k}^{\prime}x_{i})}
        """  # noqa:E501
        # subtract max to avoid overflow, see GH#3778
        Xb = np.column_stack((np.zeros(len(X)), X))
        Xb -= Xb.max(1).reshape(-1, 1)
        eXB = np.exp(Xb)
        denom = eXB.sum(1)[:, None]
        return eXB / denom

    def loglikeobs(self, params):
        r"""
        Log-likelihood of the multinomial logit model for each observation.

        Parameters
        ----------
        params : array-like
            The parameters of the multinomial logit model.

        Returns
        -------
        loglike : array-like
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        ------
        .. math:: \ln L_{i}
            =\sum_{j=0}^{J}d_{ij}\ln\left(\frac{\exp\left(\beta_{j}^{\prime}x_{i}\right)}{\sum_{k=0}^{J}\exp\left(\beta_{k}^{\prime}x_{i}\right)}\right)

        for observations :math:`i=1, ..., n`

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """  # noqa:E501
        params = params.reshape(self.K, -1, order='F')
        Xb = np.dot(self.exog, params)
        prob = self.cdf(Xb)
        logprob = np.log(prob)
        out = self.wendog * logprob
        if np.isnan(out).any():
            # TODO: More efficient way to do this check?
            # GH#3778 We can avoid some NaN issues by convention
            # that 0 * np.inf = 0
            mask = self.wendog == 0
            out[mask] = 0
        return out

    def score(self, params):
        r"""
        Score matrix for multinomial logit model log-likelihood

        Parameters
        ----------
        params : array
            The parameters of the multinomial logit model.

        Returns
        --------
        score : ndarray, (K * (J-1))
            The 2-d score vector, i.e. the first derivative of the
            loglikelihood function, of the multinomial logit model evaluated at
            `params`.

        Notes
        -----
        .. math:: \frac{\partial\ln L}{\partial\beta_{j}}
            =\sum_{i}\left(d_{ij}-\frac{\exp\left(\beta_{j}^{\prime}x_{i}\right)}{\sum_{k=0}^{J}\exp\left(\beta_{k}^{\prime}x_{i}\right)}\right)x_{i}

        for :math:`j=1, ..., J`

        In the multinomial model the score matrix is K x J-1 but is returned
        as a flattened array to work with the solvers.
        """  # noqa:E501
        params = params.reshape(self.K, -1, order='F')
        Xb = np.dot(self.exog, params)
        wresid = self.wendog[:, 1:] - self.cdf(Xb)[:, 1:]
        # NOTE: might need to switch terms if params is reshaped
        return np.dot(wresid.T, self.exog).flatten()
        # Note: this is non-trivially more performant than wrapping score_obs

    def loglike_and_score(self, params):  # TODO: Is this needed/used?
        """
        Returns log likelihood and score, efficiently reusing calculations.

        Note that both of these returned quantities will need to be negated
        before being minimized by the maximum likelihood fitting machinery.
        """
        params = params.reshape(self.K, -1, order='F')
        Xb = np.dot(self.exog, params)
        prob = self.cdf(Xb)
        loglike_value = np.sum(self.wendog * np.log(prob))
        wresid = self.wendog[:, 1:] - prob[:, 1:]
        score_array = np.dot(wresid.T, self.exog).flatten()
        return loglike_value, score_array

    def score_obs(self, params):
        r"""
        Jacobian matrix for multinomial logit model log-likelihood

        Parameters
        ----------
        params : array
            The parameters of the multinomial logit model.

        Returns
        --------
        jac : array-like
            The derivative of the loglikelihood for each observation evaluated
            at `params` .

        Notes
        -----
        .. math:: \frac{\partial\ln L_{i}}{\partial\beta_{j}}
            =\left(d_{ij}-\frac{\exp(\beta_{j}^{\prime}x_{i})}{\sum_{k=0}^{J}\exp(\beta_{k}^{\prime}x_{i})}\right)x_{i}

        for :math:`j=1, ..., J`, for observations :math:`i=1, ..., n`

        In the multinomial model the score vector is K x (J-1) but is returned
        as a flattened array. The Jacobian has the observations in rows and
        the flatteded array of derivatives in columns.
        """  # noqa:E501
        params = params.reshape(self.K, -1, order='F')
        Xb = np.dot(self.exog, params)
        wresid = self.wendog[:, 1:] - self.cdf(Xb)[:, 1:]
        # NOTE: might need to switch terms if params is reshaped
        return (wresid[:, :, None] *
                self.exog[:, None, :]).reshape(self.nobs, -1)

    def hessian(self, params):
        r"""
        Multinomial logit Hessian matrix of the log-likelihood

        Parameters
        -----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (J*K, J*K)
            The Hessian, second derivative of loglikelihood function with
            respect to the flattened parameters, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\partial^{2}\ln L}{\partial\beta_{j}\partial\beta_{l}}
            =-\sum_{i=1}^{n}\frac{\exp\left(\beta_{j}^{\prime}x_{i}\right)}{\sum_{k=0}^{J}\exp\left(\beta_{k}^{\prime}x_{i}\right)}\left[\boldsymbol{1}\left(j=l\right)-\frac{\exp\left(\beta_{l}^{\prime}x_{i}\right)}{\sum_{k=0}^{J}\exp\left(\beta_{k}^{\prime}x_{i}\right)}\right]x_{i}x_{l}^{\prime}

        where
        :math:`\boldsymbol{1}\left(j=l\right)` equals 1 if `j` = `l` and 0
        otherwise.

        The actual Hessian matrix has J**2 * K x K elements. Our Hessian
        is reshaped to be square (J*K, J*K) so that the solvers can use it.

        This implementation does not take advantage of the symmetry of
        the Hessian and could probably be refactored for speed.
        """  # noqa:E501
        params = params.reshape(self.K, -1, order='F')
        X = self.exog
        Xb = np.dot(self.exog, params)
        pr = self.cdf(Xb)
        J = self.J
        K = self.K
        partials = [[None for i in range(J - 1)] for j in range(J - 1)]
        for i in range(J - 1):
            for j in range(i, J - 1):
                # this loop assumes we drop the first col.
                if i == j:
                    part = ((pr[:, i + 1] * (1 - pr[:, j + 1]))[:, None] * X).T
                else:
                    part = ((pr[:, i + 1] * -pr[:, j + 1])[:, None] * X).T
                partials[i][j] = -np.dot(part, X)
                partials[j][i] = partials[i][j]

        H = np.array(partials)
        # the developer's notes on multinomial should clear this math up
        H = H.reshape(J - 1, J - 1, K, K)
        H = np.transpose(H, (0, 2, 1, 3))
        H = H.reshape((J - 1) * K, (J - 1) * K)
        return H


# ----------------------------------------------------------------
# Results Classes

class DiscreteResults(base.LikelihoodModelResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for the discrete dependent "
                                "variable models.",
        "extra_attr": ""}

    def __init__(self, model, mlefit, cov_type='nonrobust', cov_kwds=None,
                 use_t=None):
        # NB: does _not_ call super(...).__init__
        self.model = model
        try:
            self.df_model = model.df_model
            self.df_resid = model.df_resid
        except AttributeError:
            # In some subclasses, df_model and df_resid are properties
            # TODO: standardize this behavior
            pass

        self._cache = resettable_cache()
        self.nobs = model.exog.shape[0]  # i.e. model.nobs
        self.__dict__.update(mlefit.__dict__)  # TODO: dont do this

        if not hasattr(self, 'cov_type'):
            # do this only if super, i.e. mlefit didn't already add cov_type
            # robust covariance
            if use_t is not None:
                self.use_t = use_t

            cov_kwds = cov_kwds or {}
            self._get_robustcov_results(cov_type=cov_type, use_self=True,
                                        **cov_kwds)

    def __getstate__(self):
        # remove unpicklable methods
        mle_settings = getattr(self, 'mle_settings', None)
        if mle_settings is not None:
            # `callback` and `cov_params_func` are the most likely culprits
            # See GH#4083
            methods = [key for key in mle_settings
                       if callable(mle_settings[key])]
            for key in methods:
                mle_settings[key] = None
            # TODO: move this higher up the class hierarchy?
        return self.__dict__

    @cached_value
    def prsquared(self):
        return 1 - self.llf / self.llnull

    @cached_value
    def llr(self):
        return -2 * (self.llnull - self.llf)

    @cached_value
    def llr_pvalue(self):
        return stats.distributions.chi2.sf(self.llr, self.df_model)

    def set_null_options(self, llnull=None, attach_results=True, **kwds):
        """set fit options for Null (constant-only) model

        This resets the cache for related attributes which is potentially
        fragile. This only sets the option, the null model is estimated
        when llnull is accessed, if llnull is not yet in cache.

        Parameters
        ----------
        llnull : None or float
            If llnull is not None, then the value will be directly assigned to
            the cached attribute "llnull".
        attach_results : bool
            Sets an internal flag whether the results instance of the null
            model should be attached. By default without calling this method,
            thenull model results are not attached and only the loglikelihood
            value llnull is stored.
        kwds : keyword arguments
            `kwds` are directly used as fit keyword arguments for the null
            model, overriding any provided defaults.

        Returns
        -------
        no returns, modifies attributes of this instance
        """
        # reset cache, note we need to add here anything that depends on
        # llnullor the null model. If something is missing, then the attribute
        # might be incorrect.
        self._cache.pop('llnull', None)
        self._cache.pop('llr', None)
        self._cache.pop('llr_pvalue', None)
        self._cache.pop('prsquared', None)
        if hasattr(self, 'res_null'):
            del self.res_null

        if llnull is not None:
            self._cache['llnull'] = llnull
        self._attach_nullmodel = attach_results
        self._optim_kwds_null = kwds

    @cached_value
    def llnull(self):
        # TODO: slow, 1.086 seconds per call in tests
        model = self.model
        kwds = model._get_init_kwds().copy()
        for key in getattr(model, '_null_drop_keys', []):
            del kwds[key]

        # TODO: what parameters to pass to fit?
        mod_null = model.__class__(model.endog, np.ones(self.nobs), **kwds)
        # TODO: consider catching and warning on convergence failure?
        # in the meantime, try hard to converge. see
        # TestPoissonConstrained1a.test_smoke

        optim_kwds = getattr(self, '_optim_kwds_null', {}).copy()

        if 'start_params' in optim_kwds:
            # user provided
            sp_null = optim_kwds.pop('start_params')
        elif hasattr(model, '_get_start_params_null'):
            # get moment estimates if available
            sp_null = model._get_start_params_null()
        else:
            sp_null = None

        opt_kwds = dict(method='bfgs',
                        warn_convergence=False,
                        maxiter=10000,
                        disp=0)
        opt_kwds.update(optim_kwds)

        if optim_kwds:
            res_null = mod_null.fit(start_params=sp_null, **opt_kwds)
        else:
            # this should be a reasonable method case across versions
            res_null = mod_null.fit(start_params=sp_null, method='nm',
                                    warn_convergence=False,
                                    maxiter=10000, disp=0)
            res_null = mod_null.fit(start_params=res_null.params,
                                    method='bfgs',
                                    warn_convergence=False,
                                    maxiter=10000, disp=0)
            # TODO: Why is maxiter=10000 here reasonable while
            # elsewhere its 35?

        if getattr(self, '_attach_nullmodel', False) is not False:
            self.res_null = res_null

        return res_null.llf

    @property
    def resid(self):
        # GH#5255
        return self.resid_response

    @cache_readonly
    def resid_response(self):
        """
        The response residuals

        Notes
        -----
        Response residuals are defined to be

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`.
        """
        return self.model.endog - self.fittedvalues

    @cached_value
    def aic(self):
        return -2 * (self.llf - (self.df_model + 1))

    @cached_value
    def bic(self):
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + 1)

    def _get_endog_name(self, yname, yname_list):
        if yname is None:
            yname = self.model.endog_names
        if yname_list is None:
            yname_list = self.model.endog_names
        return yname, yname_list

    def get_margeff(self, at='overall', method='dydx', atexog=None,
                    dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available from the returned object.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semielasticity -- dy/d(lnx)
            - 'eydx' - estimate semeilasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables.
        atexog : array-like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        DiscreteMargins : marginal effects instance
            Returns an object that holds the marginal effects, standard
            errors, confidence intervals, etc. See
            `sm2.discrete.discrete_margins.DiscreteMargins` for more
            information.

        Notes
        -----
        When using after Poisson, returns the expected number of events
        per period, assuming that the model is loglinear.
        """
        from sm2.discrete.discrete_margins import DiscreteMargins
        return DiscreteMargins(self, (at, method, atexog, dummy, count))

    @copy_doc(base.Results.summary.__doc__)
    def summary(self, yname=None, xname=None, title=None, alpha=.05,
                yname_list=None):
        top_left = [('Dep. Variable:', None),
                    ('Model:', [self.model.__class__.__name__]),
                    ('Method:', ['MLE']),
                    ('Date:', None),
                    ('Time:', None),
                    ('converged:', ["%s" % self.mle_retvals['converged']])]

        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ('Pseudo R-squ.:', ["%#6.4g" % self.prsquared]),
                     ('Log-Likelihood:', None),
                     ('LL-Null:', ["%#8.5g" % self.llnull]),
                     ('LLR p-value:', ["%#6.4g" % self.llr_pvalue])]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        # boiler plate
        from sm2.iolib.summary import Summary
        smry = Summary()
        yname, yname_list = self._get_endog_name(yname, yname_list)
        # for top of table
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)
        # for parameters, etc
        smry.add_table_params(self, yname=yname_list, xname=xname, alpha=alpha,
                              use_t=self.use_t)

        if hasattr(self, 'constraints'):
            smry.add_extra_txt(['Model has been estimated subject to linear '
                                'equality constraints.'])
        return smry


class CountResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for count data",
        "extra_attr": ""}

    @cached_data
    def resid(self):
        """
        Notes
        -----
        The residuals for Count models are defined as

        .. math:: y - p

        where :math:`p = \\exp(X\\beta)`. Any exposure and offset variables
        are also handled.
        """
        return self.model.endog - self.predict()


class NegativeBinomialResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for NegativeBinomial 1 and 2",
        "extra_attr": ""}

    @cached_value
    def lnalpha(self):
        return np.log(self.params[-1])

    @cached_value
    def lnalpha_std_err(self):
        return self.bse[-1] / self.params[-1]

    @cached_value
    def aic(self):
        # + 1 because we estimate alpha
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * (self.llf - (self.df_model + self.k_constant + k_extra))

    @cached_value
    def bic(self):
        # + 1 because we estimate alpha
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * self.llf + np.log(self.nobs) * (self.df_model +
                                                    self.k_constant + k_extra)


class GeneralizedPoissonResults(NegativeBinomialResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Generalized Poisson",
        "extra_attr": ""}

    @cache_readonly
    def _dispersion_factor(self):
        p = getattr(self.model, 'parameterization', 0)
        mu = self.predict()
        return (1 + self.params[-1] * mu**p)**2


class PoissonResults(CountResults):
    def predict_prob(self, n=None, exog=None, exposure=None, offset=None,
                     transform=True):
        """
        Return predicted probability of each count level for each observation

        Parameters
        ----------
        n : array-like or int
            The counts for which you want the probabilities. If n is None
            then the probabilities for each count from 0 to max(y) are
            given.

        Returns
        -------
        ndarray
            A nobs x n array where len(`n`) columns are indexed by the count
            n. If n is None, then column 0 is the probability that each
            observation is 0, column 1 is the probability that each
            observation is 1, etc.
        """
        if n is not None:
            counts = np.atleast_2d(n)
        else:
            counts = np.atleast_2d(np.arange(0, np.max(self.model.endog) + 1))
        mu = self.predict(exog=exog, exposure=exposure, offset=offset,
                          transform=transform, linear=False)[:, None]
        # uses broadcasting
        return stats.poisson.pmf(counts, mu)


class OrderedResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for ordered discrete data.",
        "extra_attr": ""}


class BinaryResults(DiscreteResults):
    __doc__ = OrderedResults.__doc__.replace("ordered discrete", "binary")

    def pred_table(self, threshold=.5):
        """
        Prediction table

        Parameters
        ----------
        threshold : scalar
            Number between 0 and 1. Threshold above which a prediction is
            considered 1 and below which a prediction is considered 0.

        Notes
        ------
        pred_table[i, j] refers to the number of times "i" was observed and
        the model predicted "j". Correct predictions are along the diagonal.
        """
        model = self.model
        actual = model.endog
        pred = np.array(self.predict() > threshold, dtype=float)
        bins = np.array([0, 0.5, 1])
        return np.histogram2d(actual, pred, bins=bins)[0]

    @copy_doc(DiscreteResults.summary.__doc__)
    def summary(self, yname=None, xname=None, title=None, alpha=.05,
                yname_list=None):
        smry = super(BinaryResults, self).summary(yname, xname, title, alpha,
                                                  yname_list)
        fittedvalues = self.model.cdf(self.fittedvalues)
        absprederror = np.abs(self.model.endog - fittedvalues)
        predclose_sum = (absprederror < 1e-4).sum()
        predclose_frac = predclose_sum / len(fittedvalues)

        # add warnings/notes
        etext = []
        if predclose_sum == len(fittedvalues):  # nobs?
            wstr = "Complete Separation: The results show that there is"
            wstr += "complete separation.\n"
            wstr += "In this case the Maximum Likelihood Estimator does "
            wstr += "not exist and the parameters\n"
            wstr += "are not identified."
            etext.append(wstr)
        elif predclose_frac > 0.1:  # TODO: get better diagnosis
            wstr = "Possibly complete quasi-separation: A fraction "
            wstr += "%4.2f of observations can be\n" % predclose_frac
            wstr += "perfectly predicted. This might indicate that there "
            wstr += "is complete\nquasi-separation. In this case some "
            wstr += "parameters will not be identified."
            etext.append(wstr)
        if etext:
            smry.add_extra_txt(etext)
        return smry

    @cached_data
    def resid_dev(self):
        r"""
        Deviance residuals

        Notes
        -----
        Deviance residuals are defined

        .. math:: d_j
            = \pm\left(2\left[Y_j\ln\left(\frac{Y_j}{M_jp_j}\right)
                + (M_j - Y_j\ln\left(\frac{M_j-Y_j}{M_j(1-p_j)}\right)\right]\right)^{1/2}

        where

        :math:`p_j = cdf(X\beta)` and :math:`M_j` is the total number of
        observations sharing the covariate pattern :math:`j`.

        For now :math:`M_j` is always set to 1.
        """  # noqa:E501
        # These are the deviance residuals
        endog = self.model.endog
        # M = # of individuals that share a covariate pattern
        # so M[i] = 2 for i = two share a covariate pattern
        M = 1
        p = self.predict()
        # Y_0 = np.where(exog == 0)
        # Y_M = np.where(exog == M)
        # NOTE: Common covariate patterns are not yet handled
        res = (-(1 - endog) * np.sqrt(2 * M * np.abs(np.log(1 - p))) +
               endog * np.sqrt(2 * M * np.abs(np.log(p))))
        return res

    @cached_data
    def resid_pearson(self):
        """
        Pearson residuals

        Notes
        -----
        Pearson residuals are defined to be

        .. math:: r_j = \\frac{(y - M_jp_j)}{\\sqrt{M_jp_j(1-p_j)}}

        where :math:`p_j=cdf(X\\beta)` and :math:`M_j` is the total number of
        observations sharing the covariate pattern :math:`j`.

        For now :math:`M_j` is always set to 1.
        """
        endog = self.model.endog
        # M = # of individuals that share a covariate pattern
        # so M[i] = 2 for i = two share a covariate pattern
        # use unique row pattern?
        M = 1
        p = self.predict()
        return (endog - M * p) / np.sqrt(M * p * (1 - p))

    @cached_data
    def resid_response(self):
        """
        The response residuals

        Notes
        -----
        Response residuals are defined to be

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`.
        """
        return self.model.endog - self.predict()


class LogitResults(BinaryResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Logit Model",
        "extra_attr": ""}

    @cached_data
    def resid_generalized(self):
        """
        Generalized residuals

        Notes
        -----
        The generalized residuals for the Logit model are defined

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`. This is the same as the `resid_response`
        for the Logit model.
        """
        return self.model.endog - self.predict()


class ProbitResults(BinaryResults):
    __doc__ = BinaryResults.__doc__.replace("binary data", "Probit Model")

    @cached_data
    def resid_generalized(self):
        r"""
        Generalized residuals

        Notes
        -----
        The generalized residuals for the Probit model are defined

        .. math:: y\frac{\phi(X\beta)}{\Phi(X\beta)}
            -(1-y)\frac{\phi(X\beta)}{1-\Phi(X\beta)}
        """
        # generalized residuals
        model = self.model
        endog = model.endog
        XB = self.predict(linear=True)
        pdf = model.pdf(XB)
        cdf = model.cdf(XB)
        return endog * pdf / cdf - (1 - endog) * pdf / (1 - cdf)


# FIXME: MultinomialResults doesn't have `resid` upstream, and it raises here
class MultinomialResults(DiscreteResults):
    __doc__ = ProbitResults.__doc__.replace("Probit", "multinomial")

    def __init__(self, model, mlefit):
        # Make sure params have the appropriate shape;
        mlefit.params = mlefit.params.reshape(model.K, -1, order='F')
        super(MultinomialResults, self).__init__(model, mlefit)
        self.J = model.J
        self.K = model.K
        self.nobs = model.nobs

    def _maybe_convert_ynames_int(self, ynames):  # pragma: no cover
        raise NotImplementedError("_maybe_convert_ynames_int not ported from "
                                  "upstream, use "
                                  "sm2.base.naming.maybe_convert_ynames_int "
                                  "instead.")

    @deprecate_kwarg('all', 'use_all')
    def _get_endog_name(self, yname, yname_list, use_all=False):
        """
        If use_all is False, the first variable name is dropped
        """
        model = self.model
        if yname is None:
            yname = model.endog_names
        if yname_list is None:
            ynames = model._ynames_map
            ynames = naming.maybe_convert_ynames_int(ynames)
            # use range below to ensure sortedness
            ynames = [ynames[key] for key in range(int(model.J))]
            ynames = ['='.join([yname.name, name]) for name in ynames]
            if not use_all:
                yname_list = ynames[1:]  # assumes first variable is dropped
            else:
                yname_list = ynames
        return yname, yname_list

    def pred_table(self):
        """
        Returns the J x J prediction table.

        Notes
        -----
        pred_table[i, j] refers to the number of times "i" was observed and
        the model predicted "j". Correct predictions are along the diagonal.
        """
        ju = self.model.J - 1  # highest index
        # these are the actual, predicted indices
        bins = np.concatenate(([0], np.linspace(0.5, ju - 0.5, ju), [ju]))
        return np.histogram2d(self.model.endog, self.predict().argmax(1),
                              bins=bins)[0]

    @cached_data
    def resid_response(self):
        # GH#5255
        return self.model.wendog - self.fittedvalues

    @cached_value
    def bse(self):
        bse = np.sqrt(np.diag(self.cov_params()))
        return bse.reshape(self.params.shape, order='F')
        # TODO: Is the order='F') part necessary?  Can we just add the
        # reshape to the general case? --> appears to break

    @cached_value
    def aic(self):
        return -2 * (self.llf - (self.df_model + self.J - 1))

    @cached_value
    def bic(self):
        return -2 * self.llf + np.log(self.nobs) * (self.df_model +
                                                    self.J - 1)

    def conf_int(self, alpha=.05, cols=None):
        confint = super(DiscreteResults, self).conf_int(alpha=alpha,
                                                        cols=cols)
        return confint.transpose(2, 0, 1)

    def margeff(self):  # pragma: no cover
        raise NotImplementedError("Use get_margeff instead")

    @cached_data
    def resid_misclassified(self):
        """
        Residuals indicating which observations are misclassified.

        Notes
        -----
        The residuals for the multinomial model are defined as

        .. math:: argmax(y_i) \\neq argmax(p_i)

        where :math:`argmax(y_i)` is the index of the category for the
        endogenous variable and :math:`argmax(p_i)` is the index of the
        predicted probabilities for each category. That is, the residual
        is a binary indicator that is 0 if the category with the highest
        predicted probability is the same as that of the observed variable
        and 1 otherwise.
        """
        # it's 0 or 1 - 0 for correct prediction and 1 for a missed one
        return (self.model.wendog.argmax(1) !=
                self.predict().argmax(1)).astype(float)


# --------------------------------------------------------------------
# L1 Results classes

class L1ResultsMixin(object):
    @property
    def nnz_params(self):
        return (~self.trimmed).sum()

    @property
    def df_resid(self):
        # J is really only relevant for MultinomialResults, where
        # there are J-1 constants
        J = getattr(self, 'J', 2)
        return float(self.nobs) - (self.df_model + (J - 1))

    @property
    def df_model(self):
        # J is really only relevant for MultinomialResults, where
        # there are J-1 constants
        J = getattr(self, 'J', 2)

        # adjust for extra parameter in NegativeBinomial nb1 and nb2
        # extra parameter is not included in df_model
        k_extra = getattr(self, 'k_extra', 0)

        return self.nnz_params - k_extra - (J - 1)


class L1CountResults(DiscreteResults, L1ResultsMixin):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for count data fit by "
                                "l1 regularization",
        "extra_attr": _l1_results_attr}

    def __init__(self, model, cntfit):
        # TODO: Make this happen higher up the chain.  Not doing it here breaks
        # things in count_model
        self.params = cntfit.params
        self.nobs = model.endog.shape[0]

        super(L1CountResults, self).__init__(model, cntfit)
        # self.trimmed is a boolean array with T/F telling whether or not that
        # entry in params has been set zero'd out.
        self.trimmed = cntfit.mle_retvals['trimmed']

        # Set degrees of freedom.  In doing so,
        # adjust for extra parameter in NegativeBinomial nb1 and nb2
        # extra parameter is not included in df_model
        self.k_extra = getattr(self.model, 'k_extra', 0)


class L1PoissonResults(L1CountResults, PoissonResults):
    pass


class L1NegativeBinomialResults(L1CountResults, NegativeBinomialResults):
    pass


class L1GeneralizedPoissonResults(L1CountResults, GeneralizedPoissonResults):
    pass


class L1BinaryResults(BinaryResults, L1ResultsMixin):
    __doc__ = L1CountResults.__doc__.replace("count", "binary")

    def __init__(self, model, bnryfit):
        self.nobs = model.endog.shape[0]

        super(L1BinaryResults, self).__init__(model, bnryfit)
        # self.trimmed is a boolean array with T/F telling whether or not that
        # entry in params has been set zero'd out.
        self.trimmed = bnryfit.mle_retvals['trimmed']


class L1MultinomialResults(MultinomialResults, L1ResultsMixin):
    __doc__ = L1BinaryResults.__doc__.replace("binary", "multinomial")

    def __init__(self, model, mlefit):
        self.nobs = model.endog.shape[0]

        super(L1MultinomialResults, self).__init__(model, mlefit)
        # self.trimmed is a boolean array with T/F telling whether or not that
        # entry in params has been set zero'd out.
        self.trimmed = mlefit.mle_retvals['trimmed']


# --------------------------------------------------------------------
# Results Wrappers

# TODO: Are all these wrapper classes actually necessary?
class OrderedResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(OrderedResultsWrapper, OrderedResults)  # noqa: E305


class CountResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(CountResultsWrapper, CountResults)  # noqa: E305


class NegativeBinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(NegativeBinomialResultsWrapper,  # noqa: E305
                      NegativeBinomialResults)


class GeneralizedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(GeneralizedPoissonResultsWrapper,  # noqa: E305
                      GeneralizedPoissonResults)


class PoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(PoissonResultsWrapper, PoissonResults)  # noqa: E305


class L1CountResultsWrapper(lm.RegressionResultsWrapper):
    pass


class L1PoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1PoissonResultsWrapper, L1PoissonResults)  # noqa: E305


class L1NegativeBinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1NegativeBinomialResultsWrapper,  # noqa: E305
                      L1NegativeBinomialResults)


class L1GeneralizedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1GeneralizedPoissonResultsWrapper,  # noqa: E305
                      L1GeneralizedPoissonResults)


class BinaryResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {"resid_dev": "rows",
              "resid_generalized": "rows",
              "resid_pearson": "rows",
              "resid_response": "rows"}
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs,
                                   _attrs)
wrap.populate_wrapper(BinaryResultsWrapper, BinaryResults)  # noqa: E305


class L1BinaryResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1BinaryResultsWrapper, L1BinaryResults)  # noqa: E305


class MultinomialResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {"resid_misclassified": "rows"}
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {"predict": "rows_eq"}
    _wrap_methods = wrap.union_dicts(
        lm.RegressionResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(MultinomialResultsWrapper,  # noqa: E305
                      MultinomialResults)


class L1MultinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1MultinomialResultsWrapper,  # noqa: E305
                      L1MultinomialResults)
