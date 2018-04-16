#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas.util._decorators import Substitution
from scipy import stats

from sm2.tools.data import _is_using_pandas
from sm2.tools.tools import recipr, nan_dot
from sm2.tools.numdiff import (approx_fprime, approx_fprime_cs,
                               approx_hess, approx_hess_cs)
from sm2.tools.decorators import cache_readonly, cached_value, cached_data
from sm2.tools.sm_exceptions import (ValueWarning, HessianInversionWarning,
                                     ConvergenceWarning)

from sm2.base.data import handle_data
import sm2.base.wrapper as wrap
from sm2.base.optimizer import Optimizer, _fit_doc_notes, _fit_doc_params

from sm2.stats.contrast import ContrastResults, WaldTestResults
from sm2.formula import handle_formula_data


_model_params_doc = """
    Parameters
    ----------
    endog : array-like
        1-d endogenous response variable. The dependent variable.
    exog : array-like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See
        :func:`sm2.tools.add_constant`."""

_missing_param_doc = """missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none.'"""
_extra_param_doc = """
    hasconst : None or bool
        Indicates whether the RHS includes a user-supplied constant. If True,
        a constant is not checked for and k_constant is set to 1 and all
        result statistics are calculated as if a constant is present. If
        False, a constant is not checked for and k_constant is set to 0.
"""


class Model(object):
    __doc__ = """
    A (predictive) statistical model. Intended to be subclassed not used.

    %(params_doc)s
    %(extra_params_doc)s

    Notes
    -----
    `endog` and `exog` are references to any data provided.  So if the data is
    already stored in numpy arrays and it is changed then `endog` and `exog`
    will change as well.
    """ % {'params_doc': _model_params_doc,
           'extra_params_doc': _missing_param_doc + _extra_param_doc}

    @property
    def endog_names(self):
        """Names of endogenous variables"""
        return self.data.ynames

    @property
    def exog_names(self):
        """Names of exogenous variables"""
        return self.data.xnames

    @cached_value
    def k_exog(self):
        if self.exog is None:
            return 0
        elif self.exog.ndim == 1:
            # This happens in count_model, should probably be avoided
            return 1
        return self.exog.shape[1]

    def __init__(self, endog, exog=None,
                 missing='none', hasconst=None, **kwargs):

        self.data = self._handle_data(endog, exog, missing, hasconst, **kwargs)

        self.k_constant = self.data.k_constant
        self.exog = self.data.exog
        self.endog = self.data.endog

        self._data_attr = []
        self._data_attr.extend(['exog', 'endog', 'data.exog', 'data.endog'])
        if 'formula' not in kwargs:  # won't be able to unpickle without these
            self._data_attr.extend(['data.orig_endog', 'data.orig_exog'])
        # store keys for extras if we need to recreate model instance
        # we don't need 'missing', maybe we need 'hasconst'
        self._init_keys = list(kwargs.keys())
        if hasconst is not None:
            self._init_keys.append('hasconst')

    def _get_init_kwds(self):
        """return dictionary with extra keys used in model.__init__
        """
        kwds = dict(((key, getattr(self, key, None))
                     for key in self._init_keys))
        return kwds

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        data = handle_data(endog, exog, missing, hasconst, **kwargs)
        # kwargs arrays could have changed, easier to just attach here
        for key in kwargs:
            if key in ['design_info', 'formula']:  # leave attached to data
                continue
            # pop so we don't start keeping all these twice or references
            try:
                setattr(self, key, data.__dict__.pop(key))
            except KeyError:  # panel already pops keys in data handling
                pass
        return data

    @classmethod
    def from_formula(cls, formula, data, subset=None,
                     drop_cols=None, *args, **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array-like
            The data for the model. See Notes.
        subset : array-like
            An array-like object of booleans, integers, or index values that
            indicate the subset of df to use in the model. Assumes df is a
            `pandas.DataFrame`
        drop_cols : array-like
            Columns to drop from the design matrix.  Cannot be used to
            drop terms involving categoricals.
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : Model instance

        Notes
        ------
        data must define __getitem__ with the keys in the formula terms
        args and kwargs are passed on to the model instantiation. E.g.,
        a numpy structured or rec array, a dictionary, or a pandas DataFrame.
        """
        # TODO: provide a docs template for args/kwargs from child models
        # TODO: subset could use syntax. GH#469.
        if subset is not None:
            data = data.loc[subset]
        eval_env = kwargs.pop('eval_env', None)
        if eval_env is None:
            eval_env = 2
        elif eval_env == -1:
            from patsy import EvalEnvironment
            eval_env = EvalEnvironment({})
        else:
            eval_env += 1  # we're going down the stack again
        missing = kwargs.get('missing', 'drop')
        if missing == 'none':  # with patsy it's drop or raise. let's raise.
            missing = 'raise'

        tmp = handle_formula_data(data, None, formula, depth=eval_env,
                                  missing=missing)
        ((endog, exog), missing_idx, design_info) = tmp

        if drop_cols is not None and len(drop_cols) > 0:
            # TODO: not hit in tests
            cols = [x for x in exog.columns if x not in drop_cols]
            if len(cols) < len(exog.columns):
                exog = exog[cols]
                cols = list(design_info.term_names)
                for col in drop_cols:
                    try:
                        cols.remove(col)
                    except ValueError:
                        pass  # OK if not present
                design_info = design_info.builder.subset(cols).design_info

        kwargs.update({'missing_idx': missing_idx,
                       'missing': missing,
                       'formula': formula,  # attach formula for unpckling
                       'design_info': design_info})
        mod = cls(endog, exog, *args, **kwargs)
        mod.formula = formula

        # since we got a dataframe, attach the original
        mod.data.frame = data
        return mod

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError  # pragma: no cover

    def predict(self, params, exog=None, *args, **kwargs):
        """
        After a model has been fit predict returns the fitted values.

        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError  # pragma: no cover


class LikelihoodModel(Model):
    """
    Likelihood model is a subclass of Model.
    """
    _use_approx_cs = False
    # _use_approx_cs describes whether or not we can/should use complex-step
    # when numerically differentiating the likelihood function.

    def __init__(self, endog, exog=None, **kwargs):
        super(LikelihoodModel, self).__init__(endog, exog, **kwargs)
        self.initialize()

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance. For
        instance, the design matrix of a linear model may change
        and some things must be recomputed.
        """
        pass

    # TODO: if the intent is to re-initialize the model with new data then this
    # method needs to take inputs...

    def loglikeobs(self, params, *args, **kwargs):
        """Log-likelihood of model evaluated pointwise"""
        raise NotImplementedError  # pragma: no cover

    def loglike(self, params, *args, **kwargs):
        """
        Log-likelihood of model.  Default implementation sums loglikeobs.
        """
        return np.sum(self.loglikeobs(params, *args, **kwargs))

    def score_obs(self, params, *args, **kwargs):
        """
        Score vector of model evaluated pointwise.  The gradient of loglikeobs
        with respect to each parameter.
        """
        if self._use_approx_cs:
            return approx_fprime_cs(params, self.loglikeobs,
                                    args=args, kwargs=kwargs)
        else:
            return approx_fprime(params, self.loglikeobs,
                                 args=args, kwargs=kwargs)

    def score(self, params, *args, **kwargs):
        """
        Score vector of model.  Default implementation sums score_obs.

        The gradient of loglike with respect to each parameter.
        """
        try:
            # If an analytic score_obs is available, try this first before
            # falling back to numerical differentiation below
            return self.score_obs(params, *args, **kwargs).sum(0)
        except NotImplementedError:
            # Fallback in case a `loglike` is implemented but `loglikeobs`
            # is not.
            approx_func = (approx_fprime_cs
                           if self._use_approx_cs else approx_fprime)
            return approx_func(params, self.loglike, args=args, kwargs=kwargs)

    def information(self, params):
        """
        Fisher information matrix of model

        Returns -Hessian of loglike evaluated at params.
        """
        # TODO: If the docstring is right, then why not just implement this?
        raise NotImplementedError  # pragma: no cover

    def hessian(self, params, *args, **kwargs):
        """
        The Hessian matrix of the model

        The default implementation uses a numerical derivative.
        """
        if self._use_approx_cs:
            return approx_hess_cs(params, self.loglike,
                                  args=args, kwargs=kwargs)
        else:
            return approx_hess(params, self.loglike,
                               args=args, kwargs=kwargs)

    # upstream this is implemented in GenericLikelihoodModel
    def hessian_factor(self, params, scale=None, observed=True):
        """Weights for calculating Hessian

        Parameters
        ----------
        params : ndarray
            parameter at which Hessian is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        hessian_factor : ndarray, 1d
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`
        """
        raise NotImplementedError  # pragma: no cover

    def _get_start_params(self, start_params=None):
        """
        If no start_params are given, use reasonable defaults.

        Parameters
        ----------
        start_params : None or np.ndarray (default None)

        Returns
        -------
        start_params : np.ndarray

        Raises
        ------
        ValueError if start_params cannot be found.
        """
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            elif self.exog is not None:
                # fails for shape (K,)?
                start_params = [0] * self.exog.shape[1]
            else:  # pragma: no cover
                raise ValueError("If exog is None, then start_params should "
                                 "be specified")
        return start_params

    @Substitution(doc_notes=_fit_doc_notes.strip(),
                  fit_params=_fit_doc_params.strip())
    def fit(self, start_params=None, method='newton', maxiter=100,
            full_output=True, disp=True, fargs=(), callback=None, retall=False,
            skip_hessian=False, **kwargs):
        """
        Fit method for likelihood based models

        Parameters
        ----------
        %(fit_params)s
        skip_hessian : bool, optional
            If False (default), then the negative inverse hessian is calculated
            after the optimization. If True, then the hessian will not be
            calculated. However, it will be available in methods that use the
            hessian in the optimization (currently only with `"newton"`).
        kwargs : keywords
            All kwargs are passed to the chosen solver with one exception. The
            following keyword controls what happens after the fit::
        warn_convergence : bool, optional
            If True, checks the model for the converged flag. If the
            converged flag is False, a ConvergenceWarning is issued.

        Notes
        -----
        %(doc_notes)s
        """
        Hinv = None  # JP error if full_output=0, Hinv not defined

        start_params = self._get_start_params(start_params)

        # TODO: separate args from nonarg taking score and hessian, ie.,
        # user-supplied and numerically evaluated estimate frprime doesn't take
        # args in most (any?) of the optimize function

        nobs = self.endog.shape[0]
        # f = lambda params, *args: -self.loglike(params, *args) / nobs

        def f(params, *args):
            return -self.loglike(params, *args) / nobs

        if method == 'newton':
            # TODO: why are score and hess positive?
            def score(params, *args):
                return self.score(params, *args) / nobs

            def hess(params, *args):
                return self.hessian(params, *args) / nobs
        else:
            def score(params, *args):
                return -self.score(params, *args) / nobs

            def hess(params, *args):
                return -self.hessian(params, *args) / nobs

        warn_convergence = kwargs.pop('warn_convergence', True)
        optimizer = Optimizer()
        xopt, retvals, optim_settings = optimizer._fit(f, score, start_params,
                                                       fargs, kwargs,
                                                       hessian=hess,
                                                       method=method,
                                                       disp=disp,
                                                       maxiter=maxiter,
                                                       callback=callback,
                                                       retall=retall,
                                                       full_output=full_output)

        # NOTE: this is for fit_regularized and should be generalized
        cov_params_func = kwargs.setdefault('cov_params_func', None)
        if cov_params_func:
            Hinv = cov_params_func(self, xopt, retvals)
        elif method == 'newton' and full_output:
            Hinv = np.linalg.inv(-retvals['Hessian']) / nobs
        elif not skip_hessian:
            H = -1 * self.hessian(xopt)
            invertible = False
            if np.all(np.isfinite(H)):
                eigvals, eigvecs = np.linalg.eigh(H)
                if np.min(eigvals) > 0:
                    invertible = True

            if invertible:
                Hinv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
                Hinv = np.asfortranarray((Hinv + Hinv.T) / 2.0)
            else:
                warnings.warn('Inverting hessian failed, no bse or cov_params '
                              'available', HessianInversionWarning)
                Hinv = None

        if 'cov_type' in kwargs:
            cov_kwds = kwargs.get('cov_kwds', {})
            kwds = {'cov_type': kwargs['cov_type'], 'cov_kwds': cov_kwds}
        else:
            kwds = {}
        if 'use_t' in kwargs:
            kwds['use_t'] = kwargs['use_t']
        # TODO: add Hessian approximation and change the above if needed
        mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1., **kwds)

        # TODO: hardcode scale?
        if isinstance(retvals, dict):
            mlefit.mle_retvals = retvals
            if warn_convergence and not retvals['converged']:
                warnings.warn("Maximum Likelihood optimization failed to "
                              "converge. Check mle_retvals",
                              ConvergenceWarning)

        mlefit.mle_settings = optim_settings
        return mlefit


class GenericLikelihoodModel(LikelihoodModel):
    # TODO: methods that may be worth porting from upstream:
    #   _set_extra_param_names
    #   expandparams
    #   reduceparams
    def __init__(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("GenericLikelihoodModel is not ported "
                                  "from upstream, as it is unfinished and "
                                  "effectively untested.  "
                                  "See GH#4453 upstream.")


class Results(object):
    """
    Class to contain model results

    Parameters
    ----------
    model : class instance
        the previously specified model instance
    params : array
        parameter estimates from the fit model
    k_constr : int
        number of constraints the model was fit with (default 0)
    """
    def __init__(self, model, params, k_constr=0, **kwargs):
        self.model = model
        self.params = params

        self.k_constr = k_constr
        self.k_constant = model.k_constant

        self.__dict__.update(kwargs)
        # TODO: Avoid self.__dict__.update
        self.initialize(model, params, **kwargs)
        self._data_attr = []

    def initialize(self, model, params, **kwargs):
        # TODO: Get rid of this redundant method
        pass

    @cached_data
    def fittedvalues(self):
        """
        (array) The predicted values of the model. An (nobs x k_endog) array.
        """
        return self.model.predict(self.params)

    @cached_data
    def resid(self):
        """
        (array) The model residuals. An (nobs x k_endog) array.
        """
        return self.model.endog - self.fittedvalues
        # TODO: Is this only accurate for linear models?
        # is there a more generally correct version?

    def predict(self, exog=None, transform=True, *args, **kwargs):
        """
        Call self.model.predict with self.params as the first argument.

        Parameters
        ----------
        exog : array-like, optional
            The values for which you want to predict.
        transform : bool, optional
            If the model was fit via a formula, do you want to pass
            exog through the formula. Default is True. E.g., if you fit
            a model y ~ log(x1) + log(x2), and transform is True, then
            you can pass a data structure that contains x1 and x2 in
            their original form. Otherwise, you'd need to log the data
            first.
        args, kwargs :
            Some models can take additional arguments or keywords, see the
            predict method of the model for the details.

        Returns
        -------
        prediction : ndarray, pandas.Series or pandas.DataFrame
            See self.model.predict
        """
        exog_index = exog.index if _is_using_pandas(exog, None) else None

        if transform and hasattr(self.model, 'formula') and exog is not None:
            from patsy import dmatrix
            exog = pd.DataFrame(exog)  # user may pass series, if one predictor
            if exog_index is None:  # user passed in a dictionary
                exog_index = exog.index
            exog = dmatrix(self.model.data.design_info.builder,
                           exog, return_type="dataframe")
            if len(exog) < len(exog_index):
                # missing values, rows have been dropped
                if exog_index is not None:
                    exog = exog.reindex(exog_index)
                else:
                    warnings.warn("nan rows have been dropped", ValueWarning)

        if exog is not None:
            exog = np.asarray(exog)
            if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                                   self.model.exog.shape[1] == 1):
                exog = exog[:, None]
            exog = np.atleast_2d(exog)  # needed in count model shape[1]

        predict_results = self.model.predict(self.params, exog,
                                             *args, **kwargs)

        # TODO: Shouldn't this be done by wrapping?
        if exog_index is not None and not hasattr(predict_results,
                                                  'predicted_values'):
            if predict_results.ndim == 1:
                return pd.Series(predict_results, index=exog_index)
            else:
                return pd.DataFrame(predict_results, index=exog_index)

        else:
            return predict_results

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `var_##` for ## in p the number of regressors
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        sm2.iolib.summary.Summary : class to hold summary results
        """
        # TODO: Make this raise upstream instead of just "pass"
        raise NotImplementedError  # pragma: no cover
        # TODO: move the GenericLikelihoodModelResults implementation here?


# TODO: public method?
class LikelihoodModelResults(wrap.SaveLoadMixin, Results):
    """
    Class to contain results from likelihood models

    Parameters
    -----------
    model : LikelihoodModel instance or subclass instance
        LikelihoodModelResults holds a reference to the model that is fit.
    params : 1d array_like
        parameter estimates from estimated model
    normalized_cov_params : 2d array
       Normalized (before scaling) covariance of params. (dot(X.T,X))**-1
    scale : float
        For (some subset of models) scale will typically be the
        mean square error from the estimated model (sigma^2)

    Returns
    -------
    **Attributes**
    mle_retvals : dict
        Contains the values returned from the chosen optimization method if
        full_output is True during the fit.  Available only if the model
        is fit by maximum likelihood.  See notes below for the output from
        the different methods.
    mle_settings : dict
        Contains the arguments passed to the chosen optimization method.
        Available if the model is fit by maximum likelihood.  See
        LikelihoodModel.fit for more information.
    model : model instance
        LikelihoodResults contains a reference to the model that is fit.
    params : ndarray
        The parameters estimated for the model.
    scale : float
        The scaling factor of the model given during instantiation.
    tvalues : array
        The t-values of the standard errors.

    Notes
    -----
    The covariance of params is given by scale times normalized_cov_params.

    Return values by solver if full_output is True during fit:

        'newton'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            score : ndarray
                The score vector at the optimum.
            Hessian : ndarray
                The Hessian at the optimum.
            warnflag : int
                1 if maxiter is exceeded. 0 if successful convergence.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'nm'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            warnflag : int
                1: Maximum number of function evaluations made.
                2: Maximum number of iterations reached.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'bfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            Hinv : ndarray
                value of the inverse Hessian matrix at minimum.  Note
                that this is just an approximation and will often be
                different from the value of the analytic Hessian.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient
                and/or function calls are not changing.
            converged : bool
                True: converged.  False: did not converge.
            allvecs : list
                Results at each iteration.
        'lbfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                Warning flag:

                - 0 if converged
                - 1 if too many function evaluations or too many iterations
                - 2 if stopped for another reason

            converged : bool
                True: converged.  False: did not converge.
        'powell'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            direc : ndarray
                Current direction set.
            iterations : int
                Number of iterations performed.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                1: Maximum number of function evaluations. 2: Maximum number
                of iterations.
            converged : bool
                True : converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'cg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient and/
                or function calls not changing.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'ncg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            hcalls : int
                Number of calls to hessian.
            warnflag : int
                1: Maximum number of iterations exceeded.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
    """
    # TODO: implement ResultMixin.bootstrap?

    # by default we use normal distribution
    # can be overwritten by instances or subclasses
    use_t = False

    # TODO: WTF Why does this method exist if its going to be overwritten
    # in __init__?
    def normalized_cov_params(self):
        raise NotImplementedError

    @cached_value
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglike(self.params)

    @cached_data
    def llf_obs(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglikeobs(self.params)

    @cached_value
    def tvalues(self):
        """
        Return the t-statistic for a given parameter estimate.
        """
        return self.params / self.bse

    @cached_value
    def pvalues(self):
        if self.use_t:
            df_resid = getattr(self, 'df_resid_inference', self.df_resid)
            return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
        else:
            return stats.norm.sf(np.abs(self.tvalues)) * 2

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
                 **kwargs):
        super(LikelihoodModelResults, self).__init__(model, params)
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale

        # robust covariance
        # We put cov_type in kwargs so subclasses can decide in fit whether to
        # use this generic implementation
        if 'use_t' in kwargs:
            use_t = kwargs['use_t']
            if use_t is not None:
                self.use_t = use_t

        if 'cov_type' in kwargs:
            cov_type = kwargs.get('cov_type', 'nonrobust')
            cov_kwds = kwargs.get('cov_kwds', {})
            cov_kwds = cov_kwds or {}
            self._get_robustcov_results(cov_type=cov_type,
                                        use_self=True, use_t=self.use_t,
                                        **cov_kwds)

    def _get_robustcov_results(self, cov_type='nonrobust', use_self=True,
                               use_t=None, **cov_kwds):
        from sm2.base.covtype import get_robustcov_results
        # TODO: we shouldn't need use_t in get_robustcov_results
        get_robustcov_results(self, cov_type=cov_type, use_self=use_self,
                              use_t=use_t, **cov_kwds)

    @cache_readonly
    def bse(self):
        # GH#3299
        if ((not hasattr(self, 'cov_params_default')) and
                (self.normalized_cov_params is None)):
            bse = np.empty(len(self.params))
            bse[:] = np.nan
        else:
            bse = np.sqrt(np.diag(self.cov_params()))
        # reshape is unnecessary in many cases, needed for e.g MNLogit
        return bse.reshape(self.params.shape)

    def cov_params(self, r_matrix=None, column=None, scale=None, cov_p=None,
                   other=None):
        """
        Returns the variance/covariance matrix.

        The variance/covariance matrix can be of a linear contrast
        of the estimates of params or all params multiplied by scale which
        will usually be an estimate of sigma^2.  Scale is assumed to be
        a scalar.

        Parameters
        ----------
        r_matrix : array-like
            Can be 1d, or 2d.  Can be used alone or with other.
        column :  array-like, optional
            Must be used on its own.  Can be 0d or 1d see below.
        scale : float, optional
            Can be specified or not.  Default is None, which means that
            the scale argument is taken from the model.
        other : array-like, optional
            Can be used when r_matrix is specified.

        Returns
        -------
        cov : ndarray
            covariance matrix of the parameter estimates or of linear
            combination of parameter estimates. See Notes.

        Notes
        -----
        (The below are assumed to be in matrix notation.)

        If no argument is specified returns the covariance matrix of a model
        ``(scale)*(X.T X)^(-1)``

        If contrast is specified it pre and post-multiplies as follows
        ``(scale) * r_matrix (X.T X)^(-1) r_matrix.T``

        If contrast and other are specified returns
        ``(scale) * r_matrix (X.T X)^(-1) other.T``

        If column is specified returns
        ``(scale) * (X.T X)^(-1)[column,column]`` if column is 0d

        OR

        ``(scale) * (X.T X)^(-1)[column][:,column]`` if column is 1d
        """
        if (hasattr(self, 'mle_settings') and
                self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            dot_fun = nan_dot
        else:
            dot_fun = np.dot

        if (cov_p is None and self.normalized_cov_params is None and
                not hasattr(self, 'cov_params_default')):  # pragma: no cover
            raise ValueError('need covariance of parameters for computing '
                             '(unnormalized) covariances')
        if column is not None and (r_matrix is not None or other is not None):
            raise ValueError('Column should be specified without other '
                             'arguments.')  # pragma: no cover
        if other is not None and r_matrix is None:  # pragma: no cover
            raise ValueError('other can only be specified with r_matrix')

        if cov_p is None:
            if hasattr(self, 'cov_params_default'):
                cov_p = self.cov_params_default
            else:
                if scale is None:
                    scale = self.scale
                cov_p = self.normalized_cov_params * scale

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return cov_p[column, column]
            else:
                return cov_p[column[:, None], column]
        elif r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
            if r_matrix.shape == ():  # pragma: no cover
                raise ValueError("r_matrix should be 1d or 2d")
            if other is None:
                other = r_matrix
            else:
                other = np.asarray(other)
            tmp = dot_fun(r_matrix, dot_fun(cov_p, np.transpose(other)))
            return tmp
        else:  # if r_matrix is None and column is None:
            return cov_p

    # TODO: make sure this works as needed for GLMs
    def t_test(self, r_matrix, cov_p=None, scale=None, use_t=None):
        """
        Compute a t-test for a each linear hypothesis of the form Rb = q

        Parameters
        ----------
        r_matrix : array-like, str, tuple
            - array : If an array is given, a p x k 2d array or length k 1d
              array specifying the linear restrictions. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q). If q is given,
              can be either a scalar or a length p row vector.
        cov_p : array-like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        scale : float, optional
            An optional `scale` to use.  Default is the scale specified
            by the model fit.
        use_t : bool, optional
            If use_t is None, then the default of the model is used.
            If use_t is True, then the p-values are based on the t
            distribution.
            If use_t is False, then the p-values are based on the normal
            distribution.

        Returns
        -------
        res : ContrastResults instance
            The results for the test are attributes of this results instance.
            The available results have the same elements as the parameter table
            in `summary()`.

        Examples
        --------
        >>> import numpy as np
        >>> import sm2.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> r = np.zeros_like(results.params)
        >>> r[5:] = [1,-1]
        >>> print(r)
        [ 0.  0.  0.  0.  0.  1. -1.]

        r tests that the coefficients on the 5th and 6th independent
        variable are the same.

        >>> T_test = results.t_test(r)
        >>> print(T_test)
                                     Test for Constraints
        ==============================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        c0         -1829.2026    455.391     -4.017      0.003   -2859.368    -799.037
        ==============================================================================
        >>> T_test.effect
        -1829.2025687192481
        >>> T_test.sd
        455.39079425193762
        >>> T_test.tvalue
        -4.0167754636411717
        >>> T_test.pvalue
        0.0015163772380899498

        Alternatively, you can specify the hypothesis tests using a string

        >>> dta = sm.datasets.longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = OLS.from_formula(formula, dta).fit()
        >>> hypotheses = 'GNPDEFL = GNP, UNEMP = 2, YEAR/1829 = 1'
        >>> t_test = results.t_test(hypotheses)
        >>> print(t_test)
                                     Test for Constraints
        ==============================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        c0            15.0977     84.937      0.178      0.863    -177.042     207.238
        c1            -2.0202      0.488     -8.231      0.000      -3.125      -0.915
        c2             1.0001      0.249      0.000      1.000       0.437       1.563
        ==============================================================================

        See Also
        ---------
        tvalues : individual t statistics
        f_test : for F tests
        patsy.DesignInfo.linear_constraint
        """  # noqa:E501
        from patsy import DesignInfo
        names = self.model.data.param_names
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants
        num_ttests = r_matrix.shape[0]
        num_params = r_matrix.shape[1]

        if (cov_p is None and self.normalized_cov_params is None and
                not hasattr(self, 'cov_params_default')):
            raise ValueError('Need covariance of parameters for computing '
                             'T statistics')  # pragma: no cover
        if num_params != self.params.shape[0]:  # pragma: no cover
            raise ValueError('r_matrix and params are not aligned')
        if q_matrix is None:
            q_matrix = np.zeros(num_ttests)
        else:
            q_matrix = np.asarray(q_matrix)
            q_matrix = q_matrix.squeeze()
        if q_matrix.size > 1:
            if q_matrix.shape[0] != num_ttests:  # pragma: no cover
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")

        if use_t is None:
            # switch to use_t false if undefined
            use_t = (hasattr(self, 'use_t') and self.use_t)

        tstat = _sd = None

        _effect = np.dot(r_matrix, self.params)
        # nan_dot multiplies with the convention nan * 0 = 0

        cparams = self.cov_params(r_matrix=r_matrix, cov_p=cov_p)
        # Perform the test
        if num_ttests > 1:
            _sd = np.sqrt(np.diag(cparams))
        else:
            _sd = np.sqrt(cparams)
        tstat = (_effect - q_matrix) * recipr(_sd)

        df_resid = getattr(self, 'df_resid_inference', self.df_resid)

        if use_t:
            return ContrastResults(effect=_effect, t=tstat, sd=_sd,
                                   df_denom=df_resid)
        else:
            return ContrastResults(effect=_effect, statistic=tstat, sd=_sd,
                                   df_denom=df_resid,
                                   distribution='norm')

    def f_test(self, r_matrix, cov_p=None, scale=1.0, invcov=None):
        """
        Compute the F-test for a joint linear hypothesis.

        This is a special case of `wald_test` that always uses the F
        distribution.

        Parameters
        ----------
        r_matrix : array-like, str, or tuple
            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length k row vector.
        cov_p : array-like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        scale : float, optional
            Default is 1.0 for no scaling.
        invcov : array-like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.

        Returns
        -------
        res : ContrastResults instance
            The results for the test are attributes of this results instance.

        Examples
        --------
        >>> import numpy as np
        >>> import sm2.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> A = np.identity(len(results.params))
        >>> A = A[1:,:]

        This tests that each coefficient is jointly statistically
        significantly different from zero.

        >>> print(results.f_test(A))
        <F test: F=array([[ 330.28533923]]), \
                         p=4.984030528700946e-10, df_denom=9, df_num=6>

        Compare this to

        >>> results.fvalue
        330.2853392346658
        >>> results.f_pvalue
        4.98403096572e-10

        >>> B = np.array(([0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1]))

        This tests that the coefficient on the 2nd and 3rd regressors are
        equal and jointly that the coefficient on the 5th and 6th regressors
        are equal.

        >>> print(results.f_test(B))
        <F test: F=array([[ 9.74046187]]), \
                         p=0.005605288531708235, df_denom=9, df_num=2>

        Alternatively, you can specify the hypothesis tests using a string

        >>> from sm2.datasets import longley
        >>> dta = longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = OLS.from_formula(formula, dta).fit()
        >>> hypotheses = '(GNPDEFL = GNP), (UNEMP = 2), (YEAR/1829 = 1)'
        >>> f_test = results.f_test(hypotheses)
        >>> print(f_test)
        <F test: F=array([[ 144.17976065]]), \
                         p=6.322026217355609e-08, df_denom=9, df_num=3>

        See Also
        --------
        sm2.stats.contrast.ContrastResults
        wald_test
        t_test
        patsy.DesignInfo.linear_constraint

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.
        """
        res = self.wald_test(r_matrix, cov_p=cov_p, scale=scale,
                             invcov=invcov, use_f=True)
        return res

    # TODO: untested for GLMs?
    def wald_test(self, r_matrix, cov_p=None, scale=1.0, invcov=None,
                  use_f=None):
        """
        Compute a Wald-test for a joint linear hypothesis.

        Parameters
        ----------
        r_matrix : array-like, str, or tuple
            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.
        cov_p : array-like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        scale : float, optional
            Default is 1.0 for no scaling.
        invcov : array-like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.
        use_f : bool
            If True, then the F-distribution is used. If False, then the
            asymptotic distribution, chisquare is used. If use_f is None, then
            the F distribution is used if the model specifies that use_t
            is True.  The test statistic is proportionally adjusted for the
            distribution by the number of constraints in the hypothesis.

        Returns
        -------
        res : ContrastResults instance
            The results for the test are attributes of this results instance.

        See also
        --------
        sm2.stats.contrast.ContrastResults
        f_test
        t_test
        patsy.DesignInfo.linear_constraint

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.
        """
        if use_f is None:
            # switch to use_t false if undefined
            use_f = (hasattr(self, 'use_t') and self.use_t)

        from patsy import DesignInfo
        names = self.model.data.param_names
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants

        if (self.normalized_cov_params is None and cov_p is None and
                invcov is None and not hasattr(self, 'cov_params_default')):
            raise ValueError('need covariance of parameters for computing '
                             'F statistics')  # pragma: no cover

        cparams = np.dot(r_matrix, self.params[:, None])
        J = float(r_matrix.shape[0])  # number of restrictions
        if q_matrix is None:
            q_matrix = np.zeros(J)
        else:
            q_matrix = np.asarray(q_matrix)
        if q_matrix.ndim == 1:
            q_matrix = q_matrix[:, None]
            if q_matrix.shape[0] != J:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")
        Rbq = cparams - q_matrix
        if invcov is None:
            cov_p = self.cov_params(r_matrix=r_matrix, cov_p=cov_p)
            if np.isnan(cov_p).max():
                raise ValueError("r_matrix performs f_test for using "
                                 "dimensions that are asymptotically "
                                 "non-normal")
            invcov = np.linalg.inv(cov_p)

        if (hasattr(self, 'mle_settings') and
                self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            F = nan_dot(nan_dot(Rbq.T, invcov), Rbq)
        else:
            F = np.dot(np.dot(Rbq.T, invcov), Rbq)

        df_resid = getattr(self, 'df_resid_inference', self.df_resid)
        if use_f:
            F /= J
            return ContrastResults(F=F, df_denom=df_resid,
                                   df_num=invcov.shape[0])
        else:
            return ContrastResults(chi2=F, df_denom=J, statistic=F,
                                   distribution='chi2', distargs=(J,))

    def wald_test_terms(self, skip_single=False, extra_constraints=None,
                        combine_terms=None):
        """
        Compute a sequence of Wald tests for terms over multiple columns

        This computes joined Wald tests for the hypothesis that all
        coefficients corresponding to a `term` are zero.

        `Terms` are defined by the underlying formula or by string matching.

        Parameters
        ----------
        skip_single : boolean
            If true, then terms that consist only of a single column and,
            therefore, refers only to a single parameter is skipped.
            If false, then all terms are included.
        extra_constraints : ndarray
            not tested yet
        combine_terms : None or list of strings
            Each string in this list is matched to the name of the terms or
            the name of the exogenous variables. All columns whose name
            includes that string are combined in one joint test.

        Returns
        -------
        test_result : result instance
            The result instance contains `table` which is a pandas DataFrame
            with the test results: test statistic, degrees of freedom and
            pvalues.

        Examples
        --------
        >>> formula = "np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)"
        >>> res_ols = ols(formula, data).fit()
        >>> res_ols.wald_test_terms()
        <class 'sm2.stats.contrast.WaldTestResults'>
                                                  F                P>F  df constraint  df denom
        Intercept                        279.754525  2.37985521351e-22              1        51
        C(Duration, Sum)                   5.367071    0.0245738436636              1        51
        C(Weight, Sum)                    12.432445  3.99943118767e-05              2        51
        C(Duration, Sum):C(Weight, Sum)    0.176002      0.83912310946              2        51

        >>> res_poi = Poisson.from_formula("Days ~ C(Weight) * C(Duration)", \
                                           data).fit(cov_type='HC0')
        >>> wt = res_poi.wald_test_terms(skip_single=False, \
                                         combine_terms=['Duration', 'Weight'])
        >>> print(wt)
                                    chi2             P>chi2  df constraint
        Intercept              15.695625  7.43960374424e-05              1
        C(Weight)              16.132616  0.000313940174705              2
        C(Duration)             1.009147     0.315107378931              1
        C(Weight):C(Duration)   0.216694     0.897315972824              2
        Duration               11.187849     0.010752286833              3
        Weight                 30.263368  4.32586407145e-06              4
        """  # noqa:E501
        result = self
        if extra_constraints is None:
            extra_constraints = []
        if combine_terms is None:
            combine_terms = []
        design_info = getattr(result.model.data, 'design_info', None)

        if design_info is None and extra_constraints is None:
            raise ValueError('no constraints, nothing to do')

        identity = np.eye(len(result.params))
        constraints = []
        combined = defaultdict(list)
        if design_info is not None:
            for term in design_info.terms:
                cols = design_info.slice(term)
                name = term.name()
                constraint_matrix = identity[cols]

                # check if in combined
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(constraint_matrix)

                k_constraint = constraint_matrix.shape[0]
                if skip_single:
                    if k_constraint == 1:
                        continue

                constraints.append((name, constraint_matrix))

            combined_constraints = []
            for cname in combine_terms:
                combined_constraints.append((cname,
                                             np.vstack(combined[cname])))
        else:
            # check by exog/params names if there is no formula info
            for col, name in enumerate(result.model.exog_names):
                constraint_matrix = identity[col]

                # check if in combined
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(constraint_matrix)

                if skip_single:
                    continue

                constraints.append((name, constraint_matrix))

            combined_constraints = []
            for cname in combine_terms:
                combined_constraints.append((cname,
                                             np.vstack(combined[cname])))

        use_t = result.use_t
        distribution = ['chi2', 'F'][use_t]

        res_wald = []
        index = []
        for pair in constraints + combined_constraints + extra_constraints:
            name, constraint = pair
            wt = result.wald_test(constraint)
            row = [wt.statistic.item(), wt.pvalue, constraint.shape[0]]
            if use_t:
                row.append(wt.df_denom)
            res_wald.append(row)
            index.append(name)

        # distribution neutral names
        col_names = ['statistic', 'pvalue', 'df_constraint']
        if use_t:
            col_names.append('df_denom')
        # TODO: maybe move DataFrame creation to results class
        table = pd.DataFrame(res_wald, index=index, columns=col_names)
        res = WaldTestResults(None, distribution, None, table=table)
        # TODO: remove temp again, added for testing
        res.temp = constraints + combined_constraints + extra_constraints
        return res

    def conf_int(self, alpha=.05, cols=None, method=None):
        """
        Returns the confidence interval of the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        cols : array-like, optional
            `cols` specifies which confidence intervals to return

        Returns
        --------
        conf_int : array
            Each row contains [lower, upper] limits of the confidence interval
            for the corresponding parameter. The first column contains all
            lower, the second column contains all upper limits.

        Examples
        --------
        >>> import sm2.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> results.conf_int()
        array([[-5496529.48322745, -1467987.78596704],
               [    -177.02903529,      207.15277984],
               [      -0.1115811 ,        0.03994274],
               [      -3.12506664,       -0.91539297],
               [      -1.5179487 ,       -0.54850503],
               [      -0.56251721,        0.460309  ],
               [     798.7875153 ,     2859.51541392]])

        >>> results.conf_int(cols=(2,3))
        array([[-0.1115811 ,  0.03994274],
               [-3.12506664, -0.91539297]])

        Notes
        -----
        The confidence interval is based on the Student's t-distribution
        if the model's `use_t` attribute is True, otherwise they are
        based on the standard normal distribution.
        """
        if method is not None:  # pragma: no cover
            raise NotImplementedError("`method` argument is not actually "
                                      "supported.  Upstream silently ignores "
                                      "it.")
        bse = self.bse

        if self.use_t:
            dist = stats.t
            df_resid = getattr(self, 'df_resid_inference', self.df_resid)
            q = dist.ppf(1 - alpha / 2, df_resid)
        else:
            dist = stats.norm
            q = dist.ppf(1 - alpha / 2)

        if cols is None:
            lower = self.params - q * bse
            upper = self.params + q * bse
        else:
            cols = np.asarray(cols)
            lower = self.params[cols] - q * bse[cols]
            upper = self.params[cols] + q * bse[cols]
        return np.asarray(list(zip(lower, upper)))


class LikelihoodResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'params': 'columns',
              'bse': 'columns',
              'pvalues': 'columns',
              'tvalues': 'columns',
              'resid': 'rows',
              'fittedvalues': 'rows',
              'normalized_cov_params': 'cov'}
    _wrap_attrs = _attrs
    _wrap_methods = {'cov_params': 'cov',
                     'conf_int': 'columns'}
wrap.populate_wrapper(LikelihoodResultsWrapper,  # noqa:E305
                      LikelihoodModelResults)


class ResultMixin(object):
    # TODO: upstream ResultMixin.get_nlfun passes when it should raise
    def __init__(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("ResultMixin/GenericLikelihoodModelResults "
                                  "is not ported from upstream, "
                                  "as it is effectively unused, untested, "
                                  "and will raise AttributeErrors in some "
                                  "cases (GH#4452 upstream)")


class GenericLikelihoodModelResults(ResultMixin, LikelihoodModelResults):
    pass  # raises along with ResultMixin
