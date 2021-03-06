"""
Robust linear models with support for the M-estimators  listed under
:ref:`norms <norms>`.

References
----------
PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York.  1981.

PJ Huber.  1973,  'The 1972 Wald Memorial Lectures: Robust Regression:
    Asymptotics, Conjectures, and Monte Carlo.'  The Annals of Statistics,
    1.5, 799-821.

R Venables, B Ripley. 'Modern Applied Statistics in S'  Springer, New York,
    2002.
"""
import numpy as np
from scipy import stats

from sm2.tools.decorators import (cached_value, cached_data, cache_readonly,
                                  resettable_cache)
import sm2.base.model as base
import sm2.base.wrapper as wrap

import sm2.regression.linear_model as lm
import sm2.regression._tools as reg_tools

from sm2.robust import norms, scale

__all__ = ['RLM']


def _check_convergence(criterion, iteration, tol, maxiter):
    return not (np.any(np.fabs(criterion[iteration] -
                criterion[iteration - 1]) > tol) and iteration < maxiter)


class RLM(base.LikelihoodModel):
    __doc__ = """
    Robust Linear Models

    Estimate a robust linear model via iteratively reweighted least squares
    given a robust criterion estimator.

    %(params)s
    M : sm2.robust.norms.RobustNorm, optional
        The robust criterion function for downweighting outliers.
        The current options are LeastSquares, HuberT, RamsayE, AndrewWave,
        TrimmedMean, Hampel, and TukeyBiweight.  The default is HuberT().
        See sm2.robust.norms for more information.
    %(extra_params)s

    Notes
    -----

    **Attributes**

    df_model : float
        The degrees of freedom of the model.  The number of regressors p less
        one for the intercept.  Note that the reported model degrees
        of freedom does not count the intercept as a regressor, though
        the model is assumed to have an intercept.
    df_resid : float
        The residual degrees of freedom.  The number of observations n
        less the number of regressors p.  Note that here p does include
        the intercept as using a degree of freedom.
    endog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    exog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    M : sm2.robust.norms.RobustNorm
         See above.  Robust estimator instance instantiated.
    nobs : float
        The number of observations n
    pinv_wexog : array
        The pseudoinverse of the design / exogenous data array.  Note that
        RLM has no whiten method, so this is just the pseudo inverse of the
        design.
    normalized_cov_params : array
        The p x p normalized covariance of the design / exogenous data.
        This is approximately equal to (X.T X)^(-1)

    Examples
    ---------
    >>> import sm2.api as sm
    >>> data = sm.datasets.stackloss.load(as_pandas=False)
    >>> data.exog = sm.add_constant(data.exog)
    >>> rlm_model = sm.RLM(data.endog, data.exog, \
                           M=sm.robust.norms.HuberT())

    >>> rlm_results = rlm_model.fit()
    >>> rlm_results.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
    >>> rlm_results.bse
    array([ 0.11100521,  0.30293016,  0.12864961,  9.79189854])
    >>> rlm_results_HC2 = rlm_model.fit(cov="H2")
    >>> rlm_results_HC2.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
    >>> rlm_results_HC2.bse
    array([ 0.11945975,  0.32235497,  0.11796313,  9.08950419])
    >>> mod = sm.RLM(data.endog, data.exog, M=sm.robust.norms.Hampel())
    >>> rlm_hamp_hub = mod.fit(scale_est=sm.robust.scale.HuberScale())
    >>> rlm_hamp_hub.params
    array([  0.73175452,   1.25082038,  -0.14794399, -40.27122257])
    """ % {'params': base._model_params_doc,
           'extra_params': base._missing_param_doc}

    @cached_value
    def nobs(self):
        return float(self.endog.shape[0])

    @cached_value
    def df_resid(self):
        return self.nobs - (self.df_model + 1)

    @cached_value
    def df_model(self):
        rank = np.linalg.matrix_rank(self.exog)
        return rank - 1.0

    @property
    def _res_classes(self):
        return {"fit": (RLMResults, RLMResultsWrapper)}

    def __init__(self, endog, exog, M=None, missing='none', **kwargs):
        if M is None:
            M = norms.HuberT()
        self.M = M
        super(base.LikelihoodModel, self).__init__(endog, exog,
                                                   missing=missing, **kwargs)
        # things to remove_data
        self._data_attr.extend(['weights'])

    @cached_data
    def pinv_wexog(self):
        return np.linalg.pinv(self.exog)

    @cached_value
    def normalized_cov_params(self):
        return np.dot(self.pinv_wexog, np.transpose(self.pinv_wexog))

    def score(self, params):
        raise NotImplementedError

    # TODO: Redundant with version in linear_model?
    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array-like, optional after fit has been called
            Parameters of a linear model
        exog : array-like, optional.
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        # copied from linear_model
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

    def loglike(self, params):
        raise NotImplementedError

    def deviance(self, tmp_results):
        """
        Returns the (unnormalized) log-likelihood from the M estimator.
        """
        return self.M(
            (self.endog - tmp_results.fittedvalues) / tmp_results.scale).sum()

    def _update_history(self, tmp_results, history, conv):
        history['params'].append(tmp_results.params)
        history['scale'].append(tmp_results.scale)
        if conv == 'dev':
            history['deviance'].append(self.deviance(tmp_results))
        elif conv == 'sresid':
            history['sresid'].append(tmp_results.resid / tmp_results.scale)
        elif conv == 'weights':
            history['weights'].append(tmp_results.model.weights)
        return history

    def _estimate_scale(self, resid):
        """
        Estimates the scale based on the option provided to the fit method.
        """
        if isinstance(self.scale_est, str):
            if self.scale_est.lower() == 'mad':
                return scale.mad(resid, center=0)
            else:
                raise ValueError("Option %s for scale_est not understood" %
                                 self.scale_est)
        elif isinstance(self.scale_est, scale.HuberScale):
            return self.scale_est(self.df_resid, self.nobs, resid)
        else:
            return scale.scale_est(self, resid)**2

    def fit(self, maxiter=50, tol=1e-8, scale_est='mad', init=None, cov='H1',
            update_scale=True, conv='dev'):
        """
        Fits the model using iteratively reweighted least squares.

        The IRLS routine runs until the specified objective converges to `tol`
        or `maxiter` has been reached.

        Parameters
        ----------
        conv : string
            Indicates the convergence criteria.
            Available options are "coefs" (the coefficients), "weights" (the
            weights in the iteration), "sresid" (the standardized residuals),
            and "dev" (the un-normalized log-likelihood for the M
            estimator).  The default is "dev".
        cov : string, optional
            'H1', 'H2', or 'H3'
            Indicates how the covariance matrix is estimated.  Default is 'H1'.
            See rlm.RLMResults for more information.
        init : string
            Specifies method for the initial estimates of the parameters.
            Default is None, which means that the least squares estimate
            is used.  Currently it is the only available choice.
        maxiter : int
            The maximum number of iterations to try. Default is 50.
        scale_est : string or HuberScale()
            'mad' or HuberScale()
            Indicates the estimate to use for scaling the weights in the IRLS.
            The default is 'mad' (median absolute deviation.  Other options are
            'HuberScale' for Huber's proposal 2. Huber's proposal 2 has
            optional keyword arguments d, tol, and maxiter for specifying the
            tuning constant, the convergence tolerance, and the maximum number
            of iterations. See sm2.robust.scale for more information.
        tol : float
            The convergence tolerance of the estimate.  Default is 1e-8.
        update_scale : Bool
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.  Default is True.

        Returns
        -------
        results : sm2.rlm.RLMresults
        """
        if cov.upper() not in ["H1", "H2", "H3"]:
            raise ValueError("Covariance matrix %s not understood" % cov)
        else:
            self.cov = cov.upper()
        conv = conv.lower()
        if conv not in ["weights", "coefs", "dev", "sresid"]:
            raise ValueError("Convergence argument %s not understood" % conv)
        # TODO: Should scale_est attribute be set?
        self.scale_est = scale_est

        wls_results = lm.WLS(self.endog, self.exog).fit()
        if not init:
            self.scale = self._estimate_scale(wls_results.resid)

        history = dict(params=[np.inf], scale=[])
        if conv == 'coefs':
            criterion = history['params']
        elif conv == 'dev':
            history.update(dict(deviance=[np.inf]))
            criterion = history['deviance']
        elif conv == 'sresid':
            history.update(dict(sresid=[np.inf]))
            criterion = history['sresid']
        elif conv == 'weights':
            history.update(dict(weights=[np.inf]))
            criterion = history['weights']

        # done one iteration so update
        history = self._update_history(wls_results, history, conv)
        iteration = 1
        converged = 0
        while not converged:
            weights = self.M.weights(wls_results.resid / self.scale)
            wls_results = reg_tools._MinimalWLS(self.endog, self.exog,
                                                weights=weights,
                                                check_weights=True).fit()
            if update_scale is True:
                self.scale = self._estimate_scale(wls_results.resid)
            history = self._update_history(wls_results, history, conv)
            iteration += 1
            converged = _check_convergence(criterion, iteration, tol, maxiter)

        res_cls, wrap_cls = self._res_classes["fit"]

        results = res_cls(self, wls_results.params,
                          self.normalized_cov_params, self.scale,
                          weights=weights)

        history['iteration'] = iteration
        results.fit_history = history
        results.fit_options = dict(cov=cov.upper(), scale_est=scale_est,
                                   norm=self.M.__class__.__name__, conv=conv)
        # norm is not changed in fit, no old state

        # doing the next causes exception
        #self.cov = self.scale_est = None  # reset for additional fits
        # iteration and history could contain wrong state with repeated fit
        return wrap_cls(results)


class RLMResults(base.LikelihoodModelResults):
    """
    Class to contain RLM results

    Returns
    -------
    **Attributes**

    bcov_scaled : array
        p x p scaled covariance matrix specified in the model fit method.
        The default is H1. H1 is defined as
        ``k**2 * (1/df_resid*sum(M.psi(sresid)**2)*scale**2)/
        ((1/nobs*sum(M.psi_deriv(sresid)))**2) * (X.T X)^(-1)``

        where ``k = 1 + (df_model +1)/nobs * var_psiprime/m**2``
        where ``m = mean(M.psi_deriv(sresid))`` and
        ``var_psiprime = var(M.psi_deriv(sresid))``

        H2 is defined as
        ``k * (1/df_resid) * sum(M.psi(sresid)**2) *scale**2/
        ((1/nobs)*sum(M.psi_deriv(sresid)))*W_inv``

        H3 is defined as
        ``1/k * (1/df_resid * sum(M.psi(sresid)**2)*scale**2 *
        (W_inv X.T X W_inv))``

        where `k` is defined as above and
        ``W_inv = (M.psi_deriv(sresid) exog.T exog)^(-1)``

        See the technical documentation for cleaner formulae.
    bcov_unscaled : array
        The usual p x p covariance matrix with scale set equal to 1.  It
        is then just equivalent to normalized_cov_params.
    bse : array
        An array of the standard errors of the parameters.  The standard
        errors are taken from the robust covariance matrix specified in the
        argument to fit.
    chisq : array
        An array of the chi-squared values of the paramter estimates.
    df_model
        See RLM.df_model
    df_resid
        See RLM.df_resid
    fit_history : dict
        Contains information about the iterations. Its keys are `deviance`,
        `params`, `iteration` and the convergence criteria specified in
        `RLM.fit`, if different from `deviance` or `params`.
    fit_options : dict
        Contains the options given to fit.
    fittedvalues : array
        The linear predicted values.  dot(exog, params)
    model : sm2.rlm.RLM
        A reference to the model instance
    nobs : float
        The number of observations n
    normalized_cov_params : array
        See RLM.normalized_cov_params
    params : array
        The coefficients of the fitted model
    pinv_wexog : array
        See RLM.pinv_wexog
    pvalues : array
        The p values associated with `tvalues`. Note that `tvalues` are
        assumed to be distributed standard normal rather than Student's t.
    resid : array
        The residuals of the fitted model.  endog - fittedvalues
    scale : float
        The type of scale is determined in the arguments to the fit method in
        RLM.  The reported scale is taken from the residuals of the weighted
        least squares in the last IRLS iteration if update_scale is True.  If
        update_scale is False, then it is the scale given by the first OLS
        fit before the IRLS iterations.
    sresid : array
        The scaled residuals.
    tvalues : array
        The "t-statistics" of params. These are defined as params/bse where
        bse are taken from the robust covariance matrix specified in the
        argument to fit.
    weights : array
        The reported weights are determined by passing the scaled residuals
        from the last weighted least squares fit in the IRLS algortihm.

    See also
    --------
    sm2.base.model.LikelihoodModelResults
    """
    @cached_value
    def nobs(self):
        return float(self.model.endog.shape[0])

    @cached_value
    def df_resid(self):
        return self.nobs - (self.df_model + 1)

    @cached_value
    def df_model(self):
        rank = np.linalg.matrix_rank(self.model.exog)
        return rank - 1.0

    def __init__(self, model, params, normalized_cov_params, scale, weights):
        self.weights = weights
        self.model = model
        self._cache = resettable_cache()
        super(RLMResults, self).__init__(model, params,
                                         normalized_cov_params, scale)
        self._data_attr.append('weights')  # TODO: not wild about this

        self.cov_params_default = self.bcov_scaled
        # TODO: "pvals" should come from chisq on bse?

    @cached_data
    def sresid(self):
        return self.resid / self.scale

    @cached_value
    def bcov_unscaled(self):
        return self.normalized_cov_params

    @cache_readonly
    def bcov_scaled(self):
        model = self.model
        psi_derivs = model.M.psi_deriv(self.sresid)
        m = np.mean(psi_derivs)
        var_psiprime = np.var(psi_derivs)
        k = 1 + (self.df_model + 1) / self.nobs * var_psiprime / m**2

        if model.cov == "H1":
            return (k**2 * (1 / self.df_resid *
                            np.sum(model.M.psi(self.sresid)**2) *
                            self.scale**2) / (m**2) *
                    model.normalized_cov_params)
        else:
            W = np.dot(psi_derivs * model.exog.T,
                       model.exog)
            W_inv = np.linalg.inv(W)
            # [W_jk]^-1 = [SUM(psi_deriv(Sr_i)*x_ij*x_jk)]^-1
            # where Sr are the standardized residuals
            if model.cov == "H2":
                # These are correct, based on Huber (1973) 8.13
                return (k * (1 / self.df_resid) *
                        np.sum(model.M.psi(self.sresid)**2) *
                        self.scale**2 / m *
                        W_inv)
            elif model.cov == "H3":
                return (k**-1 *
                        1 / self.df_resid *
                        np.sum(model.M.psi(self.sresid)**2) *
                        self.scale**2 *
                        np.dot(np.dot(W_inv,
                                      np.dot(model.exog.T, model.exog)),
                               W_inv))

    # TODO: Use default implementation from base class?
    @cached_value
    def pvalues(self):
        return stats.norm.sf(np.abs(self.tvalues)) * 2

    @cached_value
    def bse(self):
        return np.sqrt(np.diag(self.bcov_scaled))

    # TODO: Use default implementation from base class?
    @cached_value
    def chisq(self):
        return (self.params / self.bse)**2

    def summary(self, yname=None, xname=None, title=0, alpha=.05,
                return_fmt='text'):
        """
        This is for testing the new summary setup
        """
        # TODO: is the docstring here accurate?  isnt that summary2?

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['IRLS']),
                    ('Norm:', [self.fit_options['norm']]),
                    ('Scale Est.:', [self.fit_options['scale_est']]),
                    ('Cov Type:', [self.fit_options['cov']]),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Iterations:', ["%d" % self.fit_history['iteration']])
                    ]
        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None)
                     ]

        if title is None:
            # TODO: Fix upstream this incorrectly is "if not title is None"
            title = "Robust linear Model Regression Results"

        from sm2.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                              use_t=self.use_t)

        # add warnings/notes, added to text format only
        etext = []
        wstr = ("If the model instance has been used for another "
                "fit with different fit\n"
                "parameters, then the fit options might not be the "
                "correct ones anymore .")
        etext.append(wstr)
        if etext:
            smry.add_extra_txt(etext)
        return smry


class RLMResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(RLMResultsWrapper, RLMResults)  # noqa:E305
