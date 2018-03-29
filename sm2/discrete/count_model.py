from __future__ import division

__all__ = ["ZeroInflatedPoisson", "ZeroInflatedGeneralizedPoisson",
           "ZeroInflatedNegativeBinomialP"]

import numpy as np

from sm2.tools.numdiff import approx_fprime, approx_hess
from sm2.tools.decorators import cache_readonly, copy_doc

import sm2.base.model as base
import sm2.base.wrapper as wrap

import sm2.regression.linear_model as lm
from sm2.discrete.discrete_model import (DiscreteModel, CountModel,
                                         Poisson, Logit, CountResults,
                                         L1CountResults, Probit,
                                         _discrete_results_docs,
                                         GeneralizedPoisson,
                                         NegativeBinomialP)
from sm2.distributions import zipoisson, zigenpoisson, zinegbin


_doc_zi_params = """
    exog_infl : array_like or None
        Explanatory variables for the binary inflation model, i.e. for
        mixing probability model. If None, then a constant is used.
    offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.
    inflation : string, 'logit' or 'probit'
        The model for the zero inflation, either Logit (default) or Probit
    """


class GenericZeroInflated(CountModel):
    __doc__ = """
    Generiz Zero Inflated model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    exog_infl: array
        A reference to the zero-inflated exogenous design.
    """ % {'params': base._model_params_doc,
           'extra_params': _doc_zi_params + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None,
                 inflation='logit', exposure=None, missing='none', **kwargs):
        super(GenericZeroInflated, self).__init__(endog, exog, offset=offset,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)

        if exog_infl is None:
            self.k_inflate = 1
            self.exog_infl = np.ones((endog.size, self.k_inflate),
                                     dtype=np.float64)
        else:
            self.exog_infl = exog_infl
            self.k_inflate = exog_infl.shape[1]

        self.infl = inflation
        if inflation == 'logit':
            self.model_infl = Logit(np.zeros(self.exog_infl.shape[0]),
                                    self.exog_infl)
            self._hessian_inflate = self._hessian_logit
        elif inflation == 'probit':
            self.model_infl = Probit(np.zeros(self.exog_infl.shape[0]),
                                     self.exog_infl)
            self._hessian_inflate = self._hessian_probit
        else:
            raise ValueError("inflation == %s, which is not handled"
                             % inflation)  # pragma: no cover

        self.inflation = inflation
        self.k_extra = self.k_inflate

        if len(self.exog) != len(self.exog_infl):
            raise ValueError('exog and exog_infl have different number of'
                             'observation. `missing` handling is '
                             'not supported')  # pragma: no cover

        infl_names = ['inflate_%s' % i
                      for i in self.model_infl.data.param_names]
        self.exog_names[:] = infl_names + list(self.exog_names)
        self.exog_infl = np.asarray(self.exog_infl, dtype=np.float64)

        self._init_keys.extend(['exog_infl', 'inflation'])
        self._null_drop_keys = ['exog_infl']

    def loglikeobs(self, params):
        r"""
        Loglikelihood for observations of Generic Zero Inflated model

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
        .. math:: \ln L=\ln(w_{i}+(1-w_{i})*P_{main\_model})+
            \ln(1-w_{i})+L_{main\_model}
            where P - pdf of main model, L - loglike function of main model.

        for observations :math:`i=1,...,n`
        """
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        y = self.endog
        w = self.model_infl.predict(params_infl)

        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        llf_main = self.model_main.loglikeobs(params_main)
        zero_idx = np.nonzero(y == 0)[0]  # TODO: Cache some of these?
        nonzero_idx = np.nonzero(y)[0]

        llf = np.zeros_like(y, dtype=np.float64)
        llf[zero_idx] = (np.log(w[zero_idx] +
                         (1 - w[zero_idx]) * np.exp(llf_main[zero_idx])))
        llf[nonzero_idx] = np.log(1 - w[nonzero_idx]) + llf_main[nonzero_idx]

        return llf

    @copy_doc(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):

        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            start_params = self._get_start_params()

        if callback is None:
            # work around perfect separation callback GH#3895
            callback = lambda *x: x

        mlefit = super(GenericZeroInflated, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            disp=disp, method=method,
            full_output=full_output,
            callback=callback,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
            **kwargs)

        res_cls, wrap_cls = self._res_classes["fit"]
        zipfit = res_cls(self, mlefit._results)
        return wrap_cls(zipfit)

    @copy_doc(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1',
                        maxiter='defined_by_method', full_output=1, disp=1,
                        callback=None, alpha=0, trim_mode='auto',
                        auto_trim_tol=0.01, size_trim_tol=1e-4,
                        qc_tol=0.03, **kwargs):

        if method not in ['l1', 'l1_cvxopt_cp']:  # pragma: no cover
            raise ValueError("argument method == %s, which is not "
                             "handled" % method)

        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.k_exog + self.k_inflate
            alpha = alpha * np.ones(k_params)

        extra = self.k_extra - self.k_inflate
        if self.k_extra and np.size(alpha) > 1:
            alpha_p = alpha[:-(self.k_extra - extra)]
        else:
            alpha_p = alpha

        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            start_params = self.model_main.fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=0, callback=callback,
                alpha=alpha_p, trim_mode=trim_mode,
                auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
            start_params = np.append(np.ones(self.k_inflate), start_params)

        cntfit = super(CountModel, self).fit_regularized(
            start_params=start_params, method=method, maxiter=maxiter,
            full_output=full_output, disp=disp, callback=callback,
            alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
            size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)

        res_cls, wrap_cls = self._res_classes["fit_regularized"]
        discretefit = res_cls(self, cntfit)
        return wrap_cls(discretefit)

    def score_obs(self, params):
        """
        Generic Zero Inflated model score (gradient) vector of the
        log-likelihood

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
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        y = self.endog
        w = self.model_infl.predict(params_infl)
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        score_main = self.model_main.score_obs(params_main)
        llf_main = self.model_main.loglikeobs(params_main)
        llf = self.loglikeobs(params)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        dldp = np.zeros((self.exog.shape[0], self.k_exog), dtype=np.float64)
        dldw = np.zeros_like(self.exog_infl, dtype=np.float64)

        dldp[zero_idx, :] = (score_main[zero_idx].T *
                             (1 - (w[zero_idx]) / np.exp(llf[zero_idx]))).T
        dldp[nonzero_idx, :] = score_main[nonzero_idx]

        if self.inflation == 'logit':
            dldw[zero_idx, :] = (self.exog_infl[zero_idx].T * w[zero_idx] *
                                 (1 - w[zero_idx]) *
                                 (1 - np.exp(llf_main[zero_idx])) /
                                 np.exp(llf[zero_idx])).T
            dldw[nonzero_idx, :] = -(self.exog_infl[nonzero_idx].T *
                                     w[nonzero_idx]).T
        elif self.inflation == 'probit':
            # TODO: Maybe do this _before_ all the junk above?
            return approx_fprime(params, self.loglikeobs)

        return np.hstack((dldw, dldp))

    def _hessian_main(self, params):
        pass

    def _hessian_logit(self, params):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        y = self.endog
        w = self.model_infl.predict(params_infl)
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        score_main = self.model_main.score_obs(params_main)
        llf_main = self.model_main.loglikeobs(params_main)
        llf = self.loglikeobs(params)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        hess_arr = np.zeros((self.k_inflate, self.k_exog + self.k_inflate))

        pmf = np.exp(llf)

        # d2l/dw2
        for i in range(self.k_inflate):
            for j in range(i, -1, -1):
                hess_arr[i, j] = ((
                    self.exog_infl[zero_idx, i] * self.exog_infl[zero_idx, j] *
                    (w[zero_idx] * (1 - w[zero_idx]) * (
                        (1 - np.exp(llf_main[zero_idx])) *
                        (1 - 2 * w[zero_idx]) * np.exp(llf[zero_idx]) -
                        (w[zero_idx] - w[zero_idx]**2) *
                        (1 - np.exp(llf_main[zero_idx]))**2) /
                     pmf[zero_idx]**2)).sum() -
                    (self.exog_infl[nonzero_idx, i] *
                     self.exog_infl[nonzero_idx, j] *
                     w[nonzero_idx] * (1 - w[nonzero_idx])).sum())

        # d2l/dpdw
        for i in range(self.k_inflate):
            for j in range(self.k_exog):
                hess_arr[i, j + self.k_inflate] = -(
                    score_main[zero_idx, j] *
                    w[zero_idx] * (1 - w[zero_idx]) *
                    self.exog_infl[zero_idx, i] / pmf[zero_idx]).sum()

        return hess_arr

    def _hessian_probit(self, params):
        pass
        # TODO: Should this maybe raise?

    def hessian(self, params):
        """
        Generic Zero Inflated model Hessian matrix of the loglikelihood

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
        hess_arr_main = self._hessian_main(params)
        hess_arr_infl = self._hessian_inflate(params)

        if hess_arr_main is None or hess_arr_infl is None:
            # TODO: approx_hess is quite slow.  Are there any analytic options?
            return approx_hess(params, self.loglike)

        dim = self.k_exog + self.k_inflate

        hess_arr = np.zeros((dim, dim))

        hess_arr[:self.k_inflate, :] = hess_arr_infl
        hess_arr[self.k_inflate:, self.k_inflate:] = hess_arr_main

        tri_idx = np.triu_indices(self.k_exog + self.k_inflate, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]

        return hess_arr

    def predict(self, params, exog=None, exog_infl=None, exposure=None,
                offset=None, which='mean'):
        """
        Predict response variable of a count model given exogenous variables.

        Parameters
        ----------
        params : array-like
            The parameters of the model
        exog : array, optional
            A reference to the exogenous design.
            If not assigned, will be used exog from fitting.
        exog_infl : array, optional
            A reference to the zero-inflated exogenous design.
            If not assigned, will be used exog from fitting.
        offset : array, optional
            Offset is added to the linear prediction with coefficient equal
            to 1.
        exposure : array, optional
            Log(exposure) is added to the linear prediction with coefficient
            equal to 1. If exposure is specified, then it will be logged
            by the method.
            The user does not need to log it first.
        which : string, optional
            Define values that will be predicted.
            'mean', 'mean-main', 'linear', 'mean-nonzero', 'prob-zero,
            'prob', 'prob-main'
            Default is 'mean'.
        """
        if exog is None:
            exog = self.exog

        if exog_infl is None:
            exog_infl = self.exog_infl

        if exposure is None:
            exposure = getattr(self, 'exposure', 0)
        else:
            exposure = np.log(exposure)

        if offset is None:
            offset = 0

        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        prob_main = 1 - self.model_infl.predict(params_infl, exog_infl)

        Xb = np.dot(exog, params_main[:self.exog.shape[1]])
        lin_pred = Xb + exposure + offset

        # Refactor: This is pretty hacky,
        # there should be an appropriate predict method in model_main
        # this is just prob(y=0 | model_main)
        tmp_exog = self.model_main.exog
        tmp_endog = self.model_main.endog
        tmp_offset = getattr(self.model_main, 'offset', ['no'])
        tmp_exposure = getattr(self.model_main, 'exposure', ['no'])
        self.model_main.exog = exog
        self.model_main.endog = np.zeros((exog.shape[0]))
        self.model_main.offset = offset
        self.model_main.exposure = exposure
        llf = self.model_main.loglikeobs(params_main)
        self.model_main.exog = tmp_exog
        self.model_main.endog = tmp_endog
        # tmp_offset might be an array with elementwise equality testing
        if len(tmp_offset) == 1 and tmp_offset[0] == 'no':
            del self.model_main.offset
        else:
            self.model_main.offset = tmp_offset
        if len(tmp_exposure) == 1 and tmp_exposure[0] == 'no':
            del self.model_main.exposure
        else:
            self.model_main.exposure = tmp_exposure
        # end hack

        prob_zero = (1 - prob_main) + prob_main * np.exp(llf)

        if which == 'mean':
            return prob_main * np.exp(lin_pred)
        elif which == 'mean-main':
            return np.exp(lin_pred)
        elif which == 'linear':
            return lin_pred
        elif which == 'mean-nonzero':
            return prob_main * np.exp(lin_pred) / (1 - prob_zero)
        elif which == 'prob-zero':
            return prob_zero
        elif which == 'prob-main':
            return prob_main
        elif which == 'prob':
            return self._predict_prob(params, exog, exog_infl,
                                      exposure, offset)
        else:  # pragma: no cover
            raise ValueError('which = %s is not available' % which)


class ZeroInflatedPoisson(GenericZeroInflated):
    __doc__ = """
    Poisson Zero Inflated model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    exog_infl: array
        A reference to the zero-inflated exogenous design.
    """ % {'params': base._model_params_doc,
           'extra_params': _doc_zi_params + base._missing_param_doc}

    @property
    def _res_classes(self):
        return {
            "fit": (ZeroInflatedPoissonResults,
                    ZeroInflatedPoissonResultsWrapper),
            "fit_regularized": (L1ZeroInflatedPoissonResults,
                                L1ZeroInflatedPoissonResultsWrapper)}

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None,
                 inflation='logit', missing='none', **kwargs):
        super(ZeroInflatedPoisson, self).__init__(endog, exog, offset=offset,
                                                  inflation=inflation,
                                                  exog_infl=exog_infl,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)
        self.model_main = Poisson(self.endog, self.exog, offset=offset,
                                  exposure=exposure)
        self.distribution = zipoisson

    def _hessian_main(self, params):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        exog = self.exog
        y = self.endog
        w = self.model_infl.predict(params_infl)
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        mu = self.model_main.predict(params_main)

        hess_arr = np.zeros((self.k_exog, self.k_exog))

        coeff = (1 + w[zero_idx] * (np.exp(mu[zero_idx]) - 1))
        # d2l/dp2
        for i in range(self.k_exog):
            for j in range(i, -1, -1):
                hess_arr[i, j] = ((
                    exog[zero_idx, i] * exog[zero_idx, j] *
                    mu[zero_idx] * (w[zero_idx] - 1) * (
                        1 / coeff -
                        w[zero_idx] * mu[zero_idx] *
                        np.exp(mu[zero_idx]) / coeff**2)).sum() -
                    (mu[nonzero_idx] *
                     exog[nonzero_idx, i] * exog[nonzero_idx, j]).sum())

        return hess_arr

    def _predict_prob(self, params, exog, exog_infl, exposure, offset):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        counts = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))

        w = self.model_infl.predict(params_infl, exog_infl)
        if len(exog_infl.shape) < 2:
            transform = True
            w = np.atleast_2d(w)
        else:
            transform = False

        w = w[:, None]
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        mu = self.model_main.predict(params_main, exog,
                                     offset=offset)[:, None]
        result = self.distribution.pmf(counts, mu, w)
        return result[0] if transform else result

    def _get_start_params(self, start_params=None):
        if start_params is None:
            start_params = self.model_main.fit(disp=0, method="nm").params
            start_params = np.append(np.ones(self.k_inflate) * 0.1,
                                     start_params)
        return start_params


class ZeroInflatedGeneralizedPoisson(GenericZeroInflated):
    __doc__ = """
    Zero Inflated Generalized Poisson model for count data

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    exog_infl: array
        A reference to the zero-inflated exogenous design.
    p: scalar
        P denotes parametrizations for ZIGP regression.
    """ % {'params': base._model_params_doc,
           'extra_params': _doc_zi_params +
           """p : float
        dispersion power parameter for the GeneralizedPoisson model.  p=1 for
        ZIGP-1 and p=2 for ZIGP-2. Default is p=2
    """ + base._missing_param_doc}

    @property
    def _res_classes(self):
        return {
            "fit": (ZeroInflatedGeneralizedPoissonResults,
                    ZeroInflatedGeneralizedPoissonResultsWrapper),
            "fit_regularized": (
                L1ZeroInflatedGeneralizedPoissonResults,
                L1ZeroInflatedGeneralizedPoissonResultsWrapper)}

    @property
    def k_exog(self):
        # TODO: maybe document WHY this is different from elsewhere?
        return self.exog.shape[1] + 1

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None,
                 inflation='logit', p=2, missing='none', **kwargs):
        cls = ZeroInflatedGeneralizedPoisson
        super(cls, self).__init__(endog, exog,
                                  offset=offset,
                                  inflation=inflation,
                                  exog_infl=exog_infl,
                                  exposure=exposure,
                                  missing=missing,
                                  **kwargs)
        self.model_main = GeneralizedPoisson(self.endog, self.exog,
                                             offset=offset,
                                             exposure=exposure, p=p)
        self.distribution = zigenpoisson
        self.k_extra += 1
        self.exog_names.append("alpha")

    def _get_init_kwds(self):
        kwds = super(ZeroInflatedGeneralizedPoisson, self)._get_init_kwds()
        kwds['p'] = self.model_main.parameterization + 1
        return kwds

    def _predict_prob(self, params, exog, exog_infl, exposure, offset):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        p = self.model_main.parameterization
        counts = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))

        w = self.model_infl.predict(params_infl, exog_infl)
        if len(exog_infl.shape) < 2:
            transform = True
            w = np.atleast_2d(w)
        else:
            transform = False

        w = w[:, None]
        w[w == 1.] = np.nextafter(1, 0)
        mu = self.model_main.predict(params_main, exog,
                                     exposure=exposure, offset=offset)[:, None]
        result = self.distribution.pmf(counts, mu, params_main[-1], p, w)
        return result[0] if transform else result

    def _get_start_params(self, start_params=None):
        if start_params is None:
            zmod = ZeroInflatedPoisson(self.endog, self.exog,
                                       exog_infl=self.exog_infl)
            start_params = zmod.fit(disp=0).params
            start_params = np.append(start_params, 0.1)
        return start_params


class ZeroInflatedNegativeBinomialP(GenericZeroInflated):
    __doc__ = """
    Zero Inflated Generalized Negative Binomial model for count data

    %(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    exog_infl: array
        A reference to the zero-inflated exogenous design.
    p: scalar
        P denotes parametrizations for ZINB regression. p=1 for ZINB-1 and
    p=2 for ZINB-2. Default is p=2
    """ % {'params': base._model_params_doc,
           'extra_params': _doc_zi_params +
           """p : float
        dispersion power parameter for the NegativeBinomialP model.  p=1 for
        ZINB-1 and p=2 for ZINM-2. Default is p=2
    """ + base._missing_param_doc}

    @property
    def _res_classes(self):
        return {
            "fit": (ZeroInflatedNegativeBinomialResults,
                    ZeroInflatedNegativeBinomialResultsWrapper),
            "fit_regularized": (L1ZeroInflatedNegativeBinomialResults,
                                L1ZeroInflatedNegativeBinomialResultsWrapper)}

    @property
    def k_exog(self):
        # TODO: maybe document WHY this is different from elsewhere?
        return self.exog.shape[1] + 1

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None,
                 inflation='logit', p=2, missing='none', **kwargs):
        cls = ZeroInflatedNegativeBinomialP
        super(cls, self).__init__(endog, exog,
                                  offset=offset,
                                  inflation=inflation,
                                  exog_infl=exog_infl,
                                  exposure=exposure,
                                  missing=missing, **kwargs)

        self.model_main = NegativeBinomialP(self.endog, self.exog,
                                            offset=offset,
                                            exposure=exposure, p=p)
        self.distribution = zinegbin
        self.k_extra += 1
        self.exog_names.append("alpha")

    def _get_init_kwds(self):
        kwds = super(ZeroInflatedNegativeBinomialP, self)._get_init_kwds()
        kwds['p'] = self.model_main.parameterization
        return kwds

    def _predict_prob(self, params, exog, exog_infl, exposure, offset):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        p = self.model_main.parameterization
        counts = np.arange(0, np.max(self.endog) + 1)

        w = self.model_infl.predict(params_infl, exog_infl)
        if len(exog_infl.shape) < 2:
            transform = True
            w = np.atleast_2d(w)
        else:
            transform = False

        w = w[:, None]
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        mu = self.model_main.predict(params_main, exog,
                                     exposure=exposure, offset=offset)[:, None]
        result = self.distribution.pmf(counts, mu, params_main[-1], p, w)
        return result[0] if transform else result

    def _get_start_params(self, start_params=None):
        if start_params is None:
            start_params = self.model_main.fit(disp=0, method='nm').params
            start_params = np.append(np.zeros(self.k_inflate), start_params)
        return start_params


class ZeroInflatedPoissonResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Zero Inflated Poisson",
        "extra_attr": ""}

    @cache_readonly
    def _dispersion_factor(self):
        mu = self.predict(which='linear')
        w = 1 - self.predict() / np.exp(self.predict(which='linear'))
        return (1 + w * np.exp(mu))

    def get_margeff(self, at='overall', method='dydx', atexog=None,
                    dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        raise NotImplementedError("not yet implemented for zero inflation")


class L1ZeroInflatedPoissonResults(L1CountResults, ZeroInflatedPoissonResults):
    pass


class ZeroInflatedGeneralizedPoissonResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Zero Inflated "
                                "Generalized Poisson",
        "extra_attr": ""}

    @cache_readonly
    def _dispersion_factor(self):
        p = self.model.model_main.parameterization
        alpha = self.params[self.model.k_inflate:][-1]
        mu = np.exp(self.predict(which='linear'))
        w = 1 - self.predict() / mu
        return ((1 + alpha * mu**p)**2 + w * mu)

    def get_margeff(self, at='overall', method='dydx', atexog=None,
                    dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        raise NotImplementedError("not yet implemented for zero inflation")


class L1ZeroInflatedGeneralizedPoissonResults(L1CountResults,
        ZeroInflatedGeneralizedPoissonResults):  # noqa:E128
    pass


class ZeroInflatedNegativeBinomialResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Zero Inflated "
                                "Generalized Negative Binomial",
        "extra_attr": ""}

    @cache_readonly
    def _dispersion_factor(self):
        p = self.model.model_main.parameterization
        alpha = self.params[self.model.k_inflate:][-1]
        mu = np.exp(self.predict(which='linear'))
        w = 1 - self.predict() / mu
        return (1 + alpha * mu**(p - 1) + w * mu)

    def get_margeff(self, at='overall', method='dydx', atexog=None,
                    dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        raise NotImplementedError("not yet implemented for zero inflation")


class L1ZeroInflatedNegativeBinomialResults(L1CountResults,
        ZeroInflatedNegativeBinomialResults):  # noqa:E128
    pass


# -------------------------------------------------------------
# Wrapper Classes


class ZeroInflatedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(ZeroInflatedPoissonResultsWrapper,  # noqa:E305
                      ZeroInflatedPoissonResults)


class L1ZeroInflatedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1ZeroInflatedPoissonResultsWrapper,  # noqa:E305
                      L1ZeroInflatedPoissonResults)


class ZeroInflatedGeneralizedPoissonResultsWrapper(
        lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(  # noqa:E305
    ZeroInflatedGeneralizedPoissonResultsWrapper,
    ZeroInflatedGeneralizedPoissonResults)


class L1ZeroInflatedGeneralizedPoissonResultsWrapper(
        lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(  # noqa:E305
    L1ZeroInflatedGeneralizedPoissonResultsWrapper,
    L1ZeroInflatedGeneralizedPoissonResults)


class ZeroInflatedNegativeBinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(  # noqa:E305
    ZeroInflatedNegativeBinomialResultsWrapper,
    ZeroInflatedNegativeBinomialResults)


class L1ZeroInflatedNegativeBinomialResultsWrapper(
        lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(  # noqa:E305
    L1ZeroInflatedNegativeBinomialResultsWrapper,
    L1ZeroInflatedNegativeBinomialResults)
