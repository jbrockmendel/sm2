# -*- coding: utf-8 -*-
"""
Impulse reponse-related code
"""

from __future__ import division

from six.moves import range

import numpy as np
import scipy.linalg  # TODO: can we just use np.linalg.inv?

from sm2.tools.decorators import cache_readonly
from sm2.tools.tools import chain_dot

from sm2.tsa import tsatools
from . import util, plotting


class BaseIRAnalysis(object):
    """
    Base class for plotting and computing IRF-related statistics, want to be
    able to handle known and estimated processes
    """

    # TODO: `model` is a misnomer; this is a Results object
    def __init__(self, model, P=None, periods=10, order=None, svar=False,
                 vecm=False):
        self.model = model
        self.periods = periods
        self.nobs = model.nobs
        self.k_ar = model.k_ar
        self.neqs, self.lags, self.T = model.neqs, model.k_ar, model.nobs
        # TODO: Dont use non-standard `T`

        self.order = order

        if P is None:
            sigma = model.sigma_u

            # TODO, may be difficult at the moment
            # if order is not None:
            #     indexer = [model.get_eq_index(name) for name in order]
            #     sigma = sigma[:, indexer][indexer, :]

            #     if sigma.shape != model.sigma_u.shape:
            #         raise ValueError('variable order is wrong length')

            P = np.linalg.cholesky(sigma)

        self.P = P

        self.svar = svar

        self.irfs = model.ma_rep(periods)
        if svar:
            self.svar_irfs = model.svar_ma_rep(periods, P=P)
        else:
            self.orth_irfs = model.orth_ma_rep(periods, P=P)

        self.cum_effects = self.irfs.cumsum(axis=0)
        if svar:
            self.svar_cum_effects = self.svar_irfs.cumsum(axis=0)
        else:
            self.orth_cum_effects = self.orth_irfs.cumsum(axis=0)

        # long-run effects may be infinite for VECMs.
        if not vecm:
            self.lr_effects = model.long_run_effects()
            if svar:
                self.svar_lr_effects = np.dot(model.long_run_effects(), P)
            else:
                self.orth_lr_effects = np.dot(model.long_run_effects(), P)

        # auxiliary stuff
        if vecm:
            self._A = util.comp_matrix(model.var_rep)
        else:
            self._A = util.comp_matrix(model.coefs)

    def _choose_irfs(self, orth=False, svar=False):
        # TODO: require not (orth and svar)?
        if orth:
            return self.orth_irfs
        elif svar:
            return self.svar_irfs
        else:
            return self.irfs

    def cov(self, *args, **kwargs):
        raise NotImplementedError

    def cum_effect_cov(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, orth=False, impulse=None, response=None,
             signif=0.05, plot_params=None, subplot_params=None,
             plot_stderr=True, stderr_type='asym', repl=1000,
             seed=None, component=None):
        """
        Plot impulse responses

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        impulse : string or int
            variable providing the impulse
        response : string or int
            variable affected by the impulse
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        subplot_params : dict
            To pass to subplot plotting funcions. Example: if fonts are
            too big, pass {'fontsize' : 8} or some number to your taste.
        plot_params : dict

        plot_stderr: bool, default True
            Plot standard impulse response error bands
        stderr_type: string
            'asym': default, computes asymptotic standard errors
            'mc': monte carlo standard errors (use rpl)
        repl: int, default 1000
            Number of replications for Monte Carlo and Sims-Zha standard errors
        seed: int
            np.random.seed for Monte Carlo replications
        component: array or vector of principal component indices
        """
        svar = self.svar

        if orth and svar:  # pragma: no cover
            raise ValueError("For SVAR system, set orth=False")

        irfs = self._choose_irfs(orth, svar)
        if orth:
            title = 'Impulse responses (orthogonalized)'
        elif svar:
            title = 'Impulse responses (structural)'
        else:
            title = 'Impulse responses'

        if plot_stderr is False:
            stderr = None
        elif stderr_type == 'asym':
            stderr = self.cov(orth=orth)
        elif stderr_type == 'mc':
            stderr = self.errband_mc(orth=orth, svar=svar,
                                     repl=repl, signif=signif,
                                     seed=seed)
        elif stderr_type == 'sz1':
            stderr = self.err_band_sz1(orth=orth, svar=svar,
                                       repl=repl, signif=signif,
                                       seed=seed,
                                       component=component)
        elif stderr_type == 'sz2':
            stderr = self.err_band_sz2(orth=orth, svar=svar,
                                       repl=repl, signif=signif,
                                       seed=seed,
                                       component=component)
        elif stderr_type == 'sz3':
            stderr = self.err_band_sz3(orth=orth, svar=svar,
                                       repl=repl, signif=signif,
                                       seed=seed,
                                       component=component)
        else:
            raise ValueError("Error type must be either "
                             "'asym', 'mc', 'sz1', 'sz2', "
                             "or 'sz3'")  # pragma: no cover

        plotting.irf_grid_plot(irfs, stderr, impulse, response,
                               self.model.names, title, signif=signif,
                               subplot_params=subplot_params,
                               plot_params=plot_params,
                               stderr_type=stderr_type)

    def plot_cum_effects(self, orth=False, impulse=None, response=None,
                         signif=0.05, plot_params=None,
                         subplot_params=None, plot_stderr=True,
                         stderr_type='asym', repl=1000, seed=None):
        """
        Plot cumulative impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        impulse : string or int
            variable providing the impulse
        response : string or int
            variable affected by the impulse
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        subplot_params : dict
            To pass to subplot plotting funcions. Example: if fonts are
            too big, pass {'fontsize' : 8} or some number to your taste.
        plot_params : dict

        plot_stderr: bool, default True
            Plot standard impulse response error bands
        stderr_type: string
            'asym': default, computes asymptotic standard errors
            'mc': monte carlo standard errors (use rpl)
        repl: int, default 1000
            Number of replications for monte carlo standard errors
        seed: int
            np.random.seed for Monte Carlo replications
        """
        if orth:
            title = 'Cumulative responses responses (orthogonalized)'
            cum_effects = self.orth_cum_effects
            lr_effects = self.orth_lr_effects
        else:
            title = 'Cumulative responses'
            cum_effects = self.cum_effects
            lr_effects = self.lr_effects

        if stderr_type not in ['asym', 'mc']:  # pragma: no cover
            # TODO: Upstream this TypeError, needs fixing
            raise ValueError("stderr_type '%s' not recognized" % stderr_type)
        else:
            if stderr_type == 'asym':
                stderr = self.cum_effect_cov(orth=orth)
            if stderr_type == 'mc':
                stderr = self.cum_errband_mc(orth=orth, repl=repl,
                                             signif=signif, seed=seed)
        if not plot_stderr:
            stderr = None

        plotting.irf_grid_plot(cum_effects, stderr, impulse, response,
                               self.model.names, title, signif=signif,
                               hlines=lr_effects,
                               subplot_params=subplot_params,
                               plot_params=plot_params,
                               stderr_type=stderr_type)


# TODO: Whats the use case for this vs BaseIRAnalysis?
class IRAnalysis(BaseIRAnalysis):
    """
    Impulse response analysis class. Computes impulse responses, asymptotic
    standard errors, and produces relevant plots

    Parameters
    ----------
    model : VAR instance

    Notes
    -----
    Using Lütkepohl (2005) notation
    """
    def __init__(self, model, P=None, periods=10, order=None, svar=False,
                 vecm=False):
        BaseIRAnalysis.__init__(self, model, P=P, periods=periods,
                                order=order, svar=svar, vecm=vecm)

        if vecm:
            self.cov_a = model.cov_var_repr
        else:
            self.cov_a = model._cov_alpha
        self.cov_sig = model._cov_sigma

        # memoize dict for G matrix function
        self._g_memo = {}

    def cov(self, orth=False):
        """
        Compute asymptotic standard errors for impulse response coefficients

        Notes
        -----
        Lütkepohl eq 3.7.5
        """
        if orth:
            return self._orth_cov()

        covs = self._empty_covm(self.periods + 1)
        covs[0] = np.zeros((self.neqs ** 2, self.neqs ** 2))
        for i in range(1, self.periods + 1):
            Gi = self.G[i - 1]
            covs[i] = chain_dot(Gi, self.cov_a, Gi.T)

        return covs

    def errband_mc(self, orth=False, svar=False, repl=1000,
                   signif=0.05, seed=None, burn=100):
        """
        IRF Monte Carlo integrated error bands
        """
        model = self.model
        periods = self.periods
        if svar:
            return model.sirf_errband_mc(orth=orth, repl=repl, T=periods,
                                         signif=signif, seed=seed,
                                         burn=burn, cum=False)
        else:
            return self.irf_errband_mc(orth=orth, repl=repl, T=periods,
                                       signif=signif, seed=seed,
                                       burn=burn, cum=False)

    # TODO: De-dup highly redundant docstring arguments
    # upstream this is implemented directly in VARResults
    def irf_resim(self, orth=False, repl=1000, T=10,
                  seed=None, burn=100, cum=False):
        """
        Simulates impulse response function, returning an array of simulations.
        Used for Sims-Zha error band calculation.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int, default 1000
            Number of Monte Carlo replications to perform
        T: int, default 10
            number of impulse response periods
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        seed : int, default None
            np.random.seed for replications
        burn : int, default 100
            Number of initial simulated obs to discard
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Sims, Christoper A., and Tao Zha. 1999.
            "Error Bands for Impulse Response." Econometrica 67: 1113-1155.

        Returns
        -------
        Array of simulated impulse response functions
        """
        from sm2.tsa.vector_ar import var_model, util
        model = self.model
        coefs = model.coefs
        sigma_u = model.sigma_u
        intercept = model.intercept
        k_ar = self.k_ar
        neqs = self.neqs
        nobs = self.nobs

        ma_coll = np.zeros((repl, T + 1, neqs, neqs))

        def fill_coll(sim):
            ret = var_model.VAR(sim, exog=model.exog).fit(maxlags=k_ar,
                                                          trend=model.trend)
            ret = ret.orth_ma_rep(maxn=T) if orth else ret.ma_rep(maxn=T)
            return ret.cumsum(axis=0) if cum else ret

        for i in range(repl):
            # discard first hundred to eliminate correct for starting bias
            sim = util.varsim(coefs, intercept, sigma_u,
                              seed=seed, steps=nobs + burn)
            sim = sim[burn:]
            ma_coll[i, :, :, :] = fill_coll(sim)

        return ma_coll
        # TODO: If it weren't for model.exog, this could go higher in the
        # inheritance hierarchy

    # upstream this is implemented directly in VARResults
    def irf_errband_mc(self, orth=False, repl=1000, T=10,
                       signif=0.05, seed=None, burn=100, cum=False):
        # Monte Carlo irf standard errors
        """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int, default 1000
            Number of Monte Carlo replications to perform
        T : int, default 10
            number of impulse response periods
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        seed : int, default None
            np.random.seed for replications
        burn : int, default 100
            Number of initial simulated obs to discard
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Lütkepohl (2005) Appendix D

        Returns
        -------
        Tuple of lower and upper arrays of ma_rep monte carlo standard errors
        """
        ma_coll = self.irf_resim(orth=orth, repl=repl, T=T,
                                 seed=seed, burn=burn, cum=cum)

        # TODO: This block is really similar to _fill_irfs
        ma_sort = np.sort(ma_coll, axis=0)  # sort to get quantiles
        index = (int(round(signif / 2 * repl) - 1),
                 int(round((1 - signif / 2) * repl) - 1))
        lower = ma_sort[index[0], :, :, :]
        upper = ma_sort[index[1], :, :, :]
        return lower, upper
        # TODO: If it weren't for model.exog (in irf_resim), this could go
        # higher in the inheritance hierarchy

    def err_band_sz1(self, orth=False, svar=False, repl=1000,
                     signif=0.05, seed=None, burn=100, component=None):
        """
        IRF Sims-Zha error band method 1. Assumes symmetric error bands around
        mean.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int, default 1000
            Number of Monte Carlo replications to perform
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        seed : int, default None
            np.random.seed for replications
        burn : int, default 100
            Number of initial simulated obs to discard
        component : neqs x neqs array, default to largest for each
            Index of column of eigenvector/value to use for each error band
            Note: period of impulse (t=0) is not included when computing
            principle component

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """
        periods = self.periods
        irfs = self._choose_irfs(orth, svar)
        neqs = self.neqs
        irf_resim = self.irf_resim(orth=orth, repl=repl, T=periods, seed=seed,
                                   burn=100)
        q = util.norm_signif_level(signif)

        W, eigva, k = self._eigval_decomp_SZ(irf_resim)
        k = _validate_component(component, neqs, periods, dims=2, k=k)

        # here take the kth column of W, which we determine by finding
        # the largest eigenvalue of the covariance matrix
        lower = np.copy(irfs)
        upper = np.copy(irfs)
        for i in range(neqs):
            for j in range(neqs):
                band = W[i, j, :, k[i, j]] * q * np.sqrt(eigva[i, j, k[i, j]])
                lower[1:, i, j] = irfs[1:, i, j] + band
                upper[1:, i, j] = irfs[1:, i, j] - band

        return lower, upper

    def err_band_sz2(self, orth=False, svar=False, repl=1000, signif=0.05,
                     seed=None, burn=100, component=None):
        """
        IRF Sims-Zha error band method 2.

        This method Does not assume symmetric error bands around mean.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int, default 1000
            Number of Monte Carlo replications to perform
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        seed : int, default None
            np.random.seed for replications
        burn : int, default 100
            Number of initial simulated obs to discard
        component : neqs x neqs array, default to largest for each
            Index of column of eigenvector/value to use for each error band
            Note: period of impulse (t=0) is not included when computing
            principle component

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """
        periods = self.periods
        irfs = self._choose_irfs(orth, svar)
        neqs = self.neqs
        irf_resim = self.irf_resim(orth=orth, repl=repl, T=periods, seed=seed,
                                   burn=100)

        W, eigva, k = self._eigval_decomp_SZ(irf_resim)
        k = _validate_component(component, neqs, periods, dims=2, k=k)

        gamma = np.zeros((repl, periods + 1, neqs, neqs))
        for p in range(repl):
            for i in range(neqs):
                for j in range(neqs):
                    gamma[p, 1:, i, j] = (W[i, j, k[i, j], :] *
                                          irf_resim[p, 1:, i, j])

        lower, upper = _fill_irfs(irfs, gamma, signif, repl)
        return lower, upper

    def err_band_sz3(self, orth=False, svar=False, repl=1000, signif=0.05,
                     seed=None, burn=100, component=None):
        """
        IRF Sims-Zha error band method 3. Does not assume symmetric
        error bands around mean.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int, default 1000
            Number of Monte Carlo replications to perform
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        seed : int, default None
            np.random.seed for replications
        burn : int, default 100
            Number of initial simulated obs to discard
        component : vector length neqs, default to largest for each
            Index of column of eigenvector/value to use for each error band
            Note: period of impulse (t=0) is not included when computing
            principle component

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """
        periods = self.periods
        irfs = self._choose_irfs(orth, svar)
        neqs = self.neqs
        irf_resim = self.irf_resim(orth=orth, repl=repl, T=periods, seed=seed,
                                   burn=100)
        stack = np.zeros((neqs, repl, periods * neqs))

        # stack left to right, up and down

        for p in range(repl):
            for i in range(neqs):
                stack[i, p, :] = np.ravel(irf_resim[p, 1:, :, i].T)

        stack_cov = np.zeros((neqs, periods * neqs, periods * neqs))
        W = np.zeros((neqs, periods * neqs, periods * neqs))
        eigva = np.zeros((neqs, periods * neqs))
        k = np.zeros((neqs))
        k = _validate_component(component, neqs, periods, dims=1, k=k)

        # compute for eigen decomp for each stack
        for i in range(neqs):
            stack_cov[i] = np.cov(stack[i], rowvar=0)
            W[i], eigva[i], k[i] = util.eigval_decomp(stack_cov[i])

        gamma = np.zeros((repl, periods + 1, neqs, neqs))
        for p in range(repl):
            for j in range(neqs):
                for i in range(neqs):
                    gamma[p, 1:, i, j] = (
                        W[j, k[j], i * periods:(i + 1) * periods] *
                        irf_resim[p, 1:, i, j])
                    if i == neqs - 1:
                        gamma[p, 1:, i, j] = (W[j, k[j], i * periods:] *
                                              irf_resim[p, 1:, i, j])

        lower, upper = _fill_irfs(irfs, gamma, signif, repl)
        return lower, upper

    def _eigval_decomp_SZ(self, irf_resim):
        """
        Returns
        -------
        W: array of eigenvectors
        eigva: list of eigenvalues
        k: matrix indicating column # of largest eigenvalue for each c_i, j
        """
        neqs = self.neqs
        periods = self.periods

        cov_hold = np.zeros((neqs, neqs, periods, periods))
        for i in range(neqs):
            for j in range(neqs):
                cov_hold[i, j, :, :] = np.cov(irf_resim[:, 1:, i, j], rowvar=0)

        W = np.zeros((neqs, neqs, periods, periods))
        eigva = np.zeros((neqs, neqs, periods, 1))
        k = np.zeros((neqs, neqs))

        for i in range(neqs):
            for j in range(neqs):
                tup = util.eigval_decomp(cov_hold[i, j, :, :])
                W[i, j, :, :] = tup[0]
                eigva[i, j, :, 0] = tup[1]
                k[i, j] = tup[2]
        return W, eigva, k

    @cache_readonly
    def G(self):
        # Gi matrices as defined on p. 111
        K = self.neqs

        # nlags = self.model.p
        # J = np.hstack((np.eye(K),) + (np.zeros((K, K)),) * (nlags - 1))

        def _make_g(i):
            # p. 111 Lutkepohl
            G = 0.
            for m in range(i):
                # be a bit cute to go faster
                idx = i - 1 - m
                if idx in self._g_memo:
                    apow = self._g_memo[idx]
                else:
                    apow = np.linalg.matrix_power(self._A.T, idx)
                    # apow = np.dot(J, apow)
                    apow = apow[:K]
                    self._g_memo[idx] = apow

                # take first K rows
                piece = np.kron(apow, self.irfs[m])
                G = G + piece

            return G

        return [_make_g(i) for i in range(1, self.periods + 1)]

    def _orth_cov(self):
        # Lutkepohl 3.7.8
        Ik = np.eye(self.neqs)
        PIk = np.kron(self.P.T, Ik)

        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i == 0:
                apiece = 0
            else:
                Ci = np.dot(PIk, self.G[i - 1])
                apiece = chain_dot(Ci, self.cov_a, Ci.T)

            Cibar = np.dot(np.kron(Ik, self.irfs[i]), self.H)
            bpiece = chain_dot(Cibar, self.cov_sig, Cibar.T) / self.T

            # Lutkepohl typo, cov_sig correct
            covs[i] = apiece + bpiece

        return covs

    def cum_effect_cov(self, orth=False):
        """
        Compute asymptotic standard errors for cumulative impulse response
        coefficients

        Parameters
        ----------
        orth : boolean

        Notes
        -----
        eq. 3.7.7 (non-orth), 3.7.10 (orth)
        """
        Ik = np.eye(self.neqs)
        PIk = np.kron(self.P.T, Ik)

        F = 0.
        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i > 0:
                F = F + self.G[i - 1]

            if orth:
                if i == 0:
                    apiece = 0
                else:
                    Bn = np.dot(PIk, F)
                    apiece = chain_dot(Bn, self.cov_a, Bn.T)

                Bnbar = np.dot(np.kron(Ik, self.cum_effects[i]), self.H)
                bpiece = chain_dot(Bnbar, self.cov_sig, Bnbar.T) / self.T

                covs[i] = apiece + bpiece
            else:
                if i == 0:
                    covs[i] = np.zeros((self.neqs**2, self.neqs**2))
                    continue

                covs[i] = chain_dot(F, self.cov_a, F.T)

        return covs

    def cum_errband_mc(self, orth=False, repl=1000,
                       signif=0.05, seed=None, burn=100):
        """
        IRF Monte Carlo integrated error bands of cumulative effect
        """
        periods = self.periods
        return self.irf_errband_mc(orth=orth, repl=repl,
                                   T=periods, signif=signif,
                                   seed=seed, burn=burn, cum=True)

    def lr_effect_cov(self, orth=False):
        lre = self.lr_effects
        Finfty = np.kron(np.tile(lre.T, self.lags), lre)
        Ik = np.eye(self.neqs)

        if orth:
            Binf = np.dot(np.kron(self.P.T, np.eye(self.neqs)), Finfty)
            Binfbar = np.dot(np.kron(Ik, lre), self.H)

            return (chain_dot(Binf, self.cov_a, Binf.T) +
                    chain_dot(Binfbar, self.cov_sig, Binfbar.T))
        else:
            return chain_dot(Finfty, self.cov_a, Finfty.T)

    def stderr(self, orth=False):
        return np.array([tsatools.unvec(np.sqrt(np.diag(c)))
                         for c in self.cov(orth=orth)])

    def cum_effect_stderr(self, orth=False):
        return np.array([tsatools.unvec(np.sqrt(np.diag(c)))
                         for c in self.cum_effect_cov(orth=orth)])

    def lr_effect_stderr(self, orth=False):
        cov = self.lr_effect_cov(orth=orth)
        return tsatools.unvec(np.sqrt(np.diag(cov)))

    def _empty_covm(self, periods):
        return np.zeros((periods, self.neqs ** 2, self.neqs ** 2),
                        dtype=float)

    @cache_readonly
    def H(self):
        k = self.neqs
        Lk = tsatools.elimination_matrix(k)
        Kkk = tsatools.commutation_matrix(k, k)
        Ik = np.eye(k)

        B = chain_dot(Lk,
                      np.dot(np.kron(Ik, self.P), Kkk) + np.kron(self.P, Ik),
                      Lk.T)

        return np.dot(Lk.T, scipy.linalg.inv(B))

    def fevd_table(self):
        raise NotImplementedError
        # TODO: upstream just passes, should be fixed


def _validate_component(component, neqs, periods, dims, k):
    assert dims in [1, 2]

    if component is not None:
        if dims == 2 and np.shape(component) != (neqs, neqs):
            # dims == 2 --> err_band_sz1 or err_band_sz2
            raise ValueError("Component array must be {neqs}x{neqs}"
                             .format(neqs=neqs))
        elif dims == 1 and np.size(component) != (neqs):
            # dims == 1 --> err_band_sz3
            raise ValueError("Component array must be of length {neqs}"
                             .format(neqs=neqs))
        if np.argmax(component) >= neqs * periods:
            raise ValueError("At least one of the components "
                             "does not exist")
        else:
            k = component

    return k


def _fill_irfs(irfs, gamma, signif, repl):
    gamma_sort = np.sort(gamma, axis=0)  # sort to get quantiles
    indx = (int(round(signif / 2 * repl) - 1),
            int(round((1 - signif / 2) * repl) - 1))

    neqs = irfs.shape[-1]
    lower = np.copy(irfs)
    upper = np.copy(irfs)
    for i in range(neqs):
        for j in range(neqs):
            lower[:, i, j] = irfs[:, i, j] + gamma_sort[indx[0], :, i, j]
            upper[:, i, j] = irfs[:, i, j] + gamma_sort[indx[1], :, i, j]

    return lower, upper
