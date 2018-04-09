#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decompositions of (difference) stationary time series according
to Wold's Theorem.
"""
from __future__ import division

import numpy as np
import scipy.linalg
from pandas._libs.properties import cache_readonly

from sm2.tsa import autocov


# -----------------------------------------------------------------------
# Input Validation

def _shape_params(params):
    if np.ndim(params) == 1:
        # Univariate Case, the one dimension of params represents lags
        params = params[:, None, None]
    elif np.ndim(params) == 2:
        # i.e. 1 lag
        params = params[None, :, :]
    return params


def _check_param_dims(params):
    """
    `params` are either `ar` or `ma`
    """
    # if len(params.shape) == 2:
    #    params = params[None, :, :]
    if len(params.shape) != 3:
        raise ValueError('AR and MA should each be 1 or 3-dimensional.',
                         params.shape)
    if params.shape[1] != params.shape[2]:
        raise ValueError('Dimensions 1 and 2 of AR coefficients should be '
                         'equal.  i.e. K x N x N', params.shape)


def _check_intercept_shape(intercept, neqs):
    if intercept is None:
        intercept = np.zeros(neqs)

    intercept = np.array(intercept)
    if neqs == 1:
        accepted_shapes = [(), (1,)]
    else:
        accepted_shapes = [(neqs,)]
    if intercept.shape not in accepted_shapes:
        raise ValueError(intercept.shape, accepted_shapes)
    return intercept


def _unpack_lags_and_neqs(ar, ma, intercept):
    # Check if these coefficients correspond to a vector-ARMA

    k_ar = ar.shape[0]
    if len(ar.shape) > 1:
        _check_param_dims(ar)
        neqs = ar.shape[1]
    else:
        # Default is univariate
        neqs = 1

    if ma is not None:
        k_ma = ma.shape[0]
        if len(ma.shape) > 1:
            _check_param_dims(ma)
            if len(ar.shape) != 3 or ar.shape[1] != ma.shape[1]:
                raise ValueError('ar.shape[1:] must match ma.shape[1:]',
                                 ar.shape, ma.shape)
            neqs = ma.shape[1]
    else:
        k_ma = 0

    intercept = _check_intercept_shape(intercept, neqs)
    return (k_ar, k_ma, neqs, intercept)


# -----------------------------------------------------------------------

# TODO: Merge this into ARMAParams?
class ARMARoots(object):
    @cache_readonly
    def arroots(self):
        """Roots of autoregressive lag-polynomial"""
        # TODO: should docstring say "inverse"?
        return np.roots(np.r_[1, -self.arcoefs])**-1
        # Equiv: self.arpoly.roots()

    # TODO: trim 0 off the end of ma args, avoiding np.inf in cases where
    # you just passed [1, 0]
    @cache_readonly
    def maroots(self):
        """Roots of moving average lag-polynomial"""
        # TODO: should docstring say "inverse"?
        return np.roots(np.r_[1, self.macoefs])**-1
        # Equiv: self.mapoly.roots()

    @cache_readonly
    def arfreq(self):
        r"""
        Returns the frequency of the AR roots.

        This is the solution, x, to z = abs(z)*exp(2j*np.pi*x) where z are the
        roots.
        """
        z = self.arroots
        if not z.size:
            return  # TODO: return empty array?
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def mafreq(self):
        r"""
        Returns the frequency of the MA roots.

        This is the solution, x, to z = abs(z)*exp(2j*np.pi*x) where z are the
        roots.
        """
        z = self.maroots
        if not z.size:
            return  # TODO: return empty array?
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @property
    def isstationary(self):
        """Arma process is stationary if AR roots are outside unit circle

        Returns
        -------
        isstationary : boolean
            True if autoregressive roots are outside unit circle
        """
        # explicitly cast to bool otherwise "is False" is incorrect
        # because it returns a np._bool object
        return bool((np.abs(self.arroots) > 1.0).all())

    @property
    def isinvertible(self):
        """ARMA process is invertible if MA roots are outside unit circle.

        From Powell (http://eml.berkeley.edu/~powell/e241b_f06/TS-StatInv.pdf):

        If the MA(q) process

            y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1}
                      + ... + \theta_q\epsilon_{t-q}
                = \mu + \theta(L)\epsilon_t

        can be rewritten as a linear combination of its past
        values {y_{t-s}, s=1,2,...} plus the contemporaneous error
        term \epsilon_t, i.e.,

            y_t = \alpha + \sum_{s=1}^{\infty}\pi_s y_{t-s} + \epsilon_t

        for some \alpha and {\pi_j}, then the process is said
        to be \texit{invertible}.

        Returns
        -------
        isinvertible : boolean
            True if moving average roots are outside unit circle
        """
        # explicitly cast to bool otherwise "is False" is incorrect
        # because it returns a np._bool object
        return bool(np.all(np.abs(self.maroots) > 1))

    def invertroots(self, retnew=False):
        """
        Make MA polynomial invertible by inverting roots inside unit circle

        Parameters
        ----------
        retnew : boolean
            If False (default), then return the lag-polynomial as array.
            If True, then return a new instance with invertible MA-polynomial

        Returns
        -------
        manew : array
            new invertible MA lag-polynomial, returned if retnew is false.
        wasinvertible : boolean
            True if the MA lag-polynomial was already invertible, returned if
            retnew is false.
        armaprocess : new instance of class
            If retnew is true, then return a new instance with invertible
            MA-polynomial
        """
        pr = self.maroots
        mainv = self.ma
        invertible = self.isinvertible
        if not invertible:
            pr[np.abs(pr) < 1] = 1. / pr[np.abs(pr) < 1]
            pnew = np.polynomial.Polynomial.fromroots(pr)
            mainv = pnew.coef / pnew.coef[0]

        if retnew:
            return self.__class__(self.ar, mainv, nobs=self.nobs)
            # TODO: dont do this multiple-return thing
        else:
            return (mainv, invertible)


class ARMAParams(object):

    @staticmethod
    def _unpack_params(params, order, k_trend, k_exog, reverse=False):
        (k_ar, k_ma) = order
        k = k_trend + k_exog
        maparams = params[k + k_ar:]
        arparams = params[k:k + k_ar]
        trend = params[:k_trend]
        exparams = params[k_trend:k]
        if reverse:
            arparams = arparams[::-1]
            maparams = maparams[::-1]
        return (trend, exparams, arparams, maparams)

    def _transparams(self, params):
        """Transforms params to induce stationarity/invertability.

        Reference
        ---------
        Jones(1980)
        """
        k_ar = self.k_ar
        k_ma = getattr(self, 'k_ma', 0)

        k = getattr(self, 'k_exog', 0) + self.k_trend

        # TODO: Use unpack_params?
        arparams = params[k:k + k_ar]  # TODO: Should we call this arcoefs?
        maparams = params[k + k_ar:]

        newparams = params.copy()

        if k != 0:
            # just copy exogenous parameters
            newparams[:k] = params[:k]

        if k_ar != 0:
            # AR Coeffs
            newparams[k:k + k_ar] = self._ar_transparams(arparams.copy())

        if k_ma != 0:
            # MA Coeffs
            newparams[k + k_ar:] = self._ma_transparams(maparams.copy())

        return newparams

    def _invtransparams(self, start_params):
        """
        Inverse of the Jones reparameterization
        """
        k_ar = self.k_ar
        k_ma = getattr(self, 'k_ma', 0)
        k = getattr(self, 'k_exog', 0) + self.k_trend

        arparams = start_params[k:k + k_ar]  # TODO: call this arcoefs?
        maparams = start_params[k + k_ar:]

        newparams = start_params.copy()

        if k_ar != 0:
            # AR coeffs
            newparams[k:k + k_ar] = self._ar_invtransparams(arparams)

        if k_ma != 0:
            # MA coeffs
            mainv = self._ma_invtransparams(maparams)
            newparams[k + k_ar:k + k_ar + k_ma] = mainv

        return newparams

    @staticmethod
    def _ar_transparams(params):
        """Transforms params to induce stationarity/invertability.

        Parameters
        ----------
        params : array
            The AR coefficients

        Reference
        ---------
        Jones(1980)
        """
        meparams = np.exp(-params)
        newparams = (1 - meparams) / (1 + meparams)
        tmp = (1 - meparams) / (1 + meparams)
        for j in range(1, len(params)):
            a = newparams[j]
            for kiter in range(j):
                tmp[kiter] -= a * newparams[j - kiter - 1]
            newparams[:j] = tmp[:j]
        return newparams

    @staticmethod
    def _ar_invtransparams(params):
        """Inverse of the Jones reparameterization

        Parameters
        ----------
        params : array
            The transformed AR coefficients
        """
        # AR coeffs
        tmp = params.copy()
        for j in range(len(params) - 1, 0, -1):
            a = params[j]
            for kiter in range(j):
                val = (params[kiter] + a * params[j - kiter - 1]) / (1 - a**2)
                tmp[kiter] = val
            params[:j] = tmp[:j]

        invarcoefs = -np.log((1 - params) / (1 + params))
        return invarcoefs

    @staticmethod
    def _ma_invtransparams(macoefs):
        """Inverse of the Jones reparameterization

        Parameters
        ----------
        params : array
            The transformed MA coefficients
        """
        tmp = macoefs.copy()
        for j in range(len(macoefs) - 1, 0, -1):
            b = macoefs[j]
            for kiter in range(j):
                val = (macoefs[kiter] - b * macoefs[j - kiter - 1]) / (1 - b**2)
                tmp[kiter] = val
            macoefs[:j] = tmp[:j]

        invmacoefs = -np.log((1 - macoefs) / (1 + macoefs))
        return invmacoefs

    @staticmethod
    def _ma_transparams(params):
        """Transforms params to induce stationarity/invertability.

        Parameters
        ----------
        params : array
            The ma coeffecients of an (AR)MA model.

        Reference
        ---------
        Jones(1980)
        """
        meparams = np.exp(-params)
        newparams = (1 - meparams) / (1 + meparams)
        tmp = (1 - meparams) / (1 + meparams)

        # levinson-durbin to get macf
        for j in range(1, len(params)):
            b = newparams[j]
            for kiter in range(j):
                tmp[kiter] += b * newparams[j - kiter - 1]
            newparams[:j] = tmp[:j]
        return newparams


# TODO: Can this be extended to VARMA?
class VARParams(object):
    """Class representing a known VAR(p) process, *without* any information
    about the distribution of error terms.

    Parameters
    ----------
    arcoefs : ndarray (p x k x k)
    intercept : ndarray (length k), optional
    """

    @cache_readonly
    def k_ar(self):
        return len(self.arcoefs)

    @cache_readonly
    def k_ma(self):
        macoefs = self.macoefs
        if macoefs is None:
            return 0
        return macoefs.shape[0]
        # alternative: maparams.shape[0]-1?

    @cache_readonly
    def neqs(self):
        return self.arcoefs.shape[1]

    def __init__(self, arcoefs, macoefs=None, intercept=None):
        """
        Parameters
        ----------
        arcoefs : ndarray (p x k x k)
        intercept : ndarray (k x 1), optional
        """
        # TODO: Add macoefs to docstring
        arcoefs = _shape_params(arcoefs)
        macoefs = _shape_params(macoefs)
        # TODO: We don't actually _do_ anything with macoefs

        self.coefs = arcoefs  # alias for VAR classes
        self.arcoefs = arcoefs
        self.macoefs = macoefs

        k_ar, k_ma, neqs, intercept = _unpack_lags_and_neqs(arcoefs,
                                                            macoefs,
                                                            intercept)
        self.intercept = intercept

    @cache_readonly  # TODO: Should be called arroots?
    def roots(self):
        neqs = self.neqs
        k_ar = self.k_ar
        p = neqs * k_ar
        arr = np.zeros((p, p))
        arr[:neqs, :] = np.column_stack(self.coefs)
        arr[neqs:, :-neqs] = np.eye(p - neqs)
        roots = np.linalg.eig(arr)[0]**-1
        idx = np.argsort(np.abs(roots))[::-1]  # sort by reverse modulus
        return roots[idx]

    @cache_readonly
    def _char_mat(self):
        arcoefs = self.arcoefs
        neqs = self.neqs
        return np.eye(neqs) - arcoefs.sum(axis=0)

    def mean(self):
        r"""Mean of stable process

        Lütkepohl eq. 2.1.23

        .. math:: \mu = (I - A_1 - \dots - A_p)^{-1} \alpha
        """
        try:
            return np.linalg.solve(self._char_mat, self.intercept)
        except np.linalg.LinAlgError:
            # Note: upstream does not catch this error
            return np.array([np.nan] * self.neqs)
        # In VARProcess self.intercept is replaced by self.exog
        # TODO: does that make sense?

    def long_run_effects(self):
        r"""Compute long-run effect of unit impulse

        .. math::

            \Psi_\infty = \sum_{i=0}^\infty \Phi_i
        """
        try:
            return scipy.linalg.inv(self._char_mat)
        except np.linalg.LinAlgError:
            # Note: upstream does not catch this error
            return np.array([np.nan] * self.neqs)

    def is_stable(self, verbose=False):
        """Determine stability of VAR(p) system by examining the eigenvalues
        of the VAR(1) representation

        Parameters
        ----------
        verbose : bool
            Print eigenvalues of the VAR(1) companion

        Returns
        -------
        is_stable : bool

        Notes
        -----
        Checks if det(I - Az) = 0 for any mod(z) <= 1, so all the
        eigenvalues of the companion matrix must lie outside the unit circle.
        """
        coefs = self.arcoefs

        from sm2.tsa.vector_ar.util import comp_matrix
        A_var1 = comp_matrix(coefs)
        eigs = np.linalg.eigvals(A_var1)  # TODO: cache?

        if verbose:
            # TODO: Get rid of this option
            print('Eigenvalues of VAR(1) rep')
            for val in np.abs(eigs):
                print(val)

        return (np.abs(eigs) <= 1).all()

    def ma_rep(self, maxn=10):
        r"""
        MA(\infty) representation of VAR(p) process

        Parameters
        ----------
        maxn : int
            Number of MA matrices to compute

        Notes
        -----
        VAR(p) process as

        .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

        can be equivalently represented as

        .. math:: y_t = \mu + \sum_{i=0}^\infty \Phi_i u_{t-i}

        e.g. can recursively compute the \Phi_i matrices with \Phi_0 = I_k

        Returns
        -------
        phis : ndarray (maxn + 1 x k x k)
        """
        coefs = self.arcoefs
        p, k, k = coefs.shape
        phis = np.zeros((maxn + 1, k, k))
        phis[0] = np.eye(k)

        # recursively compute Phi matrices
        for i in range(1, maxn + 1):
            for j in range(1, i + 1):
                if j > p:
                    break

                phis[i] += np.dot(phis[i - j], coefs[j - 1])

        return phis


class VARProcess(VARParams):
    """
    Class represents a known VAR(p) process

    Parameters
    ----------
    arcoefs : ndarray (p x k x k)
    intercept : ndarray (length k)
    sigma_u : ndarray (k x k)
    """
    def __init__(self, coefs, intercept, sigma_u):
        VARParams.__init__(self, coefs, intercept=intercept)
        self.sigma_u = sigma_u

    @cache_readonly
    def _chol_sigma_u(self):
        return np.linalg.cholesky(self.sigma_u)

    @cache_readonly
    def detomega(self):
        r"""
        Return determinant of white noise covariance with degrees of freedom
        correction:

        .. math::

            \hat \Omega = \frac{T}{T - Kp - 1} \hat \Omega_{\mathrm{MLE}}
        """
        return scipy.linalg.det(self.sigma_u)

    def orth_ma_rep(self, maxn=10, P=None):
        r"""Compute Orthogonalized MA coefficient matrices using P matrix such
        that :math:`\Sigma_u = PP^\prime`. P defaults to the Cholesky
        decomposition of :math:`\Sigma_u`

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute
        P : ndarray (neqs x neqs), optional
            Matrix such that Sigma_u = PP', defaults to the
            Cholesky decomposition.

        Returns
        -------
        coefs : ndarray (maxn x neqs x neqs)
        """
        if P is None:
            P = self._chol_sigma_u

        ma_mats = self.ma_rep(maxn=maxn)
        return np.array([np.dot(coefs, P) for coefs in ma_mats])

    def mse(self, steps):
        r"""
        Compute theoretical forecast error variance matrices

        Parameters
        ----------
        steps : int
            Number of steps ahead

        Notes
        -----
        .. math:: \mathrm{MSE}(h) = \sum_{i=0}^{h-1} \Phi \Sigma_u \Phi^T

        Returns
        -------
        forc_covs : ndarray (steps x neqs x neqs)
        """
        ma_coefs = self.ma_rep(steps)
        sigma_u = self.sigma_u

        neqs = len(sigma_u)
        forc_covs = np.zeros((steps, neqs, neqs))

        prior = np.zeros((neqs, neqs))
        for h in range(steps):
            # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
            phi = ma_coefs[h]
            var = phi.dot(sigma_u).dot(phi.T)
            forc_covs[h] = prior = prior + var

        return forc_covs

    forecast_cov = mse

    def cov_ybar(self):
        r"""Asymptotically consistent estimate of covariance of the sample mean

        .. math::

            \sqrt(T) (\bar{y} - \mu) \rightarrow {\cal N}(0,\Sigma_{\bar{y}})\\

            \Sigma_{\bar{y}} = B \Sigma_u B^\prime,

            \text{where } B = (I_K - A_1 - \cdots - A_p)^{-1}

        Notes
        -----
        Lütkepohl Proposition 3.3
        """
        Ainv = scipy.linalg.inv(np.eye(self.neqs) - self.coefs.sum(0))
        return Ainv.dot(self.sigma_u).dot(Ainv.T)

    @cache_readonly
    def _cov_sigma(self):
        """
        Estimated covariance matrix of vech(sigma_u)
        """
        from sm2.tsa.tsatools import duplication_matrix
        D_K = duplication_matrix(self.neqs)
        D_Kinv = np.linalg.pinv(D_K)

        sigxsig = np.kron(self.sigma_u, self.sigma_u)
        return 2 * D_Kinv.dot(sigxsig).dot(D_Kinv.T)

    def acf(self, nlags=None):
        """
        Compute autocovariance function ACF_y(h) up to nlags of stable VAR(p)
        process

        Parameters
        ----------
        nlags : int, optional
            Defaults to order p of system

        Returns
        -------
        acf : ndarray, (p, k, k)

        Notes
        -----
        Ref: Lütkepohl p.28-29
        """
        coefs = self.coefs
        p, k, _ = coefs.shape
        if nlags is None:
            nlags = p

        # p x k x k, ACF for lags 0, ..., p-1
        result = np.zeros((nlags + 1, k, k))
        result[:p] = autocov.var_acf(coefs, self.sigma_u)

        # yule-walker equations
        for h in range(p, nlags + 1):
            # compute ACF for lag=h
            # G(h) = A_1 G(h-1) + ... + A_p G(h-p)
            for j in range(p):
                result[h] += np.dot(coefs[j], result[h - j - 1])

        return result

    def acorr(self, nlags=None):
        """Compute theoretical autocorrelation function

        Returns
        -------
        acorr : ndarray (p x k x k)
        """
        return autocov.acf_to_acorr(self.acf(nlags=nlags))
