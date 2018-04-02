#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decompositions of (difference) stationary time series according
to Wold's Theorem.
"""
from __future__ import division

import numpy as np


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
