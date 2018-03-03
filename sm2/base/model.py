#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import warnings

import numpy as np
from scipy import stats

from six.moves import range, reduce

from sm2.tools.sm_exceptions import ValueWarning, HessianInversionWarning
from sm2.tools.decorators import resettable_cache, cache_readonly


_model_params_doc = """
    Parameters
    ----------
    endog : array-like
        1-d endogenous response variable. The dependent variable.
    exog : array-like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See
        :func:`statsmodels.tools.add_constant`."""

_missing_param_doc = """
    missing : str
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
    """ % {'params_doc' : _model_params_doc,
           'extra_params_doc' : _missing_param_doc + _extra_param_doc}

    @property
    def endog_names(self):
        """Names of endogenous variables"""
        return self.data.ynames

    @property
    def exog_names(self):
        """Names of exogenous variables"""
        return self.data.xnames

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

    def predict(self, params, exog=None, *args, **kwargs):
        """
        After a model has been fit predict returns the fitted values.

        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError

    def _get_init_kwds(self):
        """return dictionary with extra keys used in model.__init__
        """
        return {key: getattr(self, key, None) for key in self._init_keys}


class LikelihoodModel(Model):
    """
    Likelihood model is a subclass of Model.
    """

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

    def loglike(self, params):
        """
        Log-likelihood of model.
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.
        """
        raise NotImplementedError

    def information(self, params):
        """
        Fisher information matrix of model

        Returns -Hessian of loglike evaluated at params.
        """
        raise NotImplementedError

    def hessian(self, params):
        """
        The Hessian matrix of the model
        """
        raise NotImplementedError
