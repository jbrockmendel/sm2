# -*- coding: utf-8 -*-
"""
Unit tests for fit_constrained
Tests for Poisson and Binomial are in discrete


Created on Sun Jan  7 09:21:39 2018

Author: Josef Perktold
"""
import pytest

import numpy as np
from numpy.testing import assert_allclose

from sm2.tools.tools import add_constant
from sm2.regression.linear_model import OLS, WLS
from sm2.genmod.generalized_linear_model import GLM


@pytest.mark.not_vetted
class ConstrainedCompareMixin(object):
    idx_c = [1]
    idx_uc = [0, 2, 3, 4]

    @classmethod
    def setup_class(cls):
        nobs, k_exog = 100, 5
        np.random.seed(987125)
        x = np.random.randn(nobs, k_exog - 1)
        x = add_constant(x)

        y_true = x.sum(1) / 2
        y = y_true + 2 * np.random.randn(nobs)
        cls.endog = y
        cls.exog = x
        cls.idx_p_uc = np.array(cls.idx_uc)
        cls.exogc = xc = x[:, cls.idx_uc]
        mod_ols_c = OLS(y - 0.5 * x[:, 1], xc)
        mod_ols_c.exog_names[:] = ['const', 'x2', 'x3', 'x4']

        cls.mod2 = mod_ols_c
        cls.res2 = cls.mod2.fit(**cls.fit_kwargs)

        cls.init()

    def test_params(self):
        assert_allclose(self.res1.params[self.idx_p_uc],
                        self.res2.params,
                        rtol=1e-10)

    def test_se(self):
        res1 = self.res1
        res2 = self.res2

        assert res1.df_resid == res2.df_resid
        assert_allclose(res1.scale,
                        res2.scale,
                        rtol=1e-10)
        assert_allclose(res1.bse[self.idx_p_uc],
                        res2.bse,
                        rtol=1e-10)
        assert_allclose(res1.cov_params()[self.idx_p_uc[:, None],
                                          self.idx_p_uc],
                        res2.cov_params(),
                        rtol=1e-10)

    def test_resid(self):
        assert_allclose(self.res1.resid_response,
                        self.res2.resid,
                        rtol=1e-10)


@pytest.mark.not_vetted
class TestGLMGaussianOffset(ConstrainedCompareMixin):
    model_cls = GLM
    fit_kwargs = {}

    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exogc,
                            offset=0.5 * cls.exog[:, cls.idx_c].squeeze())
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit(**cls.fit_kwargs)
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])


@pytest.mark.not_vetted
class TestGLMGaussianConstrained(ConstrainedCompareMixin):
    model_cls = GLM
    fit_kwargs = {}

    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exog)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', **cls.fit_kwargs)


@pytest.mark.not_vetted
class TestGLMGaussianOffsetHC(ConstrainedCompareMixin):
    model_cls = GLM
    fit_kwargs = {"cov_type": "HC0"}

    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exogc,
                            offset=0.5 * cls.exog[:, cls.idx_c].squeeze())
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit(**cls.fit_kwargs)
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])


@pytest.mark.not_vetted
class TestGLMGaussianConstrainedHC(ConstrainedCompareMixin):
    model_cls = GLM
    fit_kwargs = {"cov_type": "HC0"}

    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exog)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', **cls.fit_kwargs)


@pytest.mark.not_vetted
class ConstrainedCompareWtdMixin(ConstrainedCompareMixin):
    @classmethod
    def setup_class(cls):
        nobs, k_exog = 100, 5
        np.random.seed(987125)
        x = np.random.randn(nobs, k_exog - 1)
        x = add_constant(x)
        cls.aweights = np.random.randint(1, 10, nobs)

        y_true = x.sum(1) / 2
        y = y_true + 2 * np.random.randn(nobs)
        cls.endog = y
        cls.exog = x
        cls.idx_p_uc = np.array(cls.idx_uc)
        cls.exogc = xc = x[:, cls.idx_uc]
        mod_ols_c = WLS(y - 0.5 * x[:, 1], xc, weights=cls.aweights)
        mod_ols_c.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.mod2 = mod_ols_c
        cls.res2 = cls.mod2.fit(**cls.fit_kwargs)

        cls.init()


@pytest.mark.not_vetted
class TestGLMWtdGaussianOffset(ConstrainedCompareWtdMixin):
    model_cls = GLM
    fit_kwargs = {}

    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exogc,
                            offset=0.5 * cls.exog[:, cls.idx_c].squeeze(),
                            var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit(**cls.fit_kwargs)
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])


@pytest.mark.not_vetted
class TestGLMWtdGaussianConstrained(ConstrainedCompareWtdMixin):
    model_cls = GLM
    fit_kwargs = {}

    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exog, var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', **cls.fit_kwargs)


@pytest.mark.not_vetted
class TestGLMWtdGaussianOffsetHC(ConstrainedCompareWtdMixin):
    model_cls = GLM
    fit_kwargs = {"cov_type": "HC0"}

    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exogc,
                            offset=0.5 * cls.exog[:, cls.idx_c].squeeze(),
                            var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit(**cls.fit_kwargs)
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])


@pytest.mark.not_vetted
class TestGLMWtdGaussianConstrainedHC(ConstrainedCompareWtdMixin):
    model_cls = GLM
    fit_kwargs = {"cov_type": "HC0"}

    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exog, var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', **cls.fit_kwargs)
