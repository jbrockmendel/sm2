# -*- coding: utf-8 -*-
"""
Unit tests for fit_constrained
Tests for Poisson and Binomial are in discrete


Created on Sun Jan  7 09:21:39 2018

Author: Josef Perktold
"""
import warnings

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from sm2.tools.tools import add_constant
from sm2.tools.sm_exceptions import ValueWarning

from sm2.regression.linear_model import OLS, WLS
from sm2.genmod.generalized_linear_model import GLM
from sm2.genmod.families import family


@pytest.mark.not_vetted
class ConstrainedCompareMixin(object):
    model_cls = GLM
    fit_kwargs = {}
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
        # Note: atol=1e-15 added by upstream GH#4603, doesnt seem needed

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
                        rtol=5e-9)
        # Note: rtol lowered from1e-10 to 5e-9 by GH#4620
        # (for TestGLMBinomialCountConstrainedHC I think)

    def test_resid(self):
        assert_allclose(self.res1.resid_response,
                        self.res2.resid,
                        rtol=1e-10)


@pytest.mark.not_vetted
class TestGLMGaussianOffset(ConstrainedCompareMixin):
    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exogc,
                            offset=0.5 * cls.exog[:, cls.idx_c].squeeze())
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit(**cls.fit_kwargs)
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])


@pytest.mark.not_vetted
class TestGLMGaussianConstrained(ConstrainedCompareMixin):
    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exog)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', **cls.fit_kwargs)


@pytest.mark.not_vetted
class TestGLMGaussianOffsetHC(ConstrainedCompareMixin):
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
    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exog, var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', **cls.fit_kwargs)


@pytest.mark.not_vetted
class TestGLMWtdGaussianOffsetHC(ConstrainedCompareWtdMixin):
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
    fit_kwargs = {"cov_type": "HC0"}

    @classmethod
    def init(cls):
        mod = cls.model_cls(cls.endog, cls.exog, var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', **cls.fit_kwargs)


@pytest.mark.not_vetted
class TestGLMBinomialCountConstrained(ConstrainedCompareMixin):

    @classmethod
    def setup_class(cls):
        from sm2.datasets.star98 import load
        data = load(as_pandas=False)
        exog = add_constant(data.exog, prepend=True)
        offset = np.ones(len(data.endog))
        exog_keep = exog[:, :-5]
        cls.mod2 = GLM(data.endog, exog_keep, family=family.Binomial(),
                       offset=offset)

        cls.mod1 = GLM(data.endog, exog, family=family.Binomial(),
                       offset=offset)
        cls.init()

    @classmethod
    def init(cls):
        cls.res2 = cls.mod2.fit()
        k = cls.mod1.exog.shape[1]
        cls.idx_p_uc = np.arange(k - 5)
        constraints = np.eye(k)[-5:]
        cls.res1 = cls.mod1.fit_constrained(constraints)

    def test_resid(self):
        # need to override because res2 does not have resid
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.resid_response, res2.resid_response, rtol=1e-8)

    def test_glm_attr(self):
        for attr in ['llf', 'null_deviance', 'aic', 'bic', 'df_resid',
                     'df_model', 'pearson_chi2', 'scale']:
            assert_allclose(getattr(self.res1, attr),
                            getattr(self.res2, attr), rtol=1e-10)

    def test_wald(self):
        res1 = self.res1
        res2 = self.res2
        k1 = len(res1.params)
        k2 = len(res2.params)

        use_f = False
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ValueWarning)
            wt2 = res2.wald_test(np.eye(k2)[1:], use_f=use_f)
            wt1 = res1.wald_test(np.eye(k1)[1:], use_f=use_f)
        assert_allclose(wt2.pvalue, wt1.pvalue, atol=1e-20)  # pvalue = 0
        assert_allclose(wt2.statistic, wt1.statistic, rtol=1e-8)
        assert_equal(wt2.df_denom, wt1.df_denom)

        use_f = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ValueWarning)
            wt2 = res2.wald_test(np.eye(k2)[1:], use_f=use_f)
            wt1 = res1.wald_test(np.eye(k1)[1:], use_f=use_f)
        assert_allclose(wt2.pvalue, wt1.pvalue, rtol=1)  # pvalue = 8e-273
        assert_allclose(wt2.statistic, wt1.statistic, rtol=1e-8)
        assert_equal(wt2.df_denom, wt1.df_denom)
        assert_equal(wt2.df_num, wt1.df_num)
        assert_equal(wt2.summary()[-30:], wt1.summary()[-30:])

        # smoke # TODO: separate this out
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ValueWarning)
            # RuntimeWarnings because of truedivide and scipy distributions
            warnings.simplefilter('ignore', RuntimeWarning)
            self.res1.summary()
            # self.res1.summary2()


@pytest.mark.not_vetted
class TestGLMBinomialCountConstrainedHC(TestGLMBinomialCountConstrained):
    @classmethod
    def init(cls):
        cls.res2 = cls.mod2.fit(cov_type='HC0')
        k = cls.mod1.exog.shape[1]
        cls.idx_p_uc = np.arange(k - 5)
        constraints = np.eye(k)[-5:]
        cls.res1 = cls.mod1.fit_constrained(constraints, cov_type='HC0')
