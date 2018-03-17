# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:08:49 2017

Author: Josef Perktold
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

import sm2.api as sm
from sm2.discrete.discrete_model import NegativeBinomialP

import sm2.discrete.tests.results.results_count_margins as res_stata


# load data into module namespace
cpunish_data = sm.datasets.cpunish.load()
cpunish_data.exog[:, 3] = np.log(cpunish_data.exog[:, 3])
exog = sm.add_constant(cpunish_data.exog, prepend=False)
endog = cpunish_data.endog - 1  # avoid zero-truncation
exog /= np.round(exog.max(0), 3)


@pytest.mark.not_vetted
class CheckMarginMixin(object):
    res2_slice = slice(None, None, None)
    rtol_fac = 1
    dummy = False

    @classmethod
    def setup_class(cls):
        mod = cls.model_cls(endog, exog)
        res = mod.fit(**cls.fit_kwargs)
        cls.res = res
        cls.margeff = res.get_margeff(dummy=cls.dummy)

    def test_margins_table(self):
        res2 = self.res2
        sl = self.res2_slice
        rf = self.rtol_fac
        assert_allclose(self.margeff.margeff,
                        res2.params[sl],
                        rtol=1e-5 * rf)
        assert_allclose(self.margeff.margeff_se,
                        res2.bse[sl],
                        rtol=1e-6 * rf)
        assert_allclose(self.margeff.pvalues,
                        res2.pvalues[sl],
                        rtol=5e-6 * rf)
        assert_allclose(self.margeff.conf_int(),
                        res2.margins_table[sl, 4:6],
                        rtol=1e-6 * rf)


@pytest.mark.not_vetted
class TestPoissonMargin(CheckMarginMixin):
    res2 = res_stata.results_poisson_margins_cont

    model_cls = sm.Poisson
    fit_kwargs = {
        # here we don't need to check convergence from default start_params
        'start_params': [14.1709, 0.7085, -3.4548, -0.539, 3.2368, -7.9299,
                         -5.0529]
    }


@pytest.mark.not_vetted
class TestPoissonMarginDummy(CheckMarginMixin):
    res2 = res_stata.results_poisson_margins_dummy
    res2_slice = [0, 1, 2, 3, 5, 6]

    model_cls = sm.Poisson
    fit_kwargs = {
        # here we don't need to check convergence from default start_params
        'start_params': [14.1709, 0.7085, -3.4548, -0.539, 3.2368, -7.9299,
                         -5.0529]
    }
    dummy = True


@pytest.mark.not_vetted
class TestNegBinMargin(CheckMarginMixin):
    res2 = res_stata.results_negbin_margins_cont

    model_cls = sm.NegativeBinomial
    fit_kwargs = {
        'method': 'nm',
        'maxiter': 2000,
        # here we don't need to check convergence from default start_params
        'start_params': [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552,
                         -2.88, 1.14]
    }
    rtol_fac = 5e1  # negbin has lower agreement with Stata in this case


@pytest.mark.not_vetted
class TestNegBinMarginDummy(CheckMarginMixin):
    res2 = res_stata.results_negbin_margins_dummy
    res2_slice = [0, 1, 2, 3, 5, 6]

    model_cls = sm.NegativeBinomial
    fit_kwargs = {
        'method': 'nm',
        'maxiter': 2000,
        # here we don't need to check convergence from default start_params
        'start_params': [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552,
                         -2.88, 1.14]
    }
    dummy = True
    rtol_fac = 5e1  # negbin has lower agreement with Stata in this case


@pytest.mark.not_vetted
class TestNegBinPMargin(CheckMarginMixin):
    # this is the same as the nb2 version above for NB-P, p=2
    # by not specifically passing p=2 this implicitly tests for the default
    # being 2
    res2 = res_stata.results_negbin_margins_cont

    model_cls = NegativeBinomialP
    fit_kwargs = {
        'method': 'nm',
        'maxiter': 2000,
        # here we don't need to check convergence from default start_params
        'start_params': [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552,
                         -2.88, 1.14]
    }
    rtol_fac = 5e1  # negbin has lower agreement with Stata in this case
