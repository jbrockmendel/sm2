#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd

from sm2.compat.numpy import recarray_select
from sm2.tsa.stattools import (pacf_yw,
                               pacf, grangercausalitytests,
                               arma_order_select_ic)
from sm2.datasets import macrodata
from sm2.tsa.arima_process import arma_generate_sample

cur_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(cur_dir, "results", "results_corrgram.csv")
results_corrgram = pd.read_csv(path, delimiter=',')

DECIMAL_8 = 8
DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2


@pytest.mark.not_vetted
class CheckCorrGram(object):
    """
    Set up for ACF, PACF tests.
    """
    data = macrodata.load_pandas()
    x = data.data['realgdp']
    results = results_corrgram


# TODO: eventually belongs in test_autocov (once pacf is there)
@pytest.mark.not_vetted
class TestPACF(CheckCorrGram):
    @classmethod
    def setup_class(cls):
        cls.pacfols = cls.results['PACOLS']
        cls.pacfyw = cls.results['PACYW']

    def test_ols(self):
        pacfols, confint = pacf(self.x, nlags=40, alpha=.05, method="ols")
        assert_almost_equal(pacfols[1:], self.pacfols, DECIMAL_6)
        centered = confint - confint.mean(1)[:, None]
        # from edited Stata ado file
        res = [[-.1375625, .1375625]] * 40
        assert_almost_equal(centered[1:41], res, DECIMAL_6)
        # check lag 0
        assert_equal(centered[0], [0., 0.])
        assert_equal(confint[0], [1, 1])
        assert pacfols[0] == 1

    def test_yw(self):
        pacfyw = pacf_yw(self.x, nlags=40, method="mle")
        assert_almost_equal(pacfyw[1:], self.pacfyw, DECIMAL_8)

    def test_ld(self):
        pacfyw = pacf_yw(self.x, nlags=40, method="mle")
        pacfld = pacf(self.x, nlags=40, method="ldb")
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)

        pacfyw = pacf(self.x, nlags=40, method="yw")
        pacfld = pacf(self.x, nlags=40, method="ldu")
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)


# -----------------------------------------------------------------

@pytest.mark.skip(reason="safe_arma_fit not ported from upstream")
@pytest.mark.not_vetted
@pytest.mark.slow
@pytest.mark.smoke
def test_arma_order_select_ic():
    # smoke test, assumes info-criteria are right
    arparams = np.array([.75, -.25])
    maparams = np.array([.65, .35])
    arparams = np.r_[1, -arparams]
    nobs = 250
    np.random.seed(2014)
    y = arma_generate_sample(arparams, maparams, nobs)
    res = arma_order_select_ic(y, ic=['aic', 'bic'], trend='nc')
    # regression tests in case we change algorithm to minic in sas
    aic_x = np.array([[np.nan, 552.7342255, 484.29687843],
                      [562.10924262, 485.5197969, 480.32858497],
                      [507.04581344, 482.91065829, 481.91926034],
                      [484.03995962, 482.14868032, 483.86378955],
                      [481.8849479, 483.8377379, 485.83756612]])
    bic_x = np.array([[np.nan, 559.77714733, 494.86126118],
                      [569.15216446, 496.08417966, 494.41442864],
                      [517.61019619, 496.99650196, 499.52656493],
                      [498.12580329, 499.75598491, 504.99255506],
                      [499.49225249, 504.96650341, 510.48779255]])
    aic = pd.DataFrame(aic_x, index=list(range(5)), columns=list(range(3)))
    bic = pd.DataFrame(bic_x, index=list(range(5)), columns=list(range(3)))
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert_almost_equal(res.bic.values, bic.values, 5)
    assert_equal(res.aic_min_order, (1, 2))
    assert_equal(res.bic_min_order, (1, 2))
    assert res.aic.index.equals(aic.index)
    assert res.aic.columns.equals(aic.columns)
    assert res.bic.index.equals(bic.index)
    assert res.bic.columns.equals(bic.columns)

    res = arma_order_select_ic(y, ic='aic', trend='nc')
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert res.aic.index.equals(aic.index)
    assert res.aic.columns.equals(aic.columns)
    assert res.aic_min_order == (1, 2)


@pytest.mark.skip(reason="safe_arma_fit not ported from upstream")
@pytest.mark.not_vetted
@pytest.mark.smoke
def test_arma_order_select_ic_failure():
    # this should trigger an SVD convergence failure, smoke test that it
    # returns, likely platform dependent failure...
    # looks like AR roots may be cancelling out for 4, 1?
    y = np.array([
        0.86074377817203640006, 0.85316549067906921611,
        0.87104653774363305363, 0.60692382068987393851,
        0.69225941967301307667, 0.73336177248909339976,
        0.03661329261479619179, 0.15693067239962379955,
        0.12777403512447857437, -0.27531446294481976,
        -0.24198139631653581283, -0.23903317951236391359,
        -0.26000241325906497947, -0.21282920015519238288,
        -0.15943768324388354896, 0.25169301564268781179,
        0.1762305709151877342, 0.12678133368791388857,
        0.89755829086753169399, 0.82667068795350151511])

    with warnings.catch_warnings():
        # catch a hessian inversion and convergence failure warning
        warnings.simplefilter("ignore")
        arma_order_select_ic(y)


@pytest.mark.not_vetted
def test_grangercausality():
    # some example data
    mdata = macrodata.load().data
    mdata = recarray_select(mdata, ['realgdp', 'realcons'])
    data = mdata.view((float, 2))
    data = np.diff(np.log(data), axis=0)

    # R: lmtest:grangertest
    r_result = [0.243097, 0.7844328, 195, 2]  # f_test
    gr = grangercausalitytests(data[:, 1::-1], 2, verbose=False)
    assert_almost_equal(r_result,
                        gr[2][0]['ssr_ftest'],
                        decimal=7)
    assert_almost_equal(gr[2][0]['params_ftest'],
                        gr[2][0]['ssr_ftest'],
                        decimal=7)


def test_granger_fails_on_nobs_check():
    # Test that if maxlag is too large, Granger Test raises a clear error.
    # TODO: GH reference?
    X = np.random.rand(10, 2)
    grangercausalitytests(X, 2, verbose=False)  # This should pass.
    with pytest.raises(ValueError):
        grangercausalitytests(X, 3, verbose=False)
