#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose

from sm2.tools.sm_exceptions import ColinearityWarning

from sm2.datasets import macrodata
from sm2.tsa import unit_root


# -----------------------------------------------------------------
# ADFuller

@pytest.mark.not_vetted
class CheckADF(object):
    """
    Test Augmented Dickey-Fuller

    Test values taken from Stata.
    """
    levels = ['1%', '5%', '10%']
    data = macrodata.load()
    x = data.data['realgdp']
    y = data.data['infl']

    def test_teststat(self):
        assert_almost_equal(self.res1[0],
                            self.teststat,
                            5)

    def test_pvalue(self):
        assert_almost_equal(self.res1[1],
                            self.pvalue,
                            5)

    def test_critvalues(self):
        critvalues = [self.res1[4][lev] for lev in self.levels]
        assert_almost_equal(critvalues,
                            self.critvalues,
                            2)


# class TestADFConstantTrendSquared(CheckADF):
#    pass
# TODO: get test values from R?


@pytest.mark.not_vetted
class TestADFConstant(CheckADF):
    """
    Dickey-Fuller test for unit root
    """
    @classmethod
    def setup_class(cls):
        cls.res1 = unit_root.adfuller(cls.x, regression="c",
                                      autolag=None, maxlag=4)
        cls.teststat = .97505319
        cls.pvalue = .99399563
        cls.critvalues = [-3.476, -2.883, -2.573]


@pytest.mark.not_vetted
class TestADFConstantTrend(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = unit_root.adfuller(cls.x, regression="ct",
                                      autolag=None, maxlag=4)
        cls.teststat = -1.8566374
        cls.pvalue = .67682968
        cls.critvalues = [-4.007, -3.437, -3.137]


@pytest.mark.not_vetted
class TestADFNoConstant(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = unit_root.adfuller(cls.x, regression="nc",
                                      autolag=None, maxlag=4)
        cls.teststat = 3.5227498
        cls.pvalue = .99999
        # Stata does not return a p-value for noconstant.
        # Tau^max in MacKinnon (1994) is missing, so it is
        # assumed that its right-tail is well-behaved
        cls.critvalues = [-2.587, -1.950, -1.617]


# No Unit Root


@pytest.mark.not_vetted
class TestADFConstant2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = unit_root.adfuller(cls.y, regression="c",
                                      autolag=None, maxlag=1)
        cls.teststat = -4.3346988
        cls.pvalue = .00038661
        cls.critvalues = [-3.476, -2.883, -2.573]


@pytest.mark.not_vetted
class TestADFConstantTrend2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = unit_root.adfuller(cls.y, regression="ct",
                                      autolag=None, maxlag=1)
        cls.teststat = -4.425093
        cls.pvalue = .00199633
        cls.critvalues = [-4.006, -3.437, -3.137]


@pytest.mark.not_vetted
class TestADFNoConstant2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = unit_root.adfuller(cls.y, regression="nc",
                                      autolag=None, maxlag=1)
        cls.teststat = -2.4511596
        cls.pvalue = 0.013747
        # Stata does not return a p-value for noconstant
        # this value is just taken from our results
        cls.critvalues = [-2.587, -1.950, -1.617]
        _, _1, _2, cls.store = unit_root.adfuller(cls.y, regression="nc",
                                                  autolag=None, maxlag=1,
                                                  store=True)

    def test_store_str(self):
        assert self.store.__str__() == 'Augmented Dickey-Fuller Test Results'


# -----------------------------------------------------------------
# KPSS


@pytest.mark.not_vetted
class TestKPSS(object):
    """
    R-code
    ------
    library(tseries)
    kpss.stat(x, "Level")
    kpss.stat(x, "Trend")

    In this context, x is the vector containing the
    macrodata['realgdp'] series.
    """
    data = macrodata.load()
    x = data.data['realgdp']

    def test_fail_nonvector_input(self):
        with warnings.catch_warnings(record=True):
            unit_root.kpss(self.x)  # should be fine

        x = np.random.rand(20, 2)
        with pytest.raises(ValueError):
            unit_root.kpss(x)

    @pytest.mark.smoke  # TODO: the last one isnt smoke
    def test_fail_unclear_hypothesis(self):
        # these should be fine,
        with warnings.catch_warnings(record=True):
            unit_root.kpss(self.x, 'c')
            unit_root.kpss(self.x, 'C')
            unit_root.kpss(self.x, 'ct')
            unit_root.kpss(self.x, 'CT')

        with pytest.raises(ValueError):
            unit_root.kpss(self.x, "unclear hypothesis")

    def test_teststat(self):
        with warnings.catch_warnings(record=True):
            kpss_stat, pval, lags, crits = unit_root.kpss(self.x, 'c', 3)
        assert_almost_equal(kpss_stat, 5.0169, 3)

        with warnings.catch_warnings(record=True):
            kpss_stat, pval, lags, crits = unit_root.kpss(self.x, 'ct', 3)
        assert_almost_equal(kpss_stat, 1.1828, 3)

    def test_pval(self):
        with warnings.catch_warnings(record=True):
            kpss_stat, pval, lags, crits = unit_root.kpss(self.x, 'c', 3)
        assert pval == 0.01

        with warnings.catch_warnings(record=True):
            kpss_stat, pval, lags, crits = unit_root.kpss(self.x, 'ct', 3)
        assert pval == 0.01

    def test_store(self):
        with warnings.catch_warnings(record=True):
            kpss_stat, pval, crit, store = unit_root.kpss(self.x, 'c', 3, True)

        # assert attributes, and make sure they're correct
        assert store.nobs == len(self.x)
        assert store.lags == 3

    def test_lags(self):
        with warnings.catch_warnings(record=True):
            kpss_stat, pval, lags, crits = unit_root.kpss(self.x, 'c')

        assert_equal(lags,
                     int(np.ceil(12. * np.power(len(self.x) / 100., 1 / 4.))))
        # assert_warns(UserWarning, kpss, self.x)


# -----------------------------------------------------------------
# coint

# TODO: this doesn't produce the old results anymore
@pytest.mark.not_vetted
class TestCoint_t(object):
    """
    Get AR(1) parameter on residuals

    Test Cointegration Test Results for 2-variable system

    Test values taken from Stata
    """
    levels = ['1%', '5%', '10%']
    data = macrodata.load()
    y1 = data.data['realcons']
    y2 = data.data['realgdp']

    @classmethod
    def setup_class(cls):
        cls.coint_t = unit_root.coint(cls.y1, cls.y2, trend="c",
                                      maxlag=0, autolag=None)[0]
        cls.teststat = -1.8208817
        cls.teststat = -1.830170986148

    def test_tstat(self):
        assert_almost_equal(self.coint_t, self.teststat, 4)


@pytest.mark.not_vetted
def test_coint():
    nobs = 200
    scale_e = 1
    const = [1, 0, 0.5, 0]
    np.random.seed(123)
    unit = np.random.randn(nobs).cumsum()
    y = scale_e * np.random.randn(nobs, 4)
    y[:, :2] += unit[:, None]
    y += const
    y = np.round(y, 4)

    # results from Stata egranger
    res_egranger = {}
    # trend = 'ct'
    res = res_egranger['ct'] = {}
    res[0] = [-5.615251442239, -4.406102369132,
              -3.82866685109, -3.532082997903]
    res[1] = [-5.63591313706, -4.758609717199,
              -4.179130554708, -3.880909696863]
    res[2] = [-2.892029275027, -4.758609717199,
              -4.179130554708, -3.880909696863]
    res[3] = [-5.626932544079, -5.08363327039,
              -4.502469783057, -4.2031051091]

    # trend = 'c'
    res = res_egranger['c'] = {}
    # first critical value res[0][1] has a discrepancy starting at 4th decimal
    res[0] = [-5.760696844656, -3.952043522638,
              -3.367006313729, -3.065831247948]
    # manually adjusted to have higher precision as in other cases
    res[0][1] = -3.952321293401682
    res[1] = [-5.781087068772, -4.367111915942,
              -3.783961136005, -3.483501524709]
    res[2] = [-2.477444137366, -4.367111915942,
              -3.783961136005, -3.483501524709]
    res[3] = [-5.778205811661, -4.735249216434,
              -4.152738973763, -3.852480848968]

    # trend = 'ctt'
    res = res_egranger['ctt'] = {}
    res[0] = [-5.644431269946, -4.796038299708,
              -4.221469431008, -3.926472577178]
    res[1] = [-5.665691609506, -5.111158174219,
              -4.53317278104, -4.23601008516]
    res[2] = [-3.161462374828, -5.111158174219,
              -4.53317278104, -4.23601008516]
    res[3] = [-5.657904558563, -5.406880189412,
              -4.826111619543, -4.527090164875]

    # The following for 'nc' are only regression test numbers
    # trend = 'nc' not allowed in egranger
    # trend = 'nc'
    res = res_egranger['nc'] = {}
    nan = np.nan  # shortcut for table
    res[0] = [-3.7146175989071137, nan, nan, nan]
    res[1] = [-3.8199323012888384, nan, nan, nan]
    res[2] = [-1.6865000791270679, nan, nan, nan]
    res[3] = [-3.7991270451873675, nan, nan, nan]

    for trend in ['c', 'ct', 'ctt', 'nc']:
        res1 = {}
        res1[0] = unit_root.coint(y[:, 0], y[:, 1],
                                  trend=trend, maxlag=4, autolag=None)
        res1[1] = unit_root.coint(y[:, 0], y[:, 1:3],
                                  trend=trend, maxlag=4, autolag=None)
        res1[2] = unit_root.coint(y[:, 0], y[:, 2:],
                                  trend=trend, maxlag=4, autolag=None)
        res1[3] = unit_root.coint(y[:, 0], y[:, 1:],
                                  trend=trend, maxlag=4, autolag=None)

        for i in range(4):
            res = res_egranger[trend]
            assert_allclose(res1[i][0], res[i][0], rtol=1e-11)
            assert_allclose(res1[i][2], res[i][1:], rtol=0, atol=6e-7)


@pytest.mark.not_vetted
def test_coint_identical_series():
    nobs = 200
    scale_e = 1
    np.random.seed(123)
    y = scale_e * np.random.randn(nobs)
    warnings.simplefilter('always', ColinearityWarning)
    with warnings.catch_warnings(record=True) as w:
        c = unit_root.coint(y, y, trend="c", maxlag=0, autolag=None)
    assert len(w) == 1
    assert c[0] == 0.0
    # Limit of table
    assert c[1] > .98


@pytest.mark.not_vetted
def test_coint_perfect_collinearity():
    nobs = 200
    scale_e = 1
    np.random.seed(123)
    x = scale_e * np.random.randn(nobs, 2)
    y = 1 + x.sum(axis=1)
    warnings.simplefilter('always', ColinearityWarning)
    with warnings.catch_warnings(record=True):
        c = unit_root.coint(y, x, trend="c", maxlag=0, autolag=None)
    assert c[0] == 0.0
    # Limit of table
    assert c[1] > .98
