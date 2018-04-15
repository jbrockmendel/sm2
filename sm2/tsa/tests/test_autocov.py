#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pytest
import pandas as pd
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_almost_equal, assert_allclose)

from sm2.tools.sm_exceptions import MissingDataError
from sm2.datasets import sunspots, macrodata
from sm2.tsa.autocov import yule_walker
from sm2.tsa import autocov

from sm2.tsa.tests.results import savedrvs
from sm2.tsa.tests.results.datamlw_tls import mlccf, mlpacf, mlywar, mlacf

xo = savedrvs.rvsdata.xar2
x100 = xo[-100:] / 1000.
x1000 = xo / 1000.

# -----------------------------------------------------------------
# FIXME: This section is duplicated in test_stattools
cur_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(cur_dir, "results", "results_corrgram.csv")
results_corrgram = pd.read_csv(path, delimiter=',')


@pytest.mark.not_vetted
class CheckCorrGram(object):
    """
    Set up for ACF, PACF tests.
    """
    data = macrodata.load_pandas()
    x = data.data['realgdp']
    results = results_corrgram


# -----------------------------------------------------------------

@pytest.mark.not_vetted
class TestACF(CheckCorrGram):
    """
    Test Autocorrelation Function
    """
    @classmethod
    def setup_class(cls):
        cls.acf = cls.results['acvar']
        cls.qstat = cls.results['Q1']
        cls.res1 = autocov.acf(cls.x, nlags=40, qstat=True, alpha=.05)
        cls.confint_res = cls.results[['acvar_lb', 'acvar_ub']].as_matrix()

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:41], self.acf, 8)

    def test_confint(self):
        centered = self.res1[1] - self.res1[1].mean(1)[:, None]
        assert_almost_equal(centered[1:41], self.confint_res, 8)

    def test_qstat(self):
        assert_almost_equal(self.res1[2][:40], self.qstat, 3)
        # 3 decimal places because of stata rounding

    #def pvalue(self):
    #     pass
    # NOTE: shouldn't need testing if Q stat is correct


@pytest.mark.not_vetted
class TestACFMissing(CheckCorrGram):
    # Test Autocorrelation Function using Missing
    @classmethod
    def setup_class(cls):
        cls.x = np.concatenate((np.array([np.nan]), cls.x))
        cls.acf = cls.results['acvar']  # drop and conservative
        cls.qstat = cls.results['Q1']
        cls.res_drop = autocov.acf(cls.x, nlags=40, qstat=True, alpha=.05,
                                   missing='drop')
        cls.res_conservative = autocov.acf(cls.x, nlags=40,
                                           qstat=True, alpha=.05,
                                           missing='conservative')
        cls.acf_none = np.empty(40) * np.nan  # lags 1 to 40 inclusive
        cls.qstat_none = np.empty(40) * np.nan
        cls.res_none = autocov.acf(cls.x, nlags=40, qstat=True, alpha=.05,
                                   missing='none')

    def test_raise(self):
        with pytest.raises(MissingDataError):
            autocov.acf(self.x, nlags=40, qstat=True, alpha=0.5,
                        missing='raise')

    def test_acf_none(self):
        assert_almost_equal(self.res_none[0][1:41],
                            self.acf_none,
                            8)

    def test_acf_drop(self):
        assert_almost_equal(self.res_drop[0][1:41],
                            self.acf,
                            8)

    def test_acf_conservative(self):
        assert_almost_equal(self.res_conservative[0][1:41],
                            self.acf,
                            8)

    def test_qstat_none(self):
        # TODO: why is res1/qstat 1 short
        assert_almost_equal(self.res_none[2],
                            self.qstat_none,
                            3)

    # TODO: how to do this test? the correct q_stat depends on
    # whether nobs=len(x) is used when x contains NaNs or whether
    # nobs<len(x) when x contains NaNs
    #def test_qstat_drop(self):
    #    assert_almost_equal(self.res_drop[2][:40], self.qstat, 3)


@pytest.mark.not_vetted
class TestACF_FFT(CheckCorrGram):
    # Test Autocorrelation Function using FFT
    @classmethod
    def setup_class(cls):
        cls.acf = cls.results['acvarfft']
        cls.qstat = cls.results['Q1']
        cls.res1 = autocov.acf(cls.x, nlags=40, qstat=True, fft=True)

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:], self.acf, 8)

    def test_qstat(self):
        # TODO: why is res1/qstat 1 short
        assert_almost_equal(self.res1[2], self.qstat, 3)


@pytest.mark.not_vetted
def test_acf():
    # upstream this is in tsa.tests.test_tsa_tools
    acf_x = autocov.acf(x100, unbiased=False)[0][:21]
    assert_array_almost_equal(mlacf.acf100.ravel(), acf_x, 8)
    # TODO: why only dec=8?
    acf_x = autocov.acf(x1000, unbiased=False)[0][:21]
    assert_array_almost_equal(mlacf.acf1000.ravel(), acf_x, 8)
    # TODO: why only dec=9? (comment out of date?)


@pytest.mark.not_vetted
def test_ccf():
    # upstream this is in tsa.tests.test_tsa_tools
    ccf_x = autocov.ccf(x100[4:], x100[:-4], unbiased=False)[:21]
    assert_array_almost_equal(mlccf.ccf100.ravel()[:21][::-1], ccf_x, 8)
    ccf_x = autocov.ccf(x1000[4:], x1000[:-4], unbiased=False)[:21]
    assert_array_almost_equal(mlccf.ccf1000.ravel()[:21][::-1], ccf_x, 8)


@pytest.mark.not_vetted
def test_pacf_yw():
    # upstream this is in tsa.tests.test_tsa_tools
    pacfyw = autocov.pacf_yw(x100, 20, method='mle')
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfyw, 1)
    pacfyw = autocov.pacf_yw(x1000, 20, method='mle')
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfyw, 2)


@pytest.mark.smoke
@pytest.mark.not_vetted
def test_yule_walker_inter():
    # see GH#1869
    # upstream this is in tsa.tests.test_tsa_tools
    x = np.array([1, -1, 2, 2, 0, -2, 1, 0, -3, 0, 0])
    yule_walker(x, 3)


@pytest.mark.not_vetted
def test_ywcoef():
    # upstream this is in tsa.tests.test_tsa_tools
    assert_array_almost_equal(mlywar.arcoef100[1:],
                              -yule_walker(x100, 10, method='mle')[0], 8)
    assert_array_almost_equal(mlywar.arcoef1000[1:],
                              -yule_walker(x1000, 20, method='mle')[0], 8)


@pytest.mark.not_vetted
def test_acovf2d():
    dta = sunspots.load_pandas().data
    dta.index = pd.DatetimeIndex(start='1700', end='2009', freq='A')[:309]
    del dta["YEAR"]
    res = autocov.acovf(dta)
    assert_equal(res, autocov.acovf(dta.values))

    X = np.random.random((10, 2))
    with pytest.raises(ValueError):
        autocov.acovf(X)


@pytest.mark.not_vetted
def test_acovf_fft_vs_convolution():
    np.random.seed(1)
    q = np.random.normal(size=100)

    for demean in [True, False]:
        for unbiased in [True, False]:
            F1 = autocov.acovf(q, demean=demean, unbiased=unbiased, fft=True)
            F2 = autocov.acovf(q, demean=demean, unbiased=unbiased, fft=False)
            assert_almost_equal(F1, F2, decimal=7)


@pytest.mark.not_vetted
def test_acf_fft_dataframe():
    # GH#322
    data = sunspots.load_pandas().data[['SUNACTIVITY']]
    result = autocov.acf(data, fft=True)[0]
    assert result.ndim == 1


def test_pandasacovf():
    # test that passing Series vs ndarray to acovf doesn't affect results
    # TODO: GH reference?
    # TODO: Same test for other functions?
    ser = pd.Series(list(range(1, 11)))
    assert_allclose(autocov.acovf(ser),
                    autocov.acovf(ser.values),
                    rtol=1e-12)
