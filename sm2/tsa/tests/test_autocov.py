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

from sm2.tsa.stattools import levinson_durbin
from sm2.tsa.autocov import (
    acovf, acf,
    yule_walker, levinson_durbin_pacf, pacf_yw, pacf_burg, burg)
from sm2.tsa import autocov

from sm2.tsa.tests.results import savedrvs
from sm2.tsa.tests.results.datamlw_tls import mlccf, mlpacf, mlywar, mlacf

xo = savedrvs.rvsdata.xar2
x100 = xo[-100:] / 1000.
x1000 = xo / 1000.

# -----------------------------------------------------------------
# TODO: This section is duplicated in test_stattools
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

@pytest.fixture('module')
def acovf_data():
    # GH#4937
    rnd = np.random.RandomState(12345)
    return rnd.randn(250)


@pytest.mark.not_vetted
class TestACF(CheckCorrGram):
    """
    Test Autocorrelation Function
    """
    @classmethod
    def setup_class(cls):
        cls.acf = cls.results['acvar']
        cls.qstat = cls.results['Q1']
        cls.res1 = acf(cls.x, nlags=40,
                       qstat=True, alpha=.05, fft=False)
        cls.confint_res = cls.results[['acvar_lb', 'acvar_ub']].values

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:41], self.acf, 8)

    def test_confint(self):
        centered = self.res1[1] - self.res1[1].mean(1)[:, None]
        assert_almost_equal(centered[1:41], self.confint_res, 8)

    def test_qstat(self):
        assert_almost_equal(self.res1[2][:40], self.qstat, 3)
        # 3 decimal places because of stata rounding

    # FIXME: dont comment-out code
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
        cls.res_drop = acf(cls.x, nlags=40, qstat=True, alpha=.05,
                           missing='drop', fft=False)
        cls.res_conservative = acf(cls.x, nlags=40,
                                   qstat=True, alpha=.05,
                                   missing='conservative', fft=False)
        cls.acf_none = np.empty(40) * np.nan  # lags 1 to 40 inclusive
        cls.qstat_none = np.empty(40) * np.nan
        cls.res_none = acf(cls.x, nlags=40, qstat=True, alpha=.05,
                           missing='none', fft=False)

    def test_raise(self):
        with pytest.raises(MissingDataError):
            acf(self.x, nlags=40, qstat=True, alpha=0.5,
                missing='raise', fft=False)

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

    # FIXME: dont comment-out
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
        cls.res1 = acf(cls.x, nlags=40, qstat=True, fft=True)

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:], self.acf, 8)

    def test_qstat(self):
        # TODO: why is res1/qstat 1 short
        assert_almost_equal(self.res1[2], self.qstat, 3)


@pytest.mark.not_vetted
def test_acf():
    # upstream this is in tsa.tests.test_tsa_tools
    acf_x = acf(x100, unbiased=False, fft=False)[0][:21]
    assert_array_almost_equal(mlacf.acf100.ravel(), acf_x, 8)
    # TODO: why only dec=8?
    acf_x = acf(x1000, unbiased=False, fft=False)[0][:21]
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
    pacfyw = pacf_yw(x100, 20, method='mle')
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfyw, 1)
    pacfyw = pacf_yw(x1000, 20, method='mle')
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfyw, 2)


@pytest.mark.smoke
def test_yule_walker_inter():
    # see GH#1869
    # upstream this is in tsa.tests.test_tsa_tools
    x = np.array([1, -1, 2, 2, 0, -2, 1, 0, -3, 0, 0])
    yule_walker(x, 3)


@pytest.mark.not_vetted
def test_yule_walker():
    # upstream this is in test_regression.
    # TODO: Document where R_params came from
    R_params = [1.2831003105694765, -0.45240924374091945,
                -0.20770298557575195, 0.047943648089542337]

    data = sunspots.load()
    rho, sigma = yule_walker(data.endog, order=4, method="mle")
    # TODO: assert something about sigma?

    assert_almost_equal(rho,
                        R_params,
                        4)


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
    res = acovf(dta, fft=False)
    assert_equal(res, acovf(dta.values, fft=False))

    X = np.random.random((10, 2))
    with pytest.raises(ValueError):
        acovf(X, fft=False)


@pytest.mark.not_vetted
def test_acovf_fft_vs_convolution():
    np.random.seed(1)
    q = np.random.normal(size=100)

    # TODO: parametrize?
    for demean in [True, False]:
        for unbiased in [True, False]:
            F1 = acovf(q, demean=demean, unbiased=unbiased, fft=True)
            F2 = acovf(q, demean=demean, unbiased=unbiased, fft=False)
            assert_almost_equal(F1, F2, decimal=7)


@pytest.mark.not_vetted
def test_acf_fft_dataframe():
    # GH#322
    data = sunspots.load_pandas().data[['SUNACTIVITY']]
    result = acf(data, fft=True)[0]
    assert result.ndim == 1


@pytest.mark.parametrize("missing", ['conservative', 'drop', 'raise', 'none'])
@pytest.mark.parametrize("fft", [False, True])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("unbiased", [True, False])
def test_acovf_nlags(acovf_data, unbiased, demean, fft, missing):
    # GH#4937
    full = acovf(acovf_data, unbiased=unbiased, demean=demean, fft=fft,
                 missing=missing)
    limited = acovf(acovf_data, unbiased=unbiased, demean=demean, fft=fft,
                    missing=missing, nlag=10)
    assert_allclose(full[:11], limited)


@pytest.mark.parametrize("missing", ['conservative', 'drop'])
@pytest.mark.parametrize("fft", [False, True])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("unbiased", [True, False])
def test_acovf_nlags_missing(acovf_data, unbiased, demean, fft, missing):
    # GH#4937
    acovf_data = acovf_data.copy()
    acovf_data[1:3] = np.nan
    full = acovf(acovf_data, unbiased=unbiased, demean=demean, fft=fft,
                 missing=missing)
    limited = acovf(acovf_data, unbiased=unbiased, demean=demean, fft=fft,
                    missing=missing, nlag=10)
    assert_allclose(full[:11], limited)


def test_acovf_error(acovf_data):
    # GH#4937
    with pytest.raises(ValueError):
        acovf(acovf_data, nlag=250, fft=False)


def test_acovf_warns(acovf_data):
    # GH#4937
    with pytest.warns(FutureWarning):
        acovf(acovf_data)


def test_acf_warns(acovf_data):
    # GH#4937
    with pytest.warns(FutureWarning):
        acf(acovf_data, nlags=40)


def test_pandasacovf():
    # test that passing Series vs ndarray to acovf doesn't affect results
    # TODO: GH reference?
    # TODO: Same test for other functions?
    ser = pd.Series(list(range(1, 11)))
    assert_allclose(acovf(ser, fft=False),
                    acovf(ser.values, fft=False),
                    rtol=1e-12)


def test_pacf2acf_ar():
    # GH#5016
    pacf = np.zeros(10)
    pacf[0] = 1
    pacf[1] = 0.9

    ar, acf = levinson_durbin_pacf(pacf)
    assert_allclose(acf, 0.9 ** np.arange(10.))
    assert_allclose(ar, pacf[1:], atol=1e-8)

    ar, acf = levinson_durbin_pacf(pacf, nlags=5)
    assert_allclose(acf, 0.9 ** np.arange(6.))
    assert_allclose(ar, pacf[1:6], atol=1e-8)


def test_pacf2acf_levinson_durbin():
    # GH#5016
    pacf = -0.9 ** np.arange(11.)
    pacf[0] = 1
    ar, acf = levinson_durbin_pacf(pacf)
    _, ar_ld, pacf_ld, _, _ = levinson_durbin(acf, 10, isacov=True)
    assert_allclose(ar, ar_ld, atol=1e-8)
    assert_allclose(pacf, pacf_ld, atol=1e-8)

    # From R, FitAR, PacfToAR
    ar_from_r = [-4.1609, -9.2549, -14.4826, -17.6505, -17.5012,
                 -14.2969, -9.5020, -4.9184, -1.7911, -0.3486]
    assert_allclose(ar, ar_from_r, atol=1e-4)


def test_pacf2acf_errors():
    # GH#5016
    pacf = -0.9 ** np.arange(11.)
    pacf[0] = 1
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf, nlags=20)
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf[:1])
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros(10))
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros((10, 2)))


def test_pacf_burg():
    # GH#5016
    rnd = np.random.RandomState(12345)
    e = rnd.randn(10001)
    y = e[1:] + 0.5 * e[:-1]
    pacf, sigma2 = pacf_burg(y, 10)
    yw_pacf = pacf_yw(y, 10)
    assert_allclose(pacf, yw_pacf, atol=5e-4)
    # Internal consistency check between pacf and sigma2
    ye = y - y.mean()
    s2y = ye.dot(ye) / 10000
    pacf[0] = 0
    sigma2_direct = s2y * np.cumprod(1 - pacf ** 2)
    assert_allclose(sigma2, sigma2_direct, atol=1e-3)


def test_pacf_burg_error():
    # GH#5016
    with pytest.raises(ValueError):
        pacf_burg(np.empty((20, 2)), 10)
    with pytest.raises(ValueError):
        pacf_burg(np.empty(100), 101)


def test_burg():
    # GH#5016
    # upstream this is in test_regression
    rnd = np.random.RandomState(12345)
    e = rnd.randn(10001)
    y = e[1:] + 0.5 * e[:-1]

    # R, ar.burg
    expected = [
        [0.3909931],
        [0.4602607, -0.1771582],
        [0.47473245, -0.21475602, 0.08168813],
        [0.4787017, -0.2251910, 0.1047554, -0.0485900],
        [0.47975462, - 0.22746106, 0.10963527, -0.05896347, 0.02167001]
    ]

    for i in range(1, 6):
        ar, _ = burg(y, i)
        assert_allclose(ar, expected[i - 1], atol=1e-6)
        as_nodemean, _ = burg(1 + y, i, False)
        assert np.all(ar != as_nodemean)


def test_burg_errors():
    # GH#5016
    # upstream this is in test_regression
    with pytest.raises(ValueError):
        burg(np.ones((100, 2)))
    with pytest.raises(ValueError):
        burg(np.random.randn(100), 0)
    with pytest.raises(ValueError):
        burg(np.random.randn(100), 'apple')
