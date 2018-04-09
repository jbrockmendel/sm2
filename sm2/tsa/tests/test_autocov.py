#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from sm2.tsa.autocov import yule_walker
from sm2.tsa import autocov

from sm2.tsa.tests.results import savedrvs
from sm2.tsa.tests.results.datamlw_tls import mlccf, mlpacf, mlywar


xo = savedrvs.rvsdata.xar2
x100 = xo[-100:] / 1000.
x1000 = xo / 1000.


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
