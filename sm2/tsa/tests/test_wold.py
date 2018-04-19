#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.signal

from sm2.tsa import wold


def test_arma_periodogram_unit_root():
    # Hits case where "np.isnan(h).any()"
    ar = np.array([1, -1])
    ma = np.array([1])

    wsd = wold.arma_periodogram(ar, ma, None, 0)
    assert np.isinf(wsd[1][0])


def test_arma_periodiogram_AR1():
    # TODO: non-AR test case
    # TODO: check on the normalizations

    # Start with a simple case where we can calculate the autocovariances
    # easily
    ar = np.array([1, -0.5])
    ma = np.array([1])

    wsd = wold.arma_periodogram(ar, ma, None, 0)

    (w, h) = scipy.signal.freqz(ma, ar, worN=None, whole=0)
    sd = np.abs(h)**2 / np.sqrt(2 * np.pi)

    hrng = np.arange(-100, 100)  # 100 is an arbitrary cutoff
    # Autocorrelations
    # The reader can verify that the variance of this process is 4/3
    variance = 4. / 3.
    gammas = variance * np.array([2.**-abs(n) for n in hrng])

    fw = 0 * w
    for n in range(len(fw)):
        omega = w[n]
        val = gammas * np.exp(-1j * omega * hrng)
        # Note we are not multiplying by 2 here.  I dont know of an
        # especially good reason why not.
        fw[n] = val.sum().real / np.sqrt(2 * np.pi)
        # Note that the denominator is not standard across implementations

    assert_allclose(fw, sd)
    assert_equal(wsd[1], sd)


def test_arma_periodiogram_MA1():
    # TODO: check on the normalizations

    # MA case where we can calculate autocovariances easily
    ar = np.array([1.])
    ma = np.array([1., .4])

    wsd = wold.arma_periodogram(ar, ma, None, 0)

    (w, h) = scipy.signal.freqz(ma, ar, worN=None, whole=0)
    sd = np.abs(h)**2 / np.sqrt(2 * np.pi)

    hrng = np.arange(-100, 100)  # 100 is an arbitrary cutoff
    # Autocorrelations
    gammas = np.array([0. for n in hrng])
    gammas[100] = 1. + .4**2
    gammas[99] = .4
    gammas[101] = .4

    fw = 0 * w
    for n in range(len(fw)):
        omega = w[n]
        val = gammas * np.exp(-1j * omega * hrng)
        # Note we are not multiplying by 2 here.  I dont know of an
        # especially good reason why not.
        fw[n] = val.sum().real / np.sqrt(2 * np.pi)
        # Note that the denominator is not standard across implementations

    assert_allclose(fw, sd)
    assert_equal(wsd[1], sd)


# -------------------------------------------------------------------
# Tests for ARMAParams construction

def test_from_coeffs_None():
    # Test case where None is passed into from_coeffs constructor
    arma = wold.ARMAParams.from_coeffs(None, [.5, -.1])
    control = wold.ARMAParams.from_coeffs([], [.5, -.1])
    assert arma == control


def test_invalid_poly():
    # check that passing invalid parameters to ARMAParams raise as expected
    good = np.array([1.0, -0.5])
    bad = np.array([-0.5])

    # smoke check that valid params _dont_ raise
    wold.ARMAParams(good, good)

    for other in [good, bad, None]:
        with pytest.raises(ValueError):
            wold.ARMAParams(other, bad)
        with pytest.raises(ValueError):
            wold.ARMAParams(bad, other)


def test_invalid_shapes_arma():
    # check that passing params with invalid shapes to ARMAParams/VARParams
    # raises as expected
    # 1-dimensional case
    good = np.array([1, .34, -.2])

    # Check that correct params dont raise
    wold.ARMAParams(good, None)
    wold.ARMAParams(None, good)

    for bad in [np.array(1), good.reshape(1, 3), good.reshape(3, 1, 1)]:
        with pytest.raises(ValueError):
            wold.ARMAParams(good, bad)
        with pytest.raises(ValueError):
            wold.ARMAParams(bad, good)


class TestRoots(object):
    @classmethod
    def setup_class(cls):
        # AR Process that we'll treat as a VAR
        ar = [1, -.25]
        # Note that this induces an
        # AR Polynomial L^0 - L^1 + .25 L^2 --> (1-.5L)**2
        arparams = np.array(ar)
        ma = []
        maparams = np.array(ma)
        cls.varma = wold.VARParams(arparams, maparams)

    def test_k_ma(self):
        assert self.varma.k_ma == 0, (self.varma.k_ma, self.varma.macoefs)

    def test_neqs(self):
        assert self.varma.neqs == 1, self.varma.neqs

    def test_roots(self):
        # Our ar params induce an
        # AR Polynomial L^0 - L^1 + .25 L^2 --> (1-.5L)**2
        # so the roots should both be 2
        roots = self.varma.roots
        assert roots.shape == (2,)
        assert (roots == 2).all()


class TestARMAParams(object):
    @classmethod
    def setup_class(cls):
        # Basic AR(1)
        cls.ar = [1, -0.5]
        cls.ma = [1]
        cls.arma = wold.ARMAParams(cls.ar, cls.ma)

    def test_stationary(self):
        # $y_t = 0.5 * y_{t-1}$ is stationary
        assert self.arma.isstationary  # TODO: This belongs in a separate test

    def test_invertible(self):  # TODO: get a less-dumb test case
        # The MA component is just a [1], so the roots is an empty array
        assert self.arma.maroots.size == 0
        # All on an empty set always returns True, so self.maroots
        # is invertible.
        assert self.arma.isinvertible, (self.ar, self.ma)

    def test_2ar(self):
        # Getting the AR Representation should be effectively the
        # identity operation
        ar = self.ar
        ma = self.ma
        arma = self.arma
        arrep = arma.arma2ar(5)

        assert (arrep[:2] == ar).all()

        # get the same object via the wold.arma2ar function
        arrep2 = wold.arma2ar(ar, ma, 5)
        assert (arrep2 == arrep).all()  # TODO: belongs in separate test?

    def test_2ma(self):
        # Getting the MA Representation should be an exponential decay
        # with rate .5
        arma = self.arma
        marep = arma.arma2ma(5)
        assert (marep == [2.**-n for n in range(len(marep))]).all()

    def test_str(self):
        rep = str(self.arma)
        assert rep == 'ARMAParams\nAR: [1.0, -0.5]\nMA: [1]', rep


def test_mult():
    a20 = wold.ARMAParams.from_coeffs([.5, -.1], [])
    assert (a20.arcoefs == [.5, -.1]).all()
    assert (a20.macoefs == []).all()
    assert (a20.ar == [1, -0.5, 0.1]).all()

    a11 = wold.ARMAParams.from_coeffs([-0.1], [.6])
    assert (a11.arcoefs == [-0.1]).all()
    assert (a11.macoefs == [.6]).all()
    assert (a11.ar == [1, 0.1]).all()

    m = a20 * a11

    assert m.arpoly == a20.arpoly * a11.arpoly
    assert m.mapoly == a20.mapoly * a11.mapoly

    # __mul__ with slightly different types
    m2 = a20 * (a11.ar, a11.ma)
    assert m == m2

    assert not m != m2


# -------------------------------------------------------------------
# Tests for VARParams construction

def test_invalid_shapes_var():
    # 3-lag 2-equation VAR
    ar3 = np.random.randn(3, 2, 2)

    with pytest.raises(ValueError):
        # too few dimensions
        wold.VARParams(ar3[:, :, 0])

    with pytest.raises(ValueError):
        # shape[1] != shape[2]
        wold.VARParams(ar3[:, :, :-1])

    with pytest.raises(ValueError):
        # len(shape) < 3
        wold.VARParams(ar3[:, :, 0])


def test_neqs_mismatch():
    # The number of equations implied by ar shape doesn't match that of ma
    # 3-lag 2-equation 4MA VARMA
    ar3 = np.random.randn(3, 2, 2)
    ma4 = np.random.randn(4, 2, 2)

    wold.VARParams(ar3, ma4)

    with pytest.raises(ValueError):
        # pass ma that suggests too few equations
        wold.VARParams(ar3, ma4[:, :-1, :-1])

    with pytest.raises(ValueError):
        # pass ma with internally-inconsistent shape
        wold.VARParams(ar3, ma4[:, :-1, :])


def test_invalid_intercept_var():
    # 3-lag 2-equation VAR
    ar3 = np.random.randn(3, 2, 2)
    intercept = np.array([1, 2])
    # smoke tests for valid paramaters
    wold.VARParams(ar3)
    wold.VARParams(ar3, intercept=intercept)

    with pytest.raises(ValueError):
        wold.VARParams(ar3, intercept=intercept.reshape(2, 1))

    with pytest.raises(ValueError):
        wold.VARParams(ar3, intercept=intercept.reshape(1, 2))

    with pytest.raises(ValueError):
        wold.VARParams(ar3, intercept=np.array(4))


class TestVARParamsUnivariate(object):
    @classmethod
    def setup_class(cls):
        # AR Process that we'll treat as a VAR
        ar = [1, -.25]
        # Note that this induces an
        # AR Polynomial L^0 - L^1 + .25 L^2 --> (1-.5L)**2
        arparams = np.array(ar)
        ma = []
        maparams = np.array(ma)
        cls.varma = wold.VARParams(arparams, maparams)

    def test_k_ma(self):
        assert self.varma.k_ma == 0, (self.varma.k_ma, self.varma.macoefs)

    def test_neqs(self):
        assert self.varma.neqs == 1, self.varma.neqs

    def test_roots(self):
        # Our ar params induce an
        # AR Polynomial L^0 - L^1 + .25 L^2 --> (1-.5L)**2
        # so the roots should both be 2
        roots = self.varma.roots
        assert roots.shape == (2,)
        assert (roots == 2).all()


class TestVARParams(object):
    @classmethod
    def setup_class(cls):
        # VAR with 2 variables and 3 lags
        ar = [[[.1, .2], [.3, .4]],
              [[.5, .6], [.7, .8]],
              [[.9, 0], [-.1, -.2]]]
        arparams = np.array(ar)

        ma = [[0, .1], [.2, -.3]]
        maparams = np.array(ma)
        # Note: The __init__ call should reshape this from (2, 2) to (1, 2, 2)

        cls.varma = wold.VARParams(arparams, maparams)

        intercept = np.array([1.2, 3.4])
        cls.varma2 = wold.VARParams(arparams, maparams, intercept)

    def test_k_ar(self):
        assert self.varma.k_ar == 3, (self.varma.k_ar, self.varma.arcoefs)

    def test_k_ma(self):
        assert self.varma.k_ma == 1, (self.varma.k_ma, self.varma.macoefs)

    def test_neqs(self):
        assert self.varma.neqs == 2, self.varma.neqs

    def test_intercept(self):
        # Since no intercept was passed to the constructor, an vector of
        # zeros should have been generated.  The length of this vector should
        # be equal to neqs
        varma = self.varma
        intercept = varma.intercept
        assert isinstance(intercept, np.ndarray)
        assert (intercept == 0).all()
        assert intercept.shape == (varma.neqs,), (intercept.shape, varma.neqs)

    def test_mean(self):
        # the arparams are invertible, so zero-intercept implies zero mean.
        assert (self.varma.mean() == 0).all()
        # TODO: assertion for self.varma2.mean()

    def test_long_run_effects(self):
        assert (self.varma.long_run_effects() ==
                self.varma2.long_run_effects()).all()

    def test_roots(self):
        roots = self.varma.roots
        assert roots.shape == (6,)
        # TODO: meaningful assertion about these


class TestVARNotStationary(object):
    @classmethod
    def setup_class(cls):
        ar = [[[1., 2, 3], [0, 1., 4], [0, 0, 1.]]]
        # Choose AR params to be not-stationary
        arparams = np.array(ar)

        ma = [[[0, .1, -1], [.2, -.3, 0], [0.1, 2.1, -0.4]]]
        maparams = np.array(ma)

        cls.varma = wold.VARParams(arparams, maparams)

    def test_mean(self):
        # Non-stationary --> mean should be all-NaN
        assert np.isnan(self.varma.mean()).all()

    def test_long_run_effects(self):
        # Non-stationary --> long_run_effects should be all-NaN
        assert np.isnan(self.varma.long_run_effects()).all()

    def test_is_stable(self):
        # eigenvalues should all be 1
        assert self.varma.is_stable(verbose=True)
        # pass verbose=True just for coverage, the printed output
        # doesnt affect anything
