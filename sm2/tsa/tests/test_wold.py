#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from sm2.tsa import wold


@pytest.mark.skip(reason="VARParams doesnt inherit RootsMixin yet")
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

    @pytest.mark.skip(reason="VARParams doesnt inherit RootsMixin yet")
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
