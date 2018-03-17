"""Testing numerical differentiation

Still some problems, with API (args tuple versus *args)
finite difference Hessian has some problems that I didn't look at yet

Should Hessian also work per observation, if fun returns 2d

"""
from __future__ import print_function

import pytest

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from sm2.tools import numdiff
import sm2.api as sm

DEC3 = 3
DEC4 = 4
DEC5 = 5
DEC6 = 6
DEC8 = 8
DEC13 = 13
DEC14 = 14


def maxabs(x, y):
    return np.abs(x - y).max()


def fun(beta, x):
    return np.dot(x, beta).sum(0)


def fun1(beta, y, x):
    xb = np.dot(x, beta)
    return (y - xb)**2  # (xb-xb.mean(0))**2


def fun2(beta, y, x):
    return fun1(beta, y, x).sum(0)


# ravel() added because of MNLogit 2d params
@pytest.mark.not_vetted
class CheckGradLoglikeMixin(object):
    def test_score(self):
        for test_params in self.params:
            sc = self.mod.score(test_params)
            scfd = numdiff.approx_fprime(test_params.ravel(),
                                         self.mod.loglike)
            assert_almost_equal(sc, scfd, decimal=1)

            sccs = numdiff.approx_fprime_cs(test_params.ravel(),
                                            self.mod.loglike)
            assert_almost_equal(sc, sccs, decimal=11)

    def test_hess(self):
        for test_params in self.params:
            he = self.mod.hessian(test_params)
            hefd = numdiff.approx_fprime_cs(test_params, self.mod.score)
            assert_almost_equal(he, hefd, decimal=DEC8)

            # NOTE: notice the accuracy below
            assert_almost_equal(he, hefd, decimal=7)
            hefd = numdiff.approx_fprime(test_params, self.mod.score,
                                         centered=True)
            assert_allclose(he, hefd, rtol=1e-9)
            hefd = numdiff.approx_fprime(test_params, self.mod.score,
                                         centered=False)
            assert_almost_equal(he, hefd, decimal=4)

            hescs = numdiff.approx_fprime_cs(test_params.ravel(),
                                             self.mod.score)
            assert_allclose(he, hescs, rtol=1e-13)

            hecs = numdiff.approx_hess_cs(test_params.ravel(),
                                          self.mod.loglike)
            assert_allclose(he, hecs, rtol=1e-9)

            # NOTE: Look at the lack of precision - default epsilon not always
            # best
            grad = self.mod.score(test_params)
            hecs, gradcs = numdiff.approx_hess1(test_params, self.mod.loglike,
                                                1e-6, return_grad=True)
            assert_almost_equal(he, hecs, decimal=1)
            assert_almost_equal(grad, gradcs, decimal=1)
            hecs, gradcs = numdiff.approx_hess2(test_params, self.mod.loglike,
                                                1e-4, return_grad=True)
            assert_almost_equal(he, hecs, decimal=3)
            assert_almost_equal(grad, gradcs, decimal=1)
            hecs = numdiff.approx_hess3(test_params, self.mod.loglike, 1e-5)
            assert_almost_equal(he, hecs, decimal=4)


@pytest.mark.not_vetted
class TestGradMNLogit(CheckGradLoglikeMixin):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.anes96.load()
        exog = data.exog
        exog = sm.add_constant(exog, prepend=False)
        cls.mod = sm.MNLogit(data.endog, exog)

        res = cls.mod.fit(disp=0)
        cls.params = [res.params.ravel('F')]

    def test_hess(self):
        # NOTE: I had to overwrite this to lessen the tolerance
        for test_params in self.params:
            he = self.mod.hessian(test_params)
            hefd = numdiff.approx_fprime_cs(test_params, self.mod.score)
            assert_almost_equal(he, hefd, decimal=DEC8)

            # NOTE: notice the accuracy below and the epsilon changes
            # this doesn't work well for score -> hessian with non-cs step
            # it's a little better around the optimum
            assert_almost_equal(he, hefd, decimal=7)
            hefd = numdiff.approx_fprime(test_params, self.mod.score,
                                         centered=True)
            assert_almost_equal(he, hefd, decimal=4)
            hefd = numdiff.approx_fprime(test_params, self.mod.score, 1e-9,
                                         centered=False)
            assert_almost_equal(he, hefd, decimal=2)

            hescs = numdiff.approx_fprime_cs(test_params, self.mod.score)
            assert_almost_equal(he, hescs, decimal=DEC8)

            hecs = numdiff.approx_hess_cs(test_params, self.mod.loglike)
            assert_almost_equal(he, hecs, decimal=5)
            # NOTE: these just don't work well
            # hecs = numdiff.approx_hess1(test_params, self.mod.loglike, 1e-3)
            # assert_almost_equal(he, hecs, decimal=1)
            # hecs = numdiff.approx_hess2(test_params, self.mod.loglike, 1e-4)
            # assert_almost_equal(he, hecs, decimal=0)
            hecs = numdiff.approx_hess3(test_params, self.mod.loglike, 1e-4)
            assert_almost_equal(he, hecs, decimal=0)


@pytest.mark.not_vetted
class TestGradLogit(CheckGradLoglikeMixin):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        cls.mod = sm.Logit(data.endog, data.exog)
        cls.params = [np.array([1, 0.25, 1.4, -7])]


@pytest.mark.not_vetted
class CheckDerivativeMixin(object):
    @classmethod
    def setup_class(cls):
        nobs = 200
        np.random.seed(187678)
        x = np.random.randn(nobs, 3)

        xk = np.array([1, 2, 3])
        xk = np.array([1., 1., 1.])
        beta = xk
        y = np.dot(x, beta) + 0.1 * np.random.randn(nobs)
        xkols = np.dot(np.linalg.pinv(x), y)

        cls.x = x
        cls.y = y
        cls.params = [np.array([1., 1., 1.]), xkols]
        cls.init()

    @classmethod
    def init(cls):
        pass

    def test_grad_fun1_fd(self):
        for test_params in self.params:
            gtrue = self.gradtrue(test_params)
            fun = self.fun()
            epsilon = 1e-6
            gfd = numdiff.approx_fprime(test_params, fun, epsilon=epsilon,
                                        args=self.args)
            gfd += numdiff.approx_fprime(test_params, fun, epsilon=-epsilon,
                                         args=self.args)
            gfd /= 2.
            assert_almost_equal(gtrue, gfd, decimal=DEC6)

    def test_grad_fun1_fdc(self):
        for test_params in self.params:
            gtrue = self.gradtrue(test_params)
            fun = self.fun()

            gfd = numdiff.approx_fprime(test_params, fun, epsilon=1e-8,
                                        args=self.args, centered=True)
            assert_almost_equal(gtrue, gfd, decimal=DEC5)

    def test_grad_fun1_cs(self):
        for test_params in self.params:
            gtrue = self.gradtrue(test_params)
            fun = self.fun()

            gcs = numdiff.approx_fprime_cs(test_params, fun, args=self.args)
            assert_almost_equal(gtrue, gcs, decimal=DEC13)

    def test_hess_fun1_fd(self):
        for test_params in self.params:
            hetrue = self.hesstrue(test_params)
            if hetrue is not None:  # Hessian doesn't work for 2d return of fun
                fun = self.fun()
                # default works, epsilon 1e-6 or 1e-8 is not precise enough
                hefd = numdiff.approx_hess1(test_params, fun,  # epsilon=1e-8,
                                            args=self.args)
                # TODO:should be kwds
                assert_almost_equal(hetrue, hefd, decimal=DEC3)
                # TODO: I reduced precision to DEC3 from DEC4 because of
                #    TestDerivativeFun
                hefd = numdiff.approx_hess2(test_params, fun,  # epsilon=1e-8,
                                            args=self.args)
                # TODO:should be kwds
                assert_almost_equal(hetrue, hefd, decimal=DEC3)
                hefd = numdiff.approx_hess3(test_params, fun,  # epsilon=1e-8,
                                            args=self.args)
                # TODO:should be kwds
                assert_almost_equal(hetrue, hefd, decimal=DEC3)

    def test_hess_fun1_cs(self):
        for test_params in self.params:
            hetrue = self.hesstrue(test_params)
            if hetrue is not None:  # Hessian doesn't work for 2d return of fun
                fun = self.fun()
                hecs = numdiff.approx_hess_cs(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hecs, decimal=DEC6)


@pytest.mark.not_vetted
class TestDerivativeFun(CheckDerivativeMixin):
    @classmethod
    def setup_class(cls):
        super(TestDerivativeFun, cls).setup_class()
        xkols = np.dot(np.linalg.pinv(cls.x), cls.y)
        cls.params = [np.array([1., 1., 1.]), xkols]
        cls.args = (cls.x,)

    def fun(self):
        return fun

    def gradtrue(self, params):
        return self.x.sum(0)

    def hesstrue(self, params):
        return np.zeros((3, 3))
        # make it (3, 3), because test fails with scalar 0
        # why is precision only DEC3


@pytest.mark.not_vetted
class TestDerivativeFun2(CheckDerivativeMixin):
    @classmethod
    def setup_class(cls):
        super(TestDerivativeFun2, cls).setup_class()
        xkols = np.dot(np.linalg.pinv(cls.x), cls.y)
        cls.params = [np.array([1., 1., 1.]), xkols]
        cls.args = (cls.y, cls.x)

    def fun(self):
        return fun2

    def gradtrue(self, params):
        y, x = self.y, self.x
        return (-x * 2 * (y - np.dot(x, params))[:, None]).sum(0)
        # 2*(y-np.dot(x, params)).sum(0)

    def hesstrue(self, params):
        x = self.x
        return 2 * np.dot(x.T, x)


@pytest.mark.not_vetted
class TestDerivativeFun1(CheckDerivativeMixin):
    @classmethod
    def setup_class(cls):
        super(TestDerivativeFun1, cls).setup_class()
        xkols = np.dot(np.linalg.pinv(cls.x), cls.y)
        cls.params = [np.array([1., 1., 1.]), xkols]
        cls.args = (cls.y, cls.x)

    def fun(self):
        return fun1

    def gradtrue(self, params):
        y, x = self.y, self.x
        return (-x * 2 * (y - np.dot(x, params))[:, None])

    def hesstrue(self, params):
        return None
        y, x = self.y, self.x
        return (-x * 2 * (y - np.dot(x, params))[:, None])  # TODO: check shape


@pytest.mark.not_vetted
def test_dtypes():
    def f(x):
        return 2 * x

    desired = np.array([[2, 0],
                        [0, 2]])
    assert_allclose(numdiff.approx_fprime(np.array([1, 2]), f), desired)
    assert_allclose(numdiff.approx_fprime(np.array([1., 2.]), f), desired)
    assert_allclose(numdiff.approx_fprime(np.array([1. + 0j, 2. + 0j]), f),
                    desired)
