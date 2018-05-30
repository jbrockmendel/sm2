#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for tools.linalg
"""

from scipy import sparse
import numpy as np
from numpy.testing import (
    assert_array_equal, assert_almost_equal, assert_allclose)
import pytest

from sm2.tools.linalg import (pinv_extended, nan_dot, chain_dot,
                              smw_logdet, smw_solver)


@pytest.mark.not_vetted
def test_chain_dot():
    A = np.arange(1, 13).reshape(3, 4)
    B = np.arange(3, 15).reshape(4, 3)
    C = np.arange(5, 8).reshape(3, 1)
    assert_array_equal(chain_dot(A, B, C),
                       np.array([[1820], [4300], [6780]]))


@pytest.mark.not_vetted
class TestNanDot(object):
    @classmethod
    def setup_class(cls):
        cls.mx_1 = np.array([[np.nan, 1.], [2., 3.]])
        cls.mx_2 = np.array([[np.nan, np.nan], [2., 3.]])
        cls.mx_3 = np.array([[0., 0.], [0., 0.]])
        cls.mx_4 = np.array([[1., 0.], [1., 0.]])
        cls.mx_5 = np.array([[0., 1.], [0., 1.]])
        cls.mx_6 = np.array([[1., 2.], [3., 4.]])

    def test_11(self):  # TODO: BETTER NAME
        test_res = nan_dot(self.mx_1, self.mx_1)
        expected_res = np.array([[np.nan, np.nan], [np.nan, 11.]])
        assert_array_equal(test_res, expected_res)

    def test_12(self):
        test_res = nan_dot(self.mx_1, self.mx_2)
        expected_res = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        assert_array_equal(test_res, expected_res)

    def test_13(self):
        test_res = nan_dot(self.mx_1, self.mx_3)
        expected_res = np.array([[0., 0.], [0., 0.]])
        assert_array_equal(test_res, expected_res)

    def test_14(self):
        test_res = nan_dot(self.mx_1, self.mx_4)
        expected_res = np.array([[np.nan, 0.], [5., 0.]])
        assert_array_equal(test_res, expected_res)

    def test_41(self):
        test_res = nan_dot(self.mx_4, self.mx_1)
        expected_res = np.array([[np.nan, 1.], [np.nan, 1.]])
        assert_array_equal(test_res, expected_res)

    def test_23(self):
        test_res = nan_dot(self.mx_2, self.mx_3)
        expected_res = np.array([[0., 0.], [0., 0.]])
        assert_array_equal(test_res, expected_res)

    def test_32(self):
        test_res = nan_dot(self.mx_3, self.mx_2)
        expected_res = np.array([[0., 0.], [0., 0.]])
        assert_array_equal(test_res, expected_res)

    def test_24(self):
        test_res = nan_dot(self.mx_2, self.mx_4)
        expected_res = np.array([[np.nan, 0.], [5., 0.]])
        assert_array_equal(test_res, expected_res)

    def test_25(self):
        test_res = nan_dot(self.mx_2, self.mx_5)
        expected_res = np.array([[0., np.nan], [0., 5.]])
        assert_array_equal(test_res, expected_res)

    def test_66(self):
        test_res = nan_dot(self.mx_6, self.mx_6)
        expected_res = np.array([[7., 10.], [15., 22.]])
        assert_array_equal(test_res, expected_res)


@pytest.mark.not_vetted
class TestPinvExtended(object):
    def test_extendedpinv(self):
        X = np.random.standard_normal((40, 10))
        np_inv = np.linalg.pinv(X)
        np_sing_vals = np.linalg.svd(X, 0, 0)
        sm_inv, sing_vals = pinv_extended(X)
        assert_almost_equal(np_inv, sm_inv)
        assert_almost_equal(np_sing_vals, sing_vals)

    def test_extendedpinv_singular(self):
        X = np.random.standard_normal((40, 10))
        X[:, 5] = X[:, 1] + X[:, 3]
        np_inv = np.linalg.pinv(X)
        np_sing_vals = np.linalg.svd(X, 0, 0)
        sm_inv, sing_vals = pinv_extended(X)
        assert_almost_equal(np_inv, sm_inv)
        assert_almost_equal(np_sing_vals, sing_vals)


def test_smw_solver():
    # GH#4594

    np.random.seed(23)
    p = 5
    q = 4
    r = 2
    d = 2

    A = np.random.normal(size=(p, q))
    AtA = np.dot(A.T, A)

    B = np.zeros((q, q))
    B[0:r, 0:r] = np.random.normal(size=(r, r))
    di = np.random.uniform(size=d)
    B[r:q, r:q] = np.diag(1 / di)
    Qi = np.linalg.inv(B[0:r, 0:r])
    s = 0.5

    x = np.random.normal(size=p)
    y2 = np.linalg.solve(s * np.eye(p, p) + np.dot(A, np.dot(B, A.T)), x)

    f = smw_solver(s, A, AtA, Qi, di)
    y1 = f(x)
    assert_allclose(y1, y2)

    f = smw_solver(s, sparse.csr_matrix(A), sparse.csr_matrix(AtA), Qi, di)
    y1 = f(x)
    assert_allclose(y1, y2)


@pytest.mark.parametrize('n', range(100))
def test_smw_solver_randomized(n):
    # github.com/statsmodels/statsmodels/pull/4594#issuecomment-392286075
    p = np.random.randint(1, 5)
    # p == 0 is not meaningful
    r = np.random.randint(1, 5)
    d = np.random.randint(1, 5)
    q = r + d  # in general we expect q << p (see docstring)
    p = q + np.random.randint(1, 5)

    A = np.random.normal(size=(p, q))
    AtA = np.dot(A.T, A)

    B = np.zeros((q, q))
    B[0:r, 0:r] = np.random.normal(size=(r, r))
    di = np.random.uniform(size=d)
    B[r:q, r:q] = np.diag(1 / di)
    Qi = np.linalg.inv(B[0:r, 0:r])
    s = np.random.random()  # upstream uses 0.5, requirement is just s>0

    x = np.random.normal(size=p)
    y2 = np.linalg.solve(s * np.eye(p, p) + np.dot(A, np.dot(B, A.T)), x)

    f = smw_solver(s, A, AtA, Qi, di)
    y1 = f(x)
    assert_allclose(y1, y2)

    f = smw_solver(s, sparse.csr_matrix(A), sparse.csr_matrix(AtA), Qi, di)
    y1 = f(x)
    assert_allclose(y1, y2, err_msg=str((p, r, q)))


def test_smw_logdet():
    # GH#4594
    np.random.seed(23)
    p = 5
    q = 4
    r = 2
    d = 2

    A = np.random.normal(size=(p, q))
    AtA = np.dot(A.T, A)

    B = np.zeros((q, q))
    c = np.random.normal(size=(r, r))
    B[0:r, 0:r] = np.dot(c.T, c)
    di = np.random.uniform(size=d)
    B[r:q, r:q] = np.diag(1 / di)
    Qi = np.linalg.inv(B[0:r, 0:r])
    s = 0.5

    _, d2 = np.linalg.slogdet(s * np.eye(p, p) + np.dot(A, np.dot(B, A.T)))

    _, bd = np.linalg.slogdet(B)
    d1 = smw_logdet(s, A, AtA, Qi, di, bd)
    assert_allclose(d1, d2)


# TODO: use hypothesis?  this uses the same random seed in all runs as is
@pytest.mark.parametrize('n', range(100))
def test_smw_logdet_randomized(n):
    # github.com/statsmodels/statsmodels/pull/4594#issuecomment-392286075
    r = np.random.randint(1, 5)
    d = np.random.randint(1, 5)
    q = r + d  # in general we expect q << p (see docstring)
    p = q + np.random.randint(1, 5)

    A = np.random.normal(size=(p, q))
    AtA = np.dot(A.T, A)

    B = np.zeros((q, q))
    # Note this is different from the smw_solver construction.  Here we need
    # B to be positive-semidefinite, but for smw_solver it only
    # needs to be non-singular
    c = np.random.normal(size=(r, r))
    B[0:r, 0:r] = np.dot(c.T, c)
    di = np.random.uniform(size=d)
    B[r:q, r:q] = np.diag(1 / di)
    Qi = np.linalg.inv(B[0:r, 0:r])
    s = np.random.random()  # upstream uses 0.5, requirement is just s>0

    _, d2 = np.linalg.slogdet(s * np.eye(p, p) + np.dot(A, np.dot(B, A.T)))

    _, bd = np.linalg.slogdet(B)
    d1 = smw_logdet(s, A, AtA, Qi, di, bd)
    assert_allclose(d1, d2, atol=1e-12, err_msg=str((p, r, q)))
