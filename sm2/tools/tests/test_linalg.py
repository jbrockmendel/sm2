# -*- coding: utf-8 -*-
"""
Tests for tools.linalg
"""

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import pytest

from sm2.tools.linalg import pinv_extended, nan_dot, chain_dot


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
