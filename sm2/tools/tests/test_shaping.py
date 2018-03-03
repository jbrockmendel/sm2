#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sm2.tools.shaping import atleast_2dcols

# 2017-05-31 not implemented
# def test_atleast2d_cols_scalar():
#   raw = np.random.randn(1)[0]
#   assert np.ndim(raw) == 0, 'Implicit assumption of test appears invalid'
#
#   cooked = atleast_2dcols(raw)
#   assert np.ndim(cooked) == 2


def test_atleast2d_cols_1d():
    raw = np.random.randn(1)
    assert np.ndim(raw) == 1, 'Implicit assumption of test appears invalid'

    cooked = atleast_2dcols(raw)
    assert np.ndim(cooked) == 2


def test_atleast2d_cols_2d():
    raw = np.random.randn(1, 1)
    assert np.ndim(raw) == 2, 'Implicit assumption of test appears invalid'

    cooked = atleast_2dcols(raw)
    assert np.ndim(cooked) == 2


def test_atleast2d_cols_3d():
    raw = np.random.randn(1, 1, 1)
    assert np.ndim(raw) == 3, 'Implicit assumption of test appears invalid'

    cooked = atleast_2dcols(raw)
    assert np.ndim(cooked) == 3


# Test ported from upstream that has not been vetted/refactored
'''
class TestEnsure2d(object):
    @classmethod
    def setup_class(cls):
        x = np.arange(400.0).reshape((100,4))
        cls.df = pd.DataFrame(x, columns = ['a','b','c','d'])
        cls.series = cls.df.iloc[:,0]
        cls.ndarray = x

    def test_enfore_numpy(self):
        results = tools._ensure_2d(self.df, True)
        assert_array_equal(results[0], self.ndarray)
        assert_array_equal(results[1], self.df.columns)
        results = tools._ensure_2d(self.series, True)
        assert_array_equal(results[0], self.ndarray[:,[0]])
        assert_array_equal(results[1], self.df.columns[0])

    def test_pandas(self):
        results = tools._ensure_2d(self.df, False)
        assert_frame_equal(results[0], self.df)
        assert_array_equal(results[1], self.df.columns)

        results = tools._ensure_2d(self.series, False)
        assert_frame_equal(results[0], self.df.iloc[:,[0]])
        assert_equal(results[1], self.df.columns[0])

    def test_numpy(self):
        results = tools._ensure_2d(self.ndarray)
        assert_array_equal(results[0], self.ndarray)
        assert_equal(results[1], None)

        results = tools._ensure_2d(self.ndarray[:,0])
        assert_array_equal(results[0], self.ndarray[:,[0]])
        assert_equal(results[1], None)
'''