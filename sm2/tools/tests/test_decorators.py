#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy.testing import assert_equal

from sm2.tools.decorators import (cache_writable,
                                  resettable_cache, cache_readonly)


def test_resettable_cache():
    # Refactored out from decorators.py's "if __name__" block

    reset = dict(a=('b',), b=('c',))
    cache = resettable_cache(a=0, b=1, c=2, reset=reset)
    assert_equal(cache, dict(a=0, b=1, c=2))

    cache['a'] = 1
    assert_equal(cache, dict(a=1, b=None, c=None))
    cache['c'] = 2
    assert_equal(cache, dict(a=1, b=None, c=2))
    cache['b'] = 0
    assert_equal(cache, dict(a=1, b=0, c=None))

    # --------------------------------------------------------------------

    class Example(object):
        def __init__(self):
            self._cache = resettable_cache()
            self.a = 0

        @cache_readonly
        def b(self):
            return 1

        @cache_writable(resetlist='d')
        def c(self):
            return 2

        @cache_writable(resetlist=('e', 'f'))
        def d(self):
            return self.c + 1

        @cache_readonly
        def e(self):
            return 4

        @cache_readonly
        def f(self):
            return self.e + 1

    ex = Example()

    assert_equal(ex.__dict__, dict(a=0, _cache={}))

    b = ex.b
    assert_equal(b, 1)
    assert_equal(ex.__dict__, dict(a=0, _cache=dict(b=1,)))
    # assert_equal(ex.__dict__, dict(a=0, b=1, _cache=dict(b=1)))
    ex.b = -1

    assert_equal(ex._cache, dict(b=1,))

    c = ex.c
    assert_equal(c, 2)
    assert_equal(ex._cache, dict(b=1, c=2))
    d = ex.d
    assert_equal(d, 3)
    assert_equal(ex._cache, dict(b=1, c=2, d=3))
    ex.c = 0
    assert_equal(ex._cache, dict(b=1, c=0, d=None, e=None, f=None))
    d = ex.d
    assert_equal(ex._cache, dict(b=1, c=0, d=1, e=None, f=None))
    ex.d = 5
    assert_equal(ex._cache, dict(b=1, c=0, d=5, e=None, f=None))