#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import pytest
from numpy.testing import assert_equal

from sm2.tools.decorators import (cache_writable,
                                  resettable_cache, cache_readonly,
                                  deprecated_alias, copy_doc)


class TestDeprecatedAlias(object):

    @classmethod
    def setup_class(cls):

        class Dummy(object):

            y = deprecated_alias('y', 'x', '0.11.0')

            def __init__(self, y):
                self.x = y

        cls.Dummy = Dummy

    def test_get(self):
        inst = self.Dummy(4)

        with warnings.catch_warnings(record=True) as record:
            assert inst.y == 4

        assert len(record) == 1, record
        assert 'is a deprecated alias' in str(record[0])

    def test_set(self):
        inst = self.Dummy(4)

        with warnings.catch_warnings(record=True) as record:
            inst.y = 5

        assert len(record) == 1, record
        assert 'is a deprecated alias' in str(record[0])
        assert inst.x == 5


class TestCopyDoc:
    def test_copy_doc(self):

        @copy_doc("foo")
        def func(*args, **kwargs):
            return (args, kwargs)

        assert func.__doc__ == "foo"

    def test_copy_doc_overwrite(self):

        @copy_doc("foo")
        def func(*args, **kwargs):
            """bar"""
            return (args, kwargs)

        assert func.__doc__ == "foo"

    def test_copy_doc_orig_altered(self):

        def func(*args, **kwargs):
            """bar"""
            return (args, kwargs)

        func2 = copy_doc("foo")(func)
        assert func2.__doc__ == "foo"
        assert func.__doc__ == "foo"


@pytest.mark.skip(reason="resettle_cache is in the progress of "
                         "being un-ported from upstream.")
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
