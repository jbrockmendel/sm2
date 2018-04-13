from __future__ import print_function
import warnings

from sm2.tools.sm_exceptions import CacheWriteWarning

__all__ = ['resettable_cache', 'cache_readonly', 'cache_writable',
           'deprecated_alias', 'copy_doc']


def deprecated_alias(old_name, new_name, remove_version=None):
    """
    Older or less-used classes may not conform to statsmodels naming
    conventions.  `deprecated_alias` lets us bring them into conformance
    without breaking backward-compatibility.

    Example
    -------
    Instances of the `Foo` class have a `nvars` attribute, but it _should_
    be called `neqs`:

    class Foo(object):
        nvars = deprecated_alias('nvars', 'neqs')

        def __init__(self, neqs):
            self.neqs = neqs

    >>> foo = Foo(3)
    >>> foo.nvars
    __main__:1: FutureWarning: nvars is a deprecated alias for neqs
    3

    """
    msg = '%s is a deprecated alias for %s' % (old_name, new_name)
    if remove_version is not None:
        msg += ', will be removed in version %s' % remove_version

    def fget(self):
        warnings.warn(msg, FutureWarning, stacklevel=2)
        return getattr(self, new_name)

    def fset(self, value):
        warnings.warn(msg, FutureWarning, stacklevel=2)
        setattr(self, new_name, value)

    res = property(fget=fget, fset=fset)
    return res


def copy_doc(docstring):
    """
    Add a docstring to a function, so that

        def foo(x):
            [...]
        foo.__doc__ = bar

    can be replaced with:

        @copy_doc(bar)
        def foo(x):
            [...]

    """
    def decoration(func):
        func.__doc__ = docstring
        return func
    return decoration


# Upstream ResettableCache is unnecessarily complicated.  See GH#4468
ResettableCache = dict
resettable_cache = ResettableCache


class CachedAttribute(object):
    # changed from upstream by getting rid of `resetlist`
    def __init__(self, func, cachename=None):
        self.fget = func
        self.name = func.__name__
        self.cachename = cachename or '_cache'

    def __get__(self, obj, type=None):
        if obj is None:
            # accessing the attribute on the class, not an instance
            return self.fget

        # Get the cache or set a default one if needed
        _cachename = self.cachename
        _cache = getattr(obj, _cachename, None)
        if _cache is None:
            setattr(obj, _cachename, resettable_cache())
            _cache = getattr(obj, _cachename)

        # Get the name of the attribute to set and cache
        name = self.name
        _cachedval = _cache.get(name, None)
        if _cachedval is None:
            # Call the "fget" function
            _cachedval = self.fget(obj)
            # Set the attribute in obj
            _cache[name] = _cachedval

        return _cachedval

    def __set__(self, obj, value):
        warnings.warn("The attribute '%s' cannot be overwritten" % self.name,
                      CacheWriteWarning)


class CachedWritableAttribute(CachedAttribute):
    def __set__(self, obj, value):
        _cache = getattr(obj, self.cachename)
        _cache[self.name] = value


class _cache_readonly(object):
    """
    Decorator for CachedAttribute
    """
    def __init__(self, cachename=None):
        self.func = None
        self.cachename = cachename

    def __call__(self, func):
        return CachedAttribute(func,
                               cachename=self.cachename)


cache_readonly = _cache_readonly()


class cache_writable(_cache_readonly):
    """
    Decorator for CachedWritableAttribute
    """
    def __call__(self, func):
        return CachedWritableAttribute(func,
                                       cachename=self.cachename)


def nottest(fn):  # pragma: no cover
    raise NotImplementedError("nottest not ported from upstream")


# cached_value and cached_data behave identically to cache_readonly, but
# are used by `remove_data` to
#   a) identify array-like attributes to remove (cached_data)
#   b) make sure certain values are evaluated before caching (cached_value)
from pandas._libs.properties import cache_readonly as CR


class cached_data(CR):
    pass


class cached_value(CR):
    pass
