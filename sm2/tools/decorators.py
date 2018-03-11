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


class ResettableCache(dict):
    """
    Dictionary whose elements mey depend one from another.

    If entry `B` depends on entry `A`, changing the values of entry `A` will
    reset the value of entry `B` to a default (None); deleteing entry `A` will
    delete entry `B`.  The connections between entries are stored in a
    `_resetdict` private attribute.

    Parameters
    ----------
    reset : dictionary, optional
        An optional dictionary, associated a sequence of entries to any key
        of the object.
    items : var, optional
        An optional dictionary used to initialize the dictionary

    Examples
    --------
    >>> from numpy.testing import assert_equal
    >>> reset = dict(a=('b',), b=('c',))
    >>> cache = resettable_cache(a=0, b=1, c=2, reset=reset)
    >>> assert_equal(cache, dict(a=0, b=1, c=2))

    >>> print("Try resetting a")
    >>> cache['a'] = 1
    >>> assert_equal(cache, dict(a=1, b=None, c=None))
    >>> cache['c'] = 2
    >>> assert_equal(cache, dict(a=1, b=None, c=2))
    >>> cache['b'] = 0
    >>> assert_equal(cache, dict(a=1, b=0, c=None))

    >>> print("Try deleting b")
    >>> del(cache['a'])
    >>> assert_equal(cache, {})
    """

    def __init__(self, reset=None, **items):
        self._resetdict = reset or {}
        dict.__init__(self, **items)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        # if hasattr needed for unpickling with protocol=2
        if hasattr(self, '_resetdict'):
            for mustreset in self._resetdict.get(key, []):
                self[mustreset] = None

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        for mustreset in self._resetdict.get(key, []):
            del(self[mustreset])


resettable_cache = ResettableCache


class CachedAttribute(object):

    def __init__(self, func, cachename=None, resetlist=None):
        self.fget = func
        self.name = func.__name__
        self.cachename = cachename or '_cache'
        self.resetlist = resetlist or ()

    def __get__(self, obj, type=None):
        if obj is None:
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
            try:
                _cache[name] = _cachedval
            except KeyError:
                setattr(_cache, name, _cachedval)

            # Update the reset list if needed (and possible)
            resetlist = self.resetlist
            if resetlist is not ():
                try:
                    _cache._resetdict[name] = self.resetlist
                except AttributeError:
                    pass

        return _cachedval

    def __set__(self, obj, value):
        warnings.warn("The attribute '%s' cannot be overwritten" % self.name,
                      CacheWriteWarning)


class CachedWritableAttribute(CachedAttribute):
    def __set__(self, obj, value):
        _cache = getattr(obj, self.cachename)
        name = self.name
        try:
            _cache[name] = value
        except KeyError:
            setattr(_cache, name, value)


class _cache_readonly(object):
    """
    Decorator for CachedAttribute
    """
    def __init__(self, cachename=None, resetlist=None):
        self.func = None
        self.cachename = cachename
        self.resetlist = resetlist or None

    def __call__(self, func):
        return CachedAttribute(func,
                               cachename=self.cachename,
                               resetlist=self.resetlist)


cache_readonly = _cache_readonly()


class cache_writable(_cache_readonly):
    """
    Decorator for CachedWritableAttribute
    """
    def __call__(self, func):
        return CachedWritableAttribute(func,
                                       cachename=self.cachename,
                                       resetlist=self.resetlist)


def nottest(fn):
    fn.__test__ = False
    return fn
