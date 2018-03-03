#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import functools

from six.moves import reduce
from inspect import getargspec
from statsmodels.compat.python import get_function_name


class SaveLoadMixin(object):
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict_):
        self.__dict__.update(dict_)

    def save(self, fname, remove_data=False):
        """save a pickle of this instance

        Parameters
        ----------
        fname : string or filehandle
            fname can be a string to a file path or filename, or a filehandle.
        remove_data : bool
            If False (default), then the instance is pickled without changes.
            If True, then all arrays with length nobs are set to None before
            pickling. See the remove_data method.
            In some cases not all arrays will be set to None.

        """
        from statsmodels.iolib.smpickle import save_pickle

        if remove_data:
            self.remove_data()

        save_pickle(self, fname)

    @classmethod
    def load(cls, fname):
        """
        load a pickle, (class method)

        Parameters
        ----------
        fname : string or filehandle
            fname can be a string to a file path or filename, or a filehandle.

        Returns
        -------
        unpickled instance

        """
        from statsmodels.iolib.smpickle import load_pickle
        return load_pickle(fname)

    def remove_data(self):
        """remove data arrays, all nobs arrays from result and model

        This reduces the size of the instance, so it can be pickled with less
        memory. Currently tested for use with predict from an unpickled
        results and model instance.

        .. warning:: Since data and some intermediate results have been removed
           calculating new statistics that require them will raise exceptions.
           The exception will occur the first time an attribute is accessed
           that has been set to None.

        Not fully tested for time series models, tsa, and might delete too much
        for prediction or not all that would be possible.

        The lists of arrays to delete are maintained as attributes of
        the result and model instance, except for cached values. These
        lists could be changed before calling remove_data.

        The attributes to remove are named in:

        model._data_attr : arrays attached to both the model instance
            and the results instance with the same attribute name.

        result.data_in_cache : arrays that may exist as values in
            result._cache (TODO : should privatize name)

        result._data_attr_model : arrays attached to the model
            instance but not to the results instance
        """
        if hasattr(self, '_results'):
            # TODO: mix SaveLoadMixin into Results or
            # ResultsWrapper, but not both.
            return self._results.remove_data()

        _data_attr_model = getattr(self, "_data_attr_model", [])
        model_only = ['model.' + i for i in _data_attr_model]
        model_attr = ['model.' + i for i in self.model._data_attr]
        for att in self._data_attr + model_attr + model_only:
            _wipe(self, att)

        data_in_cache = getattr(self, 'data_in_cache', [])
        data_in_cache += ['fittedvalues', 'resid', 'wresid']
        for key in data_in_cache:
            try:
                self._cache[key] = None
            except (AttributeError, KeyError):
                pass


def _wipe(obj, att):
    """_wipe is a private function intended only to be called from within
    SaveLoadMixin.remove_data.
    """
    # get to last element in attribute path
    p = att.split('.')
    att_ = p.pop(-1)
    try:
        obj_ = reduce(getattr, [obj] + p)

        if hasattr(obj_, att_):
            setattr(obj_, att_, None)
    except AttributeError:
        pass


# ------------------------------------------------------------------

class ResultsWrapper(SaveLoadMixin):
    """
    Class which wraps a statsmodels estimation Results class and steps in to
    reattach metadata to results (if available)
    """
    _wrap_attrs = {}
    _wrap_methods = {}

    def __init__(self, results):
        self._results = results
        self.__doc__ = results.__doc__

    def __dir__(self):
        return [x for x in dir(self._results)]

    def __getattribute__(self, attr):
        try:
            results = object.__getattribute__(self, '_results')
        except AttributeError:
            pass

        # TODO: Why not do this check first and avoid the previous one?
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            pass

        obj = getattr(results, attr)
        data = results.model.data
        how = self._wrap_attrs.get(attr)
        if how and isinstance(how, tuple):
            obj = data.wrap_output(obj, how[0], *how[1:])
        elif how:
            obj = data.wrap_output(obj, how=how)

        return obj


def make_wrapper(func, how):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        results = object.__getattribute__(self, '_results')
        data = results.model.data

        unwrapped = func(results, *args, **kwargs)

        if how and isinstance(how, tuple):
            obj = data.wrap_output(unwrapped, how[0], how[1:])
        elif how:
            obj = data.wrap_output(unwrapped, how)
        return obj

    argspec = getargspec(func)
    formatted = inspect.formatargspec(argspec[0], varargs=argspec[1],
                                      defaults=argspec[3])

    func_name = get_function_name(func)

    wrapper.__doc__ = "%s%s\n%s" % (func_name, formatted, wrapper.__doc__)

    return wrapper


def populate_wrapper(klass, wrapping):
    for meth in klass._wrap_methods:
        how = klass._wrap_methods[meth]
        if not hasattr(wrapping, meth):
            continue

        func = getattr(wrapping, meth)
        wrapper = make_wrapper(func, how)
        setattr(klass, meth, wrapper)


# ------------------------------------------------------------------

def maybe_unwrap_results(results):
    """
    Gets raw results back from wrapped results.

    Can be used in plotting functions or other post-estimation type
    routines.
    """
    return getattr(results, '_results', results)


def union_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result
