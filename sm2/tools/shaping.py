#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-shaping various array-like objects.
"""
import numpy as np
import pandas as pd

from input_types import is_using_pandas


def _ensure_2d(x, ndarray=False):
    """

    Parameters
    ----------
    x : array, Series, DataFrame or None
        Input to verify dimensions, and to transform as necesary
    ndarray : bool
        Flag indicating whether to always return a NumPy array. Setting False
        will return an pandas DataFrame when the input is a Series or a
        DataFrame.

    Returns
    -------
    out : array, DataFrame or None
        array or DataFrame with 2 dimensiona.  One dimensional arrays are
        returned as nobs by 1. None is returned if x is None.
    names : list of str or None
        list containing variables names when the input is a pandas datatype.
        Returns None if the input is an ndarray.

    Notes
    -----
    Accepts None for simplicity
    """
    if x is None:
        return x

    is_pandas = is_using_pandas(x, None)
    if x.ndim == 2:
        if is_pandas:
            return x, x.columns
        else:
            return x, None
    elif x.ndim > 2:
        raise ValueError('x mst be 1 or 2-dimensional.')

    name = x.name if is_pandas else None
    if ndarray:
        return np.asarray(x)[:, None], name
    else:
        return pd.DataFrame(x), name


def unsqueeze(data, axis, oldshape):
    """Unsqueeze a collapsed array

    >>> x = np.random.standard_normal((3,4,5))
    >>> m = np.mean(x, axis=1)
    >>> m.shape
    (3, 5)
    >>> m = unsqueeze(m, 1, x.shape)
    >>> m.shape
    (3, 1, 5)

    """
    newshape = list(oldshape)
    newshape[axis] = 1
    return data.reshape(newshape)


def _asarray_2d_null_rows(x):
    """
    Makes sure input is an array and is 2d. Makes sure output is 2d. True
    indicates a null in the rows of 2d x.
    """
    # Have to have the asarrays because isnull doesn't account for array-like
    # input
    x = np.asarray(x)
    x = atleast_2dcols(x)
    return np.any(pd.isnull(x), axis=1)[:, None]


# TODO: How do we want to handle 0-dimensional?
def atleast_2dcols(x):
    """

    This is similar to np.atleast_2d.  The key difference is where the new
    dimension is inserted.  If a 1-dimensional array is passed to
    np.atleast_2d, the result will have 1 row that matches the original input.
    If the same array is passed to atleast_2dcols, the result will have 1
    column that matches the original input.

    """
    # x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x
