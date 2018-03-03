#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions models code
# TODO: non-gibberish docstring
"""

from six import integer_types

import numpy as np
import numpy.lib.recfunctions as nprf

from statsmodels.compat.python import asstr2

from .input_types import is_using_pandas, is_recarray


def _make_dictnames(tmp_arr, offset=0):
    """
    Helper function to create a dictionary mapping a column number
    to the name in tmp_arr.
    """
    col_map = {}
    for i, col_name in enumerate(tmp_arr):
        col_map.update({i + offset: col_name})
    return col_map


# TODO: needs to better preserve dtype and be more flexible
# ie., if you still have a string variable in your array you don't
# want to cast it to float
# TODO: add name validator (ie., bad names for datasets.grunfeld)
def categorical(data, col=None, dictnames=False, drop=False, ):
    '''
    Returns a dummy matrix given an array of categorical variables.

    Parameters
    ----------
    data : array
        A structured array, recarray, or array.  This can be either
        a 1d vector of the categorical variable or a 2d array with
        the column specifying the categorical variable specified by the col
        argument.
    col : 'string', int, or None
        If data is a structured array or a recarray, `col` can be a string
        that is the name of the column that contains the variable.  For all
        arrays `col` can be an int that is the (zero-based) column index
        number.  `col` can only be None for a 1d array.  The default is None.
    dictnames : bool, optional
        If True, a dictionary mapping the column number to the categorical
        name is returned.  Used to have information about plain arrays.
    drop : bool
        Whether or not keep the categorical variable in the returned matrix.

    Returns
    --------
    dummy_matrix, [dictnames, optional]
        A matrix of dummy (indicator/binary) float variables for the
        categorical data.  If dictnames is True, then the dictionary
        is returned as well.

    Notes
    -----
    This returns a dummy variable for EVERY distinct variable.  If a
    a structured or recarray is provided, the names for the new variable is the
    old variable name - underscore - category name.  So if the a variable
    'vote' had answers as 'yes' or 'no' then the returned array would have to
    new variables-- 'vote_yes' and 'vote_no'.  There is currently
    no name checking.

    Examples
    --------
    >>> import numpy as np
    >>> import sm2.api as sm

    Univariate examples

    >>> import string
    >>> string_var = [string.ascii_lowercase[0:5], \
                      string.ascii_lowercase[5:10], \
                      string.ascii_lowercase[10:15], \
                      string.ascii_lowercase[15:20],   \
                      string.ascii_lowercase[20:25]]
    >>> string_var *= 5
    >>> string_var = np.asarray(sorted(string_var))
    >>> design = sm.tools.categorical(string_var, drop=True)

    Or for a numerical categorical variable

    >>> instr = np.floor(np.arange(10,60, step=2)/10)
    >>> design = sm.tools.categorical(instr, drop=True)

    With a structured array

    >>> num = np.random.randn(25,2)
    >>> struct_ar = np.zeros((25,1), dtype=[('var1', 'f4'),('var2', 'f4'),  \
                    ('instrument','f4'),('str_instr','a5')])
    >>> struct_ar['var1'] = num[:,0][:,None]
    >>> struct_ar['var2'] = num[:,1][:,None]
    >>> struct_ar['instrument'] = instr[:,None]
    >>> struct_ar['str_instr'] = string_var[:,None]
    >>> design = sm.tools.categorical(struct_ar, col='instrument', drop=True)

    Or

    >>> design2 = sm.tools.categorical(struct_ar, col='str_instr', drop=True)
    '''
    if isinstance(col, (list, tuple)):
        try:
            assert len(col) == 1
            col = col[0]
        except:
            raise ValueError("Can only convert one column at a time")

    # TODO: add a NameValidator function
    # catch recarrays and structured arrays
    if data.dtype.names or data.__class__ is np.recarray:
        if not col and np.squeeze(data).ndim > 1:
            raise IndexError("col is None and the input array is not 1d")
        if isinstance(col, integer_types):
            col = data.dtype.names[col]
        if col is None and data.dtype.names and len(data.dtype.names) == 1:
            col = data.dtype.names[0]

        tmp_arr = np.unique(data[col])

        # if the cols are shape (#,) vs (#,1) need to add an axis and flip
        _swap = True
        if data[col].ndim == 1:
            tmp_arr = tmp_arr[:, None]
            _swap = False
        tmp_dummy = (tmp_arr == data[col]).astype(float)
        if _swap:
            tmp_dummy = np.squeeze(tmp_dummy).swapaxes(1, 0)

        if not tmp_arr.dtype.names:  # TODO: how do we get to this code path?
            tmp_arr = [asstr2(item) for item in np.squeeze(tmp_arr)]
        elif tmp_arr.dtype.names:
            tmp_arr = [asstr2(item) for item in np.squeeze(tmp_arr.tolist())]

        # prepend the varname and underscore, if col is numeric attribute
        # lookup is lost for recarrays...
        if col is None:
            try:
                col = data.dtype.names[0]
            except:
                col = 'var'
        # TODO: the above needs to be made robust because there could be many
        # var_yes, var_no varaibles for instance.
        tmp_arr = [col + '_' + item for item in tmp_arr]
        # TODO: test this for rec and structured arrays!!!

        if drop is True:
            if len(data.dtype) <= 1:
                if tmp_dummy.shape[0] < tmp_dummy.shape[1]:
                    tmp_dummy = np.squeeze(tmp_dummy).swapaxes(1, 0)
                dt = list(zip(tmp_arr, [tmp_dummy.dtype.str] * len(tmp_arr)))
                # preserve array type
                return np.array(list(map(tuple, tmp_dummy.tolist())),
                                dtype=dt).view(type(data))

            data = nprf.drop_fields(data, col, usemask=False,
                                    asrecarray=type(data) is np.recarray)
        data = nprf.append_fields(data, tmp_arr, data=tmp_dummy,
                                  usemask=False,
                                  asrecarray=type(data) is np.recarray)
        return data

    # handle ndarrays and catch array-like for an error
    elif data.__class__ is np.ndarray or not isinstance(data, np.ndarray):
        if not isinstance(data, np.ndarray):
            raise NotImplementedError("Array-like objects are not supported")

        if isinstance(col, integer_types):
            offset = data.shape[1]          # need error catching here?
            tmp_arr = np.unique(data[:, col])
            tmp_dummy = (tmp_arr[:, np.newaxis] == data[:, col]).astype(float)
            tmp_dummy = tmp_dummy.swapaxes(1, 0)
            if drop is True:
                offset -= 1
                data = np.delete(data, col, axis=1).astype(float)
            data = np.column_stack((data, tmp_dummy))
            if dictnames is True:
                col_map = _make_dictnames(tmp_arr, offset)
                return data, col_map
            return data
        elif col is None and np.squeeze(data).ndim == 1:
            tmp_arr = np.unique(data)
            tmp_dummy = (tmp_arr[:, None] == data).astype(float)
            tmp_dummy = tmp_dummy.swapaxes(1, 0)
            if drop is True:
                if dictnames is True:
                    col_map = _make_dictnames(tmp_arr)
                    return tmp_dummy, col_map
                return tmp_dummy
            else:
                data = np.column_stack((data, tmp_dummy))
                if dictnames is True:
                    col_map = _make_dictnames(tmp_arr, offset=1)
                    return data, col_map
                return data
        else:
            raise IndexError("The index %s is not understood" % col)


# TODO: add an axis argument to this for sysreg
def add_constant(data, prepend=True, has_constant='skip'):
    """
    Adds a column of ones to an array

    Parameters
    ----------
    data : array-like
        ``data`` is the column-ordered design matrix
    prepend : bool
        If true, the constant is in the first column.  Else the constant is
        appended (last column).
    has_constant : str {'raise', 'add', 'skip'}
        Behavior if ``data`` already has a constant. The default will return
        data without adding another constant. If 'raise', will raise an
        error if a constant is present. Using 'add' will duplicate the
        constant, if one is present.

    Returns
    -------
    data : array, recarray or DataFrame
        The original values with a constant (column of ones) as the first or
        last column. Returned value depends on input type.

    Notes
    -----
    When the input is recarray or a pandas Series or DataFrame, the added
    column's name is 'const'.
    """
    if is_using_pandas(data, None) or is_recarray(data):
        from statsmodels.tsa.tsatools import add_trend
        # TODO: Move add_trend here to make tools self-contained?
        return add_trend(data, trend='c',
                         prepend=prepend, has_constant=has_constant)

    # Special case for NumPy
    x = np.asanyarray(data)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        raise ValueError('Only implementd 2-dimensional arrays')

    is_nonzero_const = np.ptp(x, axis=0) == 0
    is_nonzero_const &= np.all(x != 0.0, axis=0)
    if is_nonzero_const.any():
        if has_constant == 'skip':
            return x
        elif has_constant == 'raise':
            raise ValueError("data already contains a constant")

    x = [np.ones(x.shape[0]), x]
    if not prepend:
        x = x[::-1]
    return np.column_stack(x)
