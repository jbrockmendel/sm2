"""
Compatibility tools for various data structure inputs
"""
from six.moves import range
import numpy as np
import pandas as pd


# upstream this is in discrete_model._numpy_to_dummies
def numpy_to_dummies(endog):
    from sm2.tools import tools
    if endog.dtype.kind in ['S', 'O']:
        endog_dummies, ynames = tools.categorical(endog, drop=True,
                                                  dictnames=True)
    elif endog.ndim == 2:
        endog_dummies = endog
        ynames = range(endog.shape[1])
    else:
        endog_dummies, ynames = tools.categorical(endog, drop=True,
                                                  dictnames=True)
    return endog_dummies, ynames


# upstream this is in discrete_model._pandas_to_dummies
def pandas_to_dummies(endog):
    if endog.ndim == 2:
        if endog.shape[1] == 1:
            yname = endog.columns[0]
            endog_dummies = pd.get_dummies(endog.iloc[:, 0])
        else:  # series
            yname = 'y'
            endog_dummies = endog
    else:
        yname = endog.name
        endog_dummies = pd.get_dummies(endog)
    ynames = endog_dummies.columns.tolist()

    return endog_dummies, ynames, yname


# upstream this is discrete_margins._get_const_index
def get_const_index(exog):
    """
    Returns a boolean array of non-constant column indices in exog and
    an scalar array of where the constant is or None
    """
    effects_idx = exog.var(0) != 0
    if np.any(~effects_idx):
        const_idx = np.where(~effects_idx)[0]
    else:
        const_idx = None
    return effects_idx, const_idx


# upstream this is discrete_margins._isdummy
def isdummy(X):
    """
    Given an array X, returns the column indices for the dummy variables.

    Parameters
    ----------
    X : array-like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 2, size=(15, 5)).astype(float)
    >>> X[:, 1:3] = np.random.randn(15, 2)
    >>> ind = _isdummy(X)
    >>> ind
    array([0, 3, 4])
    """
    X = np.asarray(X)
    if X.ndim > 1:
        ind = np.zeros(X.shape[1]).astype(bool)
    max = (np.max(X, axis=0) == 1)
    min = (np.min(X, axis=0) == 0)
    remainder = np.all(X % 1. == 0, axis=0)
    ind = min & max & remainder
    if X.ndim == 1:
        ind = np.asarray([ind])
    return np.where(ind)[0]


# upstream this is discrete_margins._get_dummy_index
def get_dummy_index(X, const_idx):
    dummy_ind = isdummy(X)
    dummy = True

    if dummy_ind.size == 0:  # don't waste your time
        dummy = False
        dummy_ind = None  # this gets passed to stand err func
    return dummy_ind, dummy


# upstream this is discrete_margins._iscount
def iscount(X):
    """
    Given an array X, returns the column indices for count variables.

    Parameters
    ----------
    X : array-like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 10, size=(15, 5)).astype(float)
    >>> X[:, 1:3] = np.random.randn(15, 2)
    >>> ind = _iscount(X)
    >>> ind
    array([0, 3, 4])
    """
    X = np.asarray(X)
    remainder = np.logical_and(np.logical_and(np.all(X % 1. == 0, axis=0),
                               X.var(0) != 0), np.all(X >= 0, axis=0))
    dummy = isdummy(X)
    remainder = np.where(remainder)[0].tolist()
    for idx in dummy:
        remainder.remove(idx)
    return np.array(remainder)


def get_count_index(X, const_idx):
    count_ind = iscount(X)
    count = True

    if count_ind.size == 0:  # don't waste your time
        count = False
        count_ind = None  # for stand err func
    return count_ind, count


def _check_period_index(x, freq="M"):  # pragma: no cover
    raise NotImplementedError("_check_period_index not ported from upstream, "
                              "as it is only used in tsaplots, "
                              "which is not (yet) ported")


def is_data_frame(obj):
    return isinstance(obj, pd.DataFrame)


def is_design_matrix(obj):
    from patsy import DesignMatrix
    return isinstance(obj, DesignMatrix)


def is_structured_ndarray(obj):
    return isinstance(obj, np.ndarray) and obj.dtype.names is not None


def interpret_data(data, colnames=None, rownames=None):
    """
    Convert passed data structure to form required by estimation classes

    Parameters
    ----------
    data : ndarray-like
    colnames : sequence or None
        May be part of data structure
    rownames : sequence or None

    Returns
    -------
    (values, colnames, rownames) : (homogeneous ndarray, list)
    """
    if isinstance(data, np.ndarray):
        if _is_structured_ndarray(data):
            if colnames is None:
                colnames = data.dtype.names
            values = struct_to_ndarray(data)
        else:
            values = data

        if colnames is None:
            colnames = ['Y_%d' % i for i in range(values.shape[1])]
    elif is_data_frame(data):
        # XXX: hack
        data = data.dropna()
        values = data.values
        colnames = data.columns
        rownames = data.index
    else:  # pragma: no cover
        raise Exception('cannot handle other input types at the moment')

    if not isinstance(colnames, list):
        colnames = list(colnames)

    # sanity check
    if len(colnames) != values.shape[1]:
        raise ValueError('length of colnames does not match number '
                         'of columns in data')  # pragma: no cover

    if rownames is not None and len(rownames) != len(values):
        raise ValueError('length of rownames does not match number '
                         'of rows in data')  # pragma: no cover

    return values, colnames, rownames


def struct_to_ndarray(arr):
    return arr.view((float, len(arr.dtype.names)), type=np.ndarray)


def is_using_ndarray_type(endog, exog):
    return (type(endog) is np.ndarray and
            (type(exog) is np.ndarray or exog is None))


def is_using_ndarray(endog, exog):
    return (isinstance(endog, np.ndarray) and
            (isinstance(exog, np.ndarray) or exog is None))


def is_using_pandas(endog, exog):
    klasses = (pd.Series, pd.DataFrame, pd.Panel)
    # from sm2.compat.pandas import data_klasses as klasses
    return (isinstance(endog, klasses) or isinstance(exog, klasses))


def _is_array_like(endog, exog):  # pragma: no cover
    raise NotImplementedError("_is_array_like not ported from upstream, "
                              "as it is only used in iolib.foreign, "
                              "which is itself not ported.")


def is_using_patsy(endog, exog):
    # we get this when a structured array is passed through a formula
    return (is_design_matrix(endog) and
            (is_design_matrix(exog) or exog is None))


def is_recarray(data):
    """
    Returns true if data is a recarray
    """
    return isinstance(data, np.core.recarray)


# aliases for backward compatibility
_is_structured_ndarray = is_structured_ndarray
_is_recarray = is_recarray
_is_using_patsy = is_using_patsy
_is_using_pandas = is_using_pandas
_is_using_ndarray = is_using_ndarray
_is_using_ndarray_type = is_using_ndarray_type
