"""
Compatibility tools for various data structure inputs
"""
from six.moves import range
import numpy as np
import pandas as pd


def _check_period_index(x, freq="M"):
    if not isinstance(x.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError("The index must be a DatetimeIndex or PeriodIndex")

    if x.index.freq is not None:
        inferred_freq = x.index.freqstr
    else:
        inferred_freq = pd.infer_freq(x.index)
    if not inferred_freq.startswith(freq):
        raise ValueError("Expected frequency {}. Got {}".format(inferred_freq,
                                                                freq))


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
                         'of columns in data')

    if rownames is not None and len(rownames) != len(values):
        raise ValueError('length of rownames does not match number '
                         'of rows in data')

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


def is_array_like(endog, exog):
    try:  # do it like this in case of mixed types, ie., ndarray and list
        endog = np.asarray(endog)
        exog = np.asarray(exog)
        return True
    except:
        return False


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
_is_array_like = is_array_like
_is_using_pandas = is_using_pandas
_is_using_ndarray = is_using_ndarray
_is_using_ndarray_type = is_using_ndarray_type