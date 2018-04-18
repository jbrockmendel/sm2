#!/usr/bin/env python
# -*- coding: utf-8 -*-
__all__ = ['lagmat', 'lagmat2ds', 'add_trend', 'duplication_matrix',
           'elimination_matrix', 'commutation_matrix',
           'vec', 'vech', 'unvec', 'unvech']

from six import PY3
from six.moves import range

import numpy as np
import pandas as pd

from sm2.tools.data import _is_using_pandas, _is_recarray

from sm2.tsa import wold
# aliases for backwards/upstream compat
_ar_transparams = wold.ARMATransparams._ar_transparams
_ar_invtransparams = wold.ARMATransparams._ar_invtransparams
_ma_transparams = wold.ARMATransparams._ma_transparams
_ma_invtransparams = wold.ARMATransparams._ma_invtransparams


def add_lag(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("add_lag not ported from upstream, as it "
                              "is not used outside of tests.")


def detrend(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("detrend not ported from upstream, as it "
                              " is not used outside of tests.")


def add_trend(x, trend="c", prepend=False, has_constant='skip'):
    """
    Adds a trend and/or constant to an array.

    Parameters
    ----------
    X : array-like
        Original array of data.
    trend : str {"c","t","ct","ctt"}
        "c" add constant only
        "t" add trend only
        "ct" add constant and linear trend
        "ctt" add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of X.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant already
        exists in X. 'raise' will raise an error. 'add' will duplicate a
        constant. 'skip' will return the data without change. 'skip' is the
        default.

    Returns
    -------
    y : array, recarray or DataFrame
        The original data with the additional trend columns.  If x is a
        recarray or pandas Series or DataFrame, then the trend column names
        are 'const', 'trend' and 'trend_squared'.

    Notes
    -----
    Returns columns as ["ctt","ct","c"] whenever applicable. There is currently
    no checking for an existing trend.

    See also
    --------
    sm2.tools.tools.add_constant
    """
    # TODO: could be generalized for trend of aribitrary order
    trend = trend.lower()
    columns = ['const', 'trend', 'trend_squared']
    if trend == "c":  # handles structured arrays
        columns = columns[:1]
        trendorder = 0
    elif trend == "ct" or trend == "t":
        columns = columns[:2]
        if trend == "t":
            columns = columns[1:2]
        trendorder = 1
    elif trend == "ctt":
        trendorder = 2
    else:
        raise ValueError("trend %s not understood" % trend)

    is_recarray = _is_recarray(x)
    is_pandas = _is_using_pandas(x, None) or is_recarray
    if is_pandas or is_recarray:
        if is_recarray:
            descr = x.dtype.descr
            x = pd.DataFrame.from_records(x)
        elif isinstance(x, pd.Series):
            x = pd.DataFrame(x)
        else:
            x = x.copy()
    else:
        x = np.asanyarray(x)

    nobs = len(x)
    trendarr = np.vander(np.arange(1, nobs + 1, dtype=np.float64),
                         trendorder + 1)
    # put in order ctt
    trendarr = np.fliplr(trendarr)
    if trend == "t":
        trendarr = trendarr[:, 1]

    if "c" in trend:
        if is_pandas or is_recarray:
            # Mixed type protection
            def safe_is_const(s):
                try:
                    return np.ptp(s) == 0.0 and np.any(s != 0.0)
                except (ValueError, TypeError):
                    return False
            col_const = x.apply(safe_is_const, 0)
        else:
            ptp0 = np.ptp(np.asanyarray(x), axis=0) == 0
            col_const = np.logical_and(np.any(ptp0, axis=0),
                                       np.all(x != 0.0, axis=0))
        if np.any(col_const):
            if has_constant == 'raise':
                raise ValueError("x already contains a constant")
            elif has_constant == 'skip':
                columns = columns[1:]
                trendarr = trendarr[:, 1:]

    order = 1 if prepend else -1
    if is_recarray or is_pandas:
        trendarr = pd.DataFrame(trendarr, index=x.index, columns=columns)
        x = [trendarr, x]
        x = pd.concat(x[::order], 1)
    else:
        x = [trendarr, x]
        x = np.column_stack(x[::order])

    if is_recarray:
        x = x.to_records(index=False, convert_datetime64=False)
        new_descr = x.dtype.descr
        extra_col = len(new_descr) - len(descr)
        if prepend:
            descr = new_descr[:extra_col] + descr
        else:
            descr = descr + new_descr[-extra_col:]

        if not PY3:
            # See GH#3658
            names = [entry[0] for entry in descr]
            dtypes = [entry[1] for entry in descr]
            names = [bytes(name) for name in names]
            # Fail loudly if there is a non-ascii name
            descr = list(zip(names, dtypes))

        x = x.astype(np.dtype(descr))

    return x


def lagmat(x, maxlag, trim='forward', original='ex', use_pandas=False):
    """
    Create 2d array of lags

    Parameters
    ----------
    x : array_like, 1d or 2d
        data; if 2d, observation in rows and variables in columns
    maxlag : int
        all lags from zero to maxlag are included
    trim : str {'forward', 'backward', 'both', 'none'} or None
        * 'forward' : trim invalid observations in front
        * 'backward' : trim invalid initial observations
        * 'both' : trim invalid observations on both sides
        * 'none', None : no trimming of observations
    original : str {'ex','sep','in'}
        * 'ex' : drops the original array returning only the lagged values.
        * 'in' : returns the original array and the lagged values as a single
          array.
        * 'sep' : returns a tuple (original array, lagged values). The original
                  array is truncated to have the same number of rows as
                  the returned lagmat.
    use_pandas : bool, optional
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    lagmat : 2d array
        array with lagged observations
    y : 2d array, optional
        Only returned if original == 'sep'

    Examples
    --------
    >>> from sm2.tsa.tsatools import lagmat
    >>> import numpy as np
    >>> X = np.arange(1,7).reshape(-1,2)
    >>> lagmat(X, maxlag=2, trim="forward", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="backward", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    >>> lagmat(X, maxlag=2, trim="both", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="none", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    Notes
    -----
    When using a pandas DataFrame or Series with use_pandas=True, trim can
    only be 'forward' or 'both' since it is not possible to consistently
    extend index values.
    """
    # TODO: allow list of lags additional to maxlag
    is_pandas = _is_using_pandas(x, None) and use_pandas
    trim = 'none' if trim is None else trim
    trim = trim.lower()
    if is_pandas and trim in ('none', 'backward'):
        raise ValueError("trim cannot be 'none' or 'forward' when used on "
                         "Series or DataFrames")

    xa = np.asarray(x)
    dropidx = 0
    if xa.ndim == 1:
        xa = xa[:, None]

    nobs, nvar = xa.shape
    if original in ['ex', 'sep']:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")

    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
    for k in range(0, int(maxlag + 1)):
        lm[maxlag - k:nobs + maxlag - k,
           nvar * (maxlag - k):nvar * (maxlag - k + 1)] = xa

    if trim in ('none', 'forward'):
        startobs = 0
    elif trim in ('backward', 'both'):
        startobs = maxlag
    else:
        raise ValueError('trim option not valid')

    if trim in ('none', 'backward'):
        stopobs = len(lm)
    else:
        stopobs = nobs

    if is_pandas:
        x_columns = x.columns if isinstance(x, pd.DataFrame) else [x.name]
        columns = [str(col) for col in x_columns]
        for lag in range(maxlag):
            lag_str = str(lag + 1)
            columns.extend([str(col) + '.L.' + lag_str for col in x_columns])
        lm = pd.DataFrame(lm[:stopobs], index=x.index, columns=columns)
        lags = lm.iloc[startobs:]
        if original in ('sep', 'ex'):
            leads = lags[x_columns]
            lags = lags.drop(x_columns, 1)
    else:
        lags = lm[startobs:stopobs, dropidx:]
        if original == 'sep':
            leads = lm[startobs:stopobs, :dropidx]

    # TODO: Avoid multiple-return
    if original == 'sep':
        return lags, leads
    else:
        return lags


def lagmat2ds(x, maxlag0, maxlagex=None, dropex=0, trim='forward',
              use_pandas=False):
    """
    Generate lagmatrix for 2d array, columns arranged by variables

    Parameters
    ----------
    x : array_like, 2d
        2d data, observation in rows and variables in columns
    maxlag0 : int
        for first variable all lags from zero to maxlag are included
    maxlagex : None or int
        max lag for all other variables all lags from zero to maxlag
        are included
    dropex : int (default is 0)
        exclude first dropex lags from other variables
        for all variables, except the first, lags from dropex to maxlagex are
        included
    trim : string
        * 'forward' : trim invalid observations in front
        * 'backward' : trim invalid initial observations
        * 'both' : trim invalid observations on both sides
        * 'none' : no trimming of observations
    use_pandas : bool, optional
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    lagmat : 2d array
        array with lagged observations, columns ordered by variable

    Notes
    -----
    Inefficient implementation for unequal lags, implemented for convenience
    """
    if maxlagex is None:
        maxlagex = maxlag0
    maxlag = max(maxlag0, maxlagex)
    is_pandas = _is_using_pandas(x, None)

    if x.ndim == 1:
        if is_pandas:
            x = pd.DataFrame(x)
        else:
            x = x[:, None]
    elif x.ndim == 0 or x.ndim > 2:
        raise TypeError('Only supports 1 and 2-dimensional data.')

    nobs, nvar = x.shape

    if is_pandas and use_pandas:
        lags = lagmat(x.iloc[:, 0], maxlag, trim=trim,
                      original='in', use_pandas=True)
        lagsli = [lags.iloc[:, :maxlag0 + 1]]
        for k in range(1, nvar):
            lags = lagmat(x.iloc[:, k], maxlag, trim=trim,
                          original='in', use_pandas=True)
            lagsli.append(lags.iloc[:, dropex:maxlagex + 1])
        return pd.concat(lagsli, axis=1)
    elif is_pandas:
        x = np.asanyarray(x)

    new_lagmat = lagmat(x[:, 0], maxlag, trim=trim, original='in')
    lagsli = [new_lagmat[:, :maxlag0 + 1]]
    for k in range(1, nvar):
        new_lagmat = lagmat(x[:, k], maxlag, trim=trim, original='in')
        lagsli.append(new_lagmat[:, dropex:maxlagex + 1])

    return np.column_stack(lagsli)


def vec(mat):
    return mat.ravel('F')


# TODO: is this related to vector_ar.util.vech?
def vech(mat):
    # Gets Fortran-order
    return mat.T.take(_triu_indices(len(mat)))


# tril/triu/diag, suitable for ndarray.take


def _tril_indices(n):  # pragma: no cover
    raise NotImplementedError("_tril_indices not ported from upstream")


def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols


def _diag_indices(n):  # pragma: no cover
    raise NotImplementedError("_diag_indices not ported from upstream")


def unvec(v):
    k = int(np.sqrt(len(v)))
    assert k * k == len(v)
    return v.reshape((k, k), order='F')
    # TODO: Is the 'F' part relevant?


def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result


def duplication_matrix(n):
    """
    Create duplication matrix D_n which satisfies vec(S) = D_n vech(S) for
    symmetric matrix S

    Returns
    -------
    D_n : ndarray
    """
    tmp = np.eye(n * (n + 1) // 2)
    return np.array([unvech(x).ravel() for x in tmp]).T


def elimination_matrix(n):
    """
    Create the elimination matrix L_n which satisfies vech(M) = L_n vec(M) for
    any matrix M
    """
    vech_indices = vec(np.tril(np.ones((n, n))))
    return np.eye(n * n)[vech_indices != 0]


def commutation_matrix(p, q):
    """
    Create the commutation matrix K_{p,q} satisfying vec(A') = K_{p,q} vec(A)

    Parameters
    ----------
    p : int
    q : int

    Returns
    -------
    K : ndarray (pq x pq)
    """
    K = np.eye(p * q)
    indices = np.arange(p * q).reshape((p, q), order='F')
    return K.take(indices.ravel(), axis=0)


def unintegrate_levels(x, d):
    """
    Returns the successive differences needed to unintegrate the series.

    Parameters
    ----------
    x : array-like
        The original series
    d : int
        The number of differences of the differenced series.

    Returns
    -------
    y : array-like
        The increasing differences from 0 to d-1 of the first d elements
        of x.

    See Also
    --------
    unintegrate
    """
    x = x[:d]
    return np.asarray([np.diff(x, d - i)[0] for i in range(d, 0, -1)])


def unintegrate(x, levels):
    """
    After taking n-differences of a series, return the original series

    Parameters
    ----------
    x : array-like
        The n-th differenced series
    levels : list
        A list of the first-value in each differenced series, for
        [first-difference, second-difference, ..., n-th difference]

    Returns
    -------
    y : array-like
        The original series de-differenced

    Examples
    --------
    >>> x = np.array([1, 3, 9., 19, 8.])
    >>> levels = unintegrate_levels(x, 2)
    >>> levels
    array([ 1.,  2.])
    >>> unintegrate(np.diff(x, 2), levels)
    array([  1.,   3.,   9.,  19.,   8.])
    """
    levels = list(levels)[:]  # copy
    if len(levels) > 1:
        x0 = levels.pop(-1)
        return unintegrate(np.cumsum(np.r_[x0, x]), levels)
    x0 = levels[0]
    return np.cumsum(np.r_[x0, x])


def freq_to_period(freq):  # pragma: no cover
    raise NotImplementedError("freq_to_period not ported from upstream (yet)")
