# -*- coding: utf-8 -*-
"""Linear Filters for time series analysis and testing


TODO:
* check common sequence in signature of filter functions
  (ar, ma, x) or (x, ar, ma)

Created on Sat Oct 23 17:18:03 2010

Author: Josef-pktd
"""
# not original copied from various experimental scripts
# version control history is there

from six.moves import range
import numpy as np
from scipy import signal
from ._utils import _maybe_get_pandas_wrapper


def _pad_nans(x, head=None, tail=None):
    if np.ndim(x) == 1:
        if head is None and tail is None:
            return x
        elif head and tail:
            return np.r_[[np.nan] * head, x, [np.nan] * tail]
        elif tail is None:
            return np.r_[[np.nan] * head, x]
        elif head is None:
            return np.r_[x, [np.nan] * tail]
    elif np.ndim(x) == 2:
        if head is None and tail is None:
            return x
        elif head and tail:
            return np.r_[[[np.nan] * x.shape[1]] * head, x,
                         [[np.nan] * x.shape[1]] * tail]
        elif tail is None:
            return np.r_[[[np.nan] * x.shape[1]] * head, x]
        elif head is None:
            return np.r_[x, [[np.nan] * x.shape[1]] * tail]
    else:
        # TODO: Should this be NotImplementedError?
        raise ValueError("Nan-padding for ndim > 2 not implemented")


def fftconvolveinv(in1, in2, mode="full"):  # pragma: no cover
    raise NotImplementedError("fftconvolveinv not ported from upstream, as "
                              "it is neither used nor tested there "
                              "(except in one sandbox example file)")


def fftconvolve3(in1, in2=None, in3=None, mode="full"):  # pragma: no cover
    raise NotImplementedError("fftconvolve3 not ported from upstream, as "
                              "it is only used once (in a sandbox module) "
                              "and not tested there.")


# original changes and examples in sandbox.tsa.try_var_convolve
# examples and tests are there
def recursive_filter(x, ar_coeff, init=None):
    """
    Autoregressive, or recursive, filtering.

    Parameters
    ----------
    x : array-like
        Time-series data. Should be 1d or n x 1.
    ar_coeff : array-like
        AR coefficients in reverse time order. See Notes
    init : array-like
        Initial values of the time-series prior to the first value of y.
        The default is zero.

    Returns
    -------
    y : array
        Filtered array, number of columns determined by x and ar_coeff. If a
        pandas object is given, a pandas object is returned.

    Notes
    -----
    Computes the recursive filter ::

        y[n] = ar_coeff[0] * y[n-1] + ...
                + ar_coeff[n_coeff - 1] * y[n - n_coeff] + x[n]

    where n_coeff = len(n_coeff).
    """
    _pandas_wrapper = _maybe_get_pandas_wrapper(x)
    x = np.asarray(x).squeeze()
    ar_coeff = np.asarray(ar_coeff).squeeze()

    if x.ndim > 1 or ar_coeff.ndim > 1:
        raise ValueError('x and ar_coeff have to be 1d')

    if init is not None:  # integer init are treated differently in lfiltic
        if len(init) != len(ar_coeff):
            raise ValueError("ar_coeff must be the same length as init")
        init = np.asarray(init, dtype=float)

    if init is not None:
        zi = signal.lfiltic([1], np.r_[1, -ar_coeff], init, x)
    else:
        zi = None

    y = signal.lfilter([1.], np.r_[1, -ar_coeff], x, zi=zi)

    if init is not None:
        result = y[0]
    else:
        result = y

    if _pandas_wrapper:
        return _pandas_wrapper(result)
    return result


def convolution_filter(x, filt, nsides=2):
    """
    Linear filtering via convolution. Centered and backward displaced moving
    weighted average.

    Parameters
    ----------
    x : array_like
        data array, 1d or 2d, if 2d then observations in rows
    filt : array_like
        Linear filter coefficients in reverse time-order. Should have the
        same number of dimensions as x though if 1d and ``x`` is 2d will be
        coerced to 2d.
    nsides : int, optional
        If 2, a centered moving average is computed using the filter
        coefficients. If 1, the filter coefficients are for past values only.
        Both methods use scipy.signal.convolve.

    Returns
    -------
    y : ndarray, 2d
        Filtered array, number of columns determined by x and filt. If a
        pandas object is given, a pandas object is returned. The index of
        the return is the exact same as the time period in ``x``

    Notes
    -----
    In nsides == 1, x is filtered ::

        y[n] = filt[0]*x[n-1] + ... + filt[n_filt-1]*x[n-n_filt]

    where n_filt is len(filt).

    If nsides == 2, x is filtered around lag 0 ::

        y[n] = filt[0]*x[n - n_filt/2] + ... + filt[n_filt / 2] * x[n]
               + ... + x[n + n_filt/2]

    where n_filt is len(filt). If n_filt is even, then more of the filter
    is forward in time than backward.

    If filt is 1d or (nlags, 1) one lag polynomial is applied to all
    variables (columns of x). If filt is 2d, (nlags, nvars) each series is
    independently filtered with its own lag polynomial, uses loop over nvar.
    This is different than the usual 2d vs 2d convolution.

    Filtering is done with scipy.signal.convolve, so it will be reasonably
    fast for medium sized data. For large data fft convolution would be
    faster.
    """
    # for nsides shift the index instead of using 0 for 0 lag this
    # allows correct handling of NaNs
    if nsides == 1:
        trim_head = len(filt) - 1
        trim_tail = None
    elif nsides == 2:
        trim_head = int(np.ceil(len(filt) / 2.) - 1) or None
        trim_tail = int(np.ceil(len(filt) / 2.) - len(filt) % 2) or None
    else:  # pragma: no cover
        raise ValueError("nsides must be 1 or 2")

    _pandas_wrapper = _maybe_get_pandas_wrapper(x)
    x = np.asarray(x)
    filt = np.asarray(filt)
    if x.ndim > 1 and filt.ndim == 1:
        filt = filt[:, None]
    if x.ndim > 2:
        raise ValueError('x array has to be 1d or 2d')

    if filt.ndim == 1 or min(filt.shape) == 1:
        result = signal.convolve(x, filt, mode='valid')
    elif filt.ndim == 2:
        nlags = filt.shape[0]
        nvar = x.shape[1]
        result = np.zeros((x.shape[0] - nlags + 1, nvar))
        if nsides == 2:
            for i in range(nvar):
                # could also use np.convolve, but easier for swiching to fft
                result[:, i] = signal.convolve(x[:, i], filt[:, i],
                                               mode='valid')
        elif nsides == 1:
            for i in range(nvar):
                result[:, i] = signal.convolve(x[:, i], np.r_[0, filt[:, i]],
                                               mode='valid')
    result = _pad_nans(result, trim_head, trim_tail)
    if _pandas_wrapper:
        return _pandas_wrapper(result)
    return result


def miso_lfilter(ar, ma, x, useic=False):  # pragma: no cover
    raise NotImplementedError("miso_lfilter not ported from upstream, as "
                              "it is neither used nor tested there.")
