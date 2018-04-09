#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Naming conventions for sm2 models and results classes, collected in one
place to facilitate internal consistency.
"""
from six import string_types


# upstream this is  tools.tools._make_dictnames
def make_dictnames(tmp_arr, offset=0):
    """
    Helper function to create a dictionary mapping a column number
    to the name in tmp_arr.
    """
    col_map = {}
    for i, col_name in enumerate(tmp_arr):
        col_map.update({i + offset: col_name})
    return col_map


# Upstream this is located in tsa.vector_ar.util
def make_lag_names(names, lag_order, trendorder=1, exog=None):
    """
    Produce list of lag-variable names. Constant / trends go at the beginning

    Examples
    --------
    >>> make_lag_names(['foo', 'bar'], 2, 1)
    ['const', 'L1.foo', 'L1.bar', 'L2.foo', 'L2.bar']

    """
    lag_names = []
    if isinstance(names, string_types):
        names = [names]

    # take care of lagged endogenous names
    for i in range(1, lag_order + 1):
        for name in names:
            if not isinstance(name, string_types):
                name = str(name)  # will need consistent unicode handling
            lag_names.append('L' + str(i) + '.' + name)

    # handle the constant name
    # Note: unicode literals are relevant for py2 tests,
    # see test_arima.test_arima_wrapper
    if trendorder != 0:
        lag_names.insert(0, u'const')
    if trendorder > 1:
        lag_names.insert(1, u'trend')
    if trendorder > 2:
        lag_names.insert(2, u'trend**2')
    if exog is not None:
        for i in range(exog.shape[1]):
            lag_names.insert(trendorder + i, u"exog" + str(i))
    return lag_names


# Upstream this is tsa.arima_model._make_arma_names
def make_arma_names(data, k_trend, order, exog_names):
    k_ar, k_ma = order
    exog_names = exog_names or []
    ar_lag_names = make_lag_names([data.ynames], k_ar, 0)
    ar_lag_names = [''.join(('ar.', i)) for i in ar_lag_names]
    ma_lag_names = make_lag_names([data.ynames], k_ma, 0)
    ma_lag_names = [''.join(('ma.', i)) for i in ma_lag_names]
    trend_name = make_lag_names('', 0, k_trend)

    # ensure exog_names stays unchanged when the `fit` method
    # is called multiple times.
    if k_ma == 0 and k_ar == 0:
        if len(exog_names) != 0:
            return exog_names
    elif ((exog_names[-k_ma:] == ma_lag_names) and
            exog_names[-(k_ar + k_ma):-k_ma] == ar_lag_names and
            (not exog_names or not trend_name or
             trend_name[0] == exog_names[0])):
            return exog_names

    exog_names = trend_name + exog_names + ar_lag_names + ma_lag_names
    return exog_names


# upstream this is a method in MultinomialResults._maybe_convert_ynames
def maybe_convert_ynames_int(ynames):
    # see if they're integers
    try:
        for i in ynames:
            if ynames[i] % 1 == 0:
                ynames[i] = str(int(ynames[i]))
    except TypeError:
        pass
    return ynames
