#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
pd_klasses = (pd.Series, pd.DataFrame, pd.Panel)


def is_using_pandas(endog, exog):
    return isinstance(endog, pd_klasses) or isinstance(exog, pd_klasses)


def is_structured_ndarray(obj):
    return isinstance(obj, np.ndarray) and obj.dtype.names is not None


def is_array_like(endog, exog):
    try:  # do it like this in case of mixed types, ie., ndarray and list
        endog = np.asarray(endog)
        exog = np.asarray(exog)
        return True
    except:
        return False


def is_recarray(data):
    """
    Returns true if data is a recarray
    """
    return isinstance(data, np.core.recarray)
