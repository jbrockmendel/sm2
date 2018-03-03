#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
pd_klasses = (pd.Series, pd.DataFrame, pd.Panel)


def is_using_pandas(endog, exog):
    return isinstance(endog, pd_klasses) or isinstance(exog, pd_klasses)
