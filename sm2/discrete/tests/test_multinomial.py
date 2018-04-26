# -*- coding: utf-8 -*-
"""
Tests for Multinomial Models, specifically MNLogit
"""
import os

import numpy as np
import pandas as pd
import sm2.api as sm
from sm2.discrete.discrete_model import MNLogit

cur_dir = os.path.dirname(__file__)


class TestUnderflow(object):
    # https://groups.google.com/forum/#!topic/pystatsmodels/FAnjBe_RynE
    # Case where one of the options is _much_ less likely than the others,
    # resulting in overflow/underflow in loglike/score/hessian calculations

    @classmethod
    def setup_class(cls):
        path = os.path.join(cur_dir, 'results', 'MNLogitUnderflowExample.csv')
        df = pd.read_csv(path)
        exog = sm.add_constant(df[['X', 'Y', 'Z']])
        endog = df['Count'].copy()
        # edit endog to be _less_ imbalanced
        # TODO: after this editing, upstream version gets params right
        # but fails on test_llf.  We need a slightly-more-imbalanced case
        endog[endog == 6] = 7
        endog[endog == 9] = 7
        endog[endog == 4] = 7
        model = MNLogit(endog=endog, exog=exog)
        cls.res = model.fit()
        # pre-fix this would risked raising np.linalg.LinAlgError in inverting
        # optimized Hessian (in sm2, I dont think this behavior
        # occurred upstream)
    
    def test_params(self):
        # Test that params are wrapped appropriately
        params = self.res.params
        assert isinstance(params, pd.DataFrame)
        assert (params.index == ['const', 'X', 'Y', 'Z']).all()
        cols = pd.Index([1, 2, 3, 5, 7], name='Count')[1:]
        assert params.columns.equals(cols)
        # TODO: Test the actual values?

    def test_llf(self):
        # GH#3635 in particular we're concerned about np.nan and np.inf
        assert self.res.llf == -432.84270068272059

    def test_predict(self):
        # pre-fix we'd get back an all-NaN vector, and it wouldn't be wrapped
        pred = self.res.predict()
        assert isinstance(pred, pd.DataFrame)
        expected_index = pd.Index([1, 2, 3, 5, 7], name='Count')
        assert pred.columns.equals(expected_index)
        pred1 = pred.sum(1).round(15)  # TODO: Should we clip to make exact?
        assert (pred1 == 1).all()
        # TODO: Check relative probabilities
