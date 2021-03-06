# -*- coding: utf-8 -*-
"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""
import warnings

from six import BytesIO, iterkeys
from six.moves import cPickle

import pytest
import numpy as np
import pandas as pd
import pandas.util.testing as tm


import sm2.api as sm


# TODO: Is this implemented (better) elsewhere?
def assert_equal(left, right):
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        tm.assert_series_equal(left, right)
    elif isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        assert left.equals(right)
    else:
        # TODO: Index?
        np.testing.assert_equal(right, left)


def check_pickle(obj):
    fh = BytesIO()
    cPickle.dump(obj, fh, protocol=cPickle.HIGHEST_PROTOCOL)
    plen = fh.tell()
    fh.seek(0, 0)
    res = cPickle.load(fh)
    fh.close()
    return res, plen


@pytest.mark.not_vetted
class RemoveDataPickle(object):
    @classmethod
    def setup_class(cls):
        nobs = 10000
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        x = sm.add_constant(x)
        cls.exog = x
        cls.xf = 0.25 * np.ones((2, 4))
        cls.l_max = 20000
        cls.predict_kwds = {}

    def test_remove_data_pickle(self):
        results = self.results
        xf = self.xf
        pred_kwds = self.predict_kwds
        pred1 = results.predict(xf, **pred_kwds)
        # create some cached attributes
        results.summary()  # OK with this smoke test?

        '''
        # summary2 not ported as of 2018-03-05
        # res = results.summary2()  # SMOKE test also summary2
        '''

        # check pickle unpickle works on full results
        # TODO: drop of load save is tested
        res, _ = check_pickle(results._results)

        # remove data arrays, check predict still works
        with warnings.catch_warnings(record=True):
            results.remove_data()

        pred2 = results.predict(xf, **pred_kwds)
        assert_equal(pred1, pred2)

        # pickle, unpickle reduced array
        res, plen = check_pickle(results._results)

        # for testing attach res
        self.res = res  # TODO: Why?

        # Note: l_max is just a guess for the limit on the length of the pickle
        l_max = self.l_max
        assert plen < l_max

        pred3 = results.predict(xf, **pred_kwds)
        assert_equal(pred1, pred3)

    def test_remove_data_docstring(self):
        assert self.results.remove_data.__doc__ is not None

    def test_pickle_wrapper(self):
        fh = BytesIO()  # use cPickle with binary content

        # test unwrapped results load save pickle
        self.results._results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results._results.__class__.load(fh)
        assert type(res_unpickled) is type(self.results._results)  # noqa:E721
        # TODO: Check equality instead?  This check isnt exactly meaningful

        # test wrapped results load save
        fh.seek(0, 0)
        self.results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results.__class__.load(fh)
        fh.close()
        assert type(res_unpickled) is type(self.results)  # noqa:E721
        # TODO: Check equality instead?  This check isnt exactly meaningful

        before = sorted(iterkeys(self.results.__dict__))
        after = sorted(iterkeys(res_unpickled.__dict__))
        assert before == after

        before = sorted(iterkeys(self.results._results.__dict__))
        after = sorted(iterkeys(res_unpickled._results.__dict__))
        assert before == after

        before = sorted(iterkeys(self.results.model.__dict__))
        after = sorted(iterkeys(res_unpickled.model.__dict__))
        assert before == after

        before = sorted(iterkeys(self.results._cache))
        after = sorted(iterkeys(res_unpickled._cache))
        assert before == after


@pytest.mark.not_vetted
class TestRemoveDataPickleOLS(RemoveDataPickle):
    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.OLS(y, self.exog).fit()


@pytest.mark.not_vetted
class TestRemoveDataPickleWLS(RemoveDataPickle):
    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.WLS(y, self.exog, weights=np.ones(len(y))).fit()


@pytest.mark.not_vetted
class TestRemoveDataPicklePoisson(RemoveDataPickle):
    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)
        #                , exposure=np.ones(nobs),
        #                 offset=np.zeros(nobs))  # bug with default
        # use start_params to converge faster
        start_params = np.array([0.75334818, 0.99425553,
                                 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params,
                                 method='bfgs', disp=0)

        # TODO: temporary, fixed in master
        self.predict_kwds = dict(exposure=1, offset=0)


@pytest.mark.not_vetted
class TestRemoveDataPickleNegativeBinomial(RemoveDataPickle):
    def setup(self):
        # fit for each test, because results will be changed by test
        np.random.seed(987689)
        data = sm.datasets.randhie.load(as_pandas=False)
        data.exog = sm.add_constant(data.exog, prepend=False)
        mod = sm.NegativeBinomial(data.endog, data.exog)
        self.results = mod.fit(disp=0)


@pytest.mark.not_vetted
class TestRemoveDataPickleLogit(RemoveDataPickle):
    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        cutoff = np.random.rand(nobs)
        y_bin = (cutoff < 1.0 / (1 + np.exp(x.sum(1) - x.mean())))
        y_bin = y_bin.astype(int)
        model = sm.Logit(y_bin, x)
        #               , exposure=np.ones(nobs),
        #               offset=np.zeros(nobs)) # bug with default
        # use start_params to converge faster
        start_params = np.array([-0.73403806, -1.00901514,
                                 -0.97754543, -0.95648212])
        self.results = model.fit(start_params=start_params,
                                 method='bfgs', disp=0)


@pytest.mark.not_vetted
class TestRemoveDataPickleRLM(RemoveDataPickle):
    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.RLM(y, self.exog).fit()


@pytest.mark.not_vetted
class TestRemoveDataPickleGLM(RemoveDataPickle):
    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.GLM(y, self.exog).fit()


@pytest.mark.not_vetted
class TestPickleFormula(RemoveDataPickle):
    @classmethod
    def setup_class(cls):
        super(TestPickleFormula, cls).setup_class()
        nobs = 10000
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        cls.exog = pd.DataFrame(x, columns=["A", "B", "C"])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)),
                              columns=cls.exog.columns)
        cls.l_max = 900000  # have to pickle endo/exog to unpickle form.

    def setup(self):
        x = self.exog
        np.random.seed(123)
        y = x.sum(1) + np.random.randn(x.shape[0])
        y = pd.Series(y, name="Y")
        X = self.exog.copy()
        X["Y"] = y
        self.results = sm.OLS.from_formula("Y ~ A + B + C", data=X).fit()


@pytest.mark.not_vetted
class TestPickleFormula2(RemoveDataPickle):
    @classmethod
    def setup_class(cls):
        super(TestPickleFormula2, cls).setup_class()
        nobs = 500
        np.random.seed(987689)
        data = np.random.randn(nobs, 4)
        data[:, 0] = data[:, 1:].sum(1)
        cls.data = pd.DataFrame(data, columns=["Y", "A", "B", "C"])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)),
                              columns=cls.data.columns[1:])
        cls.l_max = 900000  # have to pickle endo/exog to unpickle form.

    def setup(self):
        self.results = sm.OLS.from_formula("Y ~ A + B + C",
                                           data=self.data).fit()


@pytest.mark.not_vetted
class TestPickleFormula3(TestPickleFormula2):
    def setup(self):
        self.results = sm.OLS.from_formula("Y ~ A + B * C",
                                           data=self.data).fit()


@pytest.mark.not_vetted
class TestPickleFormula4(TestPickleFormula2):
    def setup(self):
        self.results = sm.OLS.from_formula("Y ~ np.log(abs(A) + 1) + B * C",
                                           data=self.data).fit()


# we need log in module namespace for the following test
from numpy import log  # noqa:F401


@pytest.mark.not_vetted
class TestPickleFormula5(TestPickleFormula2):
    def setup(self):
        # `log` must be present in the module-level namespace for this
        # test to work
        self.results = sm.OLS.from_formula("Y ~ log(abs(A) + 1) + B * C",
                                           data=self.data).fit()


@pytest.mark.not_vetted
class TestRemoveDataPicklePoissonRegularized(RemoveDataPickle):
    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)
        self.results = model.fit_regularized(method='l1', disp=0, alpha=10)
