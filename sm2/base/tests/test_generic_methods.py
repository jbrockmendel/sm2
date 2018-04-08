# -*- coding: utf-8 -*-
"""Tests that use cross-checks for generic methods

Should be easy to check consistency across models
Does not cover tsa

Initial cases copied from test_shrink_pickle

Created on Wed Oct 30 14:01:27 2013

Author: Josef Perktold
"""
import pytest
from six import StringIO

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pandas.util.testing as tm

import sm2.api as sm
from sm2.discrete.discrete_model import DiscreteResults
import sm2.formula.api as smf  # noqa:F841 # mostly just to get coverage


@pytest.mark.not_vetted
class CheckGenericMixin(object):
    @classmethod
    def setup_class(cls):
        nobs = 500
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        x = sm.add_constant(x)
        cls.exog = x
        cls.xf = 0.25 * np.ones((2, 4))
        cls.predict_kwds = {}  # TODO: Should these be used at some point?

    def test_ttest_tvalues(self):
        # test that t_test has same results a params, bse, tvalues, ...
        res = self.results
        mat = np.eye(len(res.params))
        tt = res.t_test(mat)

        assert_allclose(tt.effect, res.params, rtol=1e-12)
        # TODO: tt.sd and tt.tvalue are 2d also for single regressor, squeeze
        assert_allclose(np.squeeze(tt.sd), res.bse, rtol=1e-10)
        assert_allclose(np.squeeze(tt.tvalue), res.tvalues, rtol=1e-12)
        assert_allclose(tt.pvalue, res.pvalues, rtol=5e-10)
        assert_allclose(tt.conf_int(), res.conf_int(), rtol=1e-10)

        # test params table frame returned by t_test
        table_res = np.column_stack((res.params, res.bse, res.tvalues,
                                     res.pvalues, res.conf_int()))
        table2 = tt.summary_frame().values
        assert_allclose(table2, table_res, rtol=1e-12)

        # move this to test_attributes ?
        assert hasattr(res, 'use_t')

        tt = res.t_test(mat[0])

        summ = tt.summary()   # smoke test for GH#1323
        assert_allclose(tt.pvalue, res.pvalues[0], rtol=5e-10)

        string_confint = "[%4.3F      %4.3F]" % (.05 / 2, 1 - .05 / 2)
        assert string_confint in str(summ)
        # issue GH#3116 alpha not used in column headers
        summ = tt.summary(alpha=0.1)
        ss = "[0.05       0.95]"   # different formatting
        assert ss in str(summ)

        summf = tt.summary_frame(alpha=0.1)
        pvstring_use_t = 'P>|z|' if res.use_t is False else 'P>|t|'
        tstring_use_t = 'z' if res.use_t is False else 't'
        cols = ['coef', 'std err', tstring_use_t, pvstring_use_t,
                'Conf. Int. Low', 'Conf. Int. Upp.']
        assert_array_equal(summf.columns.values, cols)

    def test_ftest_pvalues(self):
        res = self.results
        use_t = res.use_t
        k_vars = len(res.params)
        # check default use_t
        pvals = [res.wald_test(np.eye(k_vars)[k], use_f=use_t).pvalue
                 for k in range(k_vars)]
        assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

        # automatic use_f based on results class use_t
        pvals = [res.wald_test(np.eye(k_vars)[k]).pvalue
                 for k in range(k_vars)]
        assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

    @pytest.mark.smoke
    def test_summary(self):
        res = self.results
        # label for pvalues in summary
        string_use_t = 'P>|z|' if res.use_t is False else 'P>|t|'
        summ = str(res.summary())
        assert string_use_t in summ

    @pytest.mark.skip(reason="summary2 not ported from upstream")
    def test_summary2(self):
        res = self.results
        use_t = res.use_t
        # label for pvalues in summary
        string_use_t = 'P>|z|' if use_t is False else 'P>|t|'

        # try except for models that don't have summary2
        try:
            summ2 = str(res.summary2())
        except AttributeError:
            raise pytest.skip(reason="summary2 not implemented for class {cls}"
                                     .format(cls=res.__class__.__name__))
        assert string_use_t in summ2

    def test_fitted(self):
        # ignore wrapper for isinstance check
        # FIXME: work around GEE has no wrapper
        results = getattr(self.results, '_results', self.results)
        if (isinstance(results, DiscreteResults) or
                results.__class__.__name__ == 'GLMResults'):
            # __name__ check is a kludge to avoid needing import from upstream
            raise pytest.skip('Infeasible for {0}'.format(type(results)))

        res = self.results
        fitted = res.fittedvalues
        assert_allclose(res.model.endog - fitted, res.resid, rtol=1e-12)
        assert_allclose(fitted, res.predict(), rtol=1e-12)

    def test_predict_types(self):
        res = self.results
        # squeeze to make 1d for single regressor test case
        p_exog = np.squeeze(np.asarray(res.model.exog[:2]))

        # ignore wrapper for isinstance check
        # FIXME: work around GEE has no wrapper
        results = getattr(self.results, '_results', self.results)

        if (isinstance(results, DiscreteResults) or
                results.__class__.__name__ == 'GLMResults'):
            # __name__ check is a kludge to avoid needing import from upstream
            # SMOKE test only  TODO
            res.predict(p_exog)
            res.predict(p_exog.tolist())
            res.predict(p_exog[0].tolist())
        else:

            fitted = res.fittedvalues[:2]
            assert_allclose(fitted, res.predict(p_exog), rtol=1e-12)
            # this needs reshape to column-vector:
            assert_allclose(fitted, res.predict(np.squeeze(p_exog).tolist()),
                            rtol=1e-12)
            # only one prediction:
            assert_allclose(fitted[:1],
                            res.predict(p_exog[0].tolist()),
                            rtol=1e-12)
            assert_allclose(fitted[:1],
                            res.predict(p_exog[0]),
                            rtol=1e-12)

            exog_index = range(len(p_exog))
            predicted = res.predict(p_exog)

            if p_exog.ndim == 1:
                predicted_pandas = res.predict(pd.Series(p_exog,
                                                         index=exog_index))
            else:
                predicted_pandas = res.predict(pd.DataFrame(p_exog,
                                                            index=exog_index))

            if predicted.ndim == 1:
                assert isinstance(predicted_pandas, pd.Series)
                predicted_expected = pd.Series(predicted, index=exog_index)
                tm.assert_series_equal(predicted_expected, predicted_pandas)
            else:
                assert isinstance(predicted_pandas, pd.DataFrame)
                predicted_expected = pd.DataFrame(predicted, index=exog_index)
                assert predicted_expected.equals(predicted_pandas)


# subclasses for individual models, unchanged from test_shrink_pickle
# TODO: check if setup_class is faster than setup

@pytest.mark.not_vetted
class TestGenericOLS(CheckGenericMixin):
    model_cls = sm.OLS
    fit_kwargs = {}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = self.model_cls(y, self.exog).fit(**self.fit_kwargs)


@pytest.mark.not_vetted
class TestGenericOLSOneExog(CheckGenericMixin):
    # check with single regressor (no constant)
    model_cls = sm.OLS
    fit_kwargs = {}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog[:, 1]
        np.random.seed(987689)
        y = x + np.random.randn(x.shape[0])
        self.results = self.model_cls(y, x).fit(**self.fit_kwargs)


@pytest.mark.not_vetted
class TestGenericWLS(CheckGenericMixin):
    model_cls = sm.WLS
    fit_kwargs = {}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        model = self.model_cls(y, self.exog, weights=np.ones(len(y)))
        self.results = model.fit(**self.fit_kwargs)


@pytest.mark.not_vetted
class TestGenericPoisson(CheckGenericMixin):
    model_cls = sm.Poisson
    fit_kwargs = {
        "disp": False,
        "method": "bfgs",
        # use start_params to converge faster
        "start_params": np.array([0.75334818, 0.99425553,
                                  1.00494724, 1.00247112])}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = self.model_cls(y_count, x)
        # , exposure=np.ones(nobs), offset=np.zeros(nobs)) # bug with default
        self.results = model.fit(**self.fit_kwargs)

        # TODO: temporary, fixed in master
        self.predict_kwds = dict(exposure=1, offset=0)
        # TODO: Should these be used at some point?


@pytest.mark.not_vetted
class TestGenericNegativeBinomial(CheckGenericMixin):
    model_cls = sm.NegativeBinomial
    fit_kwargs = {
        "disp": False,
        "start_params": np.array([-0.0565406, -0.21213599, 0.08783076,
                                  -0.02991835, 0.22901974, 0.0621026,
                                  0.06799283, 0.08406688, 0.18530969,
                                  1.36645452])}

    def setup(self):
        # fit for each test, because results will be changed by test
        np.random.seed(987689)
        data = sm.datasets.randhie.load()
        exog = sm.add_constant(data.exog, prepend=False)
        # FIXME: we're editing exog but then not _using_ it.
        # Editing data.exog instead causes tests to fail.
        mod = self.model_cls(data.endog, data.exog)
        self.results = mod.fit(**self.fit_kwargs)


@pytest.mark.not_vetted
class TestGenericLogit(CheckGenericMixin):
    model_cls = sm.Logit
    fit_kwargs = {
        "disp": False,
        "method": "bfgs",
        # use start_params to converge faster
        "start_params": np.array([-0.73403806, -1.00901514,
                                  -0.97754543, -0.95648212])}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_bin = np.random.rand(nobs) < 1.0 / (1 + np.exp(x.sum(1) - x.mean()))
        y_bin = y_bin.astype(int)
        model = self.model_cls(y_bin, x)
        # , exposure=np.ones(nobs), offset=np.zeros(nobs)) # bug with default
        self.results = model.fit(**self.fit_kwargs)


@pytest.mark.not_vetted
class TestGenericRLM(CheckGenericMixin):
    model_cls = sm.RLM
    fit_kwargs = {}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = self.model_cls(y, self.exog).fit(**self.fit_kwargs)


@pytest.mark.not_vetted
class TestGenericGLM(CheckGenericMixin):
    model_cls = sm.GLM
    fit_kwargs = {}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = self.model_cls(y, self.exog).fit(**self.fit_kwargs)


@pytest.mark.skip(reason="GEE not ported from upstream")
@pytest.mark.not_vetted
class TestGenericGEEPoisson(CheckGenericMixin):
    fit_kwargs = {
        # use start_params to speed up test, difficult convergence not tested
        "start_params": np.array([0., 1., 1., 1.])}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        groups = np.random.randint(0, 4, size=x.shape[0])

        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        self.results = sm.GEE(y_count, self.exog, groups, family=family,
                              cov_struct=vi).fit(**self.fit_kwargs)


@pytest.mark.skip(reason="GEE not ported from upstream")
@pytest.mark.not_vetted
class TestGenericGEEPoissonNaive(CheckGenericMixin):
    fit_kwargs = {
        "cov_type": "naive",
        # use start_params to speed up test, difficult convergence not tested
        "start_params": np.array([0., 1., 1., 1.])}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.sum(1).mean(0)))
        groups = np.random.randint(0, 4, size=x.shape[0])

        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        self.results = sm.GEE(y_count, self.exog, groups, family=family,
                              cov_struct=vi).fit(**self.fit_kwargs)


@pytest.mark.skip(reason="GEE not ported from upstream")
@pytest.mark.not_vetted
class TestGenericGEEPoissonBC(CheckGenericMixin):
    fit_kwargs = {
        "cov_type": "bias_reduced",
        # use start_params to speed up test; difficult convergence not tested
        # expected estimated params are
        # params_est = np.array([-0.0063238 , 0.99463752,
        #                        1.02790201, 0.98080081])
        "start_params": np.array([0., 1., 1., 1.])}

    def setup(self):
        # fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.sum(1).mean(0)))
        groups = np.random.randint(0, 4, size=x.shape[0])

        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        mod = sm.GEE(y_count, self.exog, groups, family=family, cov_struct=vi)
        self.results = mod.fit(**self.fit_kwargs)


# ------------------------------------------------------------------
# Other test classes


# kidney_table moved from deprecated test_anova, where it had the comment:
# "# kidney data taken from JT's course"
# "don't know the license"
kidney_table = StringIO("""Days      Duration Weight ID
    0.0      1      1      1
    2.0      1      1      2
    1.0      1      1      3
    3.0      1      1      4
    0.0      1      1      5
    2.0      1      1      6
    0.0      1      1      7
    5.0      1      1      8
    6.0      1      1      9
    8.0      1      1     10
    2.0      1      2      1
    4.0      1      2      2
    7.0      1      2      3
   12.0      1      2      4
   15.0      1      2      5
    4.0      1      2      6
    3.0      1      2      7
    1.0      1      2      8
    5.0      1      2      9
   20.0      1      2     10
   15.0      1      3      1
   10.0      1      3      2
    8.0      1      3      3
    5.0      1      3      4
   25.0      1      3      5
   16.0      1      3      6
    7.0      1      3      7
   30.0      1      3      8
    3.0      1      3      9
   27.0      1      3     10
    0.0      2      1      1
    1.0      2      1      2
    1.0      2      1      3
    0.0      2      1      4
    4.0      2      1      5
    2.0      2      1      6
    7.0      2      1      7
    4.0      2      1      8
    0.0      2      1      9
    3.0      2      1     10
    5.0      2      2      1
    3.0      2      2      2
    2.0      2      2      3
    0.0      2      2      4
    1.0      2      2      5
    1.0      2      2      6
    3.0      2      2      7
    6.0      2      2      8
    7.0      2      2      9
    9.0      2      2     10
   10.0      2      3      1
    8.0      2      3      2
   12.0      2      3      3
    3.0      2      3      4
    7.0      2      3      5
   15.0      2      3      6
    4.0      2      3      7
    9.0      2      3      8
    6.0      2      3      9
    1.0      2      3     10
""")
kidney_table.seek(0)
kidney_table = pd.read_table(kidney_table, sep="\s+")


@pytest.mark.not_vetted
class CheckAnovaMixin(object):

    @classmethod
    def setup_class(cls):
        cls.data = kidney_table.drop([0, 1, 2])
        cls.initialize()

    def test_combined(self):
        res = self.res
        wa = res.wald_test_terms(skip_single=False,
                                 combine_terms=['Duration', 'Weight'])
        eye = np.eye(len(res.params))
        c_const = eye[0]
        c_w = eye[[2, 3]]
        c_d = eye[1]
        c_dw = eye[[4, 5]]
        c_weight = eye[2:6]
        c_duration = eye[[1, 4, 5]]

        compare_waldres(res, wa,
                        [c_const, c_d, c_w, c_dw, c_duration, c_weight])

    def test_categories(self):
        # test only multicolumn terms
        res = self.res
        wa = res.wald_test_terms(skip_single=True)
        eye = np.eye(len(res.params))
        c_w = eye[[2, 3]]
        c_dw = eye[[4, 5]]

        compare_waldres(res, wa, [c_w, c_dw])


@pytest.mark.not_vetted
class TestWaldAnovaOLS(CheckAnovaMixin):
    model_cls = sm.OLS
    mod_kwargs = {}
    fit_kwargs = {"use_t": False}

    @classmethod
    def initialize(cls):
        formula = "np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)"
        mod = cls.model_cls.from_formula(formula, cls.data, **cls.mod_kwargs)
        cls.res = mod.fit(**cls.fit_kwargs)

    def test_noformula(self):
        endog = self.res.model.endog
        exog = self.res.model.data.orig_exog
        exog = pd.DataFrame(exog)

        res = sm.OLS(endog, exog).fit()
        wa = res.wald_test_terms(skip_single=True,
                                 combine_terms=['Duration', 'Weight'])
        eye = np.eye(len(res.params))
        c_weight = eye[2:6]
        c_duration = eye[[1, 4, 5]]

        compare_waldres(res, wa, [c_duration, c_weight])


@pytest.mark.not_vetted
class TestWaldAnovaOLSF(CheckAnovaMixin):
    model_cls = sm.OLS
    mod_kwargs = {}
    fit_kwargs = {}  # default use_t = True

    @classmethod
    def initialize(cls):
        formula = "np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)"
        mod = cls.model_cls.from_formula(formula, cls.data, **cls.mod_kwargs)
        cls.res = mod.fit(**cls.fit_kwargs)

    def test_predict_missing(self):
        ex = self.data[:5].copy()
        ex.iloc[0, 1] = np.nan
        predicted1 = self.res.predict(ex)
        predicted2 = self.res.predict(ex[1:])

        tm.assert_index_equal(predicted1.index, ex.index)
        tm.assert_series_equal(predicted1[1:], predicted2)
        assert np.isnan(predicted1.values[0])


@pytest.mark.not_vetted
class TestWaldAnovaGLM(CheckAnovaMixin):
    model_cls = sm.GLM
    mod_kwargs = {}
    fit_kwargs = {"use_t": False}

    @classmethod
    def initialize(cls):
        formula = "np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)"
        mod = cls.model_cls.from_formula(formula, cls.data, **cls.mod_kwargs)
        cls.res = mod.fit(**cls.fit_kwargs)


@pytest.mark.not_vetted
class TestWaldAnovaPoisson(CheckAnovaMixin):
    model_cls = sm.Poisson
    mod_kwargs = {}
    fit_kwargs = {"cov_type": "HC0"}

    @classmethod
    def initialize(cls):
        formula = "Days ~ C(Duration, Sum)*C(Weight, Sum)"
        mod = cls.model_cls.from_formula(formula, cls.data, **cls.mod_kwargs)
        cls.res = mod.fit(**cls.fit_kwargs)


@pytest.mark.not_vetted
class TestWaldAnovaNegBin(CheckAnovaMixin):
    model_cls = sm.NegativeBinomial
    mod_kwargs = {"loglike_method": "nb2"}
    fit_kwargs = {}

    @classmethod
    def initialize(cls):
        formula = "Days ~ C(Duration, Sum)*C(Weight, Sum)"
        mod = cls.model_cls.from_formula(formula, cls.data, **cls.mod_kwargs)
        cls.res = mod.fit(**cls.fit_kwargs)


@pytest.mark.not_vetted
class TestWaldAnovaNegBin1(CheckAnovaMixin):
    model_cls = sm.NegativeBinomial
    mod_kwargs = {"loglike_method": "nb1"}
    fit_kwargs = {"cov_type": "HC0"}

    @classmethod
    def initialize(cls):
        formula = "Days ~ C(Duration, Sum)*C(Weight, Sum)"
        mod = cls.model_cls.from_formula(formula, cls.data, **cls.mod_kwargs)
        cls.res = mod.fit(**cls.fit_kwargs)


# ----------------------------------------------------------------------

def compare_waldres(res, wa, constrasts):
    for i, c in enumerate(constrasts):
        wt = res.wald_test(c)
        assert_allclose(wa.table.values[i, 0], wt.statistic)
        assert_allclose(wa.table.values[i, 1], wt.pvalue)
        df = c.shape[0] if c.ndim == 2 else 1
        assert wa.table.values[i, 2] == df
        # attributes
        assert_allclose(wa.statistic[i], wt.statistic)
        assert_allclose(wa.pvalues[i], wt.pvalue)
        assert wa.df_constraints[i] == df
        if res.use_t:
            assert wa.df_denom[i] == res.df_resid

    col_names = wa.col_names
    if res.use_t:
        assert wa.distribution == 'F'
        assert col_names[0] == 'F'
        assert col_names[1] == 'P>F'
    else:
        assert wa.distribution == 'chi2'
        assert col_names[0] == 'chi2'
        assert col_names[1] == 'P>chi2'

    '''
    # SMOKETEST
    wa.summary_frame()
    '''
