# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:22:29 2014

Author: Josef Perktold
License: BSD-3
"""
from __future__ import division
import warnings

import numpy as np
from numpy.testing import assert_allclose

import pandas as pd
import patsy
import pytest

from sm2.discrete.discrete_model import Poisson
from sm2.base._constraints import fit_constrained, fit_constrained_wrap

from sm2.tools.tools import add_constant
from sm2 import datasets
from sm2.api import GLM, families

spector_data = datasets.spector.load(as_pandas=False)
spector_data.exog = add_constant(spector_data.exog, prepend=False)

from .results import results_poisson_constrained as results
from .results import results_glm_logit_constrained as reslogit

DEBUG = False

# TODO: This used to be here in string form and loaded with pd.read_csv.
#       Is there an original source/reference for it?
data = pd.DataFrame([[1, 1, 32, 52407],
                     [2, 1, 104, 43248],
                     [3, 1, 206, 28612],
                     [4, 1, 186, 12663],
                     [5, 1, 102, 5317],
                     [1, 0, 2, 18790],
                     [2, 0, 12, 10673],
                     [3, 0, 28, 5710],
                     [4, 0, 28, 2585],
                     [5, 0, 31, 1462]],
                    columns=['agecat', 'smokes', 'deaths', 'pyears'])
data = data.astype(int)  # explicit cast is needed for appveyor patsy tests
data['logpyears'] = np.log(data['pyears'])


class CheckPoissonConstrainedMixin(object):

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1[0], res2.params[self.idx], rtol=1e-6)
        # see below Stata has nan, we have zero
        bse1 = np.sqrt(np.diag(res1[1]))
        mask = (bse1 == 0) & np.isnan(res2.bse[self.idx])
        assert_allclose(bse1[~mask],
                        res2.bse[self.idx][~mask],
                        rtol=1e-6)

    # TODO: Split this into reasonably-scoped tests
    def test_basic_method(self):
        if not hasattr(self, 'res1m'):
            raise pytest.skip("not available yet")
        res1 = (self.res1m if not hasattr(self.res1m, '_results')
                else self.res1m._results)
        res2 = self.res2
        assert_allclose(res1.params,
                        res2.params[self.idx],
                        rtol=1e-6)

        # when a parameter is fixed, the Stata has bse=nan, we have bse=0
        mask = (res1.bse == 0) & np.isnan(res2.bse[self.idx])
        assert_allclose(res1.bse[~mask],
                        res2.bse[self.idx][~mask],
                        rtol=1e-6)

        tvalues = res2.params_table[self.idx, 2]
        # when a parameter is fixed, the Stata has tvalue=nan,
        # we have tvalue=inf
        mask = np.isinf(res1.tvalues) & np.isnan(tvalues)
        assert_allclose(res1.tvalues[~mask],
                        tvalues[~mask],
                        rtol=1e-6)
        pvalues = res2.params_table[self.idx, 3]
        # note most pvalues are very small
        # examples so far agree at 8 or more decimal, but rtol is stricter
        # see above
        mask = (res1.pvalues == 0) & np.isnan(pvalues)
        assert_allclose(res1.pvalues[~mask],
                        pvalues[~mask],
                        rtol=5e-5)

        ci_low = res2.params_table[self.idx, 4]
        ci_upp = res2.params_table[self.idx, 5]
        ci = np.column_stack((ci_low, ci_upp))
        # note most pvalues are very small
        # examples so far agree at 8 or more decimal, but rtol is stricter
        # see above: nan versus value
        assert_allclose(res1.conf_int()[~np.isnan(ci)],
                        ci[~np.isnan(ci)],
                        rtol=5e-5)

        # other
        assert_allclose(res1.llf, res2.ll, rtol=1e-6)
        assert res1.df_model == res2.df_m
        # Stata doesn't have df_resid
        df_r = res2.N - res2.df_m - 1
        assert res1.df_resid == df_r

    # TODO: test needs informative name
    def test_other(self):
        # some results may not be valid or available for all models
        if not hasattr(self, 'res1m'):
            raise pytest.skip("not available yet")

        res1 = self.res1m
        res2 = self.res2

        if hasattr(res2, 'll_0'):
            assert_allclose(res1.llnull, res2.ll_0, rtol=1e-6)
        else:
            if DEBUG:
                warnings.warn('test: ll_0 not available, llnull=%6.4F' %
                              res1.llnull)


class TestPoissonConstrained1a(CheckPoissonConstrainedMixin):
    res2 = results.results_noexposure_constraint
    idx = [7, 3, 4, 5, 6, 0, 1]  # 2 is dropped baseline for categorical
    model_cls = Poisson

    @classmethod
    def setup_class(cls):
        # example without offset
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = cls.model_cls.from_formula(formula, data=data)
        # get start_params, example fails to converge on one py TravisCI
        k_vars = len(mod.exog_names)
        start_params = np.zeros(k_vars)
        start_params[0] = np.log(mod.endog.mean())
        # if we need it, this is desired params
        # p = np.array([-3.93478643, 1.37276214, 2.33077032, 2.71338891,
        #               2.71338891, 0.57966535, 0.97254074])

        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants,
                                   start_params=start_params,
                                   fit_kwds={'method': 'bfgs',
                                             'disp': 0})
        # TODO: Newton fails

        # test method of Poisson, not monkey patched
        cls.res1m = mod.fit_constrained(constr, start_params=start_params,
                                        method='bfgs', disp=0)

    @pytest.mark.smoke
    def test_summary(self):
        # trailing text in summary, assumes it's the first extra string
        # NOTE: see comment about convergence in llnull for self.res1m
        summ = self.res1m.summary()
        assert 'linear equality constraints' in summ.extra_txt

    @pytest.mark.skip(reason="summary2 not ported from upstream")
    @pytest.mark.smoke
    def test_summary2(self):
        # trailing text in summary, assumes it's the first extra string
        # NOTE: see comment about convergence in llnull for self.res1m
        summ = self.res1m.summary2()
        assert 'linear equality constraints' in summ.extra_txt[0]


class TestPoissonConstrained1b(CheckPoissonConstrainedMixin):
    res2 = results.results_exposure_constraint
    model_cls = Poisson

    @classmethod
    def setup_class(cls):
        cls.idx = [6, 2, 3, 4, 5, 0]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = cls.model_cls.from_formula(formula, data=data,
                                         #offset=np.log(data['pyears'].values)
                                         exposure=data['pyears'].values)
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants,
                                   fit_kwds={'method': 'newton',
                                             'disp': 0})
        cls.constraints = lc
        # TODO: bfgs fails
        # test method of Poisson, not monkey patched
        cls.res1m = mod.fit_constrained(constr, method='newton',
                                        disp=0)


class TestPoissonConstrained1c(CheckPoissonConstrainedMixin):
    res2 = results.results_exposure_constraint
    model_cls = Poisson

    @classmethod
    def setup_class(cls):
        cls.idx = [6, 2, 3, 4, 5, 0]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = cls.model_cls.from_formula(formula, data=data,
                                         offset=np.log(data['pyears'].values))
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants,
                                   fit_kwds={'method': 'newton',
                                             'disp': 0})
        cls.constraints = lc
        # TODO: bfgs fails

        # test method of Poisson, not monkey patched
        cls.res1m = mod.fit_constrained(constr, method='newton', disp=0)


class TestPoissonNoConstrained(CheckPoissonConstrainedMixin):
    res2 = results.results_exposure_noconstraint
    idx = [6, 2, 3, 4, 5, 0]  # 1 is dropped baseline for categorical
    model_cls = Poisson

    @classmethod
    def setup_class(cls):
        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = cls.model_cls.from_formula(formula, data=data,
                                         #exposure=data['pyears'].values)
                                         offset=np.log(data['pyears'].values))
        res1 = mod.fit(disp=0)._results
        # res1 is duplicate check, so we can follow the same pattern
        cls.res1 = (res1.params, res1.cov_params())
        cls.res1m = res1


class TestPoissonConstrained2a(CheckPoissonConstrainedMixin):
    res2 = results.results_noexposure_constraint2
    idx = [7, 3, 4, 5, 6, 0, 1]  # 2 is dropped baseline for categorical
    model_cls = Poisson

    @classmethod
    def setup_class(cls):
        # example without offset
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = cls.model_cls.from_formula(formula, data=data)

        # get start_params, example fails to converge on one py TravisCI
        k_vars = len(mod.exog_names)
        start_params = np.zeros(k_vars)
        start_params[0] = np.log(mod.endog.mean())
        # if we need it, this is desired params
        # p = np.array([-9.43762015, 1.52762442, 2.74155711, 3.58730007,
        #               4.08730007, 1.15987869, 0.12111539])

        constr = 'C(agecat)[T.5] - C(agecat)[T.4] = 0.5'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants,
                                   start_params=start_params,
                                   fit_kwds={'method': 'bfgs', 'disp': 0})
        # TODO: Newton fails

        # test method of Poisson, not monkey patched
        cls.res1m = mod.fit_constrained(constr, start_params=start_params,
                                        method='bfgs', disp=0)


class TestPoissonConstrained2b(CheckPoissonConstrainedMixin):
    res2 = results.results_exposure_constraint2
    model_cls = Poisson

    @classmethod
    def setup_class(cls):
        cls.idx = [6, 2, 3, 4, 5, 0]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = cls.model_cls.from_formula(formula, data=data,
                                         #offset=np.log(data['pyears'].values),
                                         exposure=data['pyears'].values)
        constr = 'C(agecat)[T.5] - C(agecat)[T.4] = 0.5'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants,
                                   fit_kwds={'method': 'newton',
                                             'disp': 0})
        cls.constraints = lc
        # TODO: bfgs fails to converge. overflow somewhere?

        # test method of Poisson, not monkey patched
        cls.res1m = mod.fit_constrained(constr, method='bfgs', disp=0,
                                        start_params=cls.res1[0])


class TestPoissonConstrained2c(CheckPoissonConstrainedMixin):
    res2 = results.results_exposure_constraint2
    model_cls = Poisson

    @classmethod
    def setup_class(cls):
        cls.idx = [6, 2, 3, 4, 5, 0]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = cls.model_cls.from_formula(formula, data=data,
                                         offset=np.log(data['pyears'].values))

        constr = 'C(agecat)[T.5] - C(agecat)[T.4] = 0.5'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants,
                                   fit_kwds={'method': 'newton',
                                             'disp': 0})
        cls.constraints = lc
        # TODO: bfgs fails

        # test method of Poisson, not monkey patched
        cls.res1m = mod.fit_constrained(constr,
                                        method='bfgs', disp=0,
                                        start_params=cls.res1[0])


@pytest.mark.not_vetted
class TestGLMPoissonConstrained1a(CheckPoissonConstrainedMixin):
    res2 = results.results_noexposure_constraint
    idx = [7, 3, 4, 5, 6, 0, 1]  # 2 is dropped baseline for categorical
    model_cls = GLM

    @classmethod
    def setup_class(cls):
        # example without offset
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = cls.model_cls.from_formula(formula, data=data,
                                         family=families.Poisson())

        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants,
                                   fit_kwds={'atol': 1e-10})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, atol=1e-10)


@pytest.mark.not_vetted
class TestGLMPoissonConstrained1b(CheckPoissonConstrainedMixin):
    res2 = results.results_exposure_constraint
    idx = [6, 2, 3, 4, 5, 0]  # 2 is dropped baseline for categorical
    model_cls = GLM

    @classmethod
    def setup_class(cls):
        # example with offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = cls.model_cls.from_formula(formula, data=data,
                                         family=families.Poisson(),
                                         offset=np.log(data['pyears'].values))

        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)

        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants,
                                   fit_kwds={'atol': 1e-10})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, atol=1e-10)._results

    def test_compare_glm_poisson(self):
        res1 = self.res1m
        res2 = self.res2

        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data,
                                   #offset=np.log(data['pyears'].values),
                                   exposure=data['pyears'].values)

        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        res2 = mod.fit_constrained(constr, start_params=self.res1m.params,
                                   method='newton', warn_convergence=False,
                                   disp=0)

        # we get high precision because we use the params as start_params

        # basic, just as check that we have the same model
        assert_allclose(res1.params, res2.params, rtol=1e-12)
        assert_allclose(res1.bse, res2.bse, rtol=1e-11)

        # check predict, fitted, ...

        predicted = res1.predict()
        assert_allclose(predicted, res2.predict(), rtol=1e-10)
        assert_allclose(res1.mu, predicted, rtol=1e-10)
        assert_allclose(res1.fittedvalues, predicted, rtol=1e-10)
        assert_allclose(res2.predict(linear=True), res2.predict(linear=True),
                        rtol=1e-10)


@pytest.mark.not_vetted
class CheckGLMConstrainedMixin(CheckPoissonConstrainedMixin):
    # add tests for some GLM specific attributes

    def test_glm(self):
        res2 = self.res2  # reference results
        res1 = self.res1m

        #assert_allclose(res1.aic, res2.aic, rtol=1e-10)  # far away
        # Stata aic in ereturn and in estat ic are very different
        # we have the same as estat ic
        # see issue GH#1733
        assert_allclose(res1.aic, res2.infocrit[4], rtol=1e-10)

        assert_allclose(res1.bic, res2.bic, rtol=1e-10)
        # bic is deviance based
        # TODO: Does this comment refer to the line above, or the now-deleted
        # line below that compares res1.bic to res2.infocrit[5]?

        assert_allclose(res1.deviance, res2.deviance, rtol=1e-10)
        # TODO: which chi2 are these
        #assert_allclose(res1.pearson_chi2, res2.chi2, rtol=1e-10)


@pytest.mark.not_vetted
class TestGLMLogitConstrained1(CheckGLMConstrainedMixin):
    idx = slice(None)
    # params sequence same as Stata, but Stata reports param = nan
    # and we have param = value = 0
    model_cls = GLM

    @classmethod
    def setup_class(cls):
        cls.res2 = reslogit.results_constraint1
        mod1 = cls.model_cls(spector_data.endog, spector_data.exog,
                             family=families.Binomial())

        constr = 'x1 = 2.8'
        cls.res1m = mod1.fit_constrained(constr)

        R, q = cls.res1m.constraints.coefs, cls.res1m.constraints.constants
        cls.res1 = fit_constrained(mod1, R, q)


@pytest.mark.not_vetted
class TestGLMLogitConstrained2(CheckGLMConstrainedMixin):
    idx = slice(None)  # params sequence same as Stata
    model_cls = GLM
    res2 = reslogit.results_constraint2

    @classmethod
    def setup_class(cls):
        mod1 = cls.model_cls(spector_data.endog, spector_data.exog,
                             family=families.Binomial())

        constr = 'x1 - x3 = 0'
        cls.res1m = mod1.fit_constrained(constr, atol=1e-10)

        R, q = cls.res1m.constraints.coefs, cls.res1m.constraints.constants
        cls.res1 = fit_constrained(mod1, R, q, fit_kwds={'atol': 1e-10})
        cls.constraints_rq = (R, q)

    def test_predict(self):
        # results only available for this case
        res2 = self.res2  # reference results
        res1 = self.res1m

        predicted = res1.predict()
        assert_allclose(predicted, res2.predict_mu, atol=1e-7)
        assert_allclose(res1.mu, predicted, rtol=1e-10)
        assert_allclose(res1.fittedvalues, predicted, rtol=1e-10)

    @pytest.mark.smoke
    def test_summary(self):
        # trailing text in summary, assumes it's the first extra string
        summ = self.res1m.summary()
        assert 'linear equality constraints' in summ.extra_txt

    @pytest.mark.skip(reason="summary2 not ported from upstream")
    @pytest.mark.smoke
    def test_summary2(self):
        summ = self.res1m.summary2()
        assert 'linear equality constraints' in summ.extra_txt[0]

    def test_fit_constrained_wrap(self):
        # minimal test
        res2 = self.res2  # reference results

        res_wrap = fit_constrained_wrap(self.res1m.model, self.constraints_rq)
        assert_allclose(res_wrap.params, res2.params, rtol=1e-6)
        assert_allclose(res_wrap.params, res2.params, rtol=1e-6)


@pytest.mark.not_vetted
class TestGLMLogitConstrained2HC(CheckGLMConstrainedMixin):
    idx = slice(None)  # params sequence same as Stata
    model_cls = GLM
    res2 = reslogit.results_constraint2_robust

    @classmethod
    def setup_class(cls):
        mod1 = cls.model_cls(spector_data.endog, spector_data.exog,
                             family=families.Binomial())

        # not used to match Stata for HC
        # nobs, k_params = mod1.exog.shape
        # k_params -= 1   # one constraint
        cov_type = 'HC0'
        cov_kwds = {'scaling_factor': 32 / 31.}
        # looks like nobs / (nobs - 1) and not (nobs - 1.) / (nobs - k_params)}
        constr = 'x1 - x3 = 0'
        cls.res1m = mod1.fit_constrained(constr, cov_type=cov_type,
                                         cov_kwds=cov_kwds, atol=1e-10)

        R, q = cls.res1m.constraints.coefs, cls.res1m.constraints.constants
        cls.res1 = fit_constrained(mod1, R, q, fit_kwds={'atol': 1e-10,
                                                         'cov_type': cov_type,
                                                         'cov_kwds': cov_kwds})
        cls.constraints_rq = (R, q)
