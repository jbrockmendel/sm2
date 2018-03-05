# -*- coding: utf-8 -*-
"""

Created on Mon Dec 09 21:29:20 2013

Author: Josef Perktold
"""
import os

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pandas as pd

from sm2.tools.tools import add_constant
from sm2.base.covtype import get_robustcov_results
import sm2.stats.sandwich_covariance as sw

import sm2.discrete.discrete_model as smd
from sm2.regression.linear_model import OLS

import sm2.tools._testing as smt

# from statsmodels.genmod.generalized_linear_model import GLM
# from statsmodels.genmod import families
# from statsmodels.genmod.families import links


# get data and results as module global for now, TODO: move to class
from .results import results_count_robust_cluster as results_st

cur_dir = os.path.dirname(os.path.abspath(__file__))

filepath = os.path.join(cur_dir, "results", "ships.csv")
data_raw = pd.read_csv(filepath, index_col=False)
data = data_raw.dropna()

#mod = smd.Poisson.from_formula('accident ~ yr_con + op_75_79', data=dat)
# Don't use formula for tests against Stata because intercept needs to be last
endog = data['accident']
exog_data = data['yr_con op_75_79'.split()]
exog = add_constant(exog_data, prepend=False)
group = np.asarray(data['ship'], int)
exposure = np.asarray(data['service'])


# TODO get the test methods from regression/tests

@pytest.mark.not_vetted
class CheckCountRobustMixin(object):
    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        if len(res1.params) == (len(res2.params) - 1):
            # Stata includes lnalpha in table for NegativeBinomial
            mask = np.ones(len(res2.params), np.bool_)
            mask[-2] = False
            res2_params = res2.params[mask]
            res2_bse = res2.bse[mask]
        else:
            res2_params = res2.params
            res2_bse = res2.bse

        assert_allclose(res1._results.params,
                        res2_params,
                        1e-4)

        assert_allclose(self.bse_rob / self.corr_fact,
                        res2_bse,
                        6e-5)

    @classmethod
    def get_robust_clu(cls):
        res1 = cls.res1
        cov_clu = sw.cov_cluster(res1, group)
        cls.bse_rob = sw.se_cov(cov_clu)

        nobs, k_vars = res1.model.exog.shape
        k_params = len(res1.params)
        corr_fact = (nobs - 1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(corr_fact)

    def test_oth(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1._results.llf, res2.ll, 1e-4)
        assert_allclose(res1._results.llnull, res2.ll_0, 1e-4)

    def test_ttest(self):
        smt.check_ttest_tvalues(self.res1)

    def test_waldtest(self):
        smt.check_ftest_pvalues(self.res1)


@pytest.mark.not_vetted
class TestPoissonClu(CheckCountRobustMixin):
    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu

        mod = smd.Poisson(endog, exog)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


@pytest.mark.not_vetted
class TestPoissonCluGeneric(CheckCountRobustMixin):
    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu

        mod = smd.Poisson(endog, exog)
        cls.res1 = mod.fit(disp=False)

        debug = False
        if debug:
            # for debugging
            cls.bse_nonrobust = cls.res1.bse.copy()
            cls.res1 = mod.fit(disp=False)
            cls.get_robust_clu()
            cls.res3 = cls.res1
            cls.bse_rob3 = cls.bse_rob.copy()
            cls.res1 = mod.fit(disp=False)

        get_robustcov_results(cls.res1._results, 'cluster',
                              groups=group,
                              use_correction=True,
                              df_correction=True,  #TODO has no effect
                              use_t=False, #True,
                              use_self=True)
        cls.bse_rob = cls.res1.bse

        nobs, k_vars = cls.res1.model.exog.shape
        k_params = len(cls.res1.params)
        corr_fact = (nobs - 1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(corr_fact)


@pytest.mark.not_vetted
class TestPoissonHC1Generic(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_hc1

        mod = smd.Poisson(endog, exog)
        cls.res1 = mod.fit(disp=False)

        #res_hc0_ = cls.res1.get_robustcov_results('HC1')
        get_robustcov_results(cls.res1._results, 'HC1', use_self=True)
        cls.bse_rob = cls.res1.bse
        nobs, k_vars = mod.exog.shape
        corr_fact = (nobs) / float(nobs - 1.)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(1. / corr_fact)


# TODO: refactor xxxFit to full testing results
@pytest.mark.not_vetted
class TestPoissonCluFit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu

        mod = smd.Poisson(endog, exog)

        # scaling of cov_params_default to match Stata
        # TODO should the default be changed?
        nobs, k_params = mod.exog.shape
        sc_fact = (nobs - 1.) / float(nobs - k_params)

        cls.res1 = mod.fit(disp=False, cov_type='cluster',
                           cov_kwds=dict(groups=group,
                                         use_correction=True,
                                         scaling_factor=1. / sc_fact,
                                         df_correction=True),  #TODO has no effect
                           use_t=False, #True,
                           )

        # The model results, t_test, ... should also work without
        # normalized_cov_params, see #2209
        # Note: we cannot set on the wrapper res1, we need res1._results
        cls.res1._results.normalized_cov_params = None

        cls.bse_rob = cls.res1.bse

        # backwards compatibility with inherited test methods
        cls.corr_fact = 1

    def test_basic_inference(self):
        res1 = self.res1
        res2 = self.res2
        rtol = 1e-7
        assert_allclose(res1.params, res2.params, rtol=1e-8)
        assert_allclose(res1.bse, res2.bse, rtol=rtol)
        assert_allclose(res1.tvalues, res2.tvalues, rtol=rtol, atol=1e-8)
        assert_allclose(res1.pvalues, res2.pvalues, rtol=rtol, atol=1e-20)
        ci = res2.params_table[:, 4:6]
        assert_allclose(res1.conf_int(), ci, rtol=5e-7, atol=1e-20)


@pytest.mark.not_vetted
class TestPoissonHC1Fit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_hc1

        mod = smd.Poisson(endog, exog)
        cls.res1 = mod.fit(disp=False, cov_type='HC1')

        cls.bse_rob = cls.res1.bse
        nobs, k_vars = mod.exog.shape
        corr_fact = (nobs) / float(nobs - 1.)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(1./corr_fact)


@pytest.mark.not_vetted
class TestPoissonHC1FitExposure(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_exposure_hc1

        mod = smd.Poisson(endog, exog, exposure=exposure)
        cls.res1 = mod.fit(disp=False, cov_type='HC1')

        cls.bse_rob = cls.res1.bse
        nobs, k_vars = mod.exog.shape
        corr_fact = (nobs) / float(nobs - 1.)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(1. / corr_fact)


@pytest.mark.not_vetted
class TestPoissonCluExposure(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_exposure_clu #nonrobust

        mod = smd.Poisson(endog, exog, exposure=exposure)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


@pytest.mark.not_vetted
class TestPoissonCluExposureGeneric(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_exposure_clu #nonrobust

        mod = smd.Poisson(endog, exog, exposure=exposure)
        cls.res1 = mod.fit(disp=False)

        get_robustcov_results(cls.res1._results, 'cluster',
                              groups=group,
                              use_correction=True,
                              df_correction=True,  #TODO has no effect
                              use_t=False, #True,
                              use_self=True)
        cls.bse_rob = cls.res1.bse

        nobs, k_vars = cls.res1.model.exog.shape
        k_params = len(cls.res1.params)
        corr_fact = (nobs - 1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(corr_fact)


@pytest.mark.not_vetted
class TestGLMPoissonClu(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        cls.res2 = results_st.results_poisson_clu

        mod = smd.Poisson(endog, exog)
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = mod.fit()
        cls.get_robust_clu()


@pytest.mark.not_vetted
class TestGLMPoissonCluGeneric(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        cls.res2 = results_st.results_poisson_clu

        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = mod.fit()

        get_robustcov_results(cls.res1._results, 'cluster',
                              groups=group,
                              use_correction=True,
                              df_correction=True,  #TODO has no effect
                              use_t=False, #True,
                              use_self=True)
        cls.bse_rob = cls.res1.bse

        nobs, k_vars = cls.res1.model.exog.shape
        k_params = len(cls.res1.params)
        corr_fact = (nobs - 1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(corr_fact)


@pytest.mark.not_vetted
class TestGLMPoissonHC1Generic(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        cls.res2 = results_st.results_poisson_hc1
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = mod.fit()

        #res_hc0_ = cls.res1.get_robustcov_results('HC1')
        get_robustcov_results(cls.res1._results, 'HC1', use_self=True)
        cls.bse_rob = cls.res1.bse
        nobs, k_vars = mod.exog.shape
        corr_fact = (nobs) / float(nobs - 1.)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(1. / corr_fact)


# TODO: refactor xxxFit to full testing results
@pytest.mark.not_vetted
class TestGLMPoissonCluFit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        cls.res2 = results_st.results_poisson_clu
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = mod.fit(cov_type='cluster',
                           cov_kwds=dict(groups=group,
                                         use_correction=True,
                                         df_correction=True),  #TODO has no effect
                           use_t=False, #True,
                           )

        # The model results, t_test, ... should also work without
        # normalized_cov_params, see #2209
        # Note: we cannot set on the wrapper res1, we need res1._results
        cls.res1._results.normalized_cov_params = None

        cls.bse_rob = cls.res1.bse

        nobs, k_vars = mod.exog.shape
        k_params = len(cls.res1.params)
        corr_fact = (nobs - 1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(corr_fact)


@pytest.mark.not_vetted
class TestGLMPoissonHC1Fit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        cls.res2 = results_st.results_poisson_hc1
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = mod.fit(cov_type='HC1')

        cls.bse_rob = cls.res1.bse
        nobs, k_vars = mod.exog.shape
        corr_fact = (nobs) / float(nobs - 1.)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(1. / corr_fact)


@pytest.mark.not_vetted
class TestNegbinClu(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_clu
        mod = smd.NegativeBinomial(endog, exog)
        cls.res1 = mod.fit(disp=False, gtol=1e-7)
        cls.get_robust_clu()


@pytest.mark.not_vetted
class TestNegbinCluExposure(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_exposure_clu #nonrobust
        mod = smd.NegativeBinomial(endog, exog, exposure=exposure)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


#        mod_nbe = smd.NegativeBinomial(endog, exog, exposure=data['service'])
#        res_nbe = mod_nbe.fit()
#        mod_nb = smd.NegativeBinomial(endog, exog)
#        res_nb = mod_nb.fit()
#
#        cov_clu_nb = sw.cov_cluster(res_nb, group)
#        k_params = k_vars + 1
#        print sw.se_cov(cov_clu_nb / ((nobs-1.) / float(nobs - k_params)))
#
#        wt = res_nb.wald_test(np.eye(len(res_nb.params))[1:3], cov_p=cov_clu_nb/((nobs-1.) / float(nobs - k_params)))
#        print wt
#
#        print dir(results_st)


@pytest.mark.not_vetted
class TestNegbinCluGeneric(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_clu
        mod = smd.NegativeBinomial(endog, exog)
        cls.res1 = mod.fit(disp=False, gtol=1e-7)

        get_robustcov_results(cls.res1._results, 'cluster',
                                                  groups=group,
                                                  use_correction=True,
                                                  df_correction=True,  #TODO has no effect
                                                  use_t=False, #True,
                                                  use_self=True)
        cls.bse_rob = cls.res1.bse

        nobs, k_vars = mod.exog.shape
        k_params = len(cls.res1.params)
        corr_fact = (nobs - 1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(corr_fact)


@pytest.mark.not_vetted
class TestNegbinCluFit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_clu
        mod = smd.NegativeBinomial(endog, exog)
        cls.res1 = res1 = mod.fit(disp=False, cov_type='cluster',
                                  cov_kwds=dict(groups=group,
                                                use_correction=True,
                                                df_correction=True),  #TODO has no effect
                                  use_t=False, #True,
                                  gtol=1e-7)
        cls.bse_rob = cls.res1.bse

        nobs, k_vars = mod.exog.shape
        k_params = len(cls.res1.params)
        corr_fact = (nobs - 1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(corr_fact)


@pytest.mark.not_vetted
class TestNegbinCluExposureFit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_exposure_clu #nonrobust
        mod = smd.NegativeBinomial(endog, exog, exposure=exposure)
        cls.res1 = res1 = mod.fit(disp=False, cov_type='cluster',
                                  cov_kwds=dict(groups=group,
                                                use_correction=True,
                                                df_correction=True),  #TODO has no effect
                                  use_t=False, #True,
                                  )
        cls.bse_rob = cls.res1.bse

        nobs, k_vars = mod.exog.shape
        k_params = len(cls.res1.params)
        corr_fact = (nobs-1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(corr_fact)


@pytest.mark.not_vetted
class CheckDiscreteGLM(object):
    # compare GLM with other models, no verified reference results

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        assert_equal(res1.cov_type, self.cov_type)
        assert_equal(res2.cov_type, self.cov_type)

        assert_allclose(res1.params, res2.params, rtol=1e-13)
        # bug TODO res1.scale missing ?  in Gaussian/OLS
        assert_allclose(res1.bse, res2.bse, rtol=1e-13)
#         if not self.cov_type == 'nonrobust':
#             assert_allclose(res1.bse * res1.scale, res2.bse, rtol=1e-13)
#         else:
#             assert_allclose(res1.bse, res2.bse, rtol=1e-13)


@pytest.mark.not_vetted
class TestGLMLogit(CheckDiscreteGLM):
    cov_type = 'cluster'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        endog_bin = (endog > endog.mean()).astype(int)

        mod1 = GLM(endog_bin, exog, family=families.Binomial())
        cls.res1 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))

        mod1 = smd.Logit(endog_bin, exog)
        cls.res2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))


@pytest.mark.not_vetted
@pytest.mark.xfail(reason="This test is mangled upstream.  Has comment: "
                          "invalid link. What's Probit as GLM?")
class TestGLMProbit(CheckDiscreteGLM):
    cov_type = 'cluster'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        endog_bin = (endog > endog.mean()).astype(int)

        mod1 = GLM(endog_bin, exog, family=families.Gaussian(link=links.CDFLink()))
        cls.res1 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))

        mod1 = smd.Probit(endog_bin, exog)
        cls.res2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))


@pytest.mark.not_vetted
class TestGLMGaussNonRobust(CheckDiscreteGLM):
    cov_type = 'nonrobust'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit()

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit()


@pytest.mark.not_vetted
class TestGLMGaussClu(CheckDiscreteGLM):
    cov_type = 'cluster'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='cluster', cov_kwds=dict(groups=group))


@pytest.mark.not_vetted
class TestGLMGaussHC(CheckDiscreteGLM):
    cov_type = 'HC0'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='HC0')

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='HC0')


@pytest.mark.not_vetted
class TestGLMGaussHAC(CheckDiscreteGLM):
    cov_type = 'HAC'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        kwds = {'maxlags': 2}
        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='HAC', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='HAC', cov_kwds=kwds)


@pytest.mark.not_vetted
class TestGLMGaussHACUniform(CheckDiscreteGLM):
    cov_type = 'HAC'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")

        kwds = {'kernel': sw.weights_uniform, 'maxlags': 2}
        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='HAC', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='HAC', cov_kwds=kwds)

        # for debugging
        cls.res3 = mod2.fit(cov_type='HAC', cov_kwds={'maxlags': 2})

    def test_cov_options(self):
        # check keyword `weights_func
        kwdsa = {'weights_func': sw.weights_uniform, 'maxlags': 2}
        res1a = self.res1.model.fit(cov_type='HAC', cov_kwds=kwdsa)
        res2a = self.res2.model.fit(cov_type='HAC', cov_kwds=kwdsa)
        assert_allclose(res1a.bse, self.res1.bse, rtol=1e-12)
        assert_allclose(res2a.bse, self.res2.bse, rtol=1e-12)

        # regression test for bse values
        bse = np.array([  2.82203924,   4.60199596,  11.01275064])
        assert_allclose(res1a.bse, bse, rtol=1e-6)

        assert_(res1a.cov_kwds['weights_func'] is sw.weights_uniform)

        kwdsb = {'kernel': sw.weights_bartlett, 'maxlags': 2}
        res1a = self.res1.model.fit(cov_type='HAC', cov_kwds=kwdsb)
        res2a = self.res2.model.fit(cov_type='HAC', cov_kwds=kwdsb)
        assert_allclose(res1a.bse, res2a.bse, rtol=1e-12)

        # regression test for bse values
        bse = np.array([  2.502264,  3.697807,  9.193303])
        assert_allclose(res1a.bse, bse, rtol=1e-6)


@pytest.mark.not_vetted
class TestGLMGaussHACPanel(CheckDiscreteGLM):
    cov_type = 'hac-panel'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        # time index is just made up to have a test case
        time = np.tile(np.arange(7), 5)[:-1]
        mod1 = GLM(endog.copy(), exog.copy(), family=families.Gaussian())
        kwds = dict(time=time,
                    maxlags=2,
                    kernel=sw.weights_uniform,
                    use_correction='hac',
                    df_correction=False)
        cls.res1 = mod1.fit(cov_type='hac-panel', cov_kwds=kwds)
        cls.res1b = mod1.fit(cov_type='nw-panel', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='hac-panel', cov_kwds=kwds)

    def test_kwd(self):
        # test corrected keyword name
        assert_allclose(self.res1b.bse, self.res1.bse, rtol=1e-12)


@pytest.mark.not_vetted
class TestGLMGaussHACPanelGroups(CheckDiscreteGLM):
    cov_type = 'hac-panel'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        # time index is just made up to have a test case
        groups = np.repeat(np.arange(5), 7)[:-1]
        mod1 = GLM(endog.copy(), exog.copy(), family=families.Gaussian())
        kwds = dict(groups=pd.Series(groups),  # check for #3606
                    maxlags=2,
                    kernel=sw.weights_uniform,
                    use_correction='hac',
                    df_correction=False)
        cls.res1 = mod1.fit(cov_type='hac-panel', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='hac-panel', cov_kwds=kwds)


@pytest.mark.not_vetted
class TestGLMGaussHACGroupsum(CheckDiscreteGLM):
    cov_type = 'hac-groupsum'

    @classmethod
    def setup_class(cls):
        raise pytest.skip("GLM not implemented")
        # time index is just made up to have a test case
        time = np.tile(np.arange(7), 5)[:-1]
        mod1 = GLM(endog, exog, family=families.Gaussian())
        kwds = dict(time=pd.Series(time),  # check for #3606
                    maxlags=2,
                    use_correction='hac',
                    df_correction=False)
        cls.res1 = mod1.fit(cov_type='hac-groupsum', cov_kwds=kwds)
        cls.res1b = mod1.fit(cov_type='nw-groupsum', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='hac-groupsum', cov_kwds=kwds)

    def test_kwd(self):
        # test corrected keyword name
        assert_allclose(self.res1b.bse, self.res1.bse, rtol=1e-12)
