# -*- coding: utf-8 -*-
"""Testing OLS robust covariance matrices against STATA

Created on Mon Oct 28 15:25:14 2013

Author: Josef Perktold
"""
import warnings

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_warns
import pandas as pd
from scipy import stats

from sm2.regression.linear_model import OLS, WLS
import sm2.stats.sandwich_covariance as sw
from sm2.tools.tools import add_constant
from sm2.datasets import macrodata, grunfeld
from sm2.tools.sm_exceptions import InvalidTestWarning

from .results import (results_macro_ols_robust,
                      results_grunfeld_ols_robust_cluster)
#test_hac_simple():


@pytest.mark.not_vetted
class CheckOLSRobust(object):
    # TODO: Split these into reasonably-scoped tests
    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        rtol = getattr(self, 'rtol', 1e-10)
        assert_allclose(res1.params, res2.params, rtol=rtol)
        assert_allclose(self.bse_robust, res2.bse, rtol=rtol)
        assert_allclose(self.cov_robust, res2.cov, rtol=rtol)

    def test_t_test(self):
        # Note: differences between small (t-distribution, ddof) and
        # large (normal) F statistic has no ddof correction in large,
        # but uses F distribution (?)
        rtol = getattr(self, 'rtol', 1e-10)

        res1 = self.res1
        res2 = self.res2

        mat = np.eye(len(res1.params))
        tt = res1.t_test(mat, cov_p=self.cov_robust)

        # has 'effect', 'pvalue', 'sd', 'tvalue'
        # TODO confint missing
        assert_allclose(tt.effect, res2.params, rtol=rtol)
        assert_allclose(tt.sd, res2.bse, rtol=rtol)
        assert_allclose(tt.tvalue, res2.tvalues, rtol=rtol)
        if self.small:
            assert_allclose(tt.pvalue,
                            res2.pvalues,
                            rtol=5 * rtol)
        else:
            pval = stats.norm.sf(np.abs(tt.tvalue)) * 2
            assert_allclose(pval,
                            res2.pvalues,
                            rtol=5 * rtol, atol=1e-25)
        # SMOKE
        tt.summary()
        tt.summary_frame()

    # TODO: Split these into reasonably-scoped tests
    def test_tests(self):
        # Note: differences between small (t-distribution, ddof) and
        # large (normal) F statistic has no ddof correction in large,
        # but uses F distribution (?)
        res1 = self.res1
        res2 = self.res2
        rtol = getattr(self, 'rtol', 1e-10)
        mat = np.eye(len(res1.params))

        ft = res1.f_test(mat[:-1], cov_p=self.cov_robust)
        if self.small:
            # 'df_denom', 'df_num', 'fvalue', 'pvalue'
            assert_allclose(ft.fvalue, res2.F, rtol=rtol)
            # f-pvalue is not directly available in Stata results,
            # but is in ivreg2
            if hasattr(res2, 'Fp'):
                assert_allclose(ft.pvalue, res2.Fp, rtol=rtol)
        else:
            if not getattr(self, 'skip_f', False):
                dof_corr = res1.df_resid * 1. / res1.nobs
                assert_allclose(ft.fvalue * dof_corr, res2.F, rtol=rtol)

        if hasattr(res2, 'df_r'):
            assert_equal(ft.df_num, res2.df_m)
            assert_equal(ft.df_denom, res2.df_r)
        else:
            # ivreg2
            assert_equal(ft.df_num, res2.Fdf1)
            assert_equal(ft.df_denom, res2.Fdf2)

        # SMOKE
        ft.summary()


@pytest.mark.not_vetted
class TestOLSRobust1(CheckOLSRobust):
    # compare with regress robust
    res2 = results_macro_ols_robust.results_hc0
    small = True

    def setup(self):
        res_ols = self.res1
        self.bse_robust = res_ols.HC1_se
        self.cov_robust = res_ols.cov_HC1

    @classmethod
    def setup_class(cls):
        d2 = macrodata.load().data
        g_gdp = 400 * np.diff(np.log(d2['realgdp']))
        g_inv = 400 * np.diff(np.log(d2['realinv']))
        exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1]], prepend=False)

        cls.res1 = OLS(g_inv, exogg).fit()


@pytest.mark.not_vetted
class TestOLSRobust2(TestOLSRobust1):
    # compare with ivreg robust small
    res2 = results_macro_ols_robust.results_ivhc0_small
    small = True

    def setup(self):
        res_ols = self.res1
        self.bse_robust = res_ols.HC1_se
        self.cov_robust = res_ols.cov_HC1


@pytest.mark.not_vetted
class TestOLSRobust3(TestOLSRobust1):
    # compare with ivreg robust   (not small)
    res2 = results_macro_ols_robust.results_ivhc0_large
    small = False

    def setup(self):
        res_ols = self.res1
        self.bse_robust = res_ols.HC0_se
        self.cov_robust = res_ols.cov_HC0


@pytest.mark.not_vetted
class TestOLSRobustHacSmall(TestOLSRobust1):
    # compare with ivreg robust small
    res2 = results_macro_ols_robust.results_ivhac4_small
    small = True

    def setup(self):
        res_ols = self.res1
        cov1 = sw.cov_hac_simple(res_ols, nlags=4, use_correction=True)
        se1 = sw.se_cov(cov1)
        self.bse_robust = se1
        self.cov_robust = cov1


@pytest.mark.not_vetted
class TestOLSRobustHacLarge(TestOLSRobust1):
    # compare with ivreg robust (not small)
    res2 = results_macro_ols_robust.results_ivhac4_large
    small = False

    def setup(self):
        res_ols = self.res1
        cov1 = sw.cov_hac_simple(res_ols, nlags=4, use_correction=False)
        se1 = sw.se_cov(cov1)
        self.bse_robust = se1
        self.cov_robust = cov1


@pytest.mark.not_vetted
class CheckOLSRobustNewMixin(object):
    # This uses the robust covariance as default covariance

    def test_compare(self):
        rtol = getattr(self, 'rtol', 1e-10)
        assert_allclose(self.cov_robust, self.cov_robust2, rtol=rtol)
        assert_allclose(self.bse_robust, self.bse_robust2, rtol=rtol)

    def test_fvalue(self):
        if not getattr(self, 'skip_f', False):
            rtol = getattr(self, 'rtol', 1e-10)
            assert_allclose(self.res1.fvalue, self.res2.F, rtol=rtol)
            if hasattr(self.res2, 'Fp'):
                # only available with ivreg2
                assert_allclose(self.res1.f_pvalue, self.res2.Fp, rtol=rtol)

    def test_confint(self):
        rtol = getattr(self, 'rtol', 1e-10)
        ci1 = self.res1.conf_int()
        ci2 = self.res2.params_table[:, 4:6]
        assert_allclose(ci1, ci2, rtol=rtol)

        # check critical value
        crit1 = np.diff(ci1, 1).ravel() / 2 / self.res1.bse
        crit2 = np.diff(ci1, 1).ravel() / 2 / self.res1.bse
        assert_allclose(crit1, crit2, rtol=12)

    def test_ttest(self):
        res1 = self.res1
        res2 = self.res2
        rtol = getattr(self, 'rtol', 1e-10)
        rtolh = getattr(self, 'rtol', 1e-12)

        mat = np.eye(len(res1.params))
        tt = res1.t_test(mat, cov_p=self.cov_robust)
        # has 'effect', 'pvalue', 'sd', 'tvalue'
        # TODO confint missing
        assert_allclose(tt.effect, res2.params, rtol=rtolh)
        assert_allclose(tt.sd, res2.bse, rtol=rtol)
        assert_allclose(tt.tvalue, res2.tvalues, rtol=rtolh)
        assert_allclose(tt.pvalue, res2.pvalues, rtol=5 * rtol)
        ci1 = tt.conf_int()
        ci2 = self.res2.params_table[:, 4:6]
        assert_allclose(ci1, ci2, rtol=rtol)

    def test_scale(self):
        res1 = self.res1
        res2 = self.res2
        rtol = 1e-5
        # Note we always use df_resid for scale
        # Stata uses nobs or df_resid for rmse, not always available in Stata
        #assert_allclose(res1.scale,
        #                 res2.rmse**2 * res2.N / (res2.N - res2.df_m - 1),
        #                 rtol=rtol)
        skip = False
        if hasattr(res2, 'rss'):
            scale = res2.rss / (res2.N - res2.df_m - 1)
        elif hasattr(res2, 'rmse'):
            scale = res2.rmse**2
        else:
            skip = True

        if isinstance(res1.model, WLS):
            skip = True
            # Stata uses different scaling and using unweighted resid for rmse

        if not skip:
            assert_allclose(res1.scale, scale, rtol=rtol)

        if res2.vcetype != 'Newey-West':
            # no rsquared in Stata
            r2 = res2.r2 if hasattr(res2, 'r2') else res2.r2c
            assert_allclose(res1.rsquared, r2,
                            rtol=rtol,
                            err_msg=str(skip))

        # consistency checks, not against Stata
        df_resid = res1.nobs - res1.df_model - 1
        assert res1.df_resid == df_resid

        # variance of resid_pearson is 1, with ddof, and loc=0
        psum = (res1.resid_pearson**2).sum()
        assert_allclose(psum, df_resid, rtol=1e-13)

    @pytest.mark.smoke
    def test_summary(self):
        self.res1.summary()


@pytest.mark.not_vetted
class TestOLSRobust2SmallNew(TestOLSRobust1, CheckOLSRobustNewMixin):
    # compare with ivreg robust small
    res2 = results_macro_ols_robust.results_ivhc0_small
    small = True

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HC1', use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        self.bse_robust2 = res_ols.HC1_se
        self.cov_robust2 = res_ols.cov_HC1

    def test_compare(self):
        # check that we get a warning using the nested compare methods
        res1 = self.res1
        endog = res1.model.endog
        exog = res1.model.exog[:, [0, 2]]  # drop one variable
        res_ols2 = OLS(endog, exog).fit()
        # results from Stata
        r_pval = .0307306938402991
        r_chi2 = 4.667944083588736
        r_df = 1
        assert_warns(InvalidTestWarning, res1.compare_lr_test, res_ols2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2, pval, df = res1.compare_lr_test(res_ols2)

        assert_allclose(chi2, r_chi2, rtol=1e-11)
        assert_allclose(pval, r_pval, rtol=1e-11)
        assert_equal(df, r_df)

        assert_warns(InvalidTestWarning, res1.compare_f_test, res_ols2)
        #fva, pval, df = res1.compare_f_test(res_ols2)


@pytest.mark.not_vetted
class TestOLSRobustHACSmallNew(TestOLSRobust1, CheckOLSRobustNewMixin):
    # compare with ivreg robust small
    res2 = results_macro_ols_robust.results_ivhac4_small
    small = True

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HAC', maxlags=4,
                                                  use_correction=True,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_hac_simple(res_ols, nlags=4, use_correction=True)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


@pytest.mark.not_vetted
class TestOLSRobust2LargeNew(TestOLSRobust1, CheckOLSRobustNewMixin):
    # compare with ivreg robust small
    res2 = results_macro_ols_robust.results_ivhc0_large
    small = False

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HC0')
        res_ols.use_t = False
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        self.bse_robust2 = res_ols.HC0_se
        self.cov_robust2 = res_ols.cov_HC0

    # TODO: skipping next two for now, not refactored yet for `large`
    def test_fvalue(self):
        pass

    def test_confint(self):
        pass


# -----------------------------------------------------------------
# cluster robust standard errors


@pytest.mark.not_vetted
class CheckOLSRobustCluster(CheckOLSRobust):
    # compare with regress robust

    @classmethod
    def setup_class(cls):
        dtapa = grunfeld.data.load_pandas()
        # Stata example/data seems to miss last firm
        dtapa_endog = dtapa.endog[:200]
        dtapa_exog = dtapa.exog[:200]
        exog = add_constant(dtapa_exog[['value', 'capital']], prepend=False)
        cls.res1 = OLS(dtapa_endog, exog).fit()

        firm_names, firm_id = np.unique(np.asarray(dtapa_exog[['firm']], 'S20'),
                                        return_inverse=True)
        cls.groups = firm_id
        # time indicator in range(max Ti)
        time = np.asarray(dtapa_exog[['year']])
        time -= time.min()
        cls.time = np.squeeze(time).astype(int)
        # nw_panel function requires interval bounds
        cls.tidx = [(i * 20, 20 * (i + 1)) for i in range(10)]


@pytest.mark.not_vetted
class TestOLSRobustCluster2(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_cluster
    small = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=self.groups,
                                                  use_correction=True,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=True)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


@pytest.mark.not_vetted
class TestOLSRobustCluster2Input(CheckOLSRobustCluster,
                                 CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_cluster
    small = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        fat_array = self.groups.reshape(-1, 1)
        fat_groups = pd.DataFrame(fat_array)

        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=fat_groups,
                                                  use_correction=True,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=True)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1

    def test_too_many_groups(self):
        long_groups = self.groups.reshape(-1, 1)
        groups3 = np.hstack((long_groups, long_groups, long_groups))
        with pytest.raises(ValueError):
            self.res1.get_robustcov_results('cluster',
                                            groups=groups3,
                                            use_correction=True,
                                            use_t=True)

    def test_2way_dataframe(self):
        long_groups = self.groups.reshape(-1, 1)
        groups2 = pd.DataFrame(np.hstack((long_groups, long_groups)))
        res = self.res1.get_robustcov_results('cluster',
                                              groups=groups2,
                                              use_correction=True,
                                              use_t=True)
        # TODO: Do something with res?


@pytest.mark.not_vetted
class TestOLSRobustCluster2Fit(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # copy, past uses fit method
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_cluster
    small = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.model.fit(cov_type='cluster',
                                      cov_kwds=dict(groups=self.groups,
                                                    use_correction=True,
                                                    use_t=True))
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=True)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1

    # TODO: Split this into reasonably-scoped tests
    def test_basic_inference(self):
        res1 = self.res1
        res2 = self.res2
        rtol = 1e-7
        assert_allclose(res1.params, res2.params, rtol=1e-8)
        assert_allclose(res1.bse, res2.bse, rtol=rtol)
        assert_allclose(res1.pvalues, res2.pvalues, rtol=rtol, atol=1e-20)

        ci = res2.params_table[:, 4:6]
        assert_allclose(res1.conf_int(),
                        ci,
                        rtol=5e-7, atol=1e-20)


@pytest.mark.not_vetted
class TestOLSRobustCluster2Large(CheckOLSRobustCluster,
                                 CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_cluster_large

    small = False
    skip_f = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=self.groups,
                                                  use_correction=False,
                                                  use_t=False,
                                                  df_correction=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=False)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1

    # skipping see GH#1189
    def test_f_value(self):
        pass


@pytest.mark.not_vetted
class TestOLSRobustCluster2LargeFit(CheckOLSRobustCluster,
                                    CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_cluster_large
    small = False
    skip_f = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        model = OLS(self.res1.model.endog, self.res1.model.exog)
        res_ols = model.fit(cov_type='cluster',
                            cov_kwds=dict(groups=self.groups,
                                          use_correction=False,
                                          use_t=False,
                                          df_correction=True))
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=False)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1

    # skipping see GH#1189
    def test_fvalue(self):
        # TODO: Upstream this is mangled.  fix it there
        pass


@pytest.mark.not_vetted
class TestOLSRobustClusterGS(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_nw_groupsum4
    small = True
    skip_f = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.get_robustcov_results('nw-groupsum',
                                                  time=self.time,
                                                  maxlags=4,
                                                  use_correction=False,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_nw_groupsum(self.res1, 4, self.time,
                                  use_correction=False)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


@pytest.mark.not_vetted
class TestOLSRobustClusterGSFit(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_nw_groupsum4
    small = True
    skip_f = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.model.fit(cov_type='nw-groupsum',
                                      cov_kwds=dict(time=self.time,
                                                    maxlags=4,
                                                    use_correction=False,
                                                    use_t=True))
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_nw_groupsum(self.res1, 4, self.time,
                                  use_correction=False)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


@pytest.mark.not_vetted
class TestOLSRobustClusterNWP(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_nw_panel4
    small = True
    skip_f = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.get_robustcov_results('nw-panel',
                                                  time=self.time,
                                                  maxlags=4,
                                                  use_correction='hac',
                                                  use_t=True,
                                                  df_correction=False)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_nw_panel(self.res1, 4, self.tidx)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1

    def test_keyword(self):
        # check corrected keyword
        res_ols = self.res1.get_robustcov_results('hac-panel',
                                                  time=self.time,
                                                  maxlags=4,
                                                  use_correction='hac',
                                                  use_t=True,
                                                  df_correction=False)
        assert_allclose(res_ols.bse, self.res1.bse, rtol=1e-12)


@pytest.mark.not_vetted
class TestOLSRobustClusterNWPGroupsFit(CheckOLSRobustCluster,
                                       CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_nw_panel4
    small = True
    skip_f = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.model.fit(cov_type='nw-panel',
                                      cov_kwds=dict(groups=self.groups,
                                                    maxlags=4,
                                                    use_correction='hac',
                                                    use_t=True,
                                                    df_correction=False))
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_nw_panel(self.res1, 4, self.tidx)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


# TODO: low precision/agreement
@pytest.mark.not_vetted
class TestOLSRobustCluster2G(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_cluster_2groups_small
    small = True
    # only f_pvalue and confint for constant differ >rtol=0.05
    rtol = 0.35
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=(self.groups,
                                                          self.time),
                                                  use_correction=True,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster_2groups(self.res1, self.groups,
                                      group2=self.time,
                                      use_correction=True)[0]
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


@pytest.mark.not_vetted
class TestOLSRobustCluster2GLarge(CheckOLSRobustCluster,
                                  CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_cluster_2groups_large
    small = False
    skip_f = True
    rtol = 1e-7
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=(self.groups,
                                                          self.time),
                                                  use_correction=False,
                                                  use_t=False)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster_2groups(self.res1, self.groups, group2=self.time,
                                      use_correction=False)[0]
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


# -----------------------------------------------------------------
# WLS

@pytest.mark.not_vetted
class CheckWLSRobustCluster(CheckOLSRobust):
    # compare with regress robust

    @classmethod
    def setup_class(cls):
        dtapa = grunfeld.data.load_pandas()
        # Stata example/data seems to miss last firm
        # TODO: Is the comment above (from upstream) actionable?
        dtapa_endog = dtapa.endog[:200]
        dtapa_exog = dtapa.exog[:200]
        exog = add_constant(dtapa_exog[['value', 'capital']], prepend=False)
        # asserts don't work for pandas
        cls.res1 = WLS(dtapa_endog, exog,
                       weights=1 / dtapa_exog['value']).fit()

        firm_names, firm_id = np.unique(np.asarray(dtapa_exog[['firm']], 'S20'),
                                        return_inverse=True)
        cls.groups = firm_id
        # time indicator in range(max Ti)
        time = np.asarray(dtapa_exog[['year']])
        time -= time.min()
        cls.time = np.squeeze(time).astype(int)
        # nw_panel function requires interval bounds
        cls.tidx = [(i * 20, 20 * (i + 1)) for i in range(10)]


# not available yet for WLS
@pytest.mark.not_vetted
class TestWLSRobustCluster2(CheckWLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_cluster_wls_small
    small = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=self.groups,
                                                  use_correction=True,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=True)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


# not available yet for WLS
@pytest.mark.not_vetted
class TestWLSRobustCluster2Large(CheckWLSRobustCluster,
                                 CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_cluster_wls_large
    small = False
    skip_f = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=self.groups,
                                                  use_correction=False,
                                                  use_t=False,
                                                  df_correction=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=False)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


@pytest.mark.not_vetted
class TestWLSRobustSmall(CheckWLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`
    res2 = results_grunfeld_ols_robust_cluster.results_hc1_wls_small
    small = True
    skip_f = True
    rtol = 1e-6
    rtolh = 1e-10

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HC1',
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        # TODO: check standalone function
        # Note: the above comment is from upstream, where it is not maintained;
        # it isn't clear if the "todo" is really valid.
        # cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=False)
        cov1 = res_ols.cov_HC1
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1


@pytest.mark.not_vetted
class TestWLSOLSRobustSmall(object):

    @classmethod
    def setup_class(cls):
        dtapa = grunfeld.data.load_pandas()
        # Stata example/data seems to miss last firm
        dtapa_endog = dtapa.endog[:200]
        dtapa_exog = dtapa.exog[:200]
        exog = add_constant(dtapa_exog[['value', 'capital']], prepend=False)
        # asserts don't work for pandas
        cls.res_wls = WLS(dtapa_endog, exog,
                          weights=1 / dtapa_exog['value']).fit()
        w_sqrt = 1 / np.sqrt(np.asarray(dtapa_exog['value']))
        cls.res_ols = OLS(dtapa_endog * w_sqrt,
                          np.asarray(exog) * w_sqrt[:, None]).fit()

        firm_names, firm_id = np.unique(np.asarray(dtapa_exog[['firm']], 'S20'),
                                        return_inverse=True)
        cls.groups = firm_id
        # time indicator in range(max Ti)
        time = np.asarray(dtapa_exog[['year']])
        time -= time.min()
        cls.time = np.squeeze(time).astype(int)
        # nw_panel function requires interval bounds
        cls.tidx = [(i * 20, 20 * (i + 1)) for i in range(10)]

    # TODO: Split this into reasonably-scoped tests
    def test_all(self):
        all_cov = [('HC0', dict(use_t=True)),
                   ('HC1', dict(use_t=True)),
                   ('HC2', dict(use_t=True)),
                   ('HC3', dict(use_t=True))]

        # fvalue are not the same, see GH#1212
        # TODO: See if these can be fixed
        # assert_allclose(self.res_ols.fvalue, res_wls.fvalue, rtol=1e-13)
        # assert_allclose(self.res_ols.f_pvalue, res_wls.f_pvalue, rtol=1e-13)

        for cov_type, kwds in all_cov:
            res1 = self.res_ols.get_robustcov_results(cov_type, **kwds)
            res2 = self.res_wls.get_robustcov_results(cov_type, **kwds)
            assert_allclose(res1.params, res2.params, rtol=1e-13)
            assert_allclose(res1.cov_params(), res2.cov_params(), rtol=1e-13)
            assert_allclose(res1.bse, res2.bse, rtol=1e-13)
            assert_allclose(res1.pvalues, res2.pvalues, rtol=1e-13)
            # Note: Fvalue doesn't match up, difference in calculation ?
            #      The only difference should be in the constant detection
            # TODO: See if these can be fixed
            # assert_allclose(res1.fvalue, res2.fvalue, rtol=1e-13)
            # assert_allclose(res1.f_pvalue, res2.f_pvalue, rtol=1e-13)
            mat = np.eye(len(res1.params))
            ft1 = res1.f_test(mat)
            ft2 = res2.f_test(mat)
            assert_allclose(ft1.fvalue, ft2.fvalue, rtol=1e-13)
            assert_allclose(ft1.pvalue, ft2.pvalue, rtol=1e-12)

    def test_fixed_scale(self):
        cov_type = 'fixed_scale'
        kwds = {}
        res1 = self.res_ols.get_robustcov_results(cov_type, **kwds)
        res2 = self.res_wls.get_robustcov_results(cov_type, **kwds)
        assert_allclose(res1.params, res2.params, rtol=1e-13)
        assert_allclose(res1.cov_params(), res2.cov_params(), rtol=1e-13)
        assert_allclose(res1.bse, res2.bse, rtol=1e-13)
        assert_allclose(res1.pvalues, res2.pvalues, rtol=1e-12)

        tt = res2.t_test(np.eye(len(res2.params)),
                         cov_p=res2.normalized_cov_params)
        assert_allclose(res2.cov_params(), res2.normalized_cov_params,
                        rtol=1e-13)
        assert_allclose(res2.bse, tt.sd, rtol=1e-13)
        assert_allclose(res2.pvalues, tt.pvalue, rtol=1e-13)
        assert_allclose(res2.tvalues, tt.tvalue, rtol=1e-13)

        # using cov_type in fit
        mod = self.res_wls.model
        mod3 = WLS(mod.endog, mod.exog, weights=mod.weights)
        res3 = mod3.fit(cov_type=cov_type, cov_kwds=kwds)
        tt = res3.t_test(np.eye(len(res3.params)),
                         cov_p=res3.normalized_cov_params)
        assert_allclose(res3.cov_params(), res3.normalized_cov_params,
                        rtol=1e-13)
        assert_allclose(res3.bse, tt.sd, rtol=1e-13)
        assert_allclose(res3.pvalues, tt.pvalue, rtol=1e-13)
        assert_allclose(res3.tvalues, tt.tvalue, rtol=1e-13)


@pytest.mark.not_vetted
def test_cov_type_fixed_scale():
    # this is a unit test from scipy curvefit for `absolute_sigma` keyword
    xdata = np.array([0, 1, 2, 3, 4, 5])
    ydata = np.array([1, 1, 5, 7, 8, 12])
    sigma = np.array([1, 2, 1, 2, 1, 2])

    xdata = np.column_stack((xdata, np.ones(len(xdata))))
    weights = 1. / sigma**2

    res = WLS(ydata, xdata, weights=weights).fit()
    assert_allclose(res.bse,
                    [0.20659803, 0.57204404],
                    rtol=1e-3)

    res = WLS(ydata, xdata, weights=weights).fit()
    assert_allclose(res.bse,
                    [0.20659803, 0.57204404],
                    rtol=1e-3)

    res = WLS(ydata, xdata, weights=weights).fit(cov_type='fixed scale')
    assert_allclose(res.bse,
                    [0.30714756, 0.85045308],
                    rtol=1e-3)

    res = WLS(ydata, xdata, weights=weights / 9.).fit(cov_type='fixed scale')
    assert_allclose(res.bse,
                    [3 * 0.30714756, 3 * 0.85045308],
                    rtol=1e-3)

    res = WLS(ydata, xdata, weights=weights).fit(cov_type='fixed scale',
                                                 cov_kwds={'scale': 9})
    assert_allclose(res.bse,
                    [3 * 0.30714756, 3 * 0.85045308],
                    rtol=1e-3)
