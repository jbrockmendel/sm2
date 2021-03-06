# -*- coding: utf-8 -*-
"""Tests of GLSAR and diagnostics against Gretl

Created on Thu Feb 02 21:15:47 2012

Author: Josef Perktold
License: BSD-3

"""
import os

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from sm2.regression.linear_model import OLS, GLSAR
from sm2.tools.tools import add_constant
from sm2.datasets import macrodata

import sm2.stats.sandwich_covariance as sw
from sm2.stats import diagnostic

outliers_influence = None  # dummy to prevent flake8 warnings

cur_dir = os.path.abspath(os.path.dirname(__file__))
fpath = os.path.join(cur_dir, 'results',
                     'leverage_influence_ols_nostars.txt')
lev_data = np.genfromtxt(fpath, skip_header=3, skip_footer=1,
                         converters={0: lambda s: s})
# either numpy 1.6 or python 3.2 changed behavior
if np.isnan(lev_data[-1]['f1']):
    lev_data = np.genfromtxt(fpath, skip_header=3, skip_footer=2,
                             converters={0: lambda s: s})
lev_data.dtype.names = ['date', 'residual', 'leverage', 'influence', 'DFFITS']


def compare_ftest(contrast_res, other, decimal=(5, 4)):
    assert_almost_equal(contrast_res.fvalue, other[0], decimal=decimal[0])
    assert_almost_equal(contrast_res.pvalue, other[1], decimal=decimal[1])
    assert other[2] == contrast_res.df_num
    assert other[3] == contrast_res.df_denom
    assert other[4] == "f"


@pytest.mark.not_vetted
class TestGLSARGretl(object):
    @classmethod
    def setup_class(cls):
        d = macrodata.load_pandas().data

        # growth rates
        gs_l_realinv = 400 * np.diff(np.log(d['realinv'].values))
        gs_l_realgdp = 400 * np.diff(np.log(d['realgdp'].values))

        endogg = gs_l_realinv
        exogg = add_constant(np.c_[gs_l_realgdp, d['realint'][:-1].values])

        res_ols = OLS(endogg, exogg).fit()

        mod_g1 = GLSAR(endogg, exogg, rho=-0.108136)
        res_g1 = mod_g1.fit()

        mod_g2 = GLSAR(endogg, exogg, rho=-0.108136)   # -0.1335859) from R
        res_g2 = mod_g2.iterative_fit(maxiter=5)

        cls.res_ols = res_ols
        cls.res_g1 = res_g1
        cls.res_g2 = res_g2

    # TODO: split up giant test
    def test_all(self):
        rho = -0.108136

        # coefficient   std. error   t-ratio    p-value 95% CONFIDENCE INTERVAL
        partable = np.array([
            [-9.50990, 0.990456, -9.602, 3.65e-018, -11.4631, -7.55670],
            [4.37040, 0.208146, 21.00, 2.93e-052, 3.95993, 4.78086],
            [-0.579253, 0.268009, -2.161, 0.0319, -1.10777, -0.0507346]])

        # Statistics based on the rho-differenced data:

        result_gretl_g1 = dict(
            endog_mean=("Mean dependent var", 3.113973),
            endog_std=("S.D. dependent var", 18.67447),
            ssr=("Sum squared resid", 22530.90),
            mse_resid_sqrt=("S.E. of regression", 10.66735),
            rsquared=("R-squared", 0.676973),
            rsquared_adj=("Adjusted R-squared", 0.673710),
            fvalue=("F(2, 198)", 221.0475),
            f_pvalue=("P-value(F)", 3.56e-51),
            resid_acf1=("rho", -0.003481),
            dw=("Durbin-Watson", 1.993858))

        # fstatistic, p-value, df1, df2
        reset_2_3 = [5.219019, 0.00619, 2, 197, "f"]
        reset_2 = [7.268492, 0.00762, 1, 198, "f"]
        reset_3 = [5.248951, 0.023, 1, 198, "f"]
        # LM-statistic, p-value, df
        arch_4 = [7.30776, 0.120491, 4, "chi2"]

        # multicollinearity
        vif = [1.002, 1.002]
        cond_1norm = 6862.0664
        determinant = 1.0296049e+009
        reciprocal_condition_number = 0.013819244

        # Chi-square(2): test-statistic, pvalue, df
        normality = [20.2792, 3.94837e-005, 2]

        res = self.res_g1  # with rho from Gretl

        # basic
        assert_almost_equal(res.params, partable[:, 0], 4)
        assert_almost_equal(res.bse, partable[:, 1], 6)
        assert_almost_equal(res.tvalues, partable[:, 2], 2)

        assert_almost_equal(res.ssr, result_gretl_g1['ssr'][1], decimal=2)
        #assert_almost_equal(res.llf,
        #                    result_gretl_g1['llf'][1],
        #                    decimal=7)  # not in gretl
        #assert_almost_equal(res.rsquared,
        #                    result_gretl_g1['rsquared'][1],
        #                    decimal=7)  # FAIL
        #assert_almost_equal(res.rsquared_adj,
        #                    result_gretl_g1['rsquared_adj'][1],
        #                    decimal=7)  # FAIL
        assert_almost_equal(np.sqrt(res.mse_resid),
                            result_gretl_g1['mse_resid_sqrt'][1],
                            decimal=5)
        assert_almost_equal(res.fvalue,
                            result_gretl_g1['fvalue'][1],
                            decimal=4)
        assert_allclose(res.f_pvalue,
                        result_gretl_g1['f_pvalue'][1],
                        rtol=1e-2)
        #assert_almost_equal(res.durbin_watson,
        #                    result_gretl_g1['dw'][1],
        #                    decimal=7)  # TODO

        # arch
        sm_arch = diagnostic.het_arch(res.wresid, maxlag=4)
        assert_almost_equal(sm_arch[0], arch_4[0], decimal=4)
        assert_almost_equal(sm_arch[1], arch_4[1], decimal=6)

        # tests
        res = self.res_g2  # with estimated rho

        # estimated lag coefficient
        assert_almost_equal(res.model.rho, rho, decimal=3)

        # basic
        assert_almost_equal(res.params, partable[:, 0], 4)
        assert_almost_equal(res.bse, partable[:, 1], 3)
        assert_almost_equal(res.tvalues, partable[:, 2], 2)

        assert_almost_equal(res.ssr, result_gretl_g1['ssr'][1], decimal=2)
        #assert_almost_equal(res.llf,
        #                     result_gretl_g1['llf'][1],
        #                     decimal=7)  # not in gretl
        #assert_almost_equal(res.rsquared,
        #                     result_gretl_g1['rsquared'][1],
        #                     decimal=7)  # FAIL
        #assert_almost_equal(res.rsquared_adj,
        #                     result_gretl_g1['rsquared_adj'][1],
        #                     decimal=7)  # FAIL
        assert_almost_equal(np.sqrt(res.mse_resid),
                            result_gretl_g1['mse_resid_sqrt'][1],
                            decimal=5)
        assert_almost_equal(res.fvalue,
                            result_gretl_g1['fvalue'][1],
                            decimal=0)
        assert_almost_equal(res.f_pvalue,
                            result_gretl_g1['f_pvalue'][1],
                            decimal=6)
        #assert_almost_equal(res.durbin_watson,
        #                     result_gretl_g1['dw'][1],
        #                     decimal=7)  # TODO

        # arch
        sm_arch = diagnostic.het_arch(res.wresid, maxlag=4)
        assert_almost_equal(sm_arch[0], arch_4[0], decimal=1)
        assert_almost_equal(sm_arch[1], arch_4[1], decimal=2)

        """
        Performing iterative calculation of rho...

                         ITER       RHO        ESS
                           1     -0.10734   22530.9
                           2     -0.10814   22530.9

        Model 4: Cochrane-Orcutt, using observations 1959:3-2009:3 (T = 201)
        Dependent variable: ds_l_realinv
        rho = -0.108136

                         coefficient   std. error   t-ratio    p-value
          -------------------------------------------------------------
          const           -9.50990      0.990456    -9.602    3.65e-018 ***
          ds_l_realgdp     4.37040      0.208146    21.00     2.93e-052 ***
          realint_1       -0.579253     0.268009    -2.161    0.0319    **

        Statistics based on the rho-differenced data:

        Mean dependent var   3.113973   S.D. dependent var   18.67447
        Sum squared resid    22530.90   S.E. of regression   10.66735
        R-squared            0.676973   Adjusted R-squared   0.673710
        F(2, 198)            221.0475   P-value(F)           3.56e-51
        rho                 -0.003481   Durbin-Watson        1.993858
        """

        """
        RESET test for specification (squares and cubes)
        Test statistic: F = 5.219019,
        with p-value = P(F(2,197) > 5.21902) = 0.00619

        RESET test for specification (squares only)
        Test statistic: F = 7.268492,
        with p-value = P(F(1,198) > 7.26849) = 0.00762

        RESET test for specification (cubes only)
        Test statistic: F = 5.248951,
        with p-value = P(F(1,198) > 5.24895) = 0.023:
        """

        """
        Test for ARCH of order 4

                     coefficient   std. error   t-ratio   p-value
          --------------------------------------------------------
          alpha(0)   97.0386       20.3234       4.775    3.56e-06 ***
          alpha(1)    0.176114      0.0714698    2.464    0.0146   **
          alpha(2)   -0.0488339     0.0724981   -0.6736   0.5014
          alpha(3)   -0.0705413     0.0737058   -0.9571   0.3397
          alpha(4)    0.0384531     0.0725763    0.5298   0.5968

          Null hypothesis: no ARCH effect is present
          Test statistic: LM = 7.30776
          with p-value = P(Chi-square(4) > 7.30776) = 0.120491:
        """

        """
        Variance Inflation Factors

        Minimum possible value = 1.0
        Values > 10.0 may indicate a collinearity problem

           ds_l_realgdp    1.002
              realint_1    1.002

        VIF(j) = 1/(1 - R(j)^2), where R(j) is the multiple correlation
        coefficient between variable j and the other independent variables

        Properties of matrix X'X:

         1-norm = 6862.0664
         Determinant = 1.0296049e+009
         Reciprocal condition number = 0.013819244
        """
        """
        Test for ARCH of order 4 -
          Null hypothesis: no ARCH effect is present
          Test statistic: LM = 7.30776
          with p-value = P(Chi-square(4) > 7.30776) = 0.120491

        Test of common factor restriction -
          Null hypothesis: restriction is acceptable
          Test statistic: F(2, 195) = 0.426391
          with p-value = P(F(2, 195) > 0.426391) = 0.653468

        Test for normality of residual -
          Null hypothesis: error is normally distributed
          Test statistic: Chi-square(2) = 20.2792
          with p-value = 3.94837e-005:
        """

        # no idea what this is
        """
        Augmented regression for common factor test
        OLS, using observations 1959:3-2009:3 (T = 201)
        Dependent variable: ds_l_realinv

                           coefficient   std. error   t-ratio    p-value
          ---------------------------------------------------------------
          const            -10.9481      1.35807      -8.062    7.44e-014 ***
          ds_l_realgdp       4.28893     0.229459     18.69     2.40e-045 ***
          realint_1         -0.662644    0.334872     -1.979    0.0492    **
          ds_l_realinv_1    -0.108892    0.0715042    -1.523    0.1294
          ds_l_realgdp_1     0.660443    0.390372      1.692    0.0923    *
          realint_2          0.0769695   0.341527      0.2254   0.8219

          Sum of squared residuals = 22432.8

        Test of common factor restriction

          Test statistic: F(2, 195) = 0.426391, with p-value = 0.653468
        """

        # with OLS, HAC errors
        # Model 5: OLS, using observations 1959:2-2009:3 (T = 202)
        # Dependent variable: ds_l_realinv
        # HAC standard errors, bandwidth 4 (Bartlett kernel)

        # coefficient   std. error   t-ratio    p-value 95% CONFIDENCE INTERVAL
        # for confidence interval t(199, 0.025) = 1.972

        partable = np.array([
            [-9.48167, 1.17709, -8.055, 7.17e-014, -11.8029, -7.16049],
            [4.37422, 0.328787, 13.30, 2.62e-029, 3.72587, 5.02258],
            [-0.613997, 0.293619, -2.091, 0.0378, -1.19300, -0.0349939]])

        result_gretl_g1 = dict(
            endog_mean=("Mean dependent var", 3.257395),
            endog_std=("S.D. dependent var", 18.73915),
            ssr=("Sum squared resid", 22799.68),
            mse_resid_sqrt=("S.E. of regression", 10.70380),
            rsquared=("R-squared", 0.676978),
            rsquared_adj=("Adjusted R-squared", 0.673731),
            fvalue=("F(2, 199)", 90.79971),
            f_pvalue=("P-value(F)", 9.53e-29),
            llf=("Log-likelihood", -763.9752),
            aic=("Akaike criterion", 1533.950),
            bic=("Schwarz criterion", 1543.875),
            hqic=("Hannan-Quinn", 1537.966),
            resid_acf1=("rho", -0.107341),
            dw=("Durbin-Watson", 2.213805))

        linear_logs = [1.68351, 0.430953, 2, "chi2"]
        # for logs: dropping 70 nan or incomplete observations, T=133
        # (res_ols.model.exog <=0).any(1).sum() = 69  ?not 70
        linear_squares = [7.52477, 0.0232283, 2, "chi2"]

        # Autocorrelation, Breusch-Godfrey test for autocorrelation
        # up to order 4
        lm_acorr4 = [1.17928, 0.321197, 4, 195, "F"]
        lm2_acorr4 = [4.771043, 0.312, 4, "chi2"]
        acorr_ljungbox4 = [5.23587, 0.264, 4, "chi2"]

        # break
        cusum_Harvey_Collier = [0.494432, 0.621549, 198, "t"]  # stats.t.sf(0.494432, 198)*2
        # see cusum results in files
        break_qlr = [3.01985, 0.1, 3, 196, "maxF"]  # TODO check this, max at 2001:4
        break_chow = [13.1897, 0.00424384, 3, "chi2"]  # break at 1984:1

        arch_4 = [3.43473, 0.487871, 4, "chi2"]

        normality = [23.962, 0.00001, 2, "chi2"]

        het_white = [33.503723, 0.000003, 5, "chi2"]
        het_breusch_pagan = [1.302014, 0.521520, 2, "chi2"]  # TODO: not available
        het_breusch_pagan_konker = [0.709924, 0.701200, 2, "chi2"]

        reset_2_3 = [5.219019, 0.00619, 2, 197, "f"]
        reset_2 = [7.268492, 0.00762, 1, 198, "f"]
        reset_3 = [5.248951, 0.023, 1, 198, "f"]  # not available

        cond_1norm = 5984.0525
        determinant = 7.1087467e+008
        reciprocal_condition_number = 0.013826504
        vif = [1.001, 1.001]

        res = self.res_ols  # for easier copying

        cov_hac = sw.cov_hac_simple(res, nlags=4, use_correction=False)
        bse_hac = sw.se_cov(cov_hac)

        assert_almost_equal(res.params, partable[:, 0], 5)
        assert_almost_equal(bse_hac, partable[:, 1], 5)
        # TODO

        assert_almost_equal(res.ssr,
                            result_gretl_g1['ssr'][1],
                            decimal=2)
        assert_almost_equal(res.llf,
                            result_gretl_g1['llf'][1],
                            decimal=4)  # not in gretl
        assert_almost_equal(res.rsquared,
                            result_gretl_g1['rsquared'][1],
                            decimal=6)  # FAIL
        assert_almost_equal(res.rsquared_adj,
                            result_gretl_g1['rsquared_adj'][1],
                            decimal=6)  # FAIL
        assert_almost_equal(np.sqrt(res.mse_resid),
                            result_gretl_g1['mse_resid_sqrt'][1],
                            decimal=5)
        # f-value is based on cov_hac I guess
        #res2 = res.get_robustcov_results(cov_type='HC1')
        # TODO: fvalue differs from Gretl, trying any of the HCx
        #assert_almost_equal(res2.fvalue, result_gretl_g1['fvalue'][1],
        #                     decimal=0) # FAIL
        #assert_allclose(res.f_pvalue, result_gretl_g1['f_pvalue'][1],
        #                rtol=1e-1) # FAIL
        #assert_almost_equal(res.durbin_watson, result_gretl_g1['dw'][1],
        #                     decimal=7) # TODO

        linear_sq = diagnostic.linear_lm(res.resid, res.model.exog)
        assert_almost_equal(linear_sq[0], linear_squares[0], decimal=6)
        assert_almost_equal(linear_sq[1], linear_squares[1], decimal=7)

        hbpk = diagnostic.het_breuschpagan(res.resid, res.model.exog)
        assert_almost_equal(hbpk[0], het_breusch_pagan_konker[0], decimal=6)
        assert_almost_equal(hbpk[1], het_breusch_pagan_konker[1], decimal=6)

        hw = diagnostic.het_white(res.resid, res.model.exog)
        assert_almost_equal(hw[:2], het_white[:2], 6)

        # arch
        sm_arch = diagnostic.het_arch(res.resid, maxlag=4)
        assert_almost_equal(sm_arch[0], arch_4[0], decimal=5)
        assert_almost_equal(sm_arch[1], arch_4[1], decimal=6)

    @pytest.mark.skip(reason="outliers_influence not ported from upstream")
    def test_gls_reset_ramsey(self):
        res = self.res_g2  # with estimated rho

        # fstatistic, p-value, df1, df2
        reset_2_3 = [5.219019, 0.00619, 2, 197, "f"]
        reset_2 = [7.268492, 0.00762, 1, 198, "f"]
        reset_3 = [5.248951, 0.023, 1, 198, "f"]

        # from sm2.stats import outliers_influence
        c = outliers_influence.reset_ramsey(res, degree=2)
        compare_ftest(c, reset_2, decimal=(2, 4))
        c = outliers_influence.reset_ramsey(res, degree=3)
        compare_ftest(c, reset_2_3, decimal=(2, 4))

    @pytest.mark.skip(reason="outliers_influence not ported from upstream")
    def test_ols_reset_ramsey(self):
        res = self.res_ols

        reset_2_3 = [5.219019, 0.00619, 2, 197, "f"]
        reset_2 = [7.268492, 0.00762, 1, 198, "f"]
        # reset_3 = [5.248951, 0.023, 1, 198, "f"]  # not available

        # from sm2.stats import outliers_influence
        c = outliers_influence.reset_ramsey(res, degree=2)
        compare_ftest(c, reset_2, decimal=(6, 5))
        c = outliers_influence.reset_ramsey(res, degree=3)
        compare_ftest(c, reset_2_3, decimal=(6, 5))

    @pytest.mark.skip(reason="outliers_influence not ported from upstream")
    def test_ols_influence(self):
        res = self.res_ols
        # from sm2.stats import outliers_influence
        vif2 = [outliers_influence.variance_inflation_factor(res.model.exog, k)
                for k in [1, 2]]

        infl = outliers_influence.OLSInfluence(res)
        # just added this based on Gretl

        # just rough test, low decimal in Gretl output,
        assert_almost_equal(lev_data['residual'],
                            res.resid,
                            decimal=3)
        assert_almost_equal(lev_data['DFFITS'],
                            infl.dffits[0],
                            decimal=3)
        assert_almost_equal(lev_data['leverage'],
                            infl.hat_matrix_diag,
                            decimal=3)
        assert_almost_equal(lev_data['influence'],
                            infl.influence,
                            decimal=4)


@pytest.mark.not_vetted
def test_GLSARlag():
    # test that results for lag>1 is close to lag=1, and smaller ssr
    d2 = macrodata.load_pandas().data
    g_gdp = 400 * np.diff(np.log(d2['realgdp'].values))
    g_inv = 400 * np.diff(np.log(d2['realinv'].values))
    exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1].values],
                         prepend=False)

    mod1 = GLSAR(g_inv, exogg, 1)
    res1 = mod1.iterative_fit(5)

    mod4 = GLSAR(g_inv, exogg, 4)
    res4 = mod4.iterative_fit(10)

    assert (np.abs(res1.params / res4.params - 1) < 0.03).all()
    assert res4.ssr < res1.ssr
    assert (np.abs(res4.bse / res1.bse) - 1 < 0.015).all()
    assert np.abs((res4.fittedvalues / res1.fittedvalues - 1).mean()) < 0.015
    assert len(mod4.rho) == 4
