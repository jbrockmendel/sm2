"""
Tests for discrete models

Notes
-----
DECIMAL_3 is used because it seems that there is a loss of precision
in the Stata *.dta -> *.csv output, NOT the estimator for the Poisson
tests.
"""
# pylint: disable-msg=E1101
import os
import warnings

from six.moves import range

import numpy as np
from numpy.testing import (assert_, assert_raises, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose,
                           assert_array_less)
import pandas as pd
import pandas.util.testing as tm

from scipy.stats import nbinom
import pytest

import sm2.datasets

from sm2.tools.sm_exceptions import PerfectSeparationError, MissingDataError
from sm2.tools.tools import add_constant
from sm2.discrete.discrete_model import (Logit, Probit, MNLogit,
                                         Poisson, NegativeBinomial,
                                         CountModel, GeneralizedPoisson,
                                         NegativeBinomialP, MultinomialModel)

from statsmodels import distributions
from statsmodels.discrete.discrete_margins import _iscount, _isdummy

from .results.results_discrete import Spector, DiscreteL1, RandHIE, Anes

try:
    import cvxopt
    has_cvxopt = True
except ImportError:
    has_cvxopt = False

try:
    from scipy.optimize import basinhopping
    has_basinhopping = True
except ImportError:
    has_basinhopping = False

try:
    from scipy.optimize._trustregion_dogleg import _minimize_dogleg
    has_dogleg = True
except ImportError:
    has_dogleg = False

DECIMAL_14 = 14
DECIMAL_10 = 10
DECIMAL_9 = 9
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1
DECIMAL_0 = 0

'''
class CheckModelResults(object):
    """
    res2 should be the test results from RModelWrap
    or the results as defined in model_results_data
    """

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, rtol=8e-5)

    def test_zstat(self):
        assert_almost_equal(self.res1.tvalues, self.res2.z, DECIMAL_4)

    def pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)

#    def test_cov_params(self):
#        assert_almost_equal(self.res1.cov_params(), self.res2.cov_params,
#                DECIMAL_4)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_4)

    def test_llnull(self):
        assert_almost_equal(self.res1.llnull, self.res2.llnull, DECIMAL_4)

    def test_llr(self):
        assert_almost_equal(self.res1.llr, self.res2.llr, DECIMAL_3)

    def test_llr_pvalue(self):
        assert_almost_equal(self.res1.llr_pvalue, self.res2.llr_pvalue,
                DECIMAL_4)

    def test_normalized_cov_params(self):
        pass

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_dof(self):
        assert_equal(self.res1.df_model, self.res2.df_model)
        assert_equal(self.res1.df_resid, self.res2.df_resid)

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL_3)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL_3)

    def test_predict(self):
        assert_almost_equal(self.res1.model.predict(self.res1.params),
                            self.res2.phat, DECIMAL_4)

    def test_predict_xb(self):
        assert_almost_equal(self.res1.model.predict(self.res1.params,
                            linear=True),
                            self.res2.yhat, DECIMAL_4)

    def test_loglikeobs(self):
        #basic cross check
        llobssum = self.res1.model.loglikeobs(self.res1.params).sum()
        assert_almost_equal(llobssum, self.res1.llf, DECIMAL_14)

    def test_jac(self):
        #basic cross check
        jacsum = self.res1.model.score_obs(self.res1.params).sum(0)
        score = self.res1.model.score(self.res1.params)
        assert_almost_equal(jacsum, score, DECIMAL_9) #Poisson has low precision ?


class CheckBinaryResults(CheckModelResults):
    def test_pred_table(self):
        assert_array_equal(self.res1.pred_table(), self.res2.pred_table)

    def test_resid_dev(self):
        assert_almost_equal(self.res1.resid_dev, self.res2.resid_dev,
                DECIMAL_4)

    def test_resid_generalized(self):
        assert_almost_equal(self.res1.resid_generalized,
                            self.res2.resid_generalized, DECIMAL_4)

    def test_resid_response_smoke(self):
        self.res1.resid_response


class CheckMargEff(object):
    """
    Test marginal effects (margeff) and its options
    """

    def test_nodummy_dydxoverall(self):
        me = self.res1.get_margeff()
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dydx_se, DECIMAL_4)

        me_frame = me.summary_frame()
        eff = me_frame["dy/dx"].values
        assert_allclose(eff, me.margeff, rtol=1e-13)
        assert_equal(me_frame.shape, (me.margeff.size, 6))


    def test_nodummy_dydxmean(self):
        me = self.res1.get_margeff(at='mean')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dydxmean_se, DECIMAL_4)

    def test_nodummy_dydxmedian(self):
        me = self.res1.get_margeff(at='median')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dydxmedian, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dydxmedian_se, DECIMAL_4)

    def test_nodummy_dydxzero(self):
        me = self.res1.get_margeff(at='zero')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dydxzero, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dydxzero, DECIMAL_4)

    def test_nodummy_dyexoverall(self):
        me = self.res1.get_margeff(method='dyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dyex, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dyex_se, DECIMAL_4)

    def test_nodummy_dyexmean(self):
        me = self.res1.get_margeff(at='mean', method='dyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dyexmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dyexmean_se, DECIMAL_4)

    def test_nodummy_dyexmedian(self):
        me = self.res1.get_margeff(at='median', method='dyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dyexmedian, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dyexmedian_se, DECIMAL_4)

    def test_nodummy_dyexzero(self):
        me = self.res1.get_margeff(at='zero', method='dyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dyexzero, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dyexzero_se, DECIMAL_4)

    def test_nodummy_eydxoverall(self):
        me = self.res1.get_margeff(method='eydx')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eydx_se, DECIMAL_4)

    def test_nodummy_eydxmean(self):
        me = self.res1.get_margeff(at='mean', method='eydx')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eydxmean_se, DECIMAL_4)

    def test_nodummy_eydxmedian(self):
        me = self.res1.get_margeff(at='median', method='eydx')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eydxmedian, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eydxmedian_se, DECIMAL_4)

    def test_nodummy_eydxzero(self):
        me = self.res1.get_margeff(at='zero', method='eydx')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eydxzero, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eydxzero_se, DECIMAL_4)

    def test_nodummy_eyexoverall(self):
        me = self.res1.get_margeff(method='eyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eyex, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eyex_se, DECIMAL_4)

    def test_nodummy_eyexmean(self):
        me = self.res1.get_margeff(at='mean', method='eyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eyexmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eyexmean_se, DECIMAL_4)

    def test_nodummy_eyexmedian(self):
        me = self.res1.get_margeff(at='median', method='eyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eyexmedian, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eyexmedian_se, DECIMAL_4)

    def test_nodummy_eyexzero(self):
        me = self.res1.get_margeff(at='zero', method='eyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eyexzero, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eyexzero_se, DECIMAL_4)

    def test_dummy_dydxoverall(self):
        me = self.res1.get_margeff(dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_dydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_dydx_se, DECIMAL_4)

    def test_dummy_dydxmean(self):
        me = self.res1.get_margeff(at='mean', dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_dydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_dydxmean_se, DECIMAL_4)

    def test_dummy_eydxoverall(self):
        me = self.res1.get_margeff(method='eydx', dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_eydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_eydx_se, DECIMAL_4)

    def test_dummy_eydxmean(self):
        me = self.res1.get_margeff(at='mean', method='eydx', dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_eydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_eydxmean_se, DECIMAL_4)

    def test_count_dydxoverall(self):
        me = self.res1.get_margeff(count=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_count_dydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_count_dydx_se, DECIMAL_4)

    def test_count_dydxmean(self):
        me = self.res1.get_margeff(count=True, at='mean')
        assert_almost_equal(me.margeff,
                self.res2.margeff_count_dydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_count_dydxmean_se, DECIMAL_4)

    def test_count_dummy_dydxoverall(self):
        me = self.res1.get_margeff(count=True, dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_count_dummy_dydxoverall, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_count_dummy_dydxoverall_se, DECIMAL_4)

    def test_count_dummy_dydxmean(self):
        me = self.res1.get_margeff(count=True, dummy=True, at='mean')
        assert_almost_equal(me.margeff,
                self.res2.margeff_count_dummy_dydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_count_dummy_dydxmean_se, DECIMAL_4)


class TestProbitNewton(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = Probit(data.endog, data.exog).fit(method="newton", disp=0)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2

    #def test_predict(self):
    #    assert_almost_equal(self.res1.model.predict(self.res1.params),
    #            self.res2.predict, DECIMAL_4)


class TestProbitBFGS(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = Probit(data.endog, data.exog).fit(method="bfgs",
            disp=0)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2


class TestProbitNM(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="nm",
            disp=0, maxiter=500)


class TestProbitPowell(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="powell",
            disp=0, ftol=1e-8)


class TestProbitCG(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2

        # fmin_cg fails to converge on some machines - reparameterize
        from statsmodels.tools.transform_model import StandardizeTransform
        transf = StandardizeTransform(data.exog)
        exog_st = transf(data.exog)
        res1_st = Probit(data.endog,
                         exog_st).fit(method="cg", disp=0, maxiter=1000,
                                      gtol=1e-08)
        start_params = transf.transform_params(res1_st.params)
        assert_allclose(start_params, res2.params, rtol=1e-5, atol=1e-6)

        cls.res1 = Probit(data.endog,
                          data.exog).fit(start_params=start_params,
                                         method="cg", maxiter=1000,
                                         gtol=1e-05, disp=0)

        assert_array_less(cls.res1.mle_retvals['fcalls'], 100)


class TestProbitNCG(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="ncg",
                                                     disp=0, avextol=1e-8,
                                                     warn_convergence=False)
        # converges close enough but warnflag is 2 for precision loss


@pytest.mark.skipif(not has_basinhopping, reason='Skipped TestProbitBasinhopping '
                                                 'since basinhopping solver is '
                                                 'not available')
class TestProbitBasinhopping(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        if not has_basinhopping:
            raise pytest.skip("Skipped TestProbitBasinhopping since"
                              " basinhopping solver is not available")
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        fit = Probit(data.endog, data.exog).fit
        cls.res1 = fit(method="basinhopping", disp=0, niter=5,
                        minimizer={'method' : 'L-BFGS-B', 'tol' : 1e-8})


class TestProbitMinimizeDefault(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        fit = Probit(data.endog, data.exog).fit
        cls.res1 = fit(method="minimize", disp=0, niter=5, tol = 1e-8)


class TestProbitMinimizeDogleg(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        if not has_dogleg:
            raise pytest.skip("Skipped TestProbitMinimizeDogleg since "
                              "dogleg method is not available")

        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        fit = Probit(data.endog, data.exog).fit
        cls.res1 = fit(method="minimize", disp=0, niter=5, tol = 1e-8,
                       min_method = 'dogleg')


class TestProbitMinimizeAdditionalOptions(CheckBinaryResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="minimize", disp=0,
                                                     maxiter=500,
                                                     min_method='Nelder-Mead',
                                                     xtol=1e-4, ftol=1e-4)

class CheckLikelihoodModelL1(object):
    """
    For testing results generated with L1 regularization
    """
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_conf_int(self):
        assert_almost_equal(
                self.res1.conf_int(), self.res2.conf_int, DECIMAL_4)

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_nnz_params(self):
        assert_almost_equal(
                self.res1.nnz_params, self.res2.nnz_params, DECIMAL_4)

    def test_aic(self):
        assert_almost_equal(
                self.res1.aic, self.res2.aic, DECIMAL_3)

    def test_bic(self):
        assert_almost_equal(
                self.res1.bic, self.res2.bic, DECIMAL_3)


class TestProbitL1(CheckLikelihoodModelL1):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=True)
        alpha = np.array([0.1, 0.2, 0.3, 10]) #/ data.exog.shape[0]
        cls.res1 = Probit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=alpha, disp=0, trim_mode='auto',
            auto_trim_tol=0.02, acc=1e-10, maxiter=1000)
        res2 = DiscreteL1()
        res2.probit()
        cls.res2 = res2

    def test_cov_params(self):
        assert_almost_equal(
                self.res1.cov_params(), self.res2.cov_params, DECIMAL_4)


class TestMNLogitL1(CheckLikelihoodModelL1):

    @classmethod
    def setup_class(cls):
        anes_data = sm2.datasets.anes96.load()
        anes_exog = anes_data.exog
        anes_exog = add_constant(anes_exog, prepend=False)
        mlogit_mod = MNLogit(anes_data.endog, anes_exog)
        alpha = 10. * np.ones((mlogit_mod.J - 1, mlogit_mod.K)) #/ anes_exog.shape[0]
        alpha[-1,:] = 0
        cls.res1 = mlogit_mod.fit_regularized(
                method='l1', alpha=alpha, trim_mode='auto', auto_trim_tol=0.02,
                acc=1e-10, disp=0)
        res2 = DiscreteL1()
        res2.mnlogit()
        cls.res2 = res2


class TestLogitL1(CheckLikelihoodModelL1):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=True)
        cls.alpha = 3 * np.array([0., 1., 1., 1.]) #/ data.exog.shape[0]
        cls.res1 = Logit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=cls.alpha, disp=0, trim_mode='size',
            size_trim_tol=1e-5, acc=1e-10, maxiter=1000)
        res2 = DiscreteL1()
        res2.logit()
        cls.res2 = res2

    def test_cov_params(self):
        assert_almost_equal(
                self.res1.cov_params(), self.res2.cov_params, DECIMAL_4)


@pytest.mark.skipif(not has_cvxopt, reason='Skipped test_cvxopt since cvxopt '
                                           'is not available')
class TestCVXOPT(object):

    @classmethod
    def setup_class(cls):
        if not has_cvxopt:
            raise pytest.skip('Skipped test_cvxopt since cvxopt is not available')
        cls.data = sm2.datasets.spector.load()
        cls.data.exog = add_constant(cls.data.exog, prepend=True)

    def test_cvxopt_versus_slsqp(self):
        #Compares resutls from cvxopt to the standard slsqp
        self.alpha = 3. * np.array([0, 1, 1, 1.]) #/ self.data.endog.shape[0]
        res_slsqp = Logit(self.data.endog, self.data.exog).fit_regularized(
            method="l1", alpha=self.alpha, disp=0, acc=1e-10, maxiter=1000,
            trim_mode='auto')
        res_cvxopt = Logit(self.data.endog, self.data.exog).fit_regularized(
            method="l1_cvxopt_cp", alpha=self.alpha, disp=0, abstol=1e-10,
            trim_mode='auto', auto_trim_tol=0.01, maxiter=1000)
        assert_almost_equal(res_slsqp.params, res_cvxopt.params, DECIMAL_4)


class TestSweepAlphaL1(object):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=True)
        cls.model = Logit(data.endog, data.exog)
        cls.alphas = np.array(
                   [[0.1, 0.1, 0.1, 0.1],
                    [0.4, 0.4, 0.5, 0.5],
                    [0.5, 0.5, 1, 1]]) #/ data.exog.shape[0]
        cls.res1 = DiscreteL1()
        cls.res1.sweep()

    def test_sweep_alpha(self):
        for i in range(3):
            alpha = self.alphas[i, :]
            res2 = self.model.fit_regularized(
                    method="l1", alpha=alpha, disp=0, acc=1e-10,
                    trim_mode='off', maxiter=1000)
            assert_almost_equal(res2.params, self.res1.params[i], DECIMAL_4)


class CheckL1Compatability(object):
    """
    Tests compatability between l1 and unregularized by setting alpha such
    that certain parameters should be effectively unregularized, and others
    should be ignored by the model.
    """
    def test_params(self):
        m = self.m
        assert_almost_equal(
            self.res_unreg.params[:m], self.res_reg.params[:m], DECIMAL_4)
        # The last entry should be close to zero
        # handle extra parameter of NegativeBinomial
        kvars = self.res_reg.model.exog.shape[1]
        assert_almost_equal(0, self.res_reg.params[m:kvars], DECIMAL_4)

    def test_cov_params(self):
        m = self.m
        # The restricted cov_params should be equal
        assert_almost_equal(
            self.res_unreg.cov_params()[:m, :m],
            self.res_reg.cov_params()[:m, :m],
            DECIMAL_1)

    def test_df(self):
        assert_equal(self.res_unreg.df_model, self.res_reg.df_model)
        assert_equal(self.res_unreg.df_resid, self.res_reg.df_resid)

    def test_t_test(self):
        m = self.m
        kvars = self.kvars
        # handle extra parameter of NegativeBinomial
        extra = getattr(self, 'k_extra', 0)
        t_unreg = self.res_unreg.t_test(np.eye(len(self.res_unreg.params)))
        t_reg = self.res_reg.t_test(np.eye(kvars + extra))
        assert_almost_equal(t_unreg.effect[:m], t_reg.effect[:m], DECIMAL_3)
        assert_almost_equal(t_unreg.sd[:m], t_reg.sd[:m], DECIMAL_3)
        assert_almost_equal(np.nan, t_reg.sd[m])
        assert_allclose(t_unreg.tvalue[:m], t_reg.tvalue[:m], atol=3e-3)
        assert_almost_equal(np.nan, t_reg.tvalue[m])

    def test_f_test(self):
        m = self.m
        kvars = self.kvars
        # handle extra parameter of NegativeBinomial
        extra = getattr(self, 'k_extra', 0)
        f_unreg = self.res_unreg.f_test(np.eye(len(self.res_unreg.params))[:m])
        f_reg = self.res_reg.f_test(np.eye(kvars + extra)[:m])
        assert_allclose(f_unreg.fvalue, f_reg.fvalue, rtol=3e-5, atol=1e-3)
        assert_almost_equal(f_unreg.pvalue, f_reg.pvalue, DECIMAL_3)

    def test_bad_r_matrix(self):
        kvars = self.kvars
        assert_raises(ValueError, self.res_reg.f_test, np.eye(kvars) )


class TestPoissonL1Compatability(CheckL1Compatability):

    @classmethod
    def setup_class(cls):
        cls.kvars = 10 # Number of variables
        cls.m = 7 # Number of unregularized parameters
        rand_data = sm2.datasets.randhie.load()
        rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
        rand_exog = add_constant(rand_exog, prepend=True)
        # Drop some columns and do an unregularized fit
        exog_no_PSI = rand_exog[:, :cls.m]
        mod_unreg = Poisson(rand_data.endog, exog_no_PSI)
        cls.res_unreg = mod_unreg.fit(method="newton", disp=False)
        # Do a regularized fit with alpha, effectively dropping the last column
        alpha = 10 * len(rand_data.endog) * np.ones(cls.kvars)
        alpha[:cls.m] = 0
        cls.res_reg = Poisson(rand_data.endog, rand_exog).fit_regularized(
            method='l1', alpha=alpha, disp=False, acc=1e-10, maxiter=2000,
            trim_mode='auto')


class TestNegativeBinomialL1Compatability(CheckL1Compatability):

    @classmethod
    def setup_class(cls):
        cls.kvars = 10 # Number of variables
        cls.m = 7 # Number of unregularized parameters
        rand_data = sm2.datasets.randhie.load()
        rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
        rand_exog_st = (rand_exog - rand_exog.mean(0)) / rand_exog.std(0)
        rand_exog = add_constant(rand_exog_st, prepend=True)
        # Drop some columns and do an unregularized fit
        exog_no_PSI = rand_exog[:, :cls.m]
        mod_unreg = NegativeBinomial(rand_data.endog, exog_no_PSI)
        cls.res_unreg = mod_unreg.fit(method="newton", disp=False)
        # Do a regularized fit with alpha, effectively dropping the last column
        alpha = 10 * len(rand_data.endog) * np.ones(cls.kvars + 1)
        alpha[:cls.m] = 0
        alpha[-1] = 0  # don't penalize alpha

        mod_reg = NegativeBinomial(rand_data.endog, rand_exog)
        cls.res_reg = mod_reg.fit_regularized(
            method='l1', alpha=alpha, disp=False, acc=1e-10, maxiter=2000,
            trim_mode='auto')
        cls.k_extra = 1  # 1 extra parameter in nb2


class TestNegativeBinomialGeoL1Compatability(CheckL1Compatability):

    @classmethod
    def setup_class(cls):
        cls.kvars = 10 # Number of variables
        cls.m = 7 # Number of unregularized parameters
        rand_data = sm2.datasets.randhie.load()
        rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
        rand_exog = add_constant(rand_exog, prepend=True)
        # Drop some columns and do an unregularized fit
        exog_no_PSI = rand_exog[:, :cls.m]
        mod_unreg = NegativeBinomial(rand_data.endog, exog_no_PSI,
                                     loglike_method='geometric')
        cls.res_unreg = mod_unreg.fit(method="newton", disp=False)
        # Do a regularized fit with alpha, effectively dropping the last columns
        alpha = 10 * len(rand_data.endog) * np.ones(cls.kvars)
        alpha[:cls.m] = 0
        mod_reg = NegativeBinomial(rand_data.endog, rand_exog,
                                   loglike_method='geometric')
        cls.res_reg = mod_reg.fit_regularized(
            method='l1', alpha=alpha, disp=False, acc=1e-10, maxiter=2000,
            trim_mode='auto')

        assert_equal(mod_reg.loglike_method, 'geometric')


class TestLogitL1Compatability(CheckL1Compatability):

    @classmethod
    def setup_class(cls):
        cls.kvars = 4 # Number of variables
        cls.m = 3 # Number of unregularized parameters
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=True)
        # Do a regularized fit with alpha, effectively dropping the last column
        alpha = np.array([0, 0, 0, 10])
        cls.res_reg = Logit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=alpha, disp=0, acc=1e-15, maxiter=2000,
            trim_mode='auto')
        # Actually drop the last columnand do an unregularized fit
        exog_no_PSI = data.exog[:, :cls.m]
        cls.res_unreg = Logit(data.endog, exog_no_PSI).fit(disp=0, tol=1e-15)


class TestMNLogitL1Compatability(CheckL1Compatability):

    @classmethod
    def setup_class(cls):
        cls.kvars = 4 # Number of variables
        cls.m = 3 # Number of unregularized parameters
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=True)
        alpha = np.array([0, 0, 0, 10])
        cls.res_reg = MNLogit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=alpha, disp=0, acc=1e-15, maxiter=2000,
            trim_mode='auto')
        # Actually drop the last columnand do an unregularized fit
        exog_no_PSI = data.exog[:, :cls.m]
        cls.res_unreg = MNLogit(data.endog, exog_no_PSI).fit(
            disp=0, tol=1e-15, method='bfgs', maxiter=1000)

    def test_t_test(self):
        m = self.m
        kvars = self.kvars
        t_unreg = self.res_unreg.t_test(np.eye(m))
        t_reg = self.res_reg.t_test(np.eye(kvars))
        assert_almost_equal(t_unreg.effect, t_reg.effect[:m], DECIMAL_3)
        assert_almost_equal(t_unreg.sd, t_reg.sd[:m], DECIMAL_3)
        assert_almost_equal(np.nan, t_reg.sd[m])
        assert_almost_equal(t_unreg.tvalue, t_reg.tvalue[:m, :m], DECIMAL_3)

    @pytest.skip("Skipped test_f_test for MNLogit")
    def test_f_test(self):
        pass


class TestProbitL1Compatability(CheckL1Compatability):

    @classmethod
    def setup_class(cls):
        cls.kvars = 4 # Number of variables
        cls.m = 3 # Number of unregularized parameters
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=True)
        alpha = np.array([0, 0, 0, 10])
        cls.res_reg = Probit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=alpha, disp=0, acc=1e-15, maxiter=2000,
            trim_mode='auto')
        # Actually drop the last columnand do an unregularized fit
        exog_no_PSI = data.exog[:, :cls.m]
        cls.res_unreg = Probit(data.endog, exog_no_PSI).fit(disp=0, tol=1e-15)


class CompareL1(object):
    """
    For checking results for l1 regularization.
    Assumes self.res1 and self.res2 are two legitimate models to be compared.
    """
    def test_basic_results(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)
        assert_almost_equal(self.res1.cov_params(), self.res2.cov_params(),
                            DECIMAL_4)
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int(),
                            DECIMAL_4)
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)
        assert_almost_equal(self.res1.pred_table(), self.res2.pred_table(),
                            DECIMAL_4)
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_4)
        assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL_4)
        assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL_4)
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)

        assert_(self.res1.mle_retvals['converged'] is True)


class CompareL11D(CompareL1):
    """
    Check t and f tests.  This only works for 1-d results
    """
    def test_tests(self):
        restrictmat = np.eye(len(self.res1.params.ravel()))
        assert_almost_equal(self.res1.t_test(restrictmat).pvalue,
                            self.res2.t_test(restrictmat).pvalue, DECIMAL_4)
        assert_almost_equal(self.res1.f_test(restrictmat).pvalue,
                            self.res2.f_test(restrictmat).pvalue, DECIMAL_4)


class TestL1AlphaZeroLogit(CompareL11D):
    # Compares l1 model with alpha = 0 to the unregularized model.

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=True)
        cls.res1 = Logit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=0, disp=0, acc=1e-15, maxiter=1000,
                trim_mode='auto', auto_trim_tol=0.01)
        cls.res2 = Logit(data.endog, data.exog).fit(disp=0, tol=1e-15)

    def test_converged(self):
        res = self.res1.model.fit_regularized(
                method="l1", alpha=0, disp=0, acc=1e-15, maxiter=1,
                trim_mode='auto', auto_trim_tol=0.01)

        # see #2857
        assert_(res.mle_retvals['converged'] is False)


class TestL1AlphaZeroProbit(CompareL11D):
    # Compares l1 model with alpha = 0 to the unregularized model.

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=True)
        cls.res1 = Probit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=0, disp=0, acc=1e-15, maxiter=1000,
                trim_mode='auto', auto_trim_tol=0.01)
        cls.res2 = Probit(data.endog, data.exog).fit(disp=0, tol=1e-15)


class TestL1AlphaZeroMNLogit(CompareL1):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.anes96.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = MNLogit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=0, disp=0, acc=1e-15, maxiter=1000,
                trim_mode='auto', auto_trim_tol=0.01)
        cls.res2 = MNLogit(data.endog, data.exog).fit(disp=0, tol=1e-15,
                                                      method='bfgs',
                                                      maxiter=1000)


class TestLogitNewton(CheckBinaryResults, CheckMargEff):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = Logit(data.endog, data.exog).fit(method="newton", disp=0)
        res2 = Spector()
        res2.logit()
        cls.res2 = res2

    def test_resid_pearson(self):
        assert_almost_equal(self.res1.resid_pearson,
                            self.res2.resid_pearson, 5)

    def test_nodummy_exog1(self):
        me = self.res1.get_margeff(atexog={0 : 2.0, 2 : 1.})
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_atexog1, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_atexog1_se, DECIMAL_4)

    def test_nodummy_exog2(self):
        me = self.res1.get_margeff(atexog={1 : 21., 2 : 0}, at='mean')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_atexog2, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_atexog2_se, DECIMAL_4)

    def test_dummy_exog1(self):
        me = self.res1.get_margeff(atexog={0 : 2.0, 2 : 1.}, dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_atexog1, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_atexog1_se, DECIMAL_4)

    def test_dummy_exog2(self):
        me = self.res1.get_margeff(atexog={1 : 21., 2 : 0}, at='mean',
                dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_atexog2, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_atexog2_se, DECIMAL_4)


class TestLogitNewtonPrepend(CheckMargEff):
    # same as previous version but adjusted for add_constant prepend=True
    # bug #3695

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=True)
        cls.res1 = Logit(data.endog, data.exog).fit(method="newton", disp=0)
        res2 = Spector()
        res2.logit()
        cls.res2 = res2
        cls.slice = np.roll(np.arange(len(cls.res1.params)), 1) #.astype(int)

    def test_resid_pearson(self):
        assert_almost_equal(self.res1.resid_pearson,
                            self.res2.resid_pearson, 5)

    def test_nodummy_exog1(self):
        me = self.res1.get_margeff(atexog={1 : 2.0, 3 : 1.})
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_atexog1, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_atexog1_se, DECIMAL_4)

    def test_nodummy_exog2(self):
        me = self.res1.get_margeff(atexog={2 : 21., 3 : 0}, at='mean')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_atexog2, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_atexog2_se, DECIMAL_4)

    def test_dummy_exog1(self):
        me = self.res1.get_margeff(atexog={1 : 2.0, 3 : 1.}, dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_atexog1, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_atexog1_se, DECIMAL_4)

    def test_dummy_exog2(self):
        me = self.res1.get_margeff(atexog={2 : 21., 3 : 0}, at='mean',
                dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_atexog2, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_atexog2_se, DECIMAL_4)


class TestLogitBFGS(CheckBinaryResults, CheckMargEff):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.spector.load()
        data.exog = add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.logit()
        cls.res2 = res2
        cls.res1 = Logit(data.endog, data.exog).fit(method="bfgs", disp=0)


class TestPoissonNewton(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        cls.res1 = Poisson(data.endog, exog).fit(method='newton', disp=0)
        res2 = RandHIE()
        res2.poisson()
        cls.res2 = res2

    def test_margeff_overall(self):
        me = self.res1.get_margeff()
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_overall, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_overall_se, DECIMAL_4)

    def test_margeff_dummy_overall(self):
        me = self.res1.get_margeff(dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_overall, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_overall_se, DECIMAL_4)

    def test_resid(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, 2)

    def test_predict_prob(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        probs_res = np.loadtxt(os.path.join(cur_dir, "results",
                            "predict_prob_poisson.csv"), delimiter=",")

        # just check the first 100 obs. vs R to save memory
        probs = self.res1.predict_prob()[:100]
        assert_almost_equal(probs, probs_res, 8)


class TestNegativeBinomialNB2Newton(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        cls.res1 = NegativeBinomial(data.endog, exog, 'nb2').fit(method='newton', disp=0)
        res2 = RandHIE()
        res2.negativebinomial_nb2_bfgs()
        cls.res2 = res2

    def test_jac(self):
        pass

    #NOTE: The bse is much closer precitions to stata
    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_3)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_alpha(self):
        self.res1.bse # attaches alpha_std_err
        assert_almost_equal(self.res1.lnalpha, self.res2.lnalpha,
                            DECIMAL_4)
        assert_almost_equal(self.res1.lnalpha_std_err,
                            self.res2.lnalpha_std_err, DECIMAL_4)

    def test_conf_int(self):
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int,
                            DECIMAL_3)

    def test_zstat(self): # Low precision because Z vs. t
        assert_almost_equal(self.res1.pvalues[:-1], self.res2.pvalues,
                            DECIMAL_2)

    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues[:10],
                            self.res2.fittedvalues[:10], DECIMAL_3)

    def test_predict(self):
        assert_almost_equal(self.res1.predict()[:10],
                            np.exp(self.res2.fittedvalues[:10]), DECIMAL_3)

    def test_predict_xb(self):
        assert_almost_equal(self.res1.predict(linear=True)[:10],
                            self.res2.fittedvalues[:10], DECIMAL_3)

    def no_info(self):
        pass

    test_jac = no_info


class TestNegativeBinomialNB1Newton(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        cls.res1 = NegativeBinomial(data.endog, exog, 'nb1').fit(
                                                            method="newton",
                                                                 maxiter=100,
                                                                 disp=0)
        res2 = RandHIE()
        res2.negativebinomial_nb1_bfgs()
        cls.res2 = res2

    def test_zstat(self):
        assert_almost_equal(self.res1.tvalues, self.res2.z, DECIMAL_1)

    def test_lnalpha(self):
        self.res1.bse # attaches alpha_std_err
        assert_almost_equal(self.res1.lnalpha, self.res2.lnalpha, 3)
        assert_almost_equal(self.res1.lnalpha_std_err,
                            self.res2.lnalpha_std_err, DECIMAL_4)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_conf_int(self):
        # the bse for alpha is not high precision from the hessian
        # approximation
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int,
                            DECIMAL_2)

    def test_jac(self):
        pass

    def test_predict(self):
        pass

    def test_predict_xb(self):
        pass


class TestNegativeBinomialNB2BFGS(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        cls.res1 = NegativeBinomial(data.endog, exog, 'nb2').fit(
                                                method='bfgs', disp=0,
                                                maxiter=1000)
        res2 = RandHIE()
        res2.negativebinomial_nb2_bfgs()
        cls.res2 = res2

    def test_jac(self):
        pass

    #NOTE: The bse is much closer precitions to stata
    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_3)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_alpha(self):
        self.res1.bse # attaches alpha_std_err
        assert_almost_equal(self.res1.lnalpha, self.res2.lnalpha,
                            DECIMAL_4)
        assert_almost_equal(self.res1.lnalpha_std_err,
                            self.res2.lnalpha_std_err, DECIMAL_4)

    def test_conf_int(self):
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int,
                            DECIMAL_3)

    def test_zstat(self): # Low precision because Z vs. t
        assert_almost_equal(self.res1.pvalues[:-1], self.res2.pvalues,
                            DECIMAL_2)

    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues[:10],
                            self.res2.fittedvalues[:10], DECIMAL_3)

    def test_predict(self):
        assert_almost_equal(self.res1.predict()[:10],
                            np.exp(self.res2.fittedvalues[:10]), DECIMAL_3)

    def test_predict_xb(self):
        assert_almost_equal(self.res1.predict(linear=True)[:10],
                            self.res2.fittedvalues[:10], DECIMAL_3)

    def no_info(self):
        pass

    test_jac = no_info


class TestNegativeBinomialNB1BFGS(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        cls.res1 = NegativeBinomial(data.endog, exog, 'nb1').fit(method="bfgs",
                                                                 maxiter=100,
                                                                 disp=0)
        res2 = RandHIE()
        res2.negativebinomial_nb1_bfgs()
        cls.res2 = res2

    def test_zstat(self):
        assert_almost_equal(self.res1.tvalues, self.res2.z, DECIMAL_1)

    def test_lnalpha(self):
        self.res1.bse # attaches alpha_std_err
        assert_almost_equal(self.res1.lnalpha, self.res2.lnalpha, 3)
        assert_almost_equal(self.res1.lnalpha_std_err,
                            self.res2.lnalpha_std_err, DECIMAL_4)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_conf_int(self):
        # the bse for alpha is not high precision from the hessian
        # approximation
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int,
                            DECIMAL_2)

    def test_jac(self):
        pass

    def test_predict(self):
        pass

    def test_predict_xb(self):
        pass


class TestNegativeBinomialGeometricBFGS(CheckModelResults):
    # Cannot find another implementation of the geometric to cross-check results
    # we only test fitted values because geometric has fewer parameters
    # than nb1 and nb2
    # and we want to make sure that predict() np.dot(exog, params) works


    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        mod = NegativeBinomial(data.endog, exog, 'geometric')
        cls.res1 = mod.fit(method='bfgs', disp=0)
        res2 = RandHIE()
        res2.negativebinomial_geometric_bfgs()
        cls.res2 = res2

    # the following are regression tests, could be inherited instead

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL_3)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL_3)

    def test_conf_int(self):
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int,
                            DECIMAL_3)

    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues[:10],
                            self.res2.fittedvalues[:10], DECIMAL_3)

    def test_jac(self):
        pass

    def test_predict(self):
        assert_almost_equal(self.res1.predict()[:10],
                            np.exp(self.res2.fittedvalues[:10]), DECIMAL_3)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_3)

    def test_predict_xb(self):
        assert_almost_equal(self.res1.predict(linear=True)[:10],
                            self.res2.fittedvalues[:10], DECIMAL_3)

    def test_zstat(self): # Low precision because Z vs. t
        assert_almost_equal(self.res1.tvalues, self.res2.z, DECIMAL_1)

    def no_info(self):
        pass

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_1)

    def test_llr(self):
        assert_almost_equal(self.res1.llr, self.res2.llr, DECIMAL_2)

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_3)

    test_jac = no_info


class CheckMNLogitBaseZero(CheckModelResults):

    def test_margeff_overall(self):
        me = self.res1.get_margeff()
        assert_almost_equal(me.margeff, self.res2.margeff_dydx_overall, 6)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dydx_overall_se, 6)
        me_frame = me.summary_frame()
        eff = me_frame["dy/dx"].values.reshape(me.margeff.shape, order="F")
        assert_allclose(eff, me.margeff, rtol=1e-13)
        assert_equal(me_frame.shape, (np.size(me.margeff), 6))

    def test_margeff_mean(self):
        me = self.res1.get_margeff(at='mean')
        assert_almost_equal(me.margeff, self.res2.margeff_dydx_mean, 7)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dydx_mean_se, 7)

    def test_margeff_dummy(self):
        data = self.data
        vote = data.data['vote']
        exog = np.column_stack((data.exog, vote))
        exog = add_constant(exog, prepend=False)
        res = MNLogit(data.endog, exog).fit(method="newton", disp=0)
        me = res.get_margeff(dummy=True)
        assert_almost_equal(me.margeff, self.res2.margeff_dydx_dummy_overall,
                6)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dydx_dummy_overall_se, 6)
        me = res.get_margeff(dummy=True, method="eydx")
        assert_almost_equal(me.margeff, self.res2.margeff_eydx_dummy_overall,
                5)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_eydx_dummy_overall_se, 6)

    def test_j(self):
        assert_equal(self.res1.model.J, self.res2.J)

    def test_k(self):
        assert_equal(self.res1.model.K, self.res2.K)

    def test_endog_names(self):
        assert_equal(self.res1._get_endog_name(None,None)[1],
                     ['y=1', 'y=2', 'y=3', 'y=4', 'y=5', 'y=6'])

    def test_pred_table(self):
        # fitted results taken from gretl
        pred = [6, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 6, 0, 1, 6, 0, 0,
                1, 1, 6, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 6, 0, 0, 6, 6, 0, 0, 1,
                1, 6, 1, 6, 0, 0, 0, 1, 0, 1, 0, 0, 0, 6, 0, 0, 6, 0, 0, 0, 1,
                1, 0, 0, 6, 6, 6, 6, 1, 0, 5, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                6, 0, 6, 6, 1, 0, 1, 1, 6, 5, 1, 0, 0, 0, 5, 0, 0, 6, 0, 1, 0,
                0, 0, 0, 0, 1, 1, 0, 6, 6, 6, 6, 5, 0, 1, 1, 0, 1, 0, 6, 6, 0,
                0, 0, 6, 0, 0, 0, 6, 6, 0, 5, 1, 0, 0, 0, 0, 6, 0, 5, 6, 6, 0,
                0, 0, 0, 6, 1, 0, 0, 1, 0, 1, 6, 1, 1, 1, 1, 1, 0, 0, 0, 6, 0,
                5, 1, 0, 6, 6, 6, 0, 0, 0, 0, 1, 6, 6, 0, 0, 0, 1, 1, 5, 6, 0,
                6, 1, 0, 0, 1, 6, 0, 0, 1, 0, 6, 6, 0, 5, 6, 6, 0, 0, 6, 1, 0,
                6, 0, 1, 0, 1, 6, 0, 1, 1, 1, 6, 0, 5, 0, 0, 6, 1, 0, 6, 5, 5,
                0, 6, 1, 1, 1, 0, 0, 6, 0, 0, 5, 0, 0, 6, 6, 6, 6, 6, 0, 1, 0,
                0, 6, 6, 0, 0, 1, 6, 0, 0, 6, 1, 6, 1, 1, 1, 0, 1, 6, 5, 0, 0,
                1, 5, 0, 1, 6, 6, 1, 0, 0, 1, 6, 1, 5, 6, 1, 0, 0, 1, 1, 0, 6,
                1, 6, 0, 1, 1, 5, 6, 6, 5, 1, 1, 1, 0, 6, 1, 6, 1, 0, 1, 0, 0,
                1, 5, 0, 1, 1, 0, 5, 6, 0, 5, 1, 1, 6, 5, 0, 6, 0, 0, 0, 0, 0,
                0, 1, 6, 1, 0, 5, 1, 0, 0, 1, 6, 0, 0, 6, 6, 6, 0, 2, 1, 6, 5,
                6, 1, 1, 0, 5, 1, 1, 1, 6, 1, 6, 6, 5, 6, 0, 1, 0, 1, 6, 0, 6,
                1, 6, 0, 0, 6, 1, 0, 6, 1, 0, 0, 0, 0, 6, 6, 6, 6, 5, 6, 6, 0,
                0, 6, 1, 1, 6, 0, 0, 6, 6, 0, 6, 6, 0, 0, 6, 0, 0, 6, 6, 6, 1,
                0, 6, 0, 0, 0, 6, 1, 1, 0, 1, 5, 0, 0, 5, 0, 0, 0, 1, 1, 6, 1,
                0, 0, 0, 6, 6, 1, 1, 6, 5, 5, 0, 6, 6, 0, 1, 1, 0, 6, 6, 0, 6,
                5, 5, 6, 5, 1, 0, 6, 0, 6, 1, 0, 1, 6, 6, 6, 1, 0, 6, 0, 5, 6,
                6, 5, 0, 5, 1, 0, 6, 0, 6, 1, 5, 5, 0, 1, 5, 5, 2, 6, 6, 6, 5,
                0, 0, 1, 6, 1, 0, 1, 6, 1, 0, 0, 1, 5, 6, 6, 0, 0, 0, 5, 6, 6,
                6, 1, 5, 6, 1, 0, 0, 6, 5, 0, 1, 1, 1, 6, 6, 0, 1, 0, 0, 0, 5,
                0, 0, 6, 1, 6, 0, 6, 1, 5, 5, 6, 5, 0, 0, 0, 0, 1, 1, 0, 5, 5,
                0, 0, 0, 0, 1, 0, 6, 6, 1, 1, 6, 6, 0, 5, 5, 0, 0, 0, 6, 6, 1,
                6, 0, 0, 5, 0, 1, 6, 5, 6, 6, 5, 5, 6, 6, 1, 0, 1, 6, 6, 1, 6,
                0, 6, 0, 6, 5, 0, 6, 6, 0, 5, 6, 0, 6, 6, 5, 0, 1, 6, 6, 1, 0,
                1, 0, 6, 6, 1, 0, 6, 6, 6, 0, 1, 6, 0, 1, 5, 1, 1, 5, 6, 6, 0,
                1, 6, 6, 1, 5, 0, 5, 0, 6, 0, 1, 6, 1, 0, 6, 1, 6, 0, 6, 1, 0,
                0, 0, 6, 6, 0, 1, 1, 6, 6, 6, 1, 6, 0, 5, 6, 0, 5, 6, 6, 5, 5,
                5, 6, 0, 6, 0, 0, 0, 5, 0, 6, 1, 2, 6, 6, 6, 5, 1, 6, 0, 6, 0,
                0, 0, 0, 6, 5, 0, 5, 1, 6, 5, 1, 6, 5, 1, 1, 0, 0, 6, 1, 1, 5,
                6, 6, 0, 5, 2, 5, 5, 0, 5, 5, 5, 6, 5, 6, 6, 5, 2, 6, 5, 6, 0,
                0, 6, 5, 0, 6, 0, 0, 6, 6, 6, 0, 5, 1, 1, 6, 6, 5, 2, 1, 6, 5,
                6, 0, 6, 6, 1, 1, 5, 1, 6, 6, 6, 0, 0, 6, 1, 0, 5, 5, 1, 5, 6,
                1, 6, 0, 1, 6, 5, 0, 0, 6, 1, 5, 1, 0, 6, 0, 6, 6, 5, 5, 6, 6,
                6, 6, 2, 6, 6, 6, 5, 5, 5, 0, 1, 0, 0, 0, 6, 6, 1, 0, 6, 6, 6,
                6, 6, 1, 0, 6, 1, 5, 5, 6, 6, 6, 6, 6, 5, 6, 1, 6, 2, 5, 5, 6,
                5, 6, 6, 5, 6, 6, 5, 5, 6, 1, 5, 1, 6, 0, 2, 5, 0, 5, 0, 2, 1,
                6, 0, 0, 6, 6, 1, 6, 0, 5, 5, 6, 6, 1, 6, 6, 6, 5, 6, 6, 1, 6,
                5, 6, 1, 1, 0, 6, 6, 5, 1, 0, 0, 6, 6, 5, 6, 0, 1, 6, 0, 5, 6,
                5, 2, 5, 2, 0, 0, 1, 6, 6, 1, 5, 6, 6, 0, 6, 6, 6, 6, 6, 5]
        assert_array_equal(self.res1.predict().argmax(1), pred)

        # the rows should add up for pred table
        assert_array_equal(self.res1.pred_table().sum(0), np.bincount(pred))

        # note this is just a regression test, gretl doesn't have a prediction
        # table
        pred = [[ 126.,   41.,    2.,    0.,    0.,   12.,   19.],
                [  77.,   73.,    3.,    0.,    0.,   15.,   12.],
                [  37.,   43.,    2.,    0.,    0.,   19.,    7.],
                [  12.,    9.,    1.,    0.,    0.,    9.,    6.],
                [  19.,   10.,    2.,    0.,    0.,   20.,   43.],
                [  22.,   25.,    1.,    0.,    0.,   31.,   71.],
                [   9.,    7.,    1.,    0.,    0.,   18.,  140.]]
        assert_array_equal(self.res1.pred_table(), pred)

    def test_resid(self):
        assert_array_equal(self.res1.resid_misclassified, self.res2.resid)


class TestMNLogitNewtonBaseZero(CheckMNLogitBaseZero):
    @classmethod
    def setup_class(cls):

        data = sm2.datasets.anes96.load()
        cls.data = data
        exog = data.exog
        exog = add_constant(exog, prepend=False)
        cls.res1 = MNLogit(data.endog, exog).fit(method="newton", disp=0)
        res2 = Anes()
        res2.mnlogit_basezero()
        cls.res2 = res2


class TestMNLogitLBFGSBaseZero(CheckMNLogitBaseZero):
    @classmethod
    def setup_class(cls):
        data = sm2.datasets.anes96.load()
        cls.data = data
        exog = data.exog
        exog = add_constant(exog, prepend=False)
        mymodel = MNLogit(data.endog, exog)
        cls.res1 = mymodel.fit(method="lbfgs", disp=0, maxiter=50000,
                #m=12, pgtol=1e-7, factr=1e3, # 5 failures
                #m=20, pgtol=1e-8, factr=1e2, # 3 failures
                #m=30, pgtol=1e-9, factr=1e1, # 1 failure
                m=40, pgtol=1e-10, factr=5e0,
                loglike_and_score=mymodel.loglike_and_score)
        res2 = Anes()
        res2.mnlogit_basezero()
        cls.res2 = res2


def test_perfect_prediction():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    iris_dir = os.path.join(cur_dir, '..', '..', 'genmod', 'tests', 'results')
    iris_dir = os.path.abspath(iris_dir)
    iris = np.genfromtxt(os.path.join(iris_dir, 'iris.csv'), delimiter=",",
                            skip_header=1)
    y = iris[:,-1]
    X = iris[:,:-1]
    X = X[y != 2]
    y = y[y != 2]
    X = add_constant(X, prepend=True)
    mod = Logit(y,X)
    assert_raises(PerfectSeparationError, mod.fit, maxiter=1000)
    #turn off raise PerfectSeparationError
    mod.raise_on_perfect_prediction = False
    # this will raise if you set maxiter high enough with a singular matrix
    # this is not thread-safe
    with tm.assert_produces_warning():
        warnings.simplefilter('always')
        mod.fit(disp=False, maxiter=50)  # should not raise but does warn


def test_poisson_predict():
    #GH: 175, make sure poisson predict works without offset and exposure
    data = sm2.datasets.randhie.load()
    exog = add_constant(data.exog, prepend=True)
    res = Poisson(data.endog, exog).fit(method='newton', disp=0)
    pred1 = res.predict()
    pred2 = res.predict(exog)
    assert_almost_equal(pred1, pred2)
    #exta options
    pred3 = res.predict(exog, offset=0, exposure=1)
    assert_almost_equal(pred1, pred3)
    pred3 = res.predict(exog, offset=0, exposure=2)
    assert_almost_equal(2*pred1, pred3)
    pred3 = res.predict(exog, offset=np.log(2), exposure=1)
    assert_almost_equal(2*pred1, pred3)


def test_poisson_newton():
    #GH: 24, Newton doesn't work well sometimes
    nobs = 10000
    np.random.seed(987689)
    x = np.random.randn(nobs, 3)
    x = add_constant(x, prepend=True)
    y_count = np.random.poisson(np.exp(x.sum(1)))
    mod = Poisson(y_count, x)
    # this is not thread-safe
    with tm.assert_produces_warning():
        warnings.simplefilter('always')
        res = mod.fit(start_params=-np.ones(4), method='newton', disp=0)
    assert_(not res.mle_retvals['converged'])


def test_issue_339():
    # make sure MNLogit summary works for J != K.
    data = sm2.datasets.anes96.load()
    exog = data.exog
    # leave out last exog column
    exog = exog[:,:-1]
    exog = add_constant(exog, prepend=True)
    res1 = MNLogit(data.endog, exog).fit(method="newton", disp=0)
    # strip the header from the test
    smry = "\n".join(res1.summary().as_text().split('\n')[9:])
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_case_file = os.path.join(cur_dir, 'results', 'mn_logit_summary.txt')
    test_case = open(test_case_file, 'r').read()
    np.testing.assert_equal(smry, test_case[:-1])
    # smoke test for summary2
    res1.summary2()  # see #3651


def test_issue_341():
    data = sm2.datasets.anes96.load()
    exog = data.exog
    # leave out last exog column
    exog = exog[:,:-1]
    exog = add_constant(exog, prepend=True)
    res1 = MNLogit(data.endog, exog).fit(method="newton", disp=0)
    x = exog[0]
    np.testing.assert_equal(res1.predict(x).shape, (1,7))
    np.testing.assert_equal(res1.predict(x[None]).shape, (1,7))


def test_iscount():
    X = np.random.random((50, 10))
    X[:,2] = np.random.randint(1, 10, size=50)
    X[:,6] = np.random.randint(1, 10, size=50)
    X[:,4] = np.random.randint(0, 2, size=50)
    X[:,1] = np.random.randint(-10, 10, size=50) # not integers
    count_ind = _iscount(X)
    assert_equal(count_ind, [2, 6])


def test_isdummy():
    X = np.random.random((50, 10))
    X[:,2] = np.random.randint(1, 10, size=50)
    X[:,6] = np.random.randint(0, 2, size=50)
    X[:,4] = np.random.randint(0, 2, size=50)
    X[:,1] = np.random.randint(-10, 10, size=50) # not integers
    count_ind = _isdummy(X)
    assert_equal(count_ind, [4, 6])


def test_non_binary():
    y = [1, 2, 1, 2, 1, 2]
    X = np.random.randn(6, 2)
    np.testing.assert_raises(ValueError, Logit, y, X)


def test_mnlogit_factor():
    dta = sm2.datasets.anes96.load_pandas()
    dta['endog'] = dta.endog.replace(dict(zip(range(7), 'ABCDEFG')))
    exog = add_constant(dta.exog, prepend=True)
    mod = MNLogit(dta.endog, exog)
    res = mod.fit(disp=0)
    # smoke tests
    params = res.params
    summary = res.summary()
    predicted = res.predict(exog.iloc[:5, :])

    # with patsy
    mod = MNLogit.from_formula('PID ~ ' + ' + '.join(dta.exog.columns), dta.data)
    res2 = mod.fit(disp=0)
    params_f = res2.params
    summary = res2.summary()
    assert_allclose(params_f, params, rtol=1e-10)
    predicted_f = res2.predict(dta.exog.iloc[:5, :])
    assert_allclose(predicted_f, predicted, rtol=1e-10)


def test_formula_missing_exposure():
    # see 2083
    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, np.nan],
         'constant': [1] * 4, 'exposure' : np.random.uniform(size=4),
         'x': [1, 3, 2, 1.5]}
    df = pd.DataFrame(d)

    # should work
    mod1 = Poisson.from_formula('Foo ~ Bar', data=df, exposure=df['exposure'])
    assert_(type(mod1.exposure) is np.ndarray, msg='Exposure is not ndarray')

    # make sure this raises
    exposure = pd.Series(np.random.uniform(size=5))
    df.loc[3, 'Bar'] = 4   # nan not relevant for ValueError for shape mismatch
    assert_raises(ValueError, Poisson, df.Foo, df[['constant', 'Bar']],
                  exposure=exposure)


def test_predict_with_exposure():
    # Case where CountModel.predict is called with exog = None and exposure
    # or offset not-None
    # See 3565

    # Setup copied from test_formula_missing_exposure
    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, 4],
         'constant': [1] * 4, 'exposure' : [np.exp(1)]*4,
         'x': [1, 3, 2, 1.5]}
    df = pd.DataFrame(d)

    mod1 = CountModel.from_formula('Foo ~ Bar', data=df,
                                   exposure=df['exposure'])

    params = np.array([1, .4])
    pred = mod1.predict(params, linear=True)
    # No exposure is passed, so default to using mod1.exposure, which
    # should have been logged
    X = df[['constant', 'Bar']].values # mod1.exog
    expected = np.dot(X, params) + 1
    assert_allclose(pred, expected)
    # The above should have passed without the current patch.  The next
    # test would fail under the old code

    pred2 = mod1.predict(params, exposure=[np.exp(2)]*4, linear=True)
    expected2 = expected + 1
    assert_allclose(pred2, expected2)


def test_binary_pred_table_zeros():
    # see 2968
    nobs = 10
    y = np.zeros(nobs)
    y[[1,3]] = 1

    res = Logit(y, np.ones(nobs)).fit(disp=0)
    expected = np.array([[ 8.,  0.], [ 2.,  0.]])
    assert_equal(res.pred_table(), expected)

    res = MNLogit(y, np.ones(nobs)).fit(disp=0)
    expected = np.array([[ 8.,  0.], [ 2.,  0.]])
    assert_equal(res.pred_table(), expected)


class TestGeneralizedPoisson_p2(object):
    # Test Generalized Poisson model

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        data.exog = add_constant(data.exog, prepend=False)
        mod = GeneralizedPoisson(data.endog, data.exog, p=2)
        cls.res1 = mod.fit(method='newton')
        res2 = RandHIE()
        res2.generalizedpoisson_gp2()
        cls.res2 = res2

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=1e-5)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-5)

    def test_alpha(self):
        assert_allclose(self.res1.lnalpha, self.res2.lnalpha)
        assert_allclose(self.res1.lnalpha_std_err,
                            self.res2.lnalpha_std_err, atol=1e-5)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int,
                        atol=1e-3)

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic)

    def test_bic(self):
        assert_allclose(self.res1.bic, self.res2.bic)

    def test_df(self):
        assert_equal(self.res1.df_model, self.res2.df_model)

    def test_llf(self):
        assert_allclose(self.res1.llf, self.res2.llf)

    def test_wald(self):
        result = self.res1.wald_test(np.eye(len(self.res1.params))[:-2])
        assert_allclose(result.statistic, self.res2.wald_statistic)
        assert_allclose(result.pvalue, self.res2.wald_pvalue, atol=1e-15)

    def test_t(self):
        unit_matrix = np.identity(self.res1.params.size)
        t_test = self.res1.t_test(unit_matrix)
        assert_allclose(self.res1.tvalues, t_test.tvalue)


class TestGeneralizedPoisson_transparams(object):
    # Test Generalized Poisson model

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = GeneralizedPoisson(data.endog, data.exog, p=2).fit(
            method='newton', use_transparams=True)
        res2 = RandHIE()
        res2.generalizedpoisson_gp2()
        cls.res2 = res2

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=1e-5)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-5)

    def test_alpha(self):
        assert_allclose(self.res1.lnalpha, self.res2.lnalpha)
        assert_allclose(self.res1.lnalpha_std_err,
                        self.res2.lnalpha_std_err, atol=1e-5)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int,
                        atol=1e-3)

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic)

    def test_bic(self):
        assert_allclose(self.res1.bic, self.res2.bic)

    def test_df(self):
        assert_equal(self.res1.df_model, self.res2.df_model)

    def test_llf(self):
        assert_allclose(self.res1.llf, self.res2.llf)


class TestGeneralizedPoisson_p1(object):
    # Test Generalized Poisson model

    @classmethod
    def setup_class(cls):
        cls.data = sm2.datasets.randhie.load()
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        cls.res1 = GeneralizedPoisson(
            cls.data.endog, cls.data.exog, p=1).fit(method='newton')

    def test_llf(self):
        poisson_llf = Poisson(
            self.data.endog, self.data.exog).loglike(
            self.res1.params[:-1])
        genpoisson_llf = GeneralizedPoisson(
            self.data.endog, self.data.exog, p=1).loglike(
            list(self.res1.params[:-1]) + [0])
        assert_allclose(genpoisson_llf, poisson_llf)

    def test_score(self):
        poisson_score = Poisson(
            self.data.endog, self.data.exog).score(
            self.res1.params[:-1])
        genpoisson_score = GeneralizedPoisson(
            self.data.endog, self.data.exog, p=1).score(
            list(self.res1.params[:-1]) + [0])
        assert_allclose(genpoisson_score[:-1], poisson_score, atol=1e-9)

    def test_hessian(self):
        poisson_score = Poisson(
            self.data.endog, self.data.exog).hessian(
            self.res1.params[:-1])
        genpoisson_score = GeneralizedPoisson(
            self.data.endog, self.data.exog, p=1).hessian(
            list(self.res1.params[:-1]) + [0])
        assert_allclose(genpoisson_score[:-1,:-1], poisson_score, atol=1e-10)

    def test_t(self):
        unit_matrix = np.identity(self.res1.params.size)
        t_test = self.res1.t_test(unit_matrix)
        assert_allclose(self.res1.tvalues, t_test.tvalue)

    def test_fit_regularized(self):
        model = self.res1.model

        # don't penalize constant and dispersion parameter
        alpha = np.ones(len(self.res1.params))
        alpha[-2:] = 0
        # the first prints currently a warning, irrelevant here
        res_reg1 = model.fit_regularized(alpha=alpha*0.01, disp=0)
        res_reg2 = model.fit_regularized(alpha=alpha*100, disp=0)
        res_reg3 = model.fit_regularized(alpha=alpha*1000, disp=0)

        assert_allclose(res_reg1.params, self.res1.params, atol=5e-5)
        assert_allclose(res_reg1.bse, self.res1.bse, atol=1e-5)

        # check shrinkage, regression numbers
        assert_allclose((self.res1.params[:-2]**2).mean(), 0.016580955543320779)
        assert_allclose((res_reg1.params[:-2]**2).mean(), 0.016580734975068664)
        assert_allclose((res_reg2.params[:-2]**2).mean(), 0.010672558641545994)
        assert_allclose((res_reg3.params[:-2]**2).mean(), 0.00035544919793048415)

    def test_init_kwds(self):
        kwds = self.res1.model._get_init_kwds()
        assert_('p' in kwds)
        assert_equal(kwds['p'], 1)


class TestGeneralizedPoisson_underdispersion(object):

    @classmethod
    def setup_class(cls):
        cls.expected_params = [1, -0.5, -0.05]
        np.random.seed(1234)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = np.exp(exog.dot(cls.expected_params[:-1]))
        cls.endog = distributions.genpoisson_p.rvs(mu_true,
            cls.expected_params[-1], 1, size=len(mu_true))
        model_gp = GeneralizedPoisson(cls.endog, exog, p=1)
        cls.res = model_gp.fit(method='nm', xtol=1e-6, maxiter=5000,
                               maxfun=5000)

    def test_basic(self):
        res = self.res
        endog = res.model.endog
        # check random data generation, regression test
        assert_allclose(endog.mean(), 1.42, rtol=1e-3)
        assert_allclose(endog.var(), 1.2836, rtol=1e-3)

        # check estimation
        assert_allclose(res.params, self.expected_params, atol=0.07, rtol=0.1)
        assert_(res.mle_retvals['converged'] is True)
        assert_allclose(res.mle_retvals['fopt'], 1.418753161722015, rtol=0.01)

    def test_newton(self):
        # check newton optimization with start_params
        res = self.res
        res2 = res.model.fit(start_params=res.params, method='newton')
        assert_allclose(res.model.score(res.params),
                        np.zeros(len(res2.params)), atol=0.01)
        assert_allclose(res.model.score(res2.params),
                        np.zeros(len(res2.params)), atol=1e-10)
        assert_allclose(res.params, res2.params, atol=1e-4)

    def test_mean_var(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(),
                        atol=1e-1, rtol=1e-1)

        assert_allclose(
            self.res.predict().mean() * self.res._dispersion_factor.mean(),
            self.endog.var(), atol=2e-1, rtol=2e-1)

    def test_predict_prob(self):
        res = self.res
        endog = res.model.endog
        freq = np.bincount(endog.astype(int))

        pr = res.predict(which='prob')
        pr2 = distributions.genpoisson_p.pmf(np.arange(6)[:, None],
                                        res.predict(), res.params[-1], 1).T
        assert_allclose(pr, pr2, rtol=1e-10, atol=1e-10)

        from scipy import stats
        chi2 = stats.chisquare(freq, pr.sum(0))
        assert_allclose(chi2[:], (0.64628806058715882, 0.98578597726324468),
                        rtol=0.01)


class TestNegativeBinomialPNB2Newton(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        mod = NegativeBinomialP(data.endog, exog, p=2)
        cls.res1 = mod.fit(method='newton', disp=0)
        res2 = RandHIE()
        res2.negativebinomial_nb2_bfgs()
        cls.res2 = res2

    #NOTE: The bse is much closer precitions to stata
    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse,
                        atol=1e-3, rtol=1e-3)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params,
                        atol=1e-7)

    def test_alpha(self):
        self.res1.bse # attaches alpha_std_err
        assert_allclose(self.res1.lnalpha, self.res2.lnalpha)
        assert_allclose(self.res1.lnalpha_std_err,
                        self.res2.lnalpha_std_err,
                        atol=1e-7)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int,
                        atol=1e-3, rtol=1e-3)

    def test_zstat(self): # Low precision because Z vs. t
        assert_allclose(self.res1.pvalues[:-1], self.res2.pvalues,
                        atol=5e-3, rtol=5e-3)

    def test_fittedvalues(self):
        assert_allclose(self.res1.fittedvalues[:10],
                        self.res2.fittedvalues[:10])

    def test_predict(self):
        assert_allclose(self.res1.predict()[:10],
                        np.exp(self.res2.fittedvalues[:10]))

    def test_predict_xb(self):
        assert_allclose(self.res1.predict(which='linear')[:10],
                        self.res2.fittedvalues[:10])


class TestNegativeBinomialPNB1Newton(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        mod = NegativeBinomialP(data.endog, exog, p=1)
        cls.res1 = mod.fit(method="newton", maxiter=100, disp=0,
                           use_transparams=True)
        res2 = RandHIE()
        res2.negativebinomial_nb1_bfgs()
        cls.res2 = res2

    def test_zstat(self):
        assert_allclose(self.res1.tvalues, self.res2.z,
                        atol=5e-3, rtol=5e-3)

    def test_lnalpha(self):
        self.res1.bse # attaches alpha_std_err
        assert_allclose(self.res1.lnalpha, self.res2.lnalpha)
        assert_allclose(self.res1.lnalpha_std_err,
                            self.res2.lnalpha_std_err)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params)

    def test_conf_int(self):
        # the bse for alpha is not high precision from the hessian
        # approximation
        assert_allclose(self.res1.conf_int(), self.res2.conf_int,
                        atol=1e-3, rtol=1e-3)

    def test_predict(self):
        assert_allclose(self.res1.predict()[:10],
                        np.exp(self.res2.fittedvalues[:10]),
                        atol=1e-3, rtol=1e-3)

    def test_predict_xb(self):
        assert_allclose(self.res1.predict(which='linear')[:10],
                        self.res2.fittedvalues[:10],
                        atol=1e-3, rtol=1e-3)


class TestNegativeBinomialPNB2BFGS(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        cls.res1 = NegativeBinomialP(data.endog, exog, p=2).fit(
                                                method='bfgs', disp=0,
                                                maxiter=1000,
                                                use_transparams=True)
        res2 = RandHIE()
        res2.negativebinomial_nb2_bfgs()
        cls.res2 = res2

    #NOTE: The bse is much closer precitions to stata
    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse,
                        atol=1e-3, rtol=1e-3)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params,
                        atol=1e-3, rtol=1e-3)

    def test_alpha(self):
        self.res1.bse # attaches alpha_std_err
        assert_allclose(self.res1.lnalpha, self.res2.lnalpha,
                        atol=1e-5, rtol=1e-5)
        assert_allclose(self.res1.lnalpha_std_err,
                        self.res2.lnalpha_std_err,
                        atol=1e-5, rtol=1e-5)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int,
                        atol=1e-3, rtol=1e-3)

    def test_zstat(self): # Low precision because Z vs. t
        assert_allclose(self.res1.pvalues[:-1], self.res2.pvalues,
                        atol=5e-3, rtol=5e-3)

    def test_fittedvalues(self):
        assert_allclose(self.res1.fittedvalues[:10],
                        self.res2.fittedvalues[:10],
                        atol=1e-4, rtol=1e-4)

    def test_predict(self):
        assert_allclose(self.res1.predict()[:10],
                        np.exp(self.res2.fittedvalues[:10]),
                        atol=1e-3, rtol=1e-3)

    def test_predict_xb(self):
        assert_allclose(self.res1.predict(which='linear')[:10],
                        self.res2.fittedvalues[:10],
                        atol=1e-3, rtol=1e-3)


class TestNegativeBinomialPNB1BFGS(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = sm2.datasets.randhie.load()
        exog = add_constant(data.exog, prepend=False)
        cls.res1 = NegativeBinomialP(data.endog, exog, p=1).fit(method="bfgs",
                                                                 maxiter=100,
                                                                 disp=0)
        res2 = RandHIE()
        res2.negativebinomial_nb1_bfgs()
        cls.res2 = res2

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse,
                        atol=5e-3, rtol=5e-3)

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic,
                        atol=0.5, rtol=0.5)

    def test_bic(self):
        assert_allclose(self.res1.bic, self.res2.bic,
                        atol=0.5, rtol=0.5)

    def test_llf(self):
        assert_allclose(self.res1.llf, self.res2.llf,
                        atol=1e-3, rtol=1e-3)

    def test_llr(self):
        assert_allclose(self.res1.llf, self.res2.llf,
                        atol=1e-3, rtol=1e-3)

    def test_zstat(self):
        assert_allclose(self.res1.tvalues, self.res2.z,
                        atol=0.5, rtol=0.5)

    def test_lnalpha(self):
        assert_allclose(self.res1.lnalpha, self.res2.lnalpha,
                        atol=1e-3, rtol=1e-3)
        assert_allclose(self.res1.lnalpha_std_err,
                        self.res2.lnalpha_std_err,
                        atol=1e-3, rtol=1e-3)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params,
                        atol=5e-2, rtol=5e-2)

    def test_conf_int(self):
        # the bse for alpha is not high precision from the hessian
        # approximation
        assert_allclose(self.res1.conf_int(), self.res2.conf_int,
                        atol=5e-2, rtol=5e-2)

    def test_predict(self):
        assert_allclose(self.res1.predict()[:10],
                        np.exp(self.res2.fittedvalues[:10]),
                        atol=5e-3, rtol=5e-3)

    def test_predict_xb(self):
        assert_allclose(self.res1.predict(which='linear')[:10],
                        self.res2.fittedvalues[:10],
                        atol=5e-3, rtol=5e-3)

    def test_init_kwds(self):
        kwds = self.res1.model._get_init_kwds()
        assert_('p' in kwds)
        assert_equal(kwds['p'], 1)


class TestNegativeBinomialPL1Compatability(CheckL1Compatability):
    @classmethod
    def setup_class(cls):
        cls.kvars = 10 # Number of variables
        cls.m = 7 # Number of unregularized parameters
        rand_data = sm2.datasets.randhie.load()
        rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
        rand_exog_st = (rand_exog - rand_exog.mean(0)) / rand_exog.std(0)
        rand_exog = add_constant(rand_exog_st, prepend=True)
        # Drop some columns and do an unregularized fit
        exog_no_PSI = rand_exog[:, :cls.m]
        mod_unreg = NegativeBinomialP(rand_data.endog, exog_no_PSI)
        cls.res_unreg = mod_unreg.fit(method="newton", disp=False)
        # Do a regularized fit with alpha, effectively dropping the last column
        alpha = 10 * len(rand_data.endog) * np.ones(cls.kvars + 1)
        alpha[:cls.m] = 0
        alpha[-1] = 0  # don't penalize alpha

        mod_reg = NegativeBinomialP(rand_data.endog, rand_exog)
        cls.res_reg = mod_reg.fit_regularized(
            method='l1', alpha=alpha, disp=False, acc=1e-10, maxiter=2000,
            trim_mode='auto')
        cls.k_extra = 1  # 1 extra parameter in nb2


class  TestNegativeBinomialPPredictProb(object):

    def test_predict_prob_p1(self):
        expected_params = [1, -0.5]
        np.random.seed(1234)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = np.exp(exog.dot(expected_params))
        alpha = 0.05
        size = 1. / alpha * mu_true
        prob = size / (size + mu_true)
        endog = nbinom.rvs(size, prob, size=len(mu_true))

        res = NegativeBinomialP(endog, exog).fit()

        mu = res.predict()
        size = 1. / alpha * mu
        prob = size / (size + mu)

        probs = res.predict(which='prob')
        assert_allclose(probs,
            nbinom.pmf(np.arange(8)[:,None], size, prob).T,
            atol=1e-2, rtol=1e-2)

        probs_ex = res.predict(exog=exog[[0, -1]], which='prob')
        assert_allclose(probs_ex, probs[[0, -1]], rtol=1e-10, atol=1e-15)

    def test_predict_prob_p2(self):
        expected_params = [1, -0.5]
        np.random.seed(1234)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = np.exp(exog.dot(expected_params))
        alpha = 0.05
        size = 1. / alpha
        prob = size / (size + mu_true)
        endog = nbinom.rvs(size, prob, size=len(mu_true))

        res = NegativeBinomialP(endog, exog, p=2).fit()

        mu = res.predict()
        size = 1. / alpha
        prob = size / (size + mu)

        assert_allclose(res.predict(which='prob'),
            nbinom.pmf(np.arange(8)[:,None], size, prob).T,
            atol=1e-2, rtol=1e-2)


class CheckNull(object):

    @classmethod
    def _get_data(cls):
        x = np.array([ 20.,  25.,  30.,  35.,  40.,  45.,  50.])
        nobs = len(x)
        exog = np.column_stack((np.ones(nobs), x))
        endog = np.array([ 469, 5516, 6854, 6837, 5952, 4066, 3242])
        return endog, exog

    def test_llnull(self):
        res = self.model.fit(start_params=self.start_params)
        res._results._attach_nullmodel = True
        llf0 = res.llnull
        res_null0 = res.res_null
        assert_allclose(llf0, res_null0.llf, rtol=1e-6)

        res_null1 = self.res_null
        assert_allclose(llf0, res_null1.llf, rtol=1e-6)
        # Note default convergence tolerance doesn't get lower rtol
        # from different starting values (using bfgs)
        assert_allclose(res_null0.params, res_null1.params, rtol=5e-5)


class TestPoissonNull(CheckNull):

    @classmethod
    def setup_class(cls):
        endog, exog = cls._get_data()
        cls.model = Poisson(endog, exog)
        cls.res_null = Poisson(endog, exog[:, 0]).fit(start_params=[8.5])
        # use start params to avoid warnings
        cls.start_params = [8.5, 0]


class TestNegativeBinomialNB1Null(CheckNull):

    @classmethod
    def setup_class(cls):
        endog, exog = cls._get_data()
        cls.model = NegativeBinomial(endog, exog, loglike_method='nb1')
        cls.model_null = NegativeBinomial(endog, exog[:, 0],
                                          loglike_method='nb1')
        cls.res_null = cls.model_null.fit(start_params=[8, 1000],
                                          method='bfgs', gtol=1e-08,
                                          maxiter=300)
        # for convergence with bfgs, I needed to round down alpha start_params
        cls.start_params = np.array([7.730452, 2.01633068e-02, 1763.0])


class TestNegativeBinomialNB2Null(CheckNull):

    @classmethod
    def setup_class(cls):
        endog, exog = cls._get_data()
        cls.model = NegativeBinomial(endog, exog, loglike_method='nb2')
        cls.model_null = NegativeBinomial(endog, exog[:, 0],
                                          loglike_method='nb2')
        cls.res_null = cls.model_null.fit(start_params=[8, 0.5],
                                          method='bfgs', gtol=1e-06,
                                          maxiter=300)
        cls.start_params = np.array([8.07216448, 0.01087238, 0.44024134])


class TestNegativeBinomialNBP2Null(CheckNull):

    @classmethod
    def setup_class(cls):
        endog, exog = cls._get_data()
        cls.model = NegativeBinomialP(endog, exog, p=2)
        cls.model_null = NegativeBinomialP(endog, exog[:, 0], p=2)
        cls.res_null = cls.model_null.fit(start_params=[8, 1],
                                          method='bfgs', gtol=1e-06,
                                          maxiter=300)
        cls.start_params = np.array([8.07216448, 0.01087238, 0.44024134])

    def test_start_null(self):
        endog, exog = self.model.endog, self.model.exog
        model_nb2 = NegativeBinomial(endog, exog, loglike_method='nb2')
        sp1 = model_nb2._get_start_params_null()
        sp0 = self.model._get_start_params_null()
        assert_allclose(sp0, sp1, rtol=1e-12)


class TestNegativeBinomialNBP1Null(CheckNull):
    @classmethod
    def setup_class(cls):
        endog, exog = cls._get_data()
        cls.model = NegativeBinomialP(endog, exog, p=1.)
        cls.model_null = NegativeBinomialP(endog, exog[:, 0], p=1)
        cls.res_null = cls.model_null.fit(start_params=[8, 1],
                                          method='bfgs', gtol=1e-06,
                                          maxiter=300)
        cls.start_params = np.array([7.730452, 2.01633068e-02, 1763.0])

    def test_start_null(self):
        endog, exog = self.model.endog, self.model.exog
        model_nb2 = NegativeBinomial(endog, exog, loglike_method='nb1')
        sp1 = model_nb2._get_start_params_null()
        sp0 = self.model._get_start_params_null()
        assert_allclose(sp0, sp1, rtol=1e-12)


class TestGeneralizedPoissonNull(CheckNull):
    @classmethod
    def setup_class(cls):
        endog, exog = cls._get_data()
        cls.model = GeneralizedPoisson(endog, exog, p=1.5)
        cls.model_null = GeneralizedPoisson(endog, exog[:, 0], p=1.5)
        cls.res_null = cls.model_null.fit(start_params=[8.4, 1],
                                          method='bfgs', gtol=1e-08,
                                          maxiter=300)
        cls.start_params = np.array([6.91127148, 0.04501334, 0.88393736])


def test_null_options():
    # this is a "nice" case because we only check that options are used
    # correctly
    nobs = 10
    exog = np.ones((20, 2))
    exog[:nobs // 2, 1] = 0
    mu = np.exp(exog.sum(1))
    endog = np.random.poisson(mu)  # Note no size=nobs in np.random
    res = Poisson(endog, exog).fit(start_params=np.log([1, 1]))
    llnull0 = res.llnull
    assert_(hasattr(res, 'res_llnull') is False)
    res.set_null_options(attach_results=True)
    # default optimization
    lln = res.llnull  # access to trigger computation
    assert_allclose(res.res_null.mle_settings['start_params'],
                    np.log(endog.mean()), rtol=1e-10)
    assert_equal(res.res_null.mle_settings['optimizer'], 'bfgs')
    assert_allclose(lln, llnull0)

    res.set_null_options(attach_results=True, start_params=[0.5], method='nm')
    lln = res.llnull  # access to trigger computation
    assert_allclose(res.res_null.mle_settings['start_params'], [0.5],
                    rtol=1e-10)
    assert_equal(res.res_null.mle_settings['optimizer'], 'nm')

    res.summary()  # call to fill cache
    assert_('prsquared' in res._cache)
    assert_equal(res._cache['llnull'],  lln)

    assert_('prsquared' in res._cache)
    assert_equal(res._cache['llnull'],  lln)

    # check setting cache
    res.set_null_options(llnull=999)
    assert_('prsquared' not in res._cache)
    assert_equal(res._cache['llnull'],  999)


def test_optim_kwds_prelim():
    # test that fit options for preliminary fit is correctly transmitted

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(cur_dir, "results", "sm3533.csv")
    df = pd.read_csv(filepath)

    features = ['pp']
    X = (df[features] - df[features].mean())/df[features].std()
    y = df['num'].values
    exog = add_constant(X[features].copy())
    # offset=np.log(df['population'].values + 1)
    # offset currently not used
    offset = None

    # we use "nm", "bfgs" does not work for Poisson/exp with older scipy
    optim_kwds_prelim = dict(method='nm', maxiter=5000)
    model = Poisson(y, exog, offset=offset) #
    res_poi = model.fit(disp=0, **optim_kwds_prelim)

    model = NegativeBinomial(y, exog, offset=offset)
    res = model.fit(disp=0, optim_kwds_prelim=optim_kwds_prelim)

    assert_allclose(res.mle_settings['start_params'][:-1], res_poi.params,
                    rtol=1e-4)
    assert_equal(res.mle_settings['optim_kwds_prelim'], optim_kwds_prelim)
    assert_allclose(res.predict().mean(), y.mean(), rtol=0.1)

    # NBP22 and GPP p=1.5 also fail on older scipy with bfgs, use nm instead
    optim_kwds_prelim = dict(method='nm', maxiter=5000)
    model = NegativeBinomialP(y, exog, offset=offset, p=2)
    res = model.fit(disp=0, optim_kwds_prelim=optim_kwds_prelim)

    assert_allclose(res.mle_settings['start_params'][:-1], res_poi.params,
                    rtol=1e-4)
    assert_equal(res.mle_settings['optim_kwds_prelim'], optim_kwds_prelim)
    assert_allclose(res.predict().mean(), y.mean(), rtol=0.1)

    # GPP with p=1.5 converges correctly,
    # GPP fails when p=2 even with good start_params
    model = GeneralizedPoisson(y, exog, offset=offset, p=1.5)
    res = model.fit(disp=0, maxiter=200, optim_kwds_prelim=optim_kwds_prelim )

    assert_allclose(res.mle_settings['start_params'][:-1], res_poi.params,
                    rtol=1e-4)
    assert_equal(res.mle_settings['optim_kwds_prelim'], optim_kwds_prelim)
    # rough check that convergence makes sense
    assert_allclose(res.predict().mean(), y.mean(), rtol=0.1)
'''

# ------------------------------------------------------------------
# Tests implemented/checked 2017-10-08 or later

def check_inherited_attributes(res):
    # check that specific attributes are directly defined in the Results
    # object; this can help ensure that Results objects can still be useful
    # after `remove_data` and a pickle/unpickle cycle.
    model = res.model
    # if isinstance(model, MultinomialModel):  # havent yet implemented this
    #    assert res.J == model.J
    #    assert res.K == model.K


def test_non_binary():
    # BinaryModel and subclasses require endog to be binary
    y = [1, 2, 1, 2, 1, 2]
    X = np.random.randn(6, 2)
    with pytest.raises(ValueError):
        Logit(y, X)


# ------------------------------------------------------------------
# Issue Regression Tests

def test_mnlogit_non_square():
    # GH#339
    # make sure MNLogit summary works for J != K,
    # specifically that we don't
    # get `ValueError: endog_names has wrong length` from summary()
    data = sm2.datasets.anes96.load()
    exog = data.exog
    # leave out last exog column
    exog = exog[:, :-1]
    exog = add_constant(exog, prepend=True)
    res1 = MNLogit(data.endog, exog).fit(method="newton", disp=0)

    # strip the header from the test
    smry = "\n".join(res1.summary().as_text().split('\n')[9:])
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_case_file = os.path.join(cur_dir, 'results', 'mn_logit_summary.txt')
    with open(test_case_file, 'r') as fd:
        test_case = fd.read()
    np.testing.assert_equal(smry, test_case[:-1])

    """
    # smoke test for summary2
    res1.summary2()  # see #3651
    """


def test_mnlogit_2dexog():
    # GH#341
    data = sm2.datasets.anes96.load()
    exog = data.exog
    # leave out last exog column
    exog = exog[:, :-1]
    exog = add_constant(exog, prepend=True)
    res1 = MNLogit(data.endog, exog).fit(method="newton", disp=0)
    x = exog[0]
    np.testing.assert_equal(res1.predict(x).shape, (1, 7))
    np.testing.assert_equal(res1.predict(x[None]).shape, (1, 7))


def test_formula_missing_exposure():
    # see GH#2083
    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, np.nan],
         'constant': [1] * 4, 'exposure' : np.random.uniform(size=4),
         'x': [1, 3, 2, 1.5]}
    df = pd.DataFrame(d)

    # should work
    mod1 = Poisson.from_formula('Foo ~ Bar', data=df, exposure=df['exposure'])
    assert type(mod1.exposure) is np.ndarray, type(mod1.exposure)

    # make sure this raises
    exposure = pd.Series(np.random.randn(5))
    with pytest.raises(MissingDataError):
        Poisson(df.Foo, df[['constant', 'Bar']], exposure=exposure)


def test_predict_with_exposure():
    # Case where CountModel.predict is called with exog = None and exposure
    # or offset not-None
    # See GH#3565

    # Setup copied from test_formula_missing_exposure
    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, 4],
         'constant': [1] * 4, 'exposure' : [np.exp(1)]*4,
         'x': [1, 3, 2, 1.5]}
    df = pd.DataFrame(d)

    mod1 = CountModel.from_formula('Foo ~ Bar', data=df,
                                   exposure=df['exposure'])

    params = np.array([1, .4])
    pred = mod1.predict(params, linear=True)
    # No exposure is passed, so default to using mod1.exposure, which
    # should have been logged
    X = df[['constant', 'Bar']].values  # mod1.exog
    expected = np.dot(X, params) + 1
    assert_allclose(pred, expected)
    # The above should have passed without the current patch.  The next
    # test would fail under the old code

    pred2 = mod1.predict(params, exposure=[np.exp(2)]*4, linear=True)
    expected2 = expected + 1
    assert_allclose(pred2, expected2)


def test_poisson_predict():
    # GH#175, make sure poisson predict works without offset and exposure
    data = sm2.datasets.randhie.load()
    exog = add_constant(data.exog, prepend=True)
    res = Poisson(data.endog, exog).fit(method='newton', disp=0)
    pred1 = res.predict()
    pred2 = res.predict(exog)
    assert_almost_equal(pred1, pred2)
    # extra options
    pred3 = res.predict(exog, offset=0, exposure=1)
    assert_almost_equal(pred1, pred3)
    pred3 = res.predict(exog, offset=0, exposure=2)
    assert_almost_equal(2*pred1, pred3)
    pred3 = res.predict(exog, offset=np.log(2), exposure=1)
    assert_almost_equal(2*pred1, pred3)


def test_poisson_newton():
    # see GH#24, Newton doesn't work well sometimes
    nobs = 10000
    np.random.seed(987689)
    x = np.random.randn(nobs, 3)
    x = add_constant(x, prepend=True)
    y_count = np.random.poisson(np.exp(x.sum(1)))
    mod = Poisson(y_count, x)
    # this is not thread-safe
    with tm.assert_produces_warning():
        warnings.simplefilter('always')
        res = mod.fit(start_params=-np.ones(4), method='newton', disp=0)
    assert not res.mle_retvals['converged']


def test_unchanging_degrees_of_freedom():
    # see GH#3734, calling `fit_regularized` should not alter
    # model.df_model inplace.
    data = sm2.datasets.randhie.load()
    model = NegativeBinomial(data.endog, data.exog, loglike_method='nb2')
    params = np.array([-0.05654134, -0.21213734,  0.08783102, -0.02991825,
                       0.22902315 ,  0.06210253,  0.06799444,  0.08406794,
                       0.18530092 ,  1.36645186])

    res1 = model.fit(start_params=params)
    assert_equal(res1.df_model, 8)

    reg_params = np.array([-0.04854   , -0.15019404,  0.08363671,
                           -0.03032834,  0.17592454,
                           0.06440753 ,  0.01584555,  0.        ,
                           0.         ,  1.36984628])

    res2 = model.fit_regularized(alpha=100, start_params=reg_params)
    assert res2.df_model != 8
    # If res2.df_model == res1.df_model, then this test is invalid.

    res3 = model.fit()
    # Test that the call to `fit_regularized` didn't modify model.df_model
    # inplace.
    assert_equal(res3.df_model, res1.df_model)
    assert_equal(res3.df_resid, res1.df_resid)

    check_inherited_attributes(res1)
    check_inherited_attributes(res2)
    check_inherited_attributes(res3)


def test_binary_pred_table_zeros():
    # see GH#2968
    nobs = 10
    y = np.zeros(nobs)
    y[[1, 3]] = 1

    res = Logit(y, np.ones(nobs)).fit(disp=0)
    expected = np.array([[ 8.,  0.], [ 2.,  0.]])
    assert_equal(res.pred_table(), expected)

    res = MNLogit(y, np.ones(nobs)).fit(disp=0)
    expected = np.array([[ 8.,  0.], [ 2.,  0.]])
    assert_equal(res.pred_table(), expected)

    check_inherited_attributes(res)
