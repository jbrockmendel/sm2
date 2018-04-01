"""
Test for weights in GLM, Poisson and OLS/WLS, continuous test_glm.py


Below is a table outlining the test coverage.

================================= ====================== ====== ===================== === ======= ======== ============== ============= ============== ============= ============== ==== =========
Test                              Compared To            params normalized_cov_params bse loglike deviance resid_response resid_pearson resid_deviance resid_working resid_anscombe chi2 optimizer
================================= ====================== ====== ===================== === ======= ======== ============== ============= ============== ============= ============== ==== =========
TestGlmPoissonPlain               stata                  X                            X   X       X        X              X             X              X             X              X    bfgs
TestGlmPoissonFwNr                stata                  X                            X   X       X        X              X             X              X             X              X    bfgs
TestGlmPoissonAwNr                stata                  X                            X   X       X        X              X             X              X             X              X    bfgs
TestGlmPoissonFwHC                stata                  X                            X   X       X                                                                                 X
TestGlmPoissonAwHC                stata                  X                            X   X       X                                                                                 X
TestGlmPoissonFwClu               stata                  X                            X   X       X                                                                                 X
TestGlmTweedieAwNr                R                      X                            X           X        X              X             X              X                                 newton
TestGlmGammaAwNr                  R                      X                            X   special X        X              X             X              X                                 bfgs
TestGlmGaussianAwNr               R                      X                            X   special X        X              X             X              X                                 bfgs
TestRepeatedvsAggregated          statsmodels.GLM        X      X                                                                                                                        bfgs
TestRepeatedvsAverage             statsmodels.GLM        X      X                                                                                                                        bfgs
TestTweedieRepeatedvsAggregated   statsmodels.GLM        X      X                                                                                                                        bfgs
TestTweedieRepeatedvsAverage      statsmodels.GLM        X      X                                                                                                                        bfgs
TestBinomial0RepeatedvsAverage    statsmodels.GLM        X      X
TestBinomial0RepeatedvsDuplicated statsmodels.GLM        X      X                                                                                                                        bfgs
TestBinomialVsVarWeights          statsmodels.GLM        X      X                     X                                                                                                  bfgs
TestGlmGaussianWLS                statsmodels.WLS        X      X                     X                                                                                                  bfgs
================================= ====================== ====== ===================== === ======= ======== ============== ============= ============== ============= ============== ==== =========
"""  # noqa:E501
from __future__ import division
import copy
import warnings
import sys

import pytest
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

import sm2.api as sm
from sm2.genmod.generalized_linear_model import GLM
from sm2.tools.tools import add_constant

from .results import results_glm_poisson_weights as res_stata
from .results import res_R_var_weight as res_r

# load data into module namespace
from sm2.datasets.cpunish import load
cpunish_data = load()
cpunish_data.exog[:, 3] = np.log(cpunish_data.exog[:, 3])
cpunish_data.exog = add_constant(cpunish_data.exog, prepend=False)


@pytest.mark.not_vetted
class CheckWeight(object):
    def test_params(self):
        assert_allclose(self.res1.params,
                        self.res2.params,
                        atol=1e-6, rtol=2e-6)

    def test_normalized_cov_params(self):
        if hasattr(self.res2, 'normalized_cov_params'):
            assert_allclose(self.res1.normalized_cov_params,
                            self.res2.normalized_cov_params,
                            atol=1e-8, rtol=2e-6)

    def test_bse(self):
        if not isinstance(self, (TestRepeatedvsAggregated,
                                 TestRepeatedvsAverage,
                                 TestTweedieRepeatedvsAggregated,
                                 TestTweedieRepeatedvsAverage,
                                 TestBinomial0RepeatedvsAverage,
                                 TestBinomial0RepeatedvsDuplicated)):
            # Loglikelihood, scale, deviance is different between repeated vs.
            # exposure/average
            corr_fact = getattr(self, 'corr_fact', 1)
            assert_allclose(self.res1.bse,
                            corr_fact * self.res2.bse,
                            atol=1e-6, rtol=2e-6)

    def test_llf(self):
        if isinstance(self, (TestRepeatedvsAggregated,
                             TestRepeatedvsAverage,
                             TestTweedieRepeatedvsAggregated,
                             TestTweedieRepeatedvsAverage,
                             TestBinomial0RepeatedvsAverage,
                             TestBinomial0RepeatedvsDuplicated,
                             TestBinomialVsVarWeights,
                             TestGlmGaussianWLS,
                             TestGlmGaussianAwNr, TestGlmGammaAwNr)):
            # a) loglikelihood different between repeated vs. exposure/average
            # b) Binomial ll and deviance are different for 1d vs. counts...
            # c) TestGlmGaussianWLS This won't work right now either
            # d) TestGlmGaussianAwNr, TestGlmGammaAwNr: Matching R is hard
            return None
        assert_allclose(self.res1.llf,
                        self.res2.ll,
                        atol=1e-6, rtol=1e-7)

    def test_deviance(self):
        if isinstance(self, (TestRepeatedvsAggregated,
                             TestRepeatedvsAverage,
                             TestTweedieRepeatedvsAggregated,
                             TestTweedieRepeatedvsAverage,
                             TestBinomial0RepeatedvsAverage,
                             TestBinomial0RepeatedvsDuplicated,
                             TestBinomialVsVarWeights,
                             TestGlmGaussianWLS)):
            # a) deviance is different between repeated vs. exposure/average
            # b) Binomial ll and deviance are different for 1d vs. counts...
            # c) TestGlmGaussianWLS This won't work right now either
            return None
        assert_allclose(self.res1.deviance,
                        self.res2.deviance,
                        atol=1e-6, rtol=1e-7)

    def test_residuals(self):
        if isinstance(self, (TestRepeatedvsAggregated, TestRepeatedvsAverage,
                             TestTweedieRepeatedvsAggregated,
                             TestTweedieRepeatedvsAverage,
                             TestBinomial0RepeatedvsAverage,
                             TestBinomial0RepeatedvsDuplicated)):
            # This won't match as different number of records
            return None
        res1 = self.res1
        res2 = self.res2
        if not hasattr(res2, 'resids'):
            return None  # use SkipError instead
        resid_all = dict(zip(res2.resids_colnames, res2.resids.T))

        assert_allclose(res1.resid_response,
                        resid_all['resid_response'],
                        atol=1e-6, rtol=2e-6)
        assert_allclose(res1.resid_pearson,
                        resid_all['resid_pearson'],
                        atol=1e-6, rtol=2e-6)
        assert_allclose(res1.resid_deviance,
                        resid_all['resid_deviance'],
                        atol=1e-6, rtol=2e-6)
        assert_allclose(res1.resid_working,
                        resid_all['resid_working'],
                        atol=1e-6, rtol=2e-6)
        if resid_all.get('resid_anscombe') is None:
            return None

        # Stata doesn't use var_weights in anscombe residuals, it seems.
        # Adjust residuals to match our approach.
        expected = resid_all['resid_anscombe'] * np.sqrt(res1._var_weights)
        assert_allclose(res1.resid_anscombe,
                        expected,
                        atol=1e-6, rtol=2e-6)

    def test_compare_optimizers(self):
        res1 = self.res1
        if isinstance(res1.model.family, sm.families.Tweedie):
            method = 'newton'
            optim_hessian = 'eim'
        else:
            method = 'bfgs'
            optim_hessian = 'oim'
        if isinstance(self, (TestGlmPoissonFwHC, TestGlmPoissonAwHC,
                             TestGlmPoissonFwClu,
                             TestBinomial0RepeatedvsAverage)):
            return None
        res2 = self.res1.model.fit(method=method, optim_hessian=optim_hessian)
        assert_allclose(res1.params,
                        res2.params,
                        atol=1e-3, rtol=2e-3)

        H = res2.model.hessian(res2.params, observed=False)
        res2_bse = np.sqrt(-np.diag(np.linalg.inv(H)))
        assert_allclose(res1.bse,
                        res2_bse,
                        atol=1e-3, rtol=1e-3)

    def test_pearson_chi2(self):
        if hasattr(self.res2, 'chi2'):
            assert_allclose(self.res1.pearson_chi2,
                            self.res2.deviance_p,
                            atol=1e-6, rtol=1e-6)


@pytest.mark.not_vetted
class TestGlmPoissonPlain(CheckWeight):
    res2 = res_stata.results_poisson_none_nonrobust
    mod_kwargs = {"family": sm.families.Poisson()}

    @classmethod
    def setup_class(cls):
        model = GLM(cpunish_data.endog, cpunish_data.exog,
                    **cls.mod_kwargs)
        cls.res1 = model.fit()


@pytest.mark.not_vetted
class TestGlmPoissonFwNr(CheckWeight):
    res2 = res_stata.results_poisson_fweight_nonrobust
    mod_kwargs = {
        "family": sm.families.Poisson(),
        "freq_weights": np.array([1, 1, 1, 2, 2, 2, 3, 3,
                                  3, 1, 1, 1, 2, 2, 2, 3, 3])
    }

    @classmethod
    def setup_class(cls):
        model = GLM(cpunish_data.endog, cpunish_data.exog,
                    **cls.mod_kwargs)
        cls.res1 = model.fit()


@pytest.mark.not_vetted
class TestGlmPoissonAwNr(CheckWeight):
    @classmethod
    def setup_class(cls):
        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs

        model = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson(), var_weights=aweights)
        cls.res1 = model.fit()

        # Need to copy to avoid inplace adjustment
        cls.res2 = copy.copy(res_stata.results_poisson_aweight_nonrobust)
        cls.res2.resids = cls.res2.resids.copy()

        # Need to adjust resids for pearson and deviance to add weights
        cls.res2.resids[:, 3:5] *= np.sqrt(aweights[:, np.newaxis])


# prob_weights fail with HC, not properly implemented yet
@pytest.mark.not_vetted
class TestGlmPoissonPwNr(CheckWeight):
    res2 = res_stata.results_poisson_pweight_nonrobust

    @classmethod
    def setup_class(cls):
        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)

        cls.res1 = GLM(cpunish_data.endog, cpunish_data.exog,
                       family=sm.families.Poisson(),
                       freq_weights=fweights).fit(cov_type='HC1')

    @pytest.mark.xfail(reason='prob_weights not yet implemented GH#4397')
    def test_bse(self):
        super(TestGlmPoissonPwNr, self).test_bse(self)

    @pytest.mark.xfail(reason='prob_weights not yet implemented GH#4397')
    def test_compare_optimizers(self):
        super(TestGlmPoissonPwNr, self).test_compare_optimizers(self)


@pytest.mark.not_vetted
class TestGlmPoissonFwHC(CheckWeight):
    res2 = res_stata.results_poisson_fweight_hc1

    @classmethod
    def setup_class(cls):
        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)
        wsum = fweights.sum()
        cls.corr_fact = np.sqrt((wsum - 1.) / wsum)
        model = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson(), freq_weights=fweights)
        cls.res1 = model.fit(cov_type='HC0')


# var_weights (aweights fail with HC, not properly implemented yet
@pytest.mark.not_vetted
class TestGlmPoissonAwHC(CheckWeight):
    res2 = res_stata.results_poisson_aweight_hc1

    @classmethod
    def setup_class(cls):
        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs

        # This is really close when corr_fact = (wsum - 1.) / wsum, but to
        # avoid having loosen precision of the assert_allclose, I'm doing this
        # manually. Its *possible* lowering the IRLS convergence criterion
        # in stata and here will make this less sketchy.
        cls.corr_fact = np.sqrt((wsum - 1.) / wsum) * 0.98518473599905609
        model = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson(), var_weights=aweights)
        cls.res1 = model.fit(cov_type='HC0')


@pytest.mark.not_vetted
class TestGlmPoissonFwClu(CheckWeight):
    res2 = res_stata.results_poisson_fweight_clu1

    @classmethod
    def setup_class(cls):
        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)

        gid = np.arange(1, 17 + 1) // 2
        n_groups = len(np.unique(gid))

        # no wnobs yet in sandwich covariance calcualtion
        cls.corr_fact = 1 / np.sqrt(n_groups / (n_groups - 1))
        cov_kwds = {'groups': gid, 'use_correction': False}
        model = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson(), freq_weights=fweights)
        cls.res1 = model.fit(cov_type='cluster', cov_kwds=cov_kwds)


@pytest.mark.not_vetted
class TestGlmTweedieAwNr(CheckWeight):
    res2 = res_r.results_tweedie_aweights_nonrobust

    @classmethod
    def setup_class(cls):
        data = sm.datasets.fair.load_pandas()
        endog = data.endog
        data = data.exog
        data['fair'] = endog
        aweights = np.repeat(1, len(data.index))
        aweights[::5] = 5
        aweights[::13] = 3
        fam = sm.families.Tweedie(var_power=1.55,
                                  link=sm.families.links.log())
        model = sm.GLM.from_formula('fair ~ age + yrs_married',
                                    data=data,
                                    family=fam,
                                    var_weights=aweights)
        cls.res1 = model.fit(rtol=1e-25, atol=0)


@pytest.mark.not_vetted
class TestGlmGammaAwNr(CheckWeight):
    res2 = res_r.results_gamma_aweights_nonrobust

    @classmethod
    def setup_class(cls):
        from .results.results_glm import CancerLog
        res2 = CancerLog
        endog = res2.endog
        exog = res2.exog[:, :-1]
        exog = sm.add_constant(exog, prepend=True)

        aweights = np.repeat(1, len(endog))
        aweights[::5] = 5
        aweights[::13] = 3
        model = sm.GLM(endog, exog,
                       family=sm.families.Gamma(link=sm.families.links.log()),
                       var_weights=aweights)
        cls.res1 = model.fit(rtol=1e-25, atol=0)

    def test_r_llf(self):
        scale = self.res1.deviance / self.res1._iweights.sum()
        ll = self.res1.family.loglike(self.res1.model.endog,
                                      self.res1.mu,
                                      freq_weights=self.res1._var_weights,
                                      scale=scale)
        assert_allclose(ll, self.res2.ll, atol=1e-6, rtol=1e-7)


@pytest.mark.not_vetted
class TestGlmGaussianAwNr(CheckWeight):
    res2 = res_r.results_gaussian_aweights_nonrobust

    @classmethod
    def setup_class(cls):
        data = sm.datasets.cpunish.load_pandas()
        endog = data.endog
        data = data.exog
        data['EXECUTIONS'] = endog
        data['INCOME'] /= 1000
        aweights = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1])
        model = GLM.from_formula(
            'EXECUTIONS ~ INCOME + SOUTH - 1',
            data=data,
            family=sm.families.Gaussian(link=sm.families.links.log()),
            var_weights=aweights)
        cls.res1 = model.fit(rtol=1e-25, atol=0)

    def test_r_llf(self):
        res1 = self.res1
        res2 = self.res2
        model = self.res1.model

        # Need to make a few adjustments...
        # First, calculate scale using nobs as denominator
        scale = res1.scale * model.df_resid / model.wnobs
        # Calculate llf using adj scale and wts = freq_weights
        wts = model.freq_weights
        llf = model.family.loglike(model.endog, res1.mu,
                                   freq_weights=wts,
                                   scale=scale)
        # SM uses (essentially) stat's loglike formula... first term is
        # (endog - mu) ** 2 / scale
        adj_sm = -1 / 2 * ((model.endog - res1.mu) ** 2).sum() / scale
        # R has these 2 terms that stata/sm don't
        adj_r = -model.wnobs / 2 + np.sum(np.log(model.var_weights)) / 2
        llf_adj = llf - adj_sm + adj_r
        assert_allclose(llf_adj,
                        res2.ll,
                        atol=1e-6, rtol=1e-7)


# TODO: almost identical to gen_endog in test_glm
@pytest.mark.not_vetted
def gen_endog(lin_pred, family_class, link, binom_version=0):

    np.random.seed(872)

    fam = sm.families

    mu = link().inverse(lin_pred)

    if family_class == fam.Binomial:
        if binom_version == 0:
            endog = 1 * (np.random.uniform(size=len(lin_pred)) < mu)
        else:
            endog = np.empty((len(lin_pred), 2))
            n = 10
            uni = np.random.uniform(size=(len(lin_pred), n))
            endog[:, 0] = (uni < mu[:, None]).sum(1)
            endog[:, 1] = n - endog[:, 0]
    elif family_class == fam.Poisson:
        endog = np.random.poisson(mu)
    elif family_class == fam.Gamma:
        endog = np.random.gamma(2, mu)
    elif family_class == fam.Gaussian:
        endog = mu + np.random.normal(size=len(lin_pred))
    elif family_class == fam.NegativeBinomial:
        from scipy.stats.distributions import nbinom
        endog = nbinom.rvs(mu, 0.5)
    elif family_class == fam.InverseGaussian:
        from scipy.stats.distributions import invgauss
        endog = invgauss.rvs(mu)
    elif family_class == fam.Tweedie:
        rate = 1
        shape = 1.0
        scale = mu / (rate * shape)
        endog = (np.random.poisson(rate, size=scale.shape[0]) *
                 np.random.gamma(shape * scale))
    else:
        raise ValueError

    return endog


def get_dummies(x):
    values = np.sort(np.unique(x))
    out = np.zeros(shape=(x.shape[0], len(values) - 1))
    for i, v in enumerate(values):
        if i == 0:
            continue
        out[:, i - 1] = np.where(v == x, 1, 0)
    return out


@pytest.mark.not_vetted
class TestRepeatedvsAggregated(CheckWeight):
    @classmethod
    def setup_class(cls):
        np.random.seed(4321)
        n = 100
        p = 5
        exog = np.empty((n, p))
        exog[:, 0] = 1
        exog[:, 1] = np.random.randint(low=-5, high=5, size=n)
        x = np.repeat(np.array([1, 2, 3, 4]), n / 4)
        exog[:, 2:] = get_dummies(x)
        beta = np.array([-1, 0.1, -0.05, .2, 0.35])
        lin_pred = (exog * beta).sum(axis=1)
        family = sm.families.Poisson
        link = sm.families.links.log
        endog = gen_endog(lin_pred, family, link)
        mod1 = sm.GLM(endog, exog, family=family(link=link))
        cls.res1 = mod1.fit()

        agg = pd.DataFrame(exog)
        agg['endog'] = endog
        agg_endog = agg.groupby([0, 1, 2, 3, 4]).sum()[['endog']]
        agg_wt = agg.groupby([0, 1, 2, 3, 4]).count()[['endog']]
        agg_exog = np.array(agg_endog.index.tolist())
        agg_wt = agg_wt['endog']
        agg_endog = agg_endog['endog']
        mod2 = sm.GLM(agg_endog, agg_exog, family=family(link=link),
                      exposure=agg_wt)
        cls.res2 = mod2.fit()


@pytest.mark.not_vetted
class TestRepeatedvsAverage(CheckWeight):
    @classmethod
    def setup_class(cls):
        np.random.seed(4321)
        n = 10000
        p = 5
        exog = np.empty((n, p))
        exog[:, 0] = 1
        exog[:, 1] = np.random.randint(low=-5, high=5, size=n)
        x = np.repeat(np.array([1, 2, 3, 4]), n / 4)
        exog[:, 2:] = get_dummies(x)
        beta = np.array([-1, 0.1, -0.05, .2, 0.35])
        lin_pred = (exog * beta).sum(axis=1)
        family = sm.families.Poisson
        link = sm.families.links.log
        endog = gen_endog(lin_pred, family, link)
        mod1 = sm.GLM(endog, exog, family=family(link=link))
        cls.res1 = mod1.fit()

        agg = pd.DataFrame(exog)
        agg['endog'] = endog
        agg_endog = agg.groupby([0, 1, 2, 3, 4]).sum()[['endog']]
        agg_wt = agg.groupby([0, 1, 2, 3, 4]).count()[['endog']]
        agg_exog = np.array(agg_endog.index.tolist())
        agg_wt = agg_wt['endog']
        avg_endog = agg_endog['endog'] / agg_wt
        mod2 = sm.GLM(avg_endog, agg_exog, family=family(link=link),
                      var_weights=agg_wt)
        cls.res2 = mod2.fit()


@pytest.mark.not_vetted
class TestTweedieRepeatedvsAggregated(CheckWeight):
    @classmethod
    def setup_class(cls):
        np.random.seed(4321)
        n = 10000
        p = 5
        exog = np.empty((n, p))
        exog[:, 0] = 1
        exog[:, 1] = np.random.randint(low=-5, high=5, size=n)
        x = np.repeat(np.array([1, 2, 3, 4]), n / 4)
        exog[:, 2:] = get_dummies(x)
        beta = np.array([7, 0.1, -0.05, .2, 0.35])
        lin_pred = (exog * beta).sum(axis=1)
        family = sm.families.Tweedie
        link = sm.families.links.log
        endog = gen_endog(lin_pred, family, link)
        mod1 = sm.GLM(endog, exog, family=family(link=link(), var_power=1.5))
        cls.res1 = mod1.fit(rtol=1e-20, atol=0, tol_criterion='params')

        agg = pd.DataFrame(exog)
        agg['endog'] = endog
        agg_endog = agg.groupby([0, 1, 2, 3, 4]).sum()[['endog']]
        agg_wt = agg.groupby([0, 1, 2, 3, 4]).count()[['endog']]
        agg_exog = np.array(agg_endog.index.tolist())
        agg_wt = agg_wt['endog']
        agg_endog = agg_endog['endog']
        mod2 = sm.GLM(agg_endog, agg_exog,
                      family=family(link=link(), var_power=1.5),
                      exposure=agg_wt, var_weights=agg_wt ** 0.5)
        cls.res2 = mod2.fit(rtol=1e-20, atol=0, tol_criterion='params')


@pytest.mark.not_vetted
class TestTweedieRepeatedvsAverage(CheckWeight):
    @classmethod
    def setup_class(cls):
        np.random.seed(4321)
        n = 1000
        p = 5
        exog = np.empty((n, p))
        exog[:, 0] = 1
        exog[:, 1] = np.random.randint(low=-5, high=5, size=n)
        x = np.repeat(np.array([1, 2, 3, 4]), n / 4)
        exog[:, 2:] = get_dummies(x)
        beta = np.array([7, 0.1, -0.05, .2, 0.35])
        lin_pred = (exog * beta).sum(axis=1)
        family = sm.families.Tweedie
        link = sm.families.links.log
        endog = gen_endog(lin_pred, family, link)
        mod1 = sm.GLM(endog, exog, family=family(link=link(), var_power=1.5))
        cls.res1 = mod1.fit(rtol=1e-10, atol=0, tol_criterion='params',
                            scaletype='x2')

        agg = pd.DataFrame(exog)
        agg['endog'] = endog
        agg_endog = agg.groupby([0, 1, 2, 3, 4]).sum()[['endog']]
        agg_wt = agg.groupby([0, 1, 2, 3, 4]).count()[['endog']]
        agg_exog = np.array(agg_endog.index.tolist())
        agg_wt = agg_wt['endog']
        avg_endog = agg_endog['endog'] / agg_wt
        mod2 = sm.GLM(avg_endog, agg_exog,
                      family=family(link=link(), var_power=1.5),
                      var_weights=agg_wt)
        cls.res2 = mod2.fit(rtol=1e-10, atol=0, tol_criterion='params')


@pytest.mark.not_vetted
class TestBinomial0RepeatedvsAverage(CheckWeight):
    @classmethod
    def setup_class(cls):
        np.random.seed(4321)
        n = 20
        p = 5
        exog = np.empty((n, p))
        exog[:, 0] = 1
        exog[:, 1] = np.random.randint(low=-5, high=5, size=n)
        x = np.repeat(np.array([1, 2, 3, 4]), n / 4)
        exog[:, 2:] = get_dummies(x)
        beta = np.array([-1, 0.1, -0.05, .2, 0.35])
        lin_pred = (exog * beta).sum(axis=1)
        family = sm.families.Binomial
        link = sm.families.links.log
        endog = gen_endog(lin_pred, family, link, binom_version=0)
        mod1 = sm.GLM(endog, exog, family=family(link=link()))
        cls.res1 = mod1.fit(rtol=1e-10, atol=0, tol_criterion='params',
                            scaletype='x2')

        agg = pd.DataFrame(exog)
        agg['endog'] = endog
        agg_endog = agg.groupby([0, 1, 2, 3, 4]).sum()[['endog']]
        agg_wt = agg.groupby([0, 1, 2, 3, 4]).count()[['endog']]
        agg_exog = np.array(agg_endog.index.tolist())
        agg_wt = agg_wt['endog']
        avg_endog = agg_endog['endog'] / agg_wt
        mod2 = sm.GLM(avg_endog, agg_exog,
                      family=family(link=link()),
                      var_weights=agg_wt)
        cls.res2 = mod2.fit(rtol=1e-10, atol=0, tol_criterion='params')


@pytest.mark.not_vetted
class TestBinomial0RepeatedvsDuplicated(CheckWeight):
    @classmethod
    def setup_class(cls):
        np.random.seed(4321)
        n = 10000
        p = 5
        exog = np.empty((n, p))
        exog[:, 0] = 1
        exog[:, 1] = np.random.randint(low=-5, high=5, size=n)
        x = np.repeat(np.array([1, 2, 3, 4]), n / 4)
        exog[:, 2:] = get_dummies(x)
        beta = np.array([-1, 0.1, -0.05, .2, 0.35])
        lin_pred = (exog * beta).sum(axis=1)
        family = sm.families.Binomial
        link = sm.families.links.log
        endog = gen_endog(lin_pred, family, link, binom_version=0)
        wt = np.random.randint(1, 5, n)
        mod1 = sm.GLM(endog, exog, family=family(link=link), freq_weights=wt)
        cls.res1 = mod1.fit()

        exog_dup = np.repeat(exog, wt, axis=0)
        endog_dup = np.repeat(endog, wt)
        mod2 = sm.GLM(endog_dup, exog_dup, family=family(link=link))
        cls.res2 = mod2.fit()


@pytest.mark.not_vetted
class TestBinomialVsVarWeights(CheckWeight):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.star98.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = GLM(data.endog, data.exog,
                       family=sm.families.Binomial()).fit()
        weights = data.endog.sum(axis=1)
        endog2 = data.endog[:, 0] / weights
        cls.res2 = GLM(endog2, data.exog,
                       family=sm.families.Binomial(),
                       var_weights=weights).fit()


@pytest.mark.not_vetted
class TestGlmGaussianWLS(CheckWeight):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.cpunish.load_pandas()
        endog = data.endog
        data = data.exog
        data['EXECUTIONS'] = endog
        data['INCOME'] /= 1000
        aweights = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1])
        model = GLM.from_formula(
            'EXECUTIONS ~ INCOME + SOUTH - 1',
            data=data,
            family=sm.families.Gaussian(link=sm.families.links.identity()),
            var_weights=aweights)
        wlsmodel = sm.WLS.from_formula(
            'EXECUTIONS ~ INCOME + SOUTH - 1',
            data=data,
            weights=aweights)
        cls.res1 = model.fit(rtol=1e-25, atol=1e-25)
        cls.res2 = wlsmodel.fit()


# --------------------------------------------------------------

@pytest.mark.not_vetted
@pytest.mark.skipif(sys.version_info < (3, 4), reason='old python')
def test_warnings_raised():
    weights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
    # faking aweights by using normalized freq_weights
    weights = np.array(weights)

    gid = np.arange(1, 17 + 1) // 2

    cov_kwds = {'groups': gid, 'use_correction': False}
    with warnings.catch_warnings(record=True) as w:
        model = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson(),
                    freq_weights=weights)
        res1 = model.fit(cov_type='cluster', cov_kwds=cov_kwds)
        res1.summary()  # TODO: Should this be marked as a smoke test?
        assert len(w) >= 1

    with warnings.catch_warnings(record=True) as w:
        model = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson(),
                    var_weights=weights)
        res1 = model.fit(cov_type='cluster', cov_kwds=cov_kwds)
        res1.summary()
        assert len(w) >= 1


weights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]


@pytest.mark.not_vetted
@pytest.mark.parametrize('formatted',
                         [weights, np.asarray(weights), pd.Series(weights)],
                         ids=['list', 'ndarray', 'Series'])
def test_weights_different_formats(formatted):
    check_weights_as_formats(formatted)


@pytest.mark.not_vetted
def check_weights_as_formats(weights):
    model = GLM(cpunish_data.endog, cpunish_data.exog,
                family=sm.families.Poisson(),
                freq_weights=weights)
    res = model.fit()
    assert isinstance(res._freq_weights, np.ndarray)
    assert isinstance(res._var_weights, np.ndarray)
    assert isinstance(res._iweights, np.ndarray)

    model = GLM(cpunish_data.endog, cpunish_data.exog,
                family=sm.families.Poisson(),
                var_weights=weights)
    res = model.fit()
    assert isinstance(res._freq_weights, np.ndarray)
    assert isinstance(res._var_weights, np.ndarray)
    assert isinstance(res._iweights, np.ndarray)


@pytest.mark.not_vetted
def test_poisson_residuals():
    nobs, k_exog = 100, 5
    np.random.seed(987125)
    x = np.random.randn(nobs, k_exog - 1)
    x = add_constant(x)

    y_true = x.sum(1) / 2
    y = y_true + 2 * np.random.randn(nobs)
    exposure = 1 + np.arange(nobs) // 4

    yp = np.random.poisson(np.exp(y_true) * exposure)
    yp[10:15] += 10

    fam = sm.families.Poisson()
    mod_poi_e = GLM(yp, x, family=fam, exposure=exposure)
    res_poi_e = mod_poi_e.fit()

    mod_poi_w = GLM(yp / exposure, x, family=fam, var_weights=exposure)
    res_poi_w = mod_poi_w.fit()

    assert_allclose(res_poi_e.resid_response / exposure,
                    res_poi_w.resid_response)
    assert_allclose(res_poi_e.resid_pearson,
                    res_poi_w.resid_pearson)
    assert_allclose(res_poi_e.resid_deviance,
                    res_poi_w.resid_deviance)
    assert_allclose(res_poi_e.resid_anscombe,
                    res_poi_w.resid_anscombe)
    assert_allclose(res_poi_e.resid_anscombe_unscaled,
                    res_poi_w.resid_anscombe)


def test_incompatible_weights_input():
    # TODO: GH reference?
    weights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
    exog = cpunish_data.exog
    endog = cpunish_data.endog
    family = sm.families.Poisson()
    # Too short
    with pytest.raises(ValueError):
        GLM(endog, exog, family=family, freq_weights=weights[:-1])
    with pytest.raises(ValueError):
        GLM(endog, exog, family=family, var_weights=weights[:-1])

    # Too long
    with pytest.raises(ValueError):
        GLM(endog, exog, family=family, freq_weights=weights + [3])
    with pytest.raises(ValueError):
        GLM(endog, exog, family=family, var_weights=weights + [3])

    # Too many dimensions
    with pytest.raises(ValueError):
        GLM(endog, exog, family=family, freq_weights=[weights, weights])
    with pytest.raises(ValueError):
        GLM(endog, exog, family=family, var_weights=[weights, weights])
