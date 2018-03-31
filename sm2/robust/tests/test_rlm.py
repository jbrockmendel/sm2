"""
Test functions for sm.rlm
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pytest
from scipy import stats

import sm2.api as sm
from sm2.robust import norms, scale
from sm2.robust.robust_linear_model import RLM

from .results import results_rlm

DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1


@pytest.mark.not_vetted
class CheckRlmResultsMixin(object):
    """
    res2 contains  results from Rmodelwrap or were obtained from a statistical
    packages such as R, Stata, or SAS and written to results.results_rlm

    Covariance matrices were obtained from SAS and are imported from
    results.results_rlm
    """
    def test_params(self):
        assert_almost_equal(self.res1.params,
                            self.res2.params,
                            DECIMAL_4)

    decimal_standarderrors = DECIMAL_4

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse,
                            self.res2.bse,
                            self.decimal_standarderrors)

    # TODO: get other results from SAS, though if it works for one...
    def test_confidenceintervals(self):
        if not hasattr(self.res2, 'conf_int'):
            raise pytest.skip("Results from R")
        else:
            assert_almost_equal(self.res1.conf_int(),
                                self.res2.conf_int(),
                                DECIMAL_4)

    decimal_scale = DECIMAL_4

    def test_scale(self):
        assert_almost_equal(self.res1.scale,
                            self.res2.scale,
                            self.decimal_scale)

    def test_weights(self):
        assert_almost_equal(self.res1.weights,
                            self.res2.weights,
                            DECIMAL_4)

    def test_residuals(self):
        assert_almost_equal(self.res1.resid,
                            self.res2.resid,
                            DECIMAL_4)

    def test_degrees(self):
        assert_almost_equal(self.res1.model.df_model,
                            self.res2.df_model,
                            DECIMAL_4)
        assert_almost_equal(self.res1.model.df_resid,
                            self.res2.df_resid,
                            DECIMAL_4)

    def test_bcov_unscaled(self):
        if not hasattr(self.res2, 'bcov_unscaled'):
            raise pytest.skip("No unscaled cov matrix from SAS")
        else:
            assert_almost_equal(self.res1.bcov_unscaled,
                                self.res2.bcov_unscaled,
                                DECIMAL_4)

    decimal_bcov_scaled = DECIMAL_4

    def test_bcov_scaled(self):
        assert_almost_equal(self.res1.bcov_scaled,
                            self.res2.h1,
                            self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h2,
                            self.res2.h2,
                            self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h3,
                            self.res2.h3,
                            self.decimal_bcov_scaled)

    def test_tvalues(self):
        if not hasattr(self.res2, 'tvalues'):
            raise pytest.skip("No tvalues in benchmark")
        else:
            assert_allclose(self.res1.tvalues,
                            self.res2.tvalues,
                            rtol=0.003)

    def test_tpvalues(self):
        # test comparing tvalues and pvalues with normal implementation
        # make sure they use normal distribution (inherited in results class)
        params = self.res1.params
        tvalues = params / self.res1.bse
        pvalues = stats.norm.sf(np.abs(tvalues)) * 2
        half_width = stats.norm.isf(0.025) * self.res1.bse
        conf_int = np.column_stack((params - half_width, params + half_width))

        assert_almost_equal(self.res1.tvalues, tvalues)
        assert_almost_equal(self.res1.pvalues, pvalues)
        assert_almost_equal(self.res1.conf_int(), conf_int)


@pytest.mark.not_vetted
class TestRlm(CheckRlmResultsMixin):
    decimal_standarderrors = DECIMAL_1
    decimal_scale = DECIMAL_3
    res2 = results_rlm.Huber()

    @classmethod
    def setup_class(cls):
        from sm2.datasets.stackloss import load
        cls.data = load()  # class attributes for subclasses
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)

        results = RLM(cls.data.endog, cls.data.exog,
                      M=norms.HuberT()).fit()   # default M
        h2 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.HuberT()).fit(cov="H2").bcov_scaled
        h3 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.HuberT()).fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    @pytest.mark.smoke
    def test_summary(self):
        # smoke test that summary at least returns something
        self.res1.summary()


@pytest.mark.not_vetted
class TestHampel(TestRlm):
    decimal_standarderrors = DECIMAL_2
    decimal_scale = DECIMAL_3
    decimal_bcov_scaled = DECIMAL_3
    res2 = results_rlm.Hampel()

    @classmethod
    def setup_class(cls):
        super(TestHampel, cls).setup_class()

        results = RLM(cls.data.endog, cls.data.exog,
                      M=norms.Hampel()).fit()
        h2 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.Hampel()).fit(cov="H2").bcov_scaled
        h3 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.Hampel()).fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


@pytest.mark.not_vetted
class TestRlmBisquare(TestRlm):
    decimal_standarderrors = DECIMAL_1
    res2 = results_rlm.BiSquare()

    @classmethod
    def setup_class(cls):
        super(TestRlmBisquare, cls).setup_class()

        results = RLM(cls.data.endog, cls.data.exog,
                      M=norms.TukeyBiweight()).fit()
        h2 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.TukeyBiweight()).fit(cov="H2").bcov_scaled
        h3 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.TukeyBiweight()).fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


@pytest.mark.not_vetted
class TestRlmAndrews(TestRlm):
    res2 = results_rlm.Andrews()

    @classmethod
    def setup_class(cls):
        super(TestRlmAndrews, cls).setup_class()
        results = RLM(cls.data.endog, cls.data.exog,
                      M=norms.AndrewWave()).fit()
        h2 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.AndrewWave()).fit(cov="H2").bcov_scaled
        h3 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.AndrewWave()).fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


# tests with Huber scaling

@pytest.mark.not_vetted
class TestRlmHuber(CheckRlmResultsMixin):
    res2 = results_rlm.HuberHuber()

    @classmethod
    def setup_class(cls):
        from sm2.datasets.stackloss import load
        cls.data = load()
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)

        mod = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        results = mod.fit(scale_est=scale.HuberScale())

        mod2 = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        h2 = mod2.fit(cov="H2", scale_est=scale.HuberScale()).bcov_scaled

        mod3 = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        h3 = mod3.fit(cov="H3", scale_est=scale.HuberScale()).bcov_scaled

        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


@pytest.mark.not_vetted
class TestHampelHuber(TestRlm):
    res2 = results_rlm.HampelHuber()

    @classmethod
    def setup_class(cls):
        super(TestHampelHuber, cls).setup_class()

        mod = RLM(cls.data.endog, cls.data.exog,
                  M=norms.Hampel())
        results = mod.fit(scale_est=scale.HuberScale())

        mod2 = RLM(cls.data.endog, cls.data.exog,
                   M=norms.Hampel())
        h2 = mod2.fit(cov="H2", scale_est=scale.HuberScale()).bcov_scaled

        mod3 = RLM(cls.data.endog, cls.data.exog,
                   M=norms.Hampel())
        h3 = mod3.fit(cov="H3", scale_est=scale.HuberScale()).bcov_scaled

        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


@pytest.mark.not_vetted
class TestRlmBisquareHuber(TestRlm):
    res2 = results_rlm.BisquareHuber()

    @classmethod
    def setup_class(cls):
        super(TestRlmBisquareHuber, cls).setup_class()

        mod = RLM(cls.data.endog, cls.data.exog,
                  M=norms.TukeyBiweight())
        results = mod.fit(scale_est=scale.HuberScale())

        mod2 = RLM(cls.data.endog, cls.data.exog,
                   M=norms.TukeyBiweight())
        h2 = mod2.fit(cov="H2", scale_est=scale.HuberScale()).bcov_scaled

        mod3 = RLM(cls.data.endog, cls.data.exog,
                   M=norms.TukeyBiweight())
        h3 = mod3.fit(cov="H3", scale_est=scale.HuberScale()).bcov_scaled

        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


@pytest.mark.not_vetted
class TestRlmAndrewsHuber(TestRlm):
    res2 = results_rlm.AndrewsHuber()

    @classmethod
    def setup_class(cls):
        super(TestRlmAndrewsHuber, cls).setup_class()

        mod = RLM(cls.data.endog, cls.data.exog,
                  M=norms.AndrewWave())
        results = mod.fit(scale_est=scale.HuberScale())

        mod2 = RLM(cls.data.endog, cls.data.exog,
                   M=norms.AndrewWave())
        h2 = mod2.fit(cov="H2", scale_est=scale.HuberScale()).bcov_scaled

        mod3 = RLM(cls.data.endog, cls.data.exog,
                   M=norms.AndrewWave())
        h3 = mod3.fit(cov="H3", scale_est=scale.HuberScale()).bcov_scaled

        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


@pytest.mark.not_vetted
class TestRlmSresid(CheckRlmResultsMixin):
    # Check GH#187
    res2 = results_rlm.Huber()
    decimal_standarderrors = DECIMAL_1
    decimal_scale = DECIMAL_3

    @classmethod
    def setup_class(cls):
        from sm2.datasets.stackloss import load
        cls.data = load()  # class attributes for subclasses
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)

        results = RLM(cls.data.endog, cls.data.exog,
                      M=norms.HuberT()).fit(conv='sresid')  # default M

        h2 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.HuberT()).fit(cov="H2").bcov_scaled

        h3 = RLM(cls.data.endog, cls.data.exog,
                 M=norms.HuberT()).fit(cov="H3").bcov_scaled

        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


@pytest.mark.not_vetted
def test_missing():
    # see GH#2083
    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, np.nan]}
    RLM.from_formula('Foo ~ Bar', data=d)
