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
    @classmethod
    def setup_class(cls):
        cls.data = sm.datasets.stackloss.load()
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)

    @pytest.mark.parametrize('attr', ['params', 'weights', 'resid',
                                      'df_model', 'df_resid',
                                      'bcov_unscaled'])
    def test_attr(self, attr):
        # Note: upstream incorrectly checks res1.model.df_model and df_resid
        if not hasattr(self.res2, attr):
            raise pytest.skip("No {attr} available in benchmark"
                              .format(attr=attr))
        result = getattr(self.res1, attr)
        expected = getattr(self.res2, attr)
        assert_almost_equal(result, expected, DECIMAL_4)

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


# -----------------------------------------------------------------

@pytest.mark.not_vetted
class TestRlm(CheckRlmResultsMixin):
    decimal_standarderrors = DECIMAL_1
    decimal_scale = DECIMAL_3
    res2 = results_rlm.Huber()
    M = norms.HuberT()

    @classmethod
    def setup_class(cls):
        super(TestRlm, cls).setup_class()

        results = RLM(cls.data.endog, cls.data.exog, M=cls.M).fit()
        h2 = RLM(cls.data.endog, cls.data.exog,
                 M=cls.M).fit(cov="H2").bcov_scaled
        h3 = RLM(cls.data.endog, cls.data.exog,
                 M=cls.M).fit(cov="H3").bcov_scaled
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
    M = norms.Hampel()


@pytest.mark.not_vetted
class TestRlmBisquare(TestRlm):
    decimal_standarderrors = DECIMAL_1
    res2 = results_rlm.BiSquare()
    M = norms.TukeyBiweight()


@pytest.mark.not_vetted
class TestRlmAndrews(TestRlm):
    res2 = results_rlm.Andrews()
    M = norms.AndrewWave()


# -----------------------------------------------------------------
# Tests with Huber scaling

@pytest.mark.not_vetted
class TestRlmHuber(CheckRlmResultsMixin):
    res2 = results_rlm.HuberHuber()
    scale_est = scale.HuberScale()
    M = norms.HuberT()

    @classmethod
    def setup_class(cls):
        super(TestRlmHuber, cls).setup_class()

        mod = RLM(cls.data.endog, cls.data.exog, M=cls.M)
        results = mod.fit(scale_est=cls.scale_est)

        # TODO: Do we need separate model objects?
        mod2 = RLM(cls.data.endog, cls.data.exog, M=cls.M)
        h2 = mod2.fit(cov="H2", scale_est=cls.scale_est).bcov_scaled

        mod3 = RLM(cls.data.endog, cls.data.exog, M=cls.M)
        h3 = mod3.fit(cov="H3", scale_est=cls.scale_est).bcov_scaled

        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


@pytest.mark.not_vetted
class TestHampelHuber(TestRlmHuber):
    res2 = results_rlm.HampelHuber()
    scale_est = scale.HuberScale()
    M = norms.Hampel()


@pytest.mark.not_vetted
class TestRlmBisquareHuber(TestRlmHuber):
    res2 = results_rlm.BisquareHuber()
    scale_est = scale.HuberScale()
    M = norms.TukeyBiweight()


@pytest.mark.not_vetted
class TestRlmAndrewsHuber(TestRlmHuber):
    res2 = results_rlm.AndrewsHuber()
    scale_est = scale.HuberScale()
    M = norms.AndrewWave()


# -----------------------------------------------------------------

@pytest.mark.not_vetted
class TestRlmSresid(CheckRlmResultsMixin):
    # Check GH#187
    res2 = results_rlm.Huber()
    M = norms.HuberT()
    decimal_standarderrors = DECIMAL_1
    decimal_scale = DECIMAL_3

    @classmethod
    def setup_class(cls):
        super(TestRlmSresid, cls).setup_class()
        results = RLM(cls.data.endog, cls.data.exog,
                      M=cls.M).fit(conv='sresid')  # default M

        h2 = RLM(cls.data.endog, cls.data.exog,
                 M=cls.M).fit(cov="H2").bcov_scaled

        h3 = RLM(cls.data.endog, cls.data.exog,
                 M=cls.M).fit(cov="H3").bcov_scaled

        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3


@pytest.mark.not_vetted
def test_missing():
    # see GH#2083
    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, np.nan]}
    RLM.from_formula('Foo ~ Bar', data=d)
