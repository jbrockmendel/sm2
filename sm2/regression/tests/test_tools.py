
import pytest
import numpy as np
from numpy.testing import assert_allclose

from sm2.regression.linear_model import WLS
from sm2.regression._tools import _MinimalWLS


@pytest.mark.not_vetted
class TestMinimalWLS(object):
    @classmethod
    def setup_class(cls):
        rs = np.random.RandomState(1234)
        # TODO: Is this necessary or is this a "forall" test?

        cls.exog1 = rs.randn(200, 5)
        cls.endog1 = cls.exog1.sum(1) + rs.randn(200)
        cls.weights1 = 1.0 + np.sin(np.arange(200.0) / 100.0 * np.pi)

        cls.exog2 = rs.randn(50, 1)
        cls.endog2 = 0.3 * cls.exog2.ravel() + rs.randn(50)
        cls.weights2 = 1.0 + np.log(np.arange(1.0, 51.0))

    def test_equivalence_unweighted(self):
        res = WLS(self.endog1, self.exog1).fit()
        minres = _MinimalWLS(self.endog1, self.exog1).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

    def test_equivalence_unweighted2(self):
        # TODO: Better name than 1 vs 2?
        res = WLS(self.endog2, self.exog2).fit()
        minres = _MinimalWLS(self.endog2, self.exog2).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

    def test_equivalence_weighted(self):
        res = WLS(self.endog1, self.exog1, weights=self.weights1).fit()
        minres = _MinimalWLS(self.endog1, self.exog1,
                             weights=self.weights1).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

    def test_equivalence_weighted2(self):
        # TODO: Better name than 1 vs 2?
        res = WLS(self.endog2, self.exog2, weights=self.weights2).fit()
        minres = _MinimalWLS(self.endog2, self.exog2,
                             weights=self.weights2).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)
