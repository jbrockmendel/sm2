
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from sm2.stats import contrast


@pytest.mark.skip(reason="Contrast not ported from upstream")
@pytest.mark.not_vetted
class TestContrast(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = np.random.standard_normal((40, 10))

    def test_contrast1(self):
        term = np.column_stack((self.X[:, 0], self.X[:, 2]))
        c = contrast.Contrast(term, self.X)
        test_contrast = [[1] + [0] * 9, [0] * 2 + [1] + [0] * 7]
        assert_almost_equal(test_contrast, c.contrast_matrix)

    def test_contrast2(self):
        zero = np.zeros((40,))
        term = np.column_stack((zero, self.X[:, 2]))
        c = contrast.Contrast(term, self.X)
        test_contrast = [0] * 2 + [1] + [0] * 7
        assert_almost_equal(test_contrast, c.contrast_matrix)

    def test_contrast3(self):
        P = np.dot(self.X, np.linalg.pinv(self.X))
        resid = np.identity(40) - P
        noise = np.dot(resid, np.random.standard_normal((40, 5)))
        term = np.column_stack((noise, self.X[:, 2]))
        c = contrast.Contrast(term, self.X)
        assert_equal(c.contrast_matrix.shape, (10,))
    # TODO: this should actually test the value of the contrast,
    # not only its dimension

    def test_estimable(self):
        X2 = np.column_stack((self.X, self.X[:, 5]))
        c = contrast.Contrast(self.X[:, 5], X2)
        # TODO: I don't think this should be estimable?  isestimable correct?
        # TODO: do something with c?
