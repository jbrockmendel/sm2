import pandas as pd
import numpy as np
import pytest

from patsy import dmatrix

from sm2.tools import data


@pytest.mark.not_vetted
def test_missing_data_pandas():
    # GH#144
    X = np.random.random((10, 5))
    X[1, 2] = np.nan
    df = pd.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(rnames.tolist(), [0, 2, 3, 4, 5, 6, 7, 8, 9])


@pytest.mark.not_vetted
def test_structarray():
    X = np.random.random((9,)).view([('var1', 'f8'),
                                     ('var2', 'f8'),
                                     ('var3', 'f8')])
    vals, cnames, rnames = data.interpret_data(X)
    np.testing.assert_equal(cnames, X.dtype.names)
    np.testing.assert_equal(vals, X.view((float, 3)))
    np.testing.assert_equal(rnames, None)


@pytest.mark.not_vetted
def test_recarray():
    X = np.random.random((9,)).view([('var1', 'f8'),
                                     ('var2', 'f8'),
                                     ('var3', 'f8')])
    vals, cnames, rnames = data.interpret_data(X.view(np.recarray))
    np.testing.assert_equal(cnames, X.dtype.names)
    np.testing.assert_equal(vals, X.view((float, 3)))
    np.testing.assert_equal(rnames, None)


@pytest.mark.not_vetted
def test_dataframe():  # TODO: GH Reference?
    X = np.random.random((10, 5))
    df = pd.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(vals, df.values)
    np.testing.assert_equal(rnames.tolist(), df.index.tolist())
    np.testing.assert_equal(cnames, df.columns.tolist())


@pytest.mark.not_vetted
def test_patsy_with_none_exog():
    # GH#577 when exog is None, make sure is_using_patsy is still correct
    X = np.random.random((10, 2))
    df = pd.DataFrame(X, columns=["var1", "var2"])

    endog = dmatrix("var1 - 1", df)
    assert data._is_using_patsy(endog, None)

    exog = dmatrix("var2 - 1", df)
    assert data._is_using_patsy(endog, exog)


# moved from test_discrete upstream
def test_isdummy():
    X = np.random.random((50, 10))
    X[:, 2] = np.random.randint(1, 10, size=50)
    X[:, 6] = np.random.randint(0, 2, size=50)
    X[:, 4] = np.random.randint(0, 2, size=50)
    X[:, 1] = np.random.randint(-10, 10, size=50)  # not integers
    count_ind = data.isdummy(X)
    np.testing.assert_equal(count_ind, [4, 6])


# moved from test_discrete upstream
def test_iscount():
    X = np.random.random((50, 10))
    X[:, 2] = np.random.randint(1, 10, size=50)
    X[:, 6] = np.random.randint(1, 10, size=50)
    X[:, 4] = np.random.randint(0, 2, size=50)
    X[:, 1] = np.random.randint(-10, 10, size=50)  # not integers
    count_ind = data.iscount(X)
    np.testing.assert_equal(count_ind, [2, 6])
