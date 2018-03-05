import pandas as pd
import numpy as np
import pytest

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
def test_dataframe():
    X = np.random.random((10, 5))
    df = pd.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(vals, df.values)
    np.testing.assert_equal(rnames.tolist(), df.index.tolist())
    np.testing.assert_equal(cnames, df.columns.tolist())


@pytest.mark.not_vetted
def test_patsy_577():
    X = np.random.random((10, 2))
    df = pd.DataFrame(X, columns=["var1", "var2"])
    from patsy import dmatrix
    endog = dmatrix("var1 - 1", df)
    np.testing.assert_(data._is_using_patsy(endog, None))
    exog = dmatrix("var2 - 1", df)
    np.testing.assert_(data._is_using_patsy(endog, exog))
