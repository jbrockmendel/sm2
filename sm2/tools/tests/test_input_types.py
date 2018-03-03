import pandas as pd
import numpy as np
import patsy

from numpy.testing import assert_equal

from sm2.tools import input_types


# ------------------------------------------------------------------
# Issue Regression Tests

# Deprecated interpret_data 2017-10-12
# def test_missing_data_pandas():
#    # GH144
#    X = np.random.random((10, 5))
#    X[1, 2] = np.nan
#    df = pd.DataFrame(X)
#    vals, cnames, rnames = data.interpret_data(df)
#    
#    assert_equal(rnames.tolist(), [0, 2, 3, 4, 5, 6, 7, 8, 9])


# TODO: Better name
def test_patsy_577():
    # GH#577
    X = np.random.random((10, 2))
    df = pd.DataFrame(X, columns=["var1", "var2"])
    
    endog = patsy.dmatrix("var1 - 1", df)
    assert input_types.is_using_patsy(endog, None)

    exog = patsy.dmatrix("var2 - 1", df)
    assert input_types.is_using_patsy(endog, exog)
