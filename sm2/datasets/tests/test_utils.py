
import os

import numpy as np
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal
import pytest

from sm2.datasets import get_rdataset, webuse, check_internet, utils, macrodata

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.not_vetted
@pytest.mark.smoke
@pytest.mark.xfail(reason="Files arent where it expects them to be")
def test_get_rdataset():
    test_url = ("https://raw.githubusercontent.com/vincentarelbundock/"
                "Rdatasets/master/csv/datasets/cars.csv")
    internet_available = check_internet(test_url)
    if not internet_available:
        raise pytest.skip('Unable to retrieve file - skipping test')
    duncan = get_rdataset("Duncan", "carData", cache=cur_dir)
    assert isinstance(duncan, utils.Dataset)
    # FIXME: This only seems to be True if the file is already present
    assert duncan.from_cache

    # test writing and reading cache
    guerry = get_rdataset("Guerry", "HistData", cache=cur_dir)
    assert guerry.from_cache is False
    guerry2 = get_rdataset("Guerry", "HistData", cache=cur_dir)
    assert guerry2.from_cache is True
    fn = ("raw.githubusercontent.com,vincentarelbundock,Rdatasets,"
          "master,csv,HistData,Guerry.csv.zip")
    os.remove(os.path.join(cur_dir, fn))
    fn = ("raw.githubusercontent.com,vincentarelbundock,Rdatasets,"
          "master,doc,HistData,rst,Guerry.rst.zip")
    os.remove(os.path.join(cur_dir, fn))


@pytest.mark.not_vetted
def test_webuse():
    # test copied and adjusted from iolib/tests/test_foreign
    base_gh = ("https://github.com/statsmodels/statsmodels/raw/master/"
               "statsmodels/datasets/macrodata/")
    internet_available = check_internet(base_gh)
    if not internet_available:
        raise pytest.skip('Unable to retrieve file - skipping test')

    df = macrodata.load_pandas().data.astype('f4')
    df['year'] = df['year'].astype('i2')
    df['quarter'] = df['quarter'].astype('i1')
    expected = df.to_records(index=False)
    expected = np.array([list(row) for row in expected])
    res1 = webuse('macrodata', baseurl=base_gh, as_df=False)
    assert_array_equal(res1, expected)


@pytest.mark.not_vetted
def test_webuse_pandas():
    # test copied and adjusted from iolib/tests/test_foreign
    dta = macrodata.load_pandas().data
    base_gh = ("https://github.com/statsmodels/statsmodels/raw/master/"
               "statsmodels/datasets/macrodata/")
    internet_available = check_internet(base_gh)
    if not internet_available:
        raise pytest.skip('Unable to retrieve file - skipping test')

    res1 = webuse('macrodata', baseurl=base_gh)
    res1 = res1.astype(float)
    assert_frame_equal(res1, dta.astype(float))
