
import os

from numpy.testing import assert_, assert_array_equal
from pandas.util.testing import assert_frame_equal
import pytest

from sm2.datasets import get_rdataset, webuse, check_internet, utils, macrodata

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.not_vetted
@pytest.mark.smoke
def test_get_rdataset():
    # smoke test
    test_url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/cars.csv"
    internet_available = check_internet(test_url)
    if not internet_available:
        raise pytest.skip('Unable to retrieve file - skipping test')
    duncan = get_rdataset("Duncan", "car", cache=cur_dir)
    assert_(isinstance(duncan, utils.Dataset))
    assert_(duncan.from_cache)


@pytest.mark.not_vetted
def test_webuse():
    # test copied and adjusted from iolib/tests/test_foreign
    from sm2.iolib.tests.results.macrodata import macrodata_result as res2
    base_gh = "http://github.com/statsmodels/statsmodels/raw/master/sm2.datasets/macrodata/"
    internet_available = check_internet(base_gh)
    if not internet_available:
        raise pytest.skip('Unable to retrieve file - skipping test')
    res1 = webuse('macrodata', baseurl=base_gh, as_df=False)
    assert_array_equal(res1, res2)


@pytest.mark.not_vetted
def test_webuse_pandas():
    # test copied and adjusted from iolib/tests/test_foreign
    dta = macrodata.load_pandas().data
    base_gh = "http://github.com/statsmodels/statsmodels/raw/master/sm2.datasets/macrodata/"
    internet_available = check_internet(base_gh)
    if not internet_available:
        raise pytest.skip('Unable to retrieve file - skipping test')
    res1 = webuse('macrodata', baseurl=base_gh)
    res1 = res1.astype(float)
    assert_frame_equal(res1, dta)
