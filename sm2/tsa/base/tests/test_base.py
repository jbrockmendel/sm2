from datetime import datetime

import numpy as np
import pandas as pd
import pandas.util.testing as tm

import pytest

from sm2.tsa.base.tsa_model import TimeSeriesModel


@pytest.mark.not_vetted
def test_pandas_nodates_index():
    data = [988, 819, 964]
    dates = ['a', 'b', 'c']
    s = pd.Series(data, index=dates)

    # TODO: Remove this, this is now valid
    # np.testing.assert_raises(ValueError, TimeSeriesModel, s)

    # Test with a non-date index that doesn't raise an exception because it
    # can be coerced into a nanosecond DatetimeIndex
    # (This test doesn't make sense for Numpy < 1.7 since they don't have
    # nanosecond support)
    # (This test also doesn't make sense for Pandas < 0.14 since we don't
    # support nanosecond index in Pandas < 0.14)
    # Upstream included a check for numpy < 1.7, but we require much more
    # recent numpy
    data = [988, 819, 964]
    # index=pd.date_range('1970-01-01', periods=3, freq='QS')
    index = pd.to_datetime([100, 101, 102])
    s = pd.Series(data, index=index)

    actual_str = (index[0].strftime('%Y-%m-%d %H:%M:%S.%f') +
                  str(index[0].value))
    assert actual_str == '1970-01-01 00:00:00.000000100'
    mod = TimeSeriesModel(s)
    start, end, out_of_sample, _ = mod._get_prediction_index(0, 4)
    assert len(mod.data.predict_dates) == 5


@pytest.mark.not_vetted
def test_predict_freq():
    # test that predicted dates have same frequency
    x = np.arange(1, 36.)

    # there's a bug in pandas up to 0.10.2 for YearBegin
    dates = pd.date_range("1972-4-30", "2006-4-30", freq="A-APR")
    series = pd.Series(x, index=dates)
    model = TimeSeriesModel(series)
    assert model._index.freqstr == "A-APR"

    start, end, out_of_sample, _ = (
        model._get_prediction_index("2006-4-30", "2016-4-30"))

    predict_dates = model.data.predict_dates

    expected_dates = pd.date_range("2006-4-30", "2016-4-30", freq="A-APR")
    tm.assert_index_equal(predict_dates, expected_dates)


@pytest.mark.not_vetted
def test_keyerror_start_date():
    x = np.arange(1, 36.)

    dates = pd.date_range("1972-4-30", "2006-4-30", freq="A-APR")
    series = pd.Series(x, index=dates)
    model = TimeSeriesModel(series)

    with pytest.raises(KeyError):
        model._get_prediction_index("1970-4-30", None)


@pytest.mark.not_vetted
def test_period_index():
    # GH#1285
    dates = pd.PeriodIndex(start="1/1/1990", periods=20, freq="M")
    x = np.arange(1, 21.)

    model = TimeSeriesModel(pd.Series(x, index=dates))
    assert model._index.freqstr == "M"
    model = TimeSeriesModel(pd.Series(x, index=dates))
    assert model.data.freq == "M"


@pytest.mark.not_vetted
def test_pandas_dates():
    data = [988, 819, 964]
    dates = ['2016-01-01 12:00:00',
             '2016-02-01 12:00:00',
             '2016-03-01 12:00:00']

    datetime_dates = pd.to_datetime(dates)

    result = pd.Series(data=data, index=datetime_dates, name='price')
    df = pd.DataFrame(data={'price': data},
                      index=pd.DatetimeIndex(dates, freq='MS'))

    model = TimeSeriesModel(df['price'])
    tm.assert_index_equal(result.index, model.data.dates)


@pytest.mark.not_vetted
def test_get_predict_start_end():
    index = pd.DatetimeIndex(start='1970-01-01', end='1990-01-01', freq='AS')
    endog = pd.Series(np.zeros(10), index[:10])
    model = TimeSeriesModel(endog)

    predict_starts = [1, '1971-01-01', datetime(1971, 1, 1), index[1]]
    predict_ends = [20, '1990-01-01', datetime(1990, 1, 1), index[-1]]

    desired = (1, 9, 11)
    for start in predict_starts:
        for end in predict_ends:
            assert model._get_prediction_index(start, end)[:3] == desired
