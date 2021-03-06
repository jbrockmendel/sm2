import importlib

from six import PY3
import numpy as np
import pandas as pd
import pytest

import sm2.datasets
from sm2.datasets.utils import Dataset

exclude = ['check_internet', 'clear_data_home', 'get_data_home',
           'get_rdataset', 'tests', 'utils', 'webuse']
datasets = []
for dataset_name in dir(sm2.datasets):
    if not dataset_name.startswith('_') and dataset_name not in exclude:
        datasets.append(dataset_name)


@pytest.mark.not_vetted
@pytest.mark.parametrize('dataset_name', datasets)
def test_dataset(dataset_name):
    dataset = importlib.import_module('sm2.datasets.' + dataset_name)
    warning_type = FutureWarning if PY3 else None
    with pytest.warns(warning_type):
        ds = dataset.load()

    assert isinstance(ds, Dataset)
    assert isinstance(ds.data, np.recarray)
    if hasattr(ds, 'exog'):
        assert isinstance(ds.exog, np.ndarray)
    if hasattr(ds, 'endog'):
        assert isinstance(ds.endog, np.ndarray)

    ds = dataset.load(as_pandas=True)
    assert isinstance(ds, Dataset)
    assert isinstance(ds.data, pd.DataFrame)
    if hasattr(ds, 'exog'):
        assert isinstance(ds.exog, (pd.DataFrame, pd.Series))
    if hasattr(ds, 'endog'):
        assert isinstance(ds.endog, (pd.DataFrame, pd.Series))

    ds = dataset.load_pandas()
    assert isinstance(ds, Dataset)
    assert isinstance(ds.data, pd.DataFrame)
    if hasattr(ds, 'exog'):
        assert isinstance(ds.exog, (pd.DataFrame, pd.Series))
    if hasattr(ds, 'endog'):
        assert isinstance(ds.endog, (pd.DataFrame, pd.Series))
