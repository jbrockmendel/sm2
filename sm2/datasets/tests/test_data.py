import importlib

import numpy as np
import pandas as pd
import nose
import pytest

import sm2.datasets
from sm2.datasets.utils import Dataset

exclude = ['check_internet', 'clear_data_home', 'get_data_home',
           'get_rdataset', 'tests', 'utils', 'webuse', 'template_data']
datasets = []
for dataset_name in dir(sm2.datasets):
    if not dataset_name.startswith('_') and dataset_name not in exclude:
        datasets.append(dataset_name)


# TODO: Remove nottest when nose support is dropped
@nose.tools.nottest
@pytest.mark.parametrize('dataset_name', datasets)
def test_dataset(dataset_name):
    dataset = importlib.import_module('sm2.datasets.' + dataset_name)
    data = dataset.load()
    assert isinstance(data, Dataset)
    assert isinstance(data.data, np.recarray)

    df_data = dataset.load_pandas()
    assert isinstance(df_data, Dataset)
    assert isinstance(df_data.data, pd.DataFrame)


# TODO: Remove when nose support is dropped
def test_all_datasets():
    for dataset in datasets:
        test_dataset(dataset)