"""Nile River Flows."""

__docformat__ = 'restructuredtext'

COPYRIGHT = """This is public domain."""
TITLE = """Nile River flows at Ashwan 1871-1970"""
SOURCE = """
This data is first analyzed in:

    Cobb, G. W. 1978. "The Problem of the Nile: Conditional Solution to a
        Changepoint Problem." *Biometrika*. 65.2, 243-51.
"""

DESCRSHORT = """This dataset contains measurements on the annual flow of
the Nile as measured at Ashwan for 100 years from 1871-1970."""

DESCRLONG = DESCRSHORT + " There is an apparent changepoint near 1898."

# suggested notes
NOTE = """::

    Number of observations: 100
    Number of variables: 2
    Variable name definitions:

        year - the year of the observations
        volumne - the discharge at Aswan in 10^8, m^3
"""
import os

import numpy as np
import pandas as pd

from sm2.datasets.utils import Dataset


def load():
    """
    Load the Nile data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    endog_name = 'volume'
    endog = np.array(data[endog_name], dtype=float)
    dataset = Dataset(data=data, names=[endog_name], endog=endog,
                      endog_name=endog_name)
    return dataset


def load_pandas():
    data = pd.DataFrame(_get_data())
    # TODO: time series
    endog = pd.Series(data['volume'], index=data['year'].astype(int))
    dataset = Dataset(data=data, names=list(data.columns),
                      endog=endog, endog_name='volume')
    return dataset


def _get_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'nile.csv')
    data = pd.read_csv(path, float_precision='high')
    return data.astype('f8').to_records(index=False)
