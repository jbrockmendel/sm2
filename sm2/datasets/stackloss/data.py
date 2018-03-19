"""Stack loss data"""

__docformat__ = 'restructuredtext'

COPYRIGHT = """This is public domain. """
TITLE = __doc__
SOURCE = """
Brownlee, K. A. (1965), "Statistical Theory and Methodology in
Science and Engineering", 2nd edition, New York:Wiley.
"""

DESCRSHORT = """Stack loss plant data of Brownlee (1965)"""

DESCRLONG = """The stack loss plant data of Brownlee (1965) contains
21 days of measurements from a plant's oxidation of ammonia to nitric acid.
The nitric oxide pollutants are captured in an absorption tower."""

NOTE = """::

    Number of Observations - 21

    Number of Variables - 4

    Variable name definitions::

        STACKLOSS - 10 times the percentage of ammonia going into the plant
                    that escapes from the absoroption column
        AIRFLOW   - Rate of operation of the plant
        WATERTEMP - Cooling water temperature in the absorption tower
        ACIDCONC  - Acid concentration of circulating acid minus 50 times 10.
"""
import os

import pandas as pd

from sm2.datasets import utils as du


def load():
    """
    Load the stack loss data and returns a Dataset class instance.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray(data, endog_idx=0, dtype=float)


def load_pandas():
    """
    Load the stack loss data and returns a Dataset class instance.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray_pandas(data, endog_idx=0, dtype=float)


def _get_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'stackloss.csv')
    data = pd.read_csv(path, float_precision='high')
    return data.astype('f8').to_records(index=False)
