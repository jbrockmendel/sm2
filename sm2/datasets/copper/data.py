"""World Copper Prices 1951-1975 dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT = """Used with express permission from the original author,
who retains all rights."""
TITLE = "World Copper Market 1951-1975 Dataset"
SOURCE = """
Jeff Gill's `Generalized Linear Models: A Unified Approach`

http://jgill.wustl.edu/research/books.html
"""

DESCRSHORT = """World Copper Market 1951-1975"""

DESCRLONG = """This data describes the world copper market from 1951 through
1975.  In an example, in Gill, the outcome variable (of a 2 stage estimation)
is the world consumption of copper for the 25 years.  The explanatory
variables are the world consumption of copper in 1000 metric tons, the
constant dollar adjusted price of copper, the price of a substitute,
aluminum, an index of real per capita income base 1970, an annual measure
of manufacturer inventory change, and a time trend.
"""

NOTE = """
Number of Observations - 25

Number of Variables - 6

Variable name definitions::

    WORLDCONSUMPTION - World consumption of copper (in 1000 metric tons)
    COPPERPRICE - Constant dollar adjusted price of copper
    INCOMEINDEX - An index of real per capita income (base 1970)
    ALUMPRICE - The price of aluminum
    INVENTORYINDEX - A measure of annual manufacturer inventory trend
    TIME - A time trend

Years are included in the data file though not returned by load.
"""
import os

import pandas as pd

from sm2.datasets import utils as du


def load():
    """
    Load the copper data and returns a Dataset class.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray(data, endog_idx=0, dtype=float)


def _get_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'copper.csv')
    data = pd.read_csv(path, float_precision='high')
    return data.iloc[:, 1:7].astype('f8').to_records(index=False)


def load_pandas():
    """
    Load the copper data and returns a Dataset class.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray_pandas(data, endog_idx=0, dtype=float)
