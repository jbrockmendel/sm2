"""Bill Greene's credit scoring data."""

__docformat__ = 'restructuredtext'

COPYRIGHT = """Used with express permission of the original author, who
retains all rights."""
TITLE = __doc__
SOURCE = """
William Greene's `Econometric Analysis`

More information can be found at the web site of the text:
http://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm
"""

DESCRSHORT = """William Greene's credit scoring data"""

DESCRLONG = """More information on this data can be found on the
homepage for Greene's `Econometric Analysis`. See source.
"""

NOTE = """::

    Number of observations - 72
    Number of variables - 5
    Variable name definitions - See Source for more information on the
                                variables.
"""
import os

import pandas as pd

from sm2.datasets import utils as du


def load():
    """Load the credit card data and returns a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray(data, endog_idx=0, dtype=float)


def load_pandas():
    """Load the credit card data and returns a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray_pandas(data, endog_idx=0)


def _get_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'ccard.csv')
    data = pd.read_csv(path, float_precision='high')
    return data.astype('f8').to_records(index=False)
