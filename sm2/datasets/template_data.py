#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Name of dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT = """E.g., This is public domain."""
TITLE = """Title of the dataset"""
SOURCE = """
This section should provide a link to the original dataset if possible and
attribution and correspondance information for the dataset's original author
if so desired.
"""

DESCRSHORT = """A short description."""

DESCRLONG = """A longer description of the dataset."""

# suggested notes
NOTE = """
::

    Number of observations:
    Number of variables:
    Variable name definitions:

Any other useful information that does not fit into the above categories.
"""
import os

import pandas as pd

from sm2.datasets import utils as du


def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    # SET THE INDICES #
    # NOTE: None for exog_idx is the complement of endog_idx
    return du.process_recarray(data, endog_idx=0, exog_idx=None, dtype=float)


def load_pandas():
    data = _get_data()
    # SET THE INDICES #
    # NOTE: None for exog_idx is the complement of endog_idx
    return du.process_recarray_pandas(data, endog_idx=0, exog_idx=None,
                                      dtype=float)


def _get_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # EDIT THE FOLLOWING TO POINT TO DatasetName.csv #
    path = os.path.join(cur_dir, 'DatasetName.csv')
    data = pd.read_csv(path, float_precision='high')
    return data.astype('f8').to_records(index=False)
