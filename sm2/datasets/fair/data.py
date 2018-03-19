#! /usr/bin/env python

"""Fair's Extramarital Affairs Data"""

__docformat__ = 'restructuredtext'

COPYRIGHT = """Included with permission of the author."""
TITLE = """Affairs dataset"""
SOURCE = """
Fair, Ray. 1978. "A Theory of Extramarital Affairs," `Journal of Political
Economy`, February, 45-61.

The data is available at http://fairmodel.econ.yale.edu/rayfair/pdf/2011b.htm
"""

DESCRSHORT = """Extramarital affair data."""

DESCRLONG = """Extramarital affair data used to explain the allocation
of an individual's time among work, time spent with a spouse, and time
spent with a paramour. The data is used as an example of regression
with censored data."""

# suggested notes
NOTE = """::

    Number of observations: 6366
    Number of variables: 9
    Variable name definitions:

        rate_marriage   : How rate marriage, 1 = very poor, 2 = poor, 3 = fair,
                        4 = good, 5 = very good
        age             : Age
        yrs_married     : No. years married. Interval approximations. See
                        original paper for detailed explanation.
        children        : No. children
        religious       : How relgious, 1 = not, 2 = mildly, 3 = fairly,
                        4 = strongly
        educ            : Level of education, 9 = grade school, 12 = high
                        school, 14 = some college, 16 = college graduate,
                        17 = some graduate school, 20 = advanced degree
        occupation      : 1 = student, 2 = farming, agriculture; semi-skilled,
                        or unskilled worker; 3 = white-colloar; 4 = teacher
                        counselor social worker, nurse; artist, writers;
                        technician, skilled worker, 5 = managerial,
                        administrative, business, 6 = professional with
                        advanced degree
        occupation_husb : Husband's occupation. Same as occupation.
        affairs         : measure of time spent in extramarital affairs

    See the original paper for more details.
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
    # NOTE: None for exog_idx is the complement of endog_idx
    return du.process_recarray(data, endog_idx=8, exog_idx=None, dtype=float)


def load_pandas():
    data = _get_data()
    # NOTE: None for exog_idx is the complement of endog_idx
    return du.process_recarray_pandas(data, endog_idx=8, exog_idx=None,
                                      dtype=float)


def _get_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'fair.csv')
    data = pd.read_csv(path).astype('f8').to_records(index=False)
    return data
