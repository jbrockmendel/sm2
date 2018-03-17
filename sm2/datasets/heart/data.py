"""Heart Transplant Data, Miller 1976"""

__docformat__ = 'restructuredtext'

COPYRIGHT = """???"""

TITLE = """Transplant Survival Data"""

SOURCE = """Miller, R. (1976). Least squares regression with censored data.
Biometrica, 63 (3). 449-464.
"""

DESCRSHORT = """Survival times after receiving a heart transplant"""

DESCRLONG = """This data contains the survival time after receiving a
heart transplant, the age of the patient and whether or not the survival
time was censored.
"""

NOTE = """::

    Number of Observations - 69

    Number of Variables - 3

    Variable name definitions::
        death - Days after surgery until death
        age - age at the time of surgery
        censored - indicates if an observation is censored.  1 is uncensored
"""
import os
import numpy as np

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
    dset = du.process_recarray(data, endog_idx=0, exog_idx=None, dtype=float)
    dset.censors = dset.exog[:, 0]
    dset.exog = dset.exog[:, 1]
    return dset


def load_pandas():
    data = _get_data()
    # NOTE: None for exog_idx is the complement of endog_idx
    return du.process_recarray_pandas(data, endog_idx=0, exog_idx=None,
                                      dtype=float)


def _get_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'heart.csv')
    with open(path, 'rb') as fd:
        data = np.recfromtxt(fd, delimiter=",", names=True, dtype=float)
    return data
