"""local, adjusted version from scipy.linalg.basic.py


changes:
The only changes are that additional results are returned

"""
# TODO: This is only needed (so far) by var_model.  Maybe avoid porting?

import numpy as np
import scipy.linalg

eps = np.finfo(float).eps
feps = np.finfo(np.single).eps
_array_precision = {'f': 0, 'd': 1, 'F': 0, 'D': 1}


# Linear Least Squares

def lstsq(a, b, cond=None, overwrite_a=0, overwrite_b=0):  # pragma: no cover
    raise NotImplementedError("lstsq not ported from upstream")


def pinv(a, cond=None, rcond=None):  # pragma: no cover
    raise NotImplementedError("pinv not ported from upstream")


def pinv2(a, cond=None, rcond=None):  # pragma: no cover
    raise NotImplementedError("pinv2 not ported from upstream")


def logdet_symm(m, check_symm=False):
    """
    Return log(det(m)) asserting positive definiteness of m.

    Parameters
    ----------
    m : array-like
        2d array that is positive-definite (and symmetric)

    Returns
    -------
    logdet : float
        The log-determinant of m.
    """
    if check_symm:
        if not np.all(m == m.T):  # would be nice to short-circuit check
            raise ValueError("m is not symmetric.")
    c, _ = scipy.linalg.cho_factor(m, lower=True)
    return 2 * np.sum(np.log(c.diagonal()))


def stationary_solve(r, b):  # pragma: no cover
    raise NotImplementedError("stationary_solve not ported from upstream")
