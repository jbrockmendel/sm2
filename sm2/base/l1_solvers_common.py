"""
Holds common functions for l1 solvers.
"""
from __future__ import print_function

from six.moves import range
import numpy as np


def qc_results(params, alpha, score, qc_tol, qc_verbose=False):
    """
    Theory dictates that one of two conditions holds:
        i) abs(score[i]) == alpha[i]  and  params[i] != 0
        ii) abs(score[i]) <= alpha[i]  and  params[i] == 0
    qc_results checks to see that (ii) holds, within qc_tol

    qc_results also checks for nan or results of the wrong shape.

    Parameters
    ----------
    params : np.ndarray
        model parameters.  Not including the added variables x_added.
    alpha : np.ndarray
        regularization coefficients
    score : function
        Gradient of unregularized objective function
    qc_tol : float
        Tolerance to hold conditions (i) and (ii) to for QC check.
    qc_verbose : Boolean
        If true, print out a full QC report upon failure

    Returns
    -------
    passed : Boolean
        True if QC check passed
    qc_dict : Dictionary
        Keys are fprime, alpha, params, passed_array

    Prints
    ------
    Warning message if QC check fails.
    """
    if qc_verbose:  # pragma: no cover
        # TODO: Update docstring to reflect this restriction
        raise NotImplementedError("option `qc_verbose` is available upstream, "
                                  "but is disabled in sm2.")

    # Check for fatal errors
    assert not np.isnan(params).max()
    assert (params == params.ravel('F')).min(), "params should be 1-d"

    # Start the theory compliance check
    fprime = score(params)
    k_params = len(params)

    passed_array = np.array([True] * k_params)
    for i in range(k_params):
        if alpha[i] > 0:
            # If |fprime| is too big, then something went wrong
            if (abs(fprime[i]) - alpha[i]) / alpha[i] > qc_tol:
                passed_array[i] = False

    passed = passed_array.min()
    if not passed:
        num_failed = (passed_array == False).sum()  # noqa:E712
        message = ('QC check did not pass for %d out of %d parameters'
                   % (num_failed, k_params))
        message += ('\nTry increasing solver accuracy or number of iterations'
                    ', decreasing alpha, or switch solvers')
        print(message)  # TODO: Don't print

    return passed


def _get_verbose_addon(qc_dict):  # pragma: no cover
    raise NotImplementedError("_get_verbose_addon not ported from upstream")


def do_trim_params(params, k_params, alpha, score, passed, trim_mode,
                   size_trim_tol, auto_trim_tol):
    """
    Trims (set to zero) params that are zero at the theoretical minimum.
    Uses heuristics to account for the solver not actually finding the minimum.

    In all cases, if alpha[i] == 0, then don't trim the ith param.
    In all cases, do nothing with the added variables.

    Parameters
    ----------
    params : np.ndarray
        model parameters.  Not including added variables.
    k_params : Int
        Number of parameters
    alpha : np.ndarray
        regularization coefficients
    score : Function.
        score(params) should return a 1-d vector of derivatives of the
        unpenalized objective function.
    passed : Boolean
        True if the QC check passed
    trim_mode : 'auto, 'size', or 'off'
        If not 'off', trim (set to zero) parameters that would have been zero
            if the solver reached the theoretical minimum.
        If 'auto', trim params using the Theory above.
        If 'size', trim params if they have very small absolute value
    size_trim_tol : float or 'auto' (default = 'auto')
        For use when trim_mode === 'size'
    auto_trim_tol : float
        For sue when trim_mode == 'auto'.  Use
    qc_tol : float
        Print warning and don't allow auto trim when (ii) in "Theory" (above)
        is violated by this much.

    Returns
    -------
    params : np.ndarray
        Trimmed model parameters
    trimmed : np.ndarray of Booleans
        trimmed[i] == True if the ith parameter was trimmed.
    """
    # Trim the small params
    trimmed = np.array([False] * k_params)

    if trim_mode == 'off':
        pass
    elif trim_mode == 'auto' and not passed:
        # TODO: Dont print; warn
        print("Could not trim params automatically due to failed QC "
              "check.  Trimming using trim_mode == 'size' will still work.")
    elif trim_mode == 'auto' and passed:
        fprime = score(params)
        for i in range(k_params):
            if alpha[i] != 0:
                if (alpha[i] - abs(fprime[i])) / alpha[i] > auto_trim_tol:
                    params[i] = 0.0
                    trimmed[i] = True
    elif trim_mode == 'size':
        # TODO: Is it the case that k_params == len(params)??
        mask = (alpha != 0) & (np.abs(params) < size_trim_tol)
        trimmed[mask] = True
        params[mask] = 0.0
    else:
        raise ValueError("trim_mode == %s, which is not recognized"
                         % (trim_mode))  # pragma: no cover

    return params, np.asarray(trimmed)
