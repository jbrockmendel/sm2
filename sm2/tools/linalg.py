#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""local, adjusted version from scipy.linalg.basic.py


changes:
The only changes are that additional results are returned

"""
from six.moves import reduce

import numpy as np
import scipy.linalg
from scipy import sparse
import scipy.sparse.linalg  # needed to get linalg into sparse namespace

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


# upstream this is implemented in regression.mixed_linear_model._smw_solver
def smw_solver(s, A, AtA, Qi, di):  # TODO: Clarify I think A*B*A' is matmul
    """
    Returns a solver for the linear system:

    .. math::

        (sI + ABA^\prime) y = x

    The returned function f satisfies f(x) = y as defined above.

    B and its inverse matrix are block diagonal.  The upper left block
    of :math:`B^{-1}` is Qi and its lower right block is diag(di).

    The inverse matrix of B is block diagonal.  The upper left block
    is BI and the lower right block is a diagonal matrix containing
    di.

    Parameters
    ----------
    s : scalar
        See above for usage
    A : ndarray
        p x q matrix, in general q << p, may be sparse.
    AtA : square ndarray
        :math:`A^\prime  A`, a q x q matrix.
    Qi : square symmetric ndarray
        The matrix `B` is q x q, where q = r + s.  `B` consists of a r
        x r diagonal block whose inverse is `Qi`, and a s x s diagonal
        block, whose inverse is diag(di).
    di : 1d array-like
        See documentation for Qi.

    Returns
    -------
    A function for solving a linear system, as documented above.

    Notes
    -----
    Uses Sherman-Morrison-Woodbury identity:
        https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    """
    # Use SMW identity
    qmat = AtA / s
    if sparse.issparse(qmat):
        qmat = qmat.todense()
    m = Qi.shape[0]
    qmat[0:m, 0:m] += Qi
    d = qmat.shape[0]
    qmat.flat[m * (d + 1)::d + 1] += di

    is_sparse = sparse.issparse(A)

    if is_sparse:
        qmati = sparse.linalg.spsolve(sparse.csc_matrix(qmat), A.T)
    else:
        qmati = np.linalg.solve(qmat, A.T)

    if is_sparse:
        def solver(rhs):
            ql = qmati.dot(rhs)
            ql = A.dot(ql)
            return rhs / s - ql / s**2
    else:
        def solver(rhs):
            ql = np.dot(qmati, rhs)
            ql = np.dot(A, ql)
            return rhs / s - ql / s**2

    return solver


# upstream this is implemented in regression.mixed_linear_model._smw_logdet
def smw_logdet(s, A, AtA, Qi, di, B_logdet):
    """
    Returns the log determinant of

    .. math::

        sI + ABA^\prime

    Uses the matrix determinant lemma to accelerate the calculation.
    B is assumed to be positive semidefinite, and s > 0, therefore the
    determinant is positive.

    Parameters
    ----------
    s : positive scalar
        See above for usage
    A : ndarray
        p x q matrix, in general q << p.
    AtA : square ndarray
        :math:`A^\prime  A`, a q x q matrix.
    Qi : square symmetric ndarray
        The matrix `B` is q x q, where q = r + s.  `B` consists of a r
        x r diagonal block whose inverse is `Qi`, and a s x s diagonal
        block, whose inverse is diag(di).
    di : 1d array-like
        See documentation for Qi.
    B_logdet : real
        The log determinant of B
    Returns
    -------
    The log determinant of s*I + A*B*A'.
    Notes
    -----
    Uses the matrix determinant lemma:
        https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    """
    p = A.shape[0]
    ld = p * np.log(s)
    qmat = AtA / s
    m = Qi.shape[0]
    qmat[0:m, 0:m] += Qi
    d = qmat.shape[0]
    qmat.flat[m * (d + 1)::d + 1] += di
    _, ld1 = np.linalg.slogdet(qmat)
    return B_logdet + ld + ld1


# upstream this is in tools.tools
def pinv_extended(X, rcond=1e-15):
    """
    Return the pinv of an array X as well as the singular values
    used in computation.

    Code adapted from numpy.
    """
    X = np.asarray(X)
    X = X.conjugate()
    u, s, vt = np.linalg.svd(X, 0)
    s_orig = np.copy(s)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1. / s[i]
        else:
            s[i] = 0.
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.core.newaxis],
                                               np.transpose(u)))
    return res, s_orig


# -------------------------------------------------------------------
# Multiplication-like Functions

def chain_dot(*arrs):
    """
    Returns the dot product of the given matrices.

    Parameters
    ----------
    arrs: argument list of ndarray

    Returns
    -------
    Dot product of all arguments.

    Examples
    --------
    >>> import numpy as np
    >>> from sm2.tools import chain_dot
    >>> A = np.arange(1,13).reshape(3,4)
    >>> B = np.arange(3,15).reshape(4,3)
    >>> C = np.arange(5,8).reshape(3,1)
    >>> chain_dot(A,B,C)
    array([[1820],
       [4300],
       [6780]])
    """
    return reduce(lambda x, y: np.dot(y, x), arrs[::-1])


def nan_dot(A, B):
    """
    Returns np.dot(left_matrix, right_matrix) with the convention that
    nan * 0 = 0 and nan * x = nan if x != 0.

    Parameters
    ----------
    A, B : np.ndarrays
    """
    # Find out who should be nan due to nan * nonzero
    should_be_nan_1 = np.dot(np.isnan(A), (B != 0))
    should_be_nan_2 = np.dot((A != 0), np.isnan(B))
    should_be_nan = should_be_nan_1 + should_be_nan_2

    # Multiply after setting all nan to 0
    # This is what happens if there were no nan * nonzero conflicts
    C = np.dot(np.nan_to_num(A), np.nan_to_num(B))

    C[should_be_nan] = np.nan

    return C


# upstream this is implemented in regression.mixed_linear_model
def _dotsum(x, y):
    """
    Returns sum(x * y), where '*' is the pointwise product, computed
    efficiently for dense and sparse matrices.
    """
    if sparse.issparse(x):
        return x.multiply(y).sum()
    else:
        # This way usually avoids allocating a temporary.
        return np.dot(x.ravel(), y.ravel())


# upstream this is implemented in regression.mixed_linear_model
def _dot(x, y):
    """
    Returns the dot product of the arrays, works for sparse and dense.
    """
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.dot(x, y)
    elif sparse.issparse(x):
        return x.dot(y)
    elif sparse.issparse(y):
        return y.T.dot(x.T).T


# upstream this is implemented in regression.mixed_linear_model
def _multi_dot_three(A, B, C):
    """
    Find best ordering for three arrays and do the multiplication.

    Doing in manually instead of using dynamic programing is
    approximately 15 times faster.

    From numpy, adapted to work with sparse and dense arrays.
    """
    # cost1 = cost((AB)C)
    cost1 = (A.shape[0] * A.shape[1] * B.shape[1] +  # (AB)
             A.shape[0] * B.shape[1] * C.shape[1])   # (--)C
    # cost2 = cost((AB)C)
    cost2 = (B.shape[0] * B.shape[1] * C.shape[1] +  # (BC)
             A.shape[0] * A.shape[1] * C.shape[1])   # A(--)

    if cost1 < cost2:
        return _dot(_dot(A, B), C)
    else:
        return _dot(A, _dot(B, C))
