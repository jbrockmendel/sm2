#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARMA process and estimation with scipy.signal.lfilter

Notes
-----
* written without textbook, works but not sure about everything
  briefly checked and it looks to be standard least squares, see below

* theoretical autocorrelation function of general ARMA
  Done, relatively easy to guess solution, time consuming to get
  theoretical test cases, example file contains explicit formulas for
  acovf of MA(1), MA(2) and ARMA(1,1)

Properties:
Judge, ... (1985): The Theory and Practise of Econometrics

Author: josefpktd
License: BSD
"""
from six.moves import range
import numpy as np
from scipy import signal, optimize, linalg
from pandas.util._decorators import deprecate_kwarg

from sm2.tools.decorators import copy_doc
from sm2.tsa import wold

__all__ = ['arma_acf', 'arma_acovf', 'arma_generate_sample',
           'arma_impulse_response', 'arma2ar', 'arma2ma', 'deconvolve',
           'lpol2index', 'index2lpol']

# upstream these are defined here.
arma_impulse_response = wold.arma_impulse_response
arma2ar = wold.arma2ar
arma2ma = wold.arma2ma
arma_periodogram = wold.arma_periodogram


def arma_generate_sample(ar, ma, nsample, sigma=1, distrvs=np.random.randn,
                         burnin=0):
    """
    Generate a random sample of an ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag
    nsample : int
        length of simulated time series
    sigma : float
        standard deviation of noise
    distrvs : function, random number generator
        function that generates the random numbers, and takes sample size
        as argument
        default: np.random.randn
        TODO: change to size argument
    burnin : integer
        Burn in observations at the generated and dropped from the beginning of
        the sample

    Returns
    -------
    sample : array
        sample of ARMA process given by ar, ma of length nsample

    Notes
    -----
    As mentioned above, both the AR and MA components should include the
    coefficient on the zero-lag. This is typically 1. Further, due to the
    conventions used in signal processing used in signal.lfilter vs.
    conventions in statistics for ARMA processes, the AR parameters should
    have the opposite sign of what you might expect. See the examples below.

    Examples
    --------
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> y = sm.tsa.arma_generate_sample(ar, ma, 250)
    >>> model = sm.tsa.ARMA(y, (2, 2)).fit(trend='nc', disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])
    """
    # TODO: unify with ArmaProcess method
    eta = sigma * distrvs(nsample + burnin)
    return signal.lfilter(ma, ar, eta)[burnin:]


def arma_acovf(ar, ma, nobs=10):
    """
    Theoretical autocovariance function of ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag
    nobs : int
        number of terms (lags plus zero lag) to include in returned acovf

    Returns
    -------
    acovf : array
        autocovariance of ARMA process given by ar, ma

    See Also
    --------
    arma_acf
    acovf

    Notes
    -----
    Tries to do some crude numerical speed improvements for cases
    with high persistence. However, this algorithm is slow if the process is
    highly persistent and only a few autocovariances are desired.
    """
    # increase length of impulse response for AR closer to 1
    # maybe cheap/fast enough to always keep nobs for ir large

    # TODO: This doesn't make sense should be analytical
    if np.abs(np.sum(ar) - 1) > 0.9:
        nobs_ir = max(1000, 2 * nobs)  # no idea right now how large is needed
    else:
        nobs_ir = max(100, 2 * nobs)  # no idea right now
    ir = arma_impulse_response(ar, ma, leads=nobs_ir)
    # better safe than sorry (?), I have no idea about the required precision
    # only checked for AR(1)
    while ir[-1] > 5 * 1e-5:
        nobs_ir *= 10
        ir = arma_impulse_response(ar, ma, leads=nobs_ir)
    # again no idea where the speed break points are:
    if nobs_ir > 50000 and nobs < 1001:
        end = len(ir)
        # Explitly slice from the end to avoid foo[:-0] returning an empty slice
        acovf = np.array([np.dot(ir[:end - nobs - t], ir[t:end - nobs])
                          for t in range(nobs)])
    else:
        acovf = np.correlate(ir, ir, 'full')[len(ir) - 1:]
    return acovf[:nobs]


@deprecate_kwarg('nobs', 'lags')
def arma_acf(ar, ma, lags=10):
    """
    Theoretical autocorrelation function of an ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag
    lags : int
        number of terms (lags plus zero lag) to include in returned acf

    Returns
    -------
    acf : array
        autocorrelation of ARMA process given by ar, ma


    See Also
    --------
    arma_acovf
    acf
    acovf
    """
    acovf = arma_acovf(ar, ma, lags)
    return acovf / acovf[0]


@deprecate_kwarg('nobs', 'lags')
def arma_pacf(ar, ma, lags=10):
    """
    Partial autocorrelation function of an ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag
    lags : int
        number of terms (lags plus zero lag) to include in returned pacf

    Returns
    -------
    pacf : array
        partial autocorrelation of ARMA process given by ar, ma

    Notes
    -----
    solves yule-walker equation for each lag order up to nobs lags

    not tested/checked yet
    """
    # TODO: Should use rank 1 inverse update
    apacf = np.zeros(lags)
    acov = arma_acf(ar, ma, lags=lags + 1)

    apacf[0] = 1.
    for k in range(2, lags + 1):
        r = acov[:k]
        apacf[k - 1] = linalg.solve(linalg.toeplitz(r[:-1]), r[1:])[-1]
    return apacf


# TODO: not hit in tests (here or upstream)
# moved from sandbox.tsa.try_fi
def ar2arma(ar_des, p, q, n=20, mse='ar', start=None):
    """
    Find arma approximation to ar process

    This finds the ARMA(p,q) coefficients that minimize the integrated
    squared difference between the impulse_response functions (MA
    representation) of the AR and the ARMA process. This does not  check
    whether the MA lag polynomial of the ARMA process is invertible, neither
    does it check the roots of the AR lag polynomial.

    Parameters
    ----------
    ar_des : array_like
        coefficients of original AR lag polynomial, including lag zero
    p : int
        length of desired AR lag polynomials
    q : int
        length of desired MA lag polynomials
    n : int
        number of terms of the impulse_response function to include in the
        objective function for the approximation
    mse : string, 'ar'
        not used yet,

    Returns
    -------
    ar_app, ma_app : arrays
        coefficients of the AR and MA lag polynomials of the approximation
    res : tuple
        result of optimize.leastsq

    Notes
    -----
    Extension is possible if we want to match autocovariance instead
    of impulse response function.
    """

    # TODO: convert MA lag polynomial, ma_app, to be invertible, by mirroring
    #       roots outside the unit interval to ones that are inside. How to do
    #       this?

    # p,q = pq
    def msear_err(arma, ar_des):
        ar, ma = np.r_[1, arma[:p - 1]], np.r_[1, arma[p - 1:]]
        ar_approx = arma_impulse_response(ma, ar, n)
        return (ar_des - ar_approx)  # ((ar - ar_approx)**2).sum()

    if start is None:
        arma0 = np.r_[-0.9 * np.ones(p - 1), np.zeros(q - 1)]
    else:
        arma0 = start
    res = optimize.leastsq(msear_err, arma0, ar_des, maxfev=5000)
    arma_app = np.atleast_1d(res[0])
    ar_app = np.r_[1, arma_app[:p - 1]],
    ma_app = np.r_[1, arma_app[p - 1:]]
    return ar_app, ma_app, res


def lpol2index(ar):
    """
    Remove zeros from lag polynomial

    Parameters
    ----------
    ar : array_like
        coefficients of lag polynomial

    Returns
    -------
    coeffs : array
        non-zero coefficients of lag polynomial
    index : array
        index (lags) of lag polynomial with non-zero elements
    """
    ar = np.asarray(ar)
    index = np.nonzero(ar)[0]
    coeffs = ar[index]
    return coeffs, index


def index2lpol(coeffs, index):
    """
    Expand coefficients to lag poly

    Parameters
    ----------
    coeffs : array
        non-zero coefficients of lag polynomial
    index : array
        index (lags) of lag polynomial with non-zero elements

    Returns
    -------
    ar : array_like
        coefficients of lag polynomial
    """
    n = max(index)
    ar = np.zeros(n + 1)
    ar[index] = coeffs
    return ar


def lpol_fima(d, n=20):
    """MA representation of fractional integration

    .. math:: (1-L)^{-d} for |d|<0.5  or |d|<1 (?)

    Parameters
    ----------
    d : float
        fractional power
    n : int
        number of terms to calculate, including lag zero

    Returns
    -------
    ma : array
        coefficients of lag polynomial
    """
    # hide import inside function until we use this heavily
    from scipy.special import gammaln
    j = np.arange(n)
    return np.exp(gammaln(d + j) - gammaln(j + 1) - gammaln(d))


# moved from sandbox.tsa.try_fi
def lpol_fiar(d, n=20):
    """AR representation of fractional integration

    .. math:: (1-L)^{d} for |d|<0.5  or |d|<1 (?)

    Parameters
    ----------
    d : float
        fractional power
    n : int
        number of terms to calculate, including lag zero

    Returns
    -------
    ar : array
        coefficients of lag polynomial

    Notes:
    first coefficient is 1, negative signs except for first term,
    ar(L)*x_t
    """
    # hide import inside function until we use this heavily
    from scipy.special import gammaln
    j = np.arange(n)
    ar = - np.exp(gammaln(-d + j) - gammaln(j + 1) - gammaln(-d))
    ar[0] = 1
    return ar


def lpol_sdiff(s):  # pragma: no cover
    raise NotImplementedError("lpol_sdiff not ported from upstream, as it is "
                              "neither used nor tested there.")


def deconvolve(num, den, n=None):  # pragma: no cover
    raise NotImplementedError("deconvolve not ported from upstream, as it is "
                              "neither used nor tested there.  Its docstring "
                              "says it is mostly-copied from "
                              "scipy.signal.signaltools.")


# NB: The upstream implementation has a `nobs` kwarg/attr
class ArmaProcess(wold.ARMAParams):
    r"""
    Theoretical properties of an ARMA process for specified lag-polynomials

    Parameters
    ----------
    ar : array_like, 1d, optional
        Coefficient for autoregressive lag polynomial, including zero lag.
        See the notes for some information about the sign.
    ma : array_like, 1d, optional
        Coefficient for moving-average lag polynomial, including zero lag

    Notes
    -----
    Both the AR and MA components must include the coefficient on the
    zero-lag. In almost all cases these values should be 1. Further, due to
    using the lag-polynomial representation, the AR parameters should
    have the opposite sign of what one would write in the ARMA representation.
    See the examples below.

    The ARMA(p,q) process is described by

    .. math::

        y_{t}=\phi_{1}y_{t-1}+\ldots+\phi_{p}y_{t-p}+\theta_{1}\epsilon_{t-1}
               +\ldots+\theta_{q}\epsilon_{t-q}+\epsilon_{t}

    and the parameterization used in this function uses the lag-polynomial
    representation,

    .. math::

        \left(1-\phi_{1}L-\ldots-\phi_{p}L^{p}\right)y_{t} =
            \left(1-\theta_{1}L-\ldots-\theta_{q}L^{q}\right)

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -ar] # add zero-lag and negate
    >>> ma = np.r_[1, ma] # add zero-lag
    >>> arma_process = sm.tsa.ArmaProcess(ar, ma)
    >>> arma_process.isstationary
    True
    >>> arma_process.isinvertible
    True
    >>> y = arma_process.generate_sample(250)
    >>> model = sm.tsa.ARMA(y, (2, 2)).fit(trend='nc', disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])
    """
    @copy_doc(arma_acovf.__doc__)
    def acovf(self, nobs=100):
        return arma_acovf(self.ar, self.ma, nobs=nobs)

    @copy_doc(arma_acf.__doc__)
    def acf(self, lags=100):
        return arma_acf(self.ar, self.ma, lags=lags)

    @copy_doc(arma_pacf.__doc__)
    def pacf(self, lags=100):
        return arma_pacf(self.ar, self.ma, lags=lags)

    def generate_sample(self, nsample=100, scale=1., distrvs=None, axis=0,
                        burnin=0):
        """
        Simulate an ARMA

        Parameters
        ----------
        nsample : int or tuple of ints
            If nsample is an integer, then this creates a 1d timeseries of
            length size. If nsample is a tuple, creates a len(nsample)
            dimensional time series where time is indexed along the input
            variable ``axis``. All series are unless ``distrvs`` generates
            dependent data.
        scale : float
            standard deviation of noise
        distrvs : function, random number generator
            function that generates the random numbers, and takes sample size
            as argument
            default: np.random.randn
            TODO: change to size argument
        burnin : integer (default: 0)
            to reduce the effect of initial conditions, burnin observations
            at the beginning of the sample are dropped
        axis : int
            See nsample.

        Returns
        -------
        rvs : ndarray
            random sample(s) of arma process

        Notes
        -----
        Should work for n-dimensional with time series along axis, but not
        tested yet. Processes are sampled independently.
        """
        if distrvs is None:
            distrvs = np.random.normal
        if np.ndim(nsample) == 0:
            nsample = [nsample]
        if burnin:
            # handle burin time for nd arrays
            # maybe there is a better trick in scipy.fft code
            newsize = list(nsample)
            newsize[axis] += burnin
            newsize = tuple(newsize)
            fslice = [slice(None)] * len(newsize)
            fslice[axis] = slice(burnin, None, None)
            fslice = tuple(fslice)
        else:
            newsize = tuple(nsample)
            fslice = tuple([slice(None)] * np.ndim(newsize))
        eta = scale * distrvs(size=newsize)
        return signal.lfilter(self.ma, self.ar, eta, axis=axis)[fslice]
