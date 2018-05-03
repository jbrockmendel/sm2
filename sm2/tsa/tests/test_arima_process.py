
from six.moves import range

import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_allclose,
                           assert_equal)

from sm2.tsa.arima_process import (arma_generate_sample, arma_acovf,
                                   arma_acf, arma_impulse_response,
                                   lpol_fiar, lpol_fima,
                                   ArmaProcess, lpol2index, index2lpol)

from sm2.tsa.tests.results.results_process import armarep  # benchmarkdata


arlist = [[1.],
          [1, -0.9],  # ma representation will need many terms to get precision
          [1, 0.9],
          [1, -0.9, 0.3]]

malist = [[1.],
          [1, 0.9],
          [1, -0.9],
          [1, 0.9, -0.3]]

DECIMAL_4 = 4


@pytest.mark.not_vetted
def test_arma_acovf():
    # Check for specific AR(1)
    N = 20
    phi = 0.9
    sigma = 1
    # rep 1: from module function
    rep1 = arma_acovf([1, -phi], [1], N)
    # rep 2: manually
    rep2 = [1. * sigma * phi ** i / (1 - phi ** 2) for i in range(N)]
    assert_almost_equal(rep1, rep2, 7)  # 7 is max precision here


@pytest.mark.not_vetted
def test_arma_acovf_persistent():
    # Test arma_acovf in case where there is a near-unit root.
    # .999 is high enough to trigger the "while ir[-1] > 5*1e-5:" clause,
    # but not high enough to trigger the "nobs_ir > 50000" clause.
    ar = np.array([1, -.9995])
    ma = np.array([1])
    process = ArmaProcess(ar, ma)
    res = process.acovf(10)

    # Theoretical variance sig2 given by:
    # sig2 = .9995**2 * sig2 + 1
    sig2 = 1 / (1 - .9995**2)

    corrs = .9995**np.arange(10)
    expected = sig2 * corrs
    assert res.ndim == 1
    assert_allclose(res, expected, atol=1e-6)
    # atol=7 breaks at .999, worked at .995


@pytest.mark.not_vetted
def test_arma_acf():
    # Check for specific AR(1)
    N = 20
    phi = 0.9
    sigma = 1
    # rep 1: from module function
    rep1 = arma_acf([1, -phi], [1], N)
    # rep 2: manually
    acovf = np.array([1. * sigma * phi ** i / (1 - phi ** 2)
                      for i in range(N)])
    rep2 = acovf / (1. / (1 - phi ** 2))
    assert_almost_equal(rep1, rep2, 8)  # 8 is max precision here


@pytest.mark.not_vetted
def _manual_arma_generate_sample(ar, ma, eta):
    T = len(eta)
    ar = ar[::-1]
    ma = ma[::-1]
    p, q = len(ar), len(ma)
    rep2 = [0] * max(p, q)  # initialize with zeroes
    for t in range(T):
        yt = eta[t]
        if p:
            yt += np.dot(rep2[-p:], ar)
        if q:
            # left pad shocks with zeros
            yt += np.dot([0] * (q - t) + list(eta[max(0, t - q):t]), ma)
        rep2.append(yt)
    return np.array(rep2[max(p, q):])


@pytest.mark.not_vetted
@pytest.mark.parametrize('dist', [np.random.randn])
@pytest.mark.parametrize('ar', arlist)
@pytest.mark.parametrize('ma', malist)
def test_arma_generate_sample(dist, ar, ma):
    # Test that this generates a true ARMA process
    # (amounts to just a test that scipy.signal.lfilter does what we want)
    T = 100
    dists = [np.random.randn]
    np.random.seed(1234)
    eta = dist(T)

    # rep1: from module function
    np.random.seed(1234)
    rep1 = arma_generate_sample(ar, ma, T, distrvs=dist)
    # rep2: "manually" create the ARMA process
    ar_params = -1 * np.array(ar[1:])
    ma_params = np.array(ma[1:])
    rep2 = _manual_arma_generate_sample(ar_params, ma_params, eta)
    assert_array_almost_equal(rep1, rep2, 13)


@pytest.mark.not_vetted
def test_fi():
    # test identity of ma and ar representation of fi lag polynomial
    n = 100
    mafromar = arma_impulse_response(lpol_fiar(0.4, n=n), [1], n)
    assert_array_almost_equal(mafromar, lpol_fima(0.4, n=n), 13)


@pytest.mark.not_vetted
def test_arma_impulse_response():
    arrep = arma_impulse_response(armarep.ma, armarep.ar, leads=21)[1:]
    marep = arma_impulse_response(armarep.ar, armarep.ma, leads=21)[1:]
    assert_array_almost_equal(armarep.marep.ravel(), marep, 14)
    # difference in sign convention to matlab for AR term
    assert_array_almost_equal(-armarep.arrep.ravel(), arrep, 14)


@pytest.mark.skip(reason="ArmaFft not ported from upstream")
@pytest.mark.not_vetted
def test_spectrum():
    ArmaFft = None  # dummy to avoid flake8 warnings
    # from statsmodels.sandbox.tsa.fftarma import ArmaFft

    nfreq = 20
    w = np.linspace(0, np.pi, nfreq, endpoint=False)
    for ar in arlist:
        for ma in malist:
            arma = ArmaFft(ar, ma, 20)
            spdr, wr = arma.spdroots(w)
            spdp, wp = arma.spdpoly(w, 200)
            spdd, wd = arma.spddirect(nfreq * 2)
            assert_equal(w, wr)
            assert_equal(w, wp)
            assert_almost_equal(w, wd[:nfreq], decimal=14)
            assert_almost_equal(spdr, spdd[:nfreq], decimal=7,
                                err_msg='spdr spdd not equal for %s, %s'
                                        % (ar, ma))
            assert_almost_equal(spdr, spdp, decimal=7,
                                err_msg='spdr spdp not equal for %s, %s'
                                        % (ar, ma))


@pytest.mark.skip(reason="ArmaFft not ported from upstream")
@pytest.mark.not_vetted
def test_armafft():
    # test other methods
    ArmaFft = None  # dummy to avoid flake8 warnings
    # from statsmodels.sandbox.tsa.fftarma import ArmaFft
    for ar in arlist:
        for ma in malist:
            arma = ArmaFft(ar, ma, 20)
            ac1 = arma.invpowerspd(1024)[:10]
            ac2 = arma.acovf(10)[:10]
            assert_almost_equal(ac1, ac2, decimal=7,
                                err_msg='acovf not equal for %s, %s'
                                        % (ar, ma))


@pytest.mark.not_vetted
def test_lpol2index_index2lpol():
    process = ArmaProcess([1, 0, 0, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [0.8])
    assert_equal(locs, [2])

    process = ArmaProcess([1, .1, .1, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [-.1, -.1, 0.8])
    assert_equal(locs, [0, 1, 2])
    ar = index2lpol(coefs, locs)
    assert_equal(process.arcoefs, ar)


@pytest.mark.not_vetted
class TestArmaProcess(object):
    @pytest.mark.skipif("np.__version__ < '1.7'")
    def test_str_repr(self):
        process1 = ArmaProcess.from_coeffs([.9], [.2])
        out = process1.__str__()
        assert out.find('AR: [1.0, -0.9]') != -1
        assert out.find('MA: [1.0, 0.2]') != -1

        out = process1.__repr__()
        # assert out.find('nobs=100') != -1  # nobs from upstream deprecated
        assert out.find('at ' + str(hex(id(process1)))) != -1

    def test_acf(self):
        process1 = ArmaProcess.from_coeffs([.9])
        acf = process1.acf(10)
        expected = np.array(0.9) ** np.arange(10.0)
        assert_array_almost_equal(acf, expected)

        acf = process1.acf()
        # assert acf.shape[0] == process1.nobs  # nobs from upstream deprecated

    def test_pacf(self):
        process1 = ArmaProcess.from_coeffs([.9])
        pacf = process1.pacf(10)
        expected = np.array([1, 0.9] + [0] * 8)
        assert_array_almost_equal(pacf, expected)

        pacf = process1.pacf()
        # assert pacf.shape[0] == process1.nobs  # nobs from upstream deprecated
