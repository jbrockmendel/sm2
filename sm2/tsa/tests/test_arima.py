import os
import warnings

from six import BytesIO
from six.moves import cPickle


import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas as pd
import pandas.util.testing as tm
import pytest

from sm2 import datasets
from sm2.regression.linear_model import OLS
from sm2.tsa.arima_model import AR, ARMA, ARIMA
from sm2.tsa.arima_process import arma_generate_sample

from sm2.tools.sm_exceptions import MissingDataError
from .results import results_arma, results_arima

DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

current_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_path, 'results', 'y_arma_data.csv')
y_arma = pd.read_csv(path, float_precision='high').values

cpi_dates = pd.PeriodIndex(start='1959q1', end='2009q3', freq='Q')
sun_dates = pd.PeriodIndex(start='1700', end='2008', freq='A')
cpi_predict_dates = pd.PeriodIndex(start='2009q3', end='2015q4', freq='Q')
sun_predict_dates = pd.PeriodIndex(start='2008', end='2033', freq='A')


@pytest.mark.not_vetted
@pytest.mark.skip(reason="fa, Arma not ported from upstream")
def test_compare_arma():
    # dummies to avoid flake8 warnings until porting
    fa = None
    Arma = None
    # import statsmodels.sandbox.tsa.fftarma as fa
    # from statsmodels.tsa.arma_mle import Arma

    # this is a preliminary test to compare
    # arma_kf, arma_cond_ls and arma_cond_mle
    # the results returned by the fit methods are incomplete
    # for now without random.seed

    np.random.seed(9876565)
    famod = fa.ArmaFft([1, -0.5], [1., 0.4], 40)
    x = famod.generate_sample(nsample=200, burnin=1000)

    modkf = ARMA(x, (1, 1))
    reskf = modkf.fit(trend='nc', disp=-1)
    dres = reskf

    modc = Arma(x)
    resls = modc.fit(order=(1, 1))
    rescm = modc.fit_mle(order=(1, 1), start_params=[0.4, 0.4, 1.], disp=0)

    # decimal 1 corresponds to threshold of 5% difference
    # still different sign corrected
    assert_almost_equal(resls[0] / dres.params,
                        np.ones(dres.params.shape),
                        decimal=1)

    # TODO: Is the next comment still accurate.  It is retained from upstream
    # where there was a commented-out assertion after the comment
    # rescm also contains variance estimate as last element of params
    assert_almost_equal(rescm.params[:-1] / dres.params,
                        np.ones(dres.params.shape),
                        decimal=1)


@pytest.mark.not_vetted
class CheckArmaResultsMixin(object):
    """
    res2 are the results from gretl.  They are in results/results_arma.
    res1 are from sm2
    """
    decimal_params = DECIMAL_4

    def test_params(self):
        assert_almost_equal(self.res1.params,
                            self.res2.params,
                            self.decimal_params)

    decimal_aic = DECIMAL_4

    def test_aic(self):
        assert_almost_equal(self.res1.aic,
                            self.res2.aic,
                            self.decimal_aic)

    decimal_bic = DECIMAL_4

    def test_bic(self):
        assert_almost_equal(self.res1.bic,
                            self.res2.bic,
                            self.decimal_bic)

    decimal_arroots = DECIMAL_4

    def test_arroots(self):
        assert_almost_equal(self.res1.arroots,
                            self.res2.arroots,
                            self.decimal_arroots)

    decimal_maroots = DECIMAL_4

    def test_maroots(self):
        assert_almost_equal(self.res1.maroots,
                            self.res2.maroots,
                            self.decimal_maroots)

    decimal_bse = DECIMAL_2

    def test_bse(self):
        assert_almost_equal(self.res1.bse,
                            self.res2.bse,
                            self.decimal_bse)

    decimal_cov_params = DECIMAL_4

    def test_covparams(self):
        assert_almost_equal(self.res1.cov_params(),
                            self.res2.cov_params,
                            self.decimal_cov_params)

    decimal_hqic = DECIMAL_4

    def test_hqic(self):
        assert_almost_equal(self.res1.hqic,
                            self.res2.hqic,
                            self.decimal_hqic)

    decimal_llf = DECIMAL_4

    def test_llf(self):
        assert_almost_equal(self.res1.llf,
                            self.res2.llf,
                            self.decimal_llf)

    decimal_resid = DECIMAL_4

    def test_resid(self):
        assert_almost_equal(self.res1.resid,
                            self.res2.resid,
                            self.decimal_resid)

    decimal_fittedvalues = DECIMAL_4

    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues,
                            self.res2.fittedvalues,
                            self.decimal_fittedvalues)

    decimal_pvalues = DECIMAL_2

    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues,
                            self.res2.pvalues,
                            self.decimal_pvalues)

    decimal_t = DECIMAL_2  # only 2 decimal places in gretl output

    def test_tvalues(self):
        assert_almost_equal(self.res1.tvalues,
                            self.res2.tvalues,
                            self.decimal_t)

    decimal_sigma2 = DECIMAL_4

    def test_sigma2(self):
        assert_almost_equal(self.res1.sigma2,
                            self.res2.sigma2,
                            self.decimal_sigma2)

    @pytest.mark.smoke
    def test_summary(self):
        self.res1.summary()


@pytest.mark.not_vetted
class CheckForecastMixin(object):
    decimal_forecast = DECIMAL_4

    def test_forecast(self):
        assert_almost_equal(self.res1.forecast_res,
                            self.res2.forecast,
                            self.decimal_forecast)

    decimal_forecasterr = DECIMAL_4

    def test_forecasterr(self):
        assert_almost_equal(self.res1.forecast_err,
                            self.res2.forecasterr,
                            self.decimal_forecasterr)


@pytest.mark.not_vetted
class CheckDynamicForecastMixin(object):
    decimal_forecast_dyn = 4

    def test_dynamic_forecast(self):
        assert_almost_equal(self.res1.forecast_res_dyn,
                            self.res2.forecast_dyn,
                            self.decimal_forecast_dyn)

    #def test_forecasterr(self):
    #    assert_almost_equal(self.res1.forecast_err_dyn,
    #                        self.res2.forecasterr_dyn,
    #                        DECIMAL_4)


@pytest.mark.not_vetted
class CheckArimaResultsMixin(CheckArmaResultsMixin):
    def test_order(self):
        assert self.res1.k_diff == self.res2.k_diff
        assert self.res1.k_ar == self.res2.k_ar
        assert self.res1.k_ma == self.res2.k_ma

    decimal_predict_levels = DECIMAL_4

    def test_predict_levels(self):
        assert_almost_equal(self.res1.predict(typ='levels'),
                            self.res2.linear,
                            self.decimal_predict_levels)


@pytest.mark.not_vetted
class Test_Y_ARMA11_NoConst(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 0]
        cls.res1 = ARMA(endog, order=(1, 1)).fit(trend='nc', disp=-1)
        fc_res, fc_err, ci = cls.res1.forecast(10)
        cls.res1.forecast_res = fc_res
        cls.res1.forecast_err = fc_err
        cls.res2 = results_arma.Y_arma11()

    # TODO: share with test_ar? other test classes?
    def test_pickle(self):
        fh = BytesIO()
        # test wrapped results load save pickle
        self.res1.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.res1.__class__.load(fh)
        assert type(res_unpickled) is type(self.res1)  # noqa:E721
        # TODO: Test equality instead of just type equality?


@pytest.mark.not_vetted
class Test_Y_ARMA14_NoConst(CheckArmaResultsMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 1]
        cls.res1 = ARMA(endog, order=(1, 4)).fit(trend='nc', disp=-1)
        cls.res2 = results_arma.Y_arma14()


@pytest.mark.not_vetted
@pytest.mark.slow
class Test_Y_ARMA41_NoConst(CheckArmaResultsMixin, CheckForecastMixin):
    decimal_maroots = DECIMAL_3

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 2]
        cls.res1 = ARMA(endog, order=(4, 1)).fit(trend='nc', disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
         confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma41()


@pytest.mark.not_vetted
class Test_Y_ARMA22_NoConst(CheckArmaResultsMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 3]
        cls.res1 = ARMA(endog, order=(2, 2)).fit(trend='nc', disp=-1)
        cls.res2 = results_arma.Y_arma22()


@pytest.mark.not_vetted
class Test_Y_ARMA50_NoConst(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 4]
        cls.res1 = ARMA(endog, order=(5, 0)).fit(trend='nc', disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
         confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma50()


@pytest.mark.not_vetted
class Test_Y_ARMA02_NoConst(CheckArmaResultsMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 5]
        cls.res1 = ARMA(endog, order=(0, 2)).fit(trend='nc', disp=-1)
        cls.res2 = results_arma.Y_arma02()


@pytest.mark.not_vetted
class Test_Y_ARMA11_Const(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 6]
        cls.res1 = ARMA(endog, order=(1, 1)).fit(trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
         confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma11c()


@pytest.mark.not_vetted
class Test_Y_ARMA14_Const(CheckArmaResultsMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 7]
        cls.res1 = ARMA(endog, order=(1, 4)).fit(trend="c", disp=-1)
        cls.res2 = results_arma.Y_arma14c()


@pytest.mark.not_vetted
class Test_Y_ARMA41_Const(CheckArmaResultsMixin, CheckForecastMixin):
    decimal_cov_params = DECIMAL_3
    decimal_fittedvalues = DECIMAL_3
    decimal_resid = DECIMAL_3
    decimal_params = DECIMAL_3

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 8]
        cls.res2 = results_arma.Y_arma41c()
        cls.res1 = ARMA(endog, order=(4, 1)).fit(trend="c", disp=-1,
                                                 start_params=cls.res2.params)
        (cls.res1.forecast_res, cls.res1.forecast_err,
         confint) = cls.res1.forecast(10)


@pytest.mark.not_vetted
class Test_Y_ARMA22_Const(CheckArmaResultsMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 9]
        cls.res1 = ARMA(endog, order=(2, 2)).fit(trend="c", disp=-1)
        cls.res2 = results_arma.Y_arma22c()


@pytest.mark.not_vetted
class Test_Y_ARMA50_Const(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 10]
        cls.res1 = ARMA(endog, order=(5, 0)).fit(trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
         confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma50c()


@pytest.mark.not_vetted
class Test_Y_ARMA02_Const(CheckArmaResultsMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 11]
        cls.res1 = ARMA(endog, order=(0, 2)).fit(trend="c", disp=-1)
        cls.res2 = results_arma.Y_arma02c()


# cov_params and tvalues are off still but not as much vs. R
@pytest.mark.not_vetted
class Test_Y_ARMA11_NoConst_CSS(CheckArmaResultsMixin):
    decimal_t = DECIMAL_1

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 0]
        cls.res1 = ARMA(endog, order=(1, 1)).fit(method="css",
                                                 trend="nc", disp=-1)
        cls.res2 = results_arma.Y_arma11("css")


# better vs. R
@pytest.mark.not_vetted
class Test_Y_ARMA14_NoConst_CSS(CheckArmaResultsMixin):
    decimal_fittedvalues = DECIMAL_3
    decimal_resid = DECIMAL_3
    decimal_t = DECIMAL_1

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 1]
        cls.res1 = ARMA(endog, order=(1, 4)).fit(method="css",
                                                 trend="nc", disp=-1)
        cls.res2 = results_arma.Y_arma14("css")


# bse, etc. better vs. R
# maroot is off because maparams is off a bit (adjust tolerance?)
@pytest.mark.not_vetted
class Test_Y_ARMA41_NoConst_CSS(CheckArmaResultsMixin):
    decimal_t = DECIMAL_1
    decimal_pvalues = 0
    decimal_cov_params = DECIMAL_3
    decimal_maroots = DECIMAL_1

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 2]
        cls.res1 = ARMA(endog, order=(4, 1)).fit(method="css",
                                                 trend="nc", disp=-1)
        cls.res2 = results_arma.Y_arma41("css")


# same notes as above
@pytest.mark.not_vetted
class Test_Y_ARMA22_NoConst_CSS(CheckArmaResultsMixin):
    decimal_t = DECIMAL_1
    decimal_resid = DECIMAL_3
    decimal_pvalues = DECIMAL_1
    decimal_fittedvalues = DECIMAL_3

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 3]
        cls.res1 = ARMA(endog, order=(2, 2)).fit(method="css",
                                                 trend="nc", disp=-1)
        cls.res2 = results_arma.Y_arma22("css")


# NOTE: gretl just uses least squares for AR CSS
# so BIC, etc. is
# -2*res1.llf + np.log(nobs)*(res1.q+res1.p+res1.k)
# with no adjustment for p and no extra sigma estimate
# NOTE: so our tests use x-12 arima results which agree with us and are
# consistent with the rest of the models
@pytest.mark.not_vetted
class Test_Y_ARMA50_NoConst_CSS(CheckArmaResultsMixin):
    decimal_t = 0
    decimal_llf = DECIMAL_1  # looks like rounding error?

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 4]
        cls.res1 = ARMA(endog, order=(5, 0)).fit(method="css",
                                                 trend="nc", disp=-1)
        cls.res2 = results_arma.Y_arma50("css")


@pytest.mark.not_vetted
class Test_Y_ARMA02_NoConst_CSS(CheckArmaResultsMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 5]
        cls.res1 = ARMA(endog, order=(0, 2)).fit(method="css",
                                                 trend="nc", disp=-1)
        cls.res2 = results_arma.Y_arma02("css")


# NOTE: our results are close to --x-12-arima option and R
@pytest.mark.not_vetted
class Test_Y_ARMA11_Const_CSS(CheckArmaResultsMixin):
    decimal_params = DECIMAL_3
    decimal_cov_params = DECIMAL_3
    decimal_t = DECIMAL_1

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 6]
        cls.res1 = ARMA(endog, order=(1, 1)).fit(trend="c",
                                                 method="css", disp=-1)
        cls.res2 = results_arma.Y_arma11c("css")


@pytest.mark.not_vetted
class Test_Y_ARMA14_Const_CSS(CheckArmaResultsMixin):
    decimal_t = DECIMAL_1
    decimal_pvalues = DECIMAL_1

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 7]
        cls.res1 = ARMA(endog, order=(1, 4)).fit(trend="c",
                                                 method="css", disp=-1)
        cls.res2 = results_arma.Y_arma14c("css")


@pytest.mark.not_vetted
class Test_Y_ARMA41_Const_CSS(CheckArmaResultsMixin):
    decimal_t = DECIMAL_1
    decimal_cov_params = DECIMAL_1
    decimal_maroots = DECIMAL_3
    decimal_bse = DECIMAL_1

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 8]
        cls.res1 = ARMA(endog, order=(4, 1)).fit(trend="c",
                                                 method="css", disp=-1)
        cls.res2 = results_arma.Y_arma41c("css")


@pytest.mark.not_vetted
class Test_Y_ARMA22_Const_CSS(CheckArmaResultsMixin):
    decimal_t = 0
    decimal_pvalues = DECIMAL_1

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 9]
        cls.res1 = ARMA(endog, order=(2, 2)).fit(trend="c",
                                                 method="css", disp=-1)
        cls.res2 = results_arma.Y_arma22c("css")


@pytest.mark.not_vetted
class Test_Y_ARMA50_Const_CSS(CheckArmaResultsMixin):
    decimal_t = DECIMAL_1
    decimal_params = DECIMAL_3
    decimal_cov_params = DECIMAL_2

    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 10]
        cls.res1 = ARMA(endog, order=(5, 0)).fit(trend="c",
                                                 method="css", disp=-1)
        cls.res2 = results_arma.Y_arma50c("css")


@pytest.mark.not_vetted
class Test_Y_ARMA02_Const_CSS(CheckArmaResultsMixin):
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 11]
        cls.res1 = ARMA(endog, order=(0, 2)).fit(trend="c",
                                                 method="css", disp=-1)
        cls.res2 = results_arma.Y_arma02c("css")


@pytest.mark.not_vetted
class Test_ARIMA101(CheckArmaResultsMixin):
    # just make sure this works
    @classmethod
    def setup_class(cls):
        endog = y_arma[:, 6]
        cls.res1 = ARIMA(endog, (1, 0, 1)).fit(trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
         confint) = cls.res1.forecast(10)

        cls.res2 = results_arma.Y_arma11c()
        cls.res2.k_diff = 0
        cls.res2.k_ar = 1
        cls.res2.k_ma = 1


@pytest.mark.not_vetted
class Test_ARIMA111(CheckArimaResultsMixin, CheckForecastMixin,
                    CheckDynamicForecastMixin):
    decimal_llf = 3
    decimal_aic = 3
    decimal_bic = 3
    decimal_cov_params = 2  # this used to be better?
    decimal_t = 0

    @classmethod
    def setup_class(cls):
        cpi = datasets.macrodata.load_pandas().data['cpi'].values
        cls.res1 = ARIMA(cpi, (1, 1, 1)).fit(disp=-1)
        cls.res2 = results_arima.ARIMA111()
        # make sure endog names changes to D.cpi
        (cls.res1.forecast_res,
         cls.res1.forecast_err,
         conf_int) = cls.res1.forecast(25)
        # TODO: fix the indexing for the end here, I don't think this is right
        # if we're going to treat it like indexing
        # the forecast from 2005Q1 through 2009Q4 is indices
        # 184 through 227 not 226
        # note that the first one counts in the count so 164 + 64 is 65
        # predictions
        cls.res1.forecast_res_dyn = cls.res1.predict(start=164, end=164 + 63,
                                                     typ='levels',
                                                     dynamic=True)

    def test_freq(self):
        assert_almost_equal(self.res1.arfreq, [0.0000], 4)
        assert_almost_equal(self.res1.mafreq, [0.0000], 4)


@pytest.mark.not_vetted
class Test_ARIMA111CSS(CheckArimaResultsMixin, CheckForecastMixin,
                       CheckDynamicForecastMixin):
    decimal_forecast = 2
    decimal_forecast_dyn = 2
    decimal_forecasterr = 3
    decimal_arroots = 3
    decimal_cov_params = 3
    decimal_hqic = 3
    decimal_maroots = 3
    decimal_t = 1
    decimal_fittedvalues = 2  # because of rounding when copying
    decimal_resid = 2
    decimal_predict_levels = DECIMAL_2

    @classmethod
    def setup_class(cls):
        cpi = datasets.macrodata.load_pandas().data['cpi'].values
        cls.res1 = ARIMA(cpi, (1, 1, 1)).fit(disp=-1, method='css')
        cls.res2 = results_arima.ARIMA111(method='css')
        cls.res2.fittedvalues = - cpi[1:-1] + cls.res2.linear
        # make sure endog names changes to D.cpi
        (fc_res, fc_err, conf_int) = cls.res1.forecast(25)
        cls.res1.forecast_res = fc_res
        cls.res1.forecast_err = fc_err
        cls.res1.forecast_res_dyn = cls.res1.predict(start=164, end=164 + 63,
                                                     typ='levels',
                                                     dynamic=True)


@pytest.mark.not_vetted
class Test_ARIMA112CSS(CheckArimaResultsMixin):
    decimal_llf = 3
    decimal_aic = 3
    decimal_bic = 3
    decimal_arroots = 3
    decimal_maroots = 2
    decimal_t = 1
    decimal_resid = 2
    decimal_fittedvalues = 3
    decimal_predict_levels = DECIMAL_3

    @classmethod
    def setup_class(cls):
        cpi = datasets.macrodata.load_pandas().data['cpi'].values
        cls.res1 = ARIMA(cpi, (1, 1, 2)).fit(disp=-1, method='css',
                                             start_params=[.905322, -.692425,
                                                           1.07366, 0.172024])
        cls.res2 = results_arima.ARIMA112(method='css')
        cls.res2.fittedvalues = - cpi[1:-1] + cls.res2.linear
        # make sure endog names changes to D.cpi
        #(cls.res1.forecast_res,
        # cls.res1.forecast_err,
        # conf_int) = cls.res1.forecast(25)
        #cls.res1.forecast_res_dyn = cls.res1.predict(start=164, end=226,
        #                                              typ='levels',
        #                                              dynamic=True)
        # TODO: fix the indexing for the end here, I don't think this is right
        # if we're going to treat it like indexing
        # the forecast from 2005Q1 through 2009Q4 is indices
        # 184 through 227 not 226
        # note that the first one counts in the count so 164 + 64 is 65
        # predictions
        #cls.res1.forecast_res_dyn = self.predict(start=164, end=164+63,
        #                                         typ='levels', dynamic=True)
        # since we got from gretl don't have linear prediction in differences

    def test_freq(self):
        assert_almost_equal(self.res1.arfreq, [0.5000], 4)
        assert_almost_equal(self.res1.mafreq, [0.5000, 0.5000], 4)


#class Test_ARIMADates(CheckArmaResults, CheckForecast, CheckDynamicForecast):
#    @classmethod
#    def setup_class(cls):
#        cpi = datasets.macrodata.load_pandas().data['cpi'].values
#        dates = pd.date_range('1959', periods=203, freq='Q')
#        cls.res1 = ARIMA(cpi, dates=dates, freq='Q').fit(order=(1, 1, 1),
#                                                         disp=-1)
#        cls.res2 = results_arima.ARIMA111()
#        # make sure endog names changes to D.cpi
#        cls.decimal_llf = 3
#        cls.decimal_aic = 3
#        cls.decimal_bic = 3
#        (cls.res1.forecast_res,
#         cls.res1.forecast_err,
#         conf_int) = cls.res1.forecast(25)


@pytest.mark.not_vetted
@pytest.mark.slow
def test_start_params_bug():
    data = np.array([
        1368., 1187, 1090, 1439, 2362, 2783, 2869, 2512, 1804,
        1544, 1028, 869, 1737, 2055, 1947, 1618, 1196, 867, 997, 1862, 2525,
        3250, 4023, 4018, 3585, 3004, 2500, 2441, 2749, 2466, 2157, 1847,
        1463, 1146, 851, 993, 1448, 1719, 1709, 1455, 1950, 1763, 2075, 2343,
        3570, 4690, 3700, 2339, 1679, 1466, 998, 853, 835, 922, 851, 1125,
        1299, 1105, 860, 701, 689, 774, 582, 419, 846, 1132, 902, 1058, 1341,
        1551, 1167, 975, 786, 759, 751, 649, 876, 720, 498, 553, 459, 543,
        447, 415, 377, 373, 324, 320, 306, 259, 220, 342, 558, 825, 994,
        1267, 1473, 1601, 1896, 1890, 2012, 2198, 2393, 2825, 3411, 3406,
        2464, 2891, 3685, 3638, 3746, 3373, 3190, 2681, 2846, 4129, 5054,
        5002, 4801, 4934, 4903, 4713, 4745, 4736, 4622, 4642, 4478, 4510,
        4758, 4457, 4356, 4170, 4658, 4546, 4402, 4183, 3574, 2586, 3326,
        3948, 3983, 3997, 4422, 4496, 4276, 3467, 2753, 2582, 2921, 2768,
        2789, 2824, 2482, 2773, 3005, 3641, 3699, 3774, 3698, 3628, 3180,
        3306, 2841, 2014, 1910, 2560, 2980, 3012, 3210, 3457, 3158, 3344,
        3609, 3327, 2913, 2264, 2326, 2596, 2225, 1767, 1190, 792, 669,
        589, 496, 354, 246, 250, 323, 495, 924, 1536, 2081, 2660, 2814, 2992,
        3115, 2962, 2272, 2151, 1889, 1481, 955, 631, 288, 103, 60, 82, 107,
        185, 618, 1526, 2046, 2348, 2584, 2600, 2515, 2345, 2351, 2355,
        2409, 2449, 2645, 2918, 3187, 2888, 2610, 2740, 2526, 2383, 2936,
        2968, 2635, 2617, 2790, 3906, 4018, 4797, 4919, 4942, 4656, 4444,
        3898, 3908, 3678, 3605, 3186, 2139, 2002, 1559, 1235, 1183, 1096,
        673, 389, 223, 352, 308, 365, 525, 779, 894, 901, 1025, 1047, 981,
        902, 759, 569, 519, 408, 263, 156, 72, 49, 31, 41, 192, 423, 492,
        552, 564, 723, 921, 1525, 2768, 3531, 3824, 3835, 4294, 4533, 4173,
        4221, 4064, 4641, 4685, 4026, 4323, 4585, 4836, 4822, 4631, 4614,
        4326, 4790, 4736, 4104, 5099, 5154, 5121, 5384, 5274, 5225, 4899,
        5382, 5295, 5349, 4977, 4597, 4069, 3733, 3439, 3052, 2626, 1939,
        1064, 713, 916, 832, 658, 817, 921, 772, 764, 824, 967, 1127, 1153,
        824, 912, 957, 990, 1218, 1684, 2030, 2119, 2233, 2657, 2652, 2682,
        2498, 2429, 2346, 2298, 2129, 1829, 1816, 1225, 1010, 748, 627, 469,
        576, 532, 475, 582, 641, 605, 699, 680, 714, 670, 666, 636, 672,
        679, 446, 248, 134, 160, 178, 286, 413, 676, 1025, 1159, 952, 1398,
        1833, 2045, 2072, 1798, 1799, 1358, 727, 353, 347, 844, 1377, 1829,
        2118, 2272, 2745, 4263, 4314, 4530, 4354, 4645, 4547, 5391, 4855,
        4739, 4520, 4573, 4305, 4196, 3773, 3368, 2596, 2596, 2305, 2756,
        3747, 4078, 3415, 2369, 2210, 2316, 2263, 2672, 3571, 4131, 4167,
        4077, 3924, 3738, 3712, 3510, 3182, 3179, 2951, 2453, 2078, 1999,
        2486, 2581, 1891, 1997, 1366, 1294, 1536, 2794, 3211, 3242, 3406,
        3121, 2425, 2016, 1787, 1508, 1304, 1060, 1342, 1589, 2361, 3452,
        2659, 2857, 3255, 3322, 2852, 2964, 3132, 3033, 2931, 2636, 2818, 3310,
        3396, 3179, 3232, 3543, 3759, 3503, 3758, 3658, 3425, 3053, 2620, 1837,
        923, 712, 1054, 1376, 1556, 1498, 1523, 1088, 728, 890, 1413, 2524,
        3295, 4097, 3993, 4116, 3874, 4074, 4142, 3975, 3908, 3907, 3918, 3755,
        3648, 3778, 4293, 4385, 4360, 4352, 4528, 4365, 3846, 4098, 3860, 3230,
        2820, 2916, 3201, 3721, 3397, 3055, 2141, 1623, 1825, 1716, 2232, 2939,
        3735, 4838, 4560, 4307, 4975, 5173, 4859, 5268, 4992, 5100, 5070, 5270,
        4760, 5135, 5059, 4682, 4492, 4933, 4737, 4611, 4634, 4789, 4811, 4379,
        4689, 4284, 4191, 3313, 2770, 2543, 3105, 2967, 2420, 1996, 2247, 2564,
        2726, 3021, 3427, 3509, 3759, 3324, 2988, 2849, 2340, 2443, 2364, 1252,
        623, 742, 867, 684, 488, 348, 241, 187, 279, 355, 423, 678, 1375, 1497,
        1434, 2116, 2411, 1929, 1628, 1635, 1609, 1757, 2090, 2085, 1790, 1846,
        2038, 2360, 2342, 2401, 2920, 3030, 3132, 4385, 5483, 5865, 5595, 5485,
        5727, 5553, 5560, 5233, 5478, 5159, 5155, 5312, 5079, 4510, 4628, 4535,
        3656, 3698, 3443, 3146, 2562, 2304, 2181, 2293, 1950, 1930, 2197, 2796,
        3441, 3649, 3815, 2850, 4005, 5305, 5550, 5641, 4717, 5131, 2831, 3518,
        3354, 3115, 3515, 3552, 3244, 3658, 4407, 4935, 4299, 3166, 3335, 2728,
        2488, 2573, 2002, 1717, 1645, 1977, 2049, 2125, 2376, 2551, 2578, 2629,
        2750, 3150, 3699, 4062, 3959, 3264, 2671, 2205, 2128, 2133, 2095, 1964,
        2006, 2074, 2201, 2506, 2449, 2465, 2064, 1446, 1382, 983, 898, 489,
        319, 383, 332, 276, 224, 144, 101, 232, 429, 597, 750, 908, 960, 1076,
        951, 1062, 1183, 1404, 1391, 1419, 1497, 1267, 963, 682, 777, 906,
        1149, 1439, 1600, 1876, 1885, 1962, 2280, 2711, 2591, 2411])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ARMA(data, order=(4, 1)).fit(start_ar_lags=5, disp=-1)


@pytest.mark.not_vetted
def test_arima_predict_mle_dates():
    cpi = datasets.macrodata.load_pandas().data['cpi'].values
    res1 = ARIMA(cpi, (4, 1, 1), dates=cpi_dates, freq='Q').fit(disp=-1)

    path = os.path.join(current_path, 'results',
                        'results_arima_forecasts_all_mle.csv')
    arima_forecasts = pd.read_csv(path).values
    fc = arima_forecasts[:, 0]
    fcdyn = arima_forecasts[:, 1]
    fcdyn2 = arima_forecasts[:, 2]

    start, end = 2, 51
    fv = res1.predict('1959Q3', '1971Q4', typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    tm.assert_index_equal(res1.data.predict_dates,
                          cpi_dates[start:end + 1])

    start, end = 202, 227
    fv = res1.predict('2009Q3', '2015Q4', typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    tm.assert_index_equal(res1.data.predict_dates,
                          cpi_predict_dates)

    # make sure dynamic works

    start, end = '1960q2', '1971q4'
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[5:51 + 1], DECIMAL_4)

    start, end = '1965q1', '2015q4'
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[24:227 + 1], DECIMAL_4)


@pytest.mark.not_vetted
def test_arma_predict_mle_dates():
    sunspots = datasets.sunspots.load_pandas().data['SUNACTIVITY'].values
    mod = ARMA(sunspots, (9, 0), dates=sun_dates, freq='A')
    mod.method = 'mle'

    with pytest.raises(ValueError):
        mod._get_prediction_index('1701', '1751', True)

    start, end = 2, 51
    mod._get_prediction_index('1702', '1751', False)
    tm.assert_index_equal(mod.data.predict_dates, sun_dates[start:end + 1])

    start, end = 308, 333
    mod._get_prediction_index('2008', '2033', False)
    tm.assert_index_equal(mod.data.predict_dates, sun_predict_dates)


@pytest.mark.not_vetted
def test_arima_predict_css_dates():
    cpi = datasets.macrodata.load_pandas().data['cpi'].values
    res1 = ARIMA(cpi, (4, 1, 1), dates=cpi_dates, freq='Q').fit(disp=-1,
                                                                method='css',
                                                                trend='nc')

    params = np.array([1.231272508473910,
                       -0.282516097759915,
                       0.170052755782440,
                       -0.118203728504945,
                       -0.938783134717947])

    path = os.path.join(current_path, 'results',
                        'results_arima_forecasts_all_css.csv')
    arima_forecasts = pd.read_csv(path).values
    fc = arima_forecasts[:, 0]
    fcdyn = arima_forecasts[:, 1]
    fcdyn2 = arima_forecasts[:, 2]

    start, end = 5, 51
    fv = res1.model.predict(params, '1960Q2', '1971Q4', typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    tm.assert_index_equal(res1.data.predict_dates, cpi_dates[start:end + 1])

    start, end = 202, 227
    fv = res1.model.predict(params, '2009Q3', '2015Q4', typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    tm.assert_index_equal(res1.data.predict_dates, cpi_predict_dates)

    # make sure dynamic works
    start, end = 5, 51
    fv = res1.model.predict(params, '1960Q2', '1971Q4', typ='levels',
                            dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)

    start, end = '1965q1', '2015q4'
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[24:227 + 1], DECIMAL_4)


@pytest.mark.not_vetted
def test_arma_predict_css_dates():
    # TODO: GH reference?
    sunspots = datasets.sunspots.load_pandas().data['SUNACTIVITY'].values
    mod = ARMA(sunspots, (9, 0), dates=sun_dates, freq='A')
    mod.method = 'css'
    with pytest.raises(ValueError):
        mod._get_prediction_index('1701', '1751', False)


def test_arima_wrapper():
    # test that names get attached to res.params correctly
    # TODO: GH reference?
    cpi = datasets.macrodata.load_pandas().data['cpi']
    cpi.index = pd.Index(cpi_dates)
    res = ARIMA(cpi, (4, 1, 1), freq='Q').fit(disp=-1)

    expected_index = pd.Index(['const', 'ar.L1.D.cpi', 'ar.L2.D.cpi',
                               'ar.L3.D.cpi', 'ar.L4.D.cpi',
                               'ma.L1.D.cpi'])
    assert expected_index.equals(res.params.index)
    tm.assert_index_equal(res.params.index, expected_index)
    assert res.model.endog_names == 'D.cpi'


@pytest.mark.not_vetted
@pytest.mark.smoke
def test_1dexog():
    # smoke test, this will raise an error if broken
    dta = datasets.macrodata.load_pandas().data
    endog = dta['realcons'].values
    exog = dta['m1'].values.squeeze()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ARMA(endog, (1, 1), exog).fit(disp=-1)
        mod.predict(193, 203, exog[-10:])

        # check for dynamic is true and pandas Series  see GH#2589
        mod.predict(193, 202, exog[-10:], dynamic=True)

        dta.index = pd.Index(cpi_dates)
        mod = ARMA(dta['realcons'], (1, 1), dta['m1'])
        res = mod.fit(disp=-1)
        res.predict(dta.index[-10], dta.index[-1],
                    exog=dta['m1'][-10:], dynamic=True)

        mod = ARMA(dta['realcons'], (1, 1), dta['m1'])
        res = mod.fit(trend='nc', disp=-1)
        res.predict(dta.index[-10], dta.index[-1],
                    exog=dta['m1'][-10:], dynamic=True)


@pytest.mark.not_vetted
def test_arima_predict_bug():
    # predict_start_date wasn't getting set on start = None
    # TODO: GH reference?
    dta = datasets.sunspots.load_pandas().data['SUNACTIVITY']
    dta.index = pd.DatetimeIndex(start='1700', end='2009', freq='A')[:309]
    arma_mod20 = ARMA(dta, (2, 0)).fit(disp=-1)
    arma_mod20.predict(None, None)

    # test prediction with time stamp, see GH#2587
    predict = arma_mod20.predict(dta.index[-20], dta.index[-1])
    assert predict.index.equals(dta.index[-20:])
    predict = arma_mod20.predict(dta.index[-20], dta.index[-1], dynamic=True)
    assert predict.index.equals(dta.index[-20:])
    # partially out of sample
    predict_dates = pd.DatetimeIndex(start='2000', end='2015', freq='A')
    predict = arma_mod20.predict(predict_dates[0], predict_dates[-1])
    assert predict.index.equals(predict_dates)


@pytest.mark.not_vetted
def test_arima_predict_q2():
    # bug with q > 1 for arima predict
    # TODO: GH reference?
    inv = datasets.macrodata.load().data['realinv']
    arima_mod = ARIMA(np.log(inv), (1, 1, 2)).fit(start_params=[0, 0, 0, 0],
                                                  disp=-1)
    fc, stderr, conf_int = arima_mod.forecast(5)
    # values copy-pasted from gretl
    assert_almost_equal(fc,
                        [7.306320, 7.313825, 7.321749, 7.329827, 7.337962],
                        5)


@pytest.mark.not_vetted
def test_arima_predict_pandas_nofreq():
    # GH#712
    dates = ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07",
             "2010-01-08", "2010-01-11", "2010-01-12", "2010-01-11",
             "2010-01-12", "2010-01-13", "2010-01-17"]
    close = [626.75, 623.99, 608.26, 594.1, 602.02, 601.11, 590.48, 587.09,
             589.85, 580.0, 587.62]
    data = pd.DataFrame(close, index=pd.DatetimeIndex(dates),
                        columns=["close"])

    # TODO: fix this names bug for non-string names names
    arma = ARMA(data, order=(1, 0)).fit(disp=-1)

    # first check that in-sample prediction works
    predict = arma.predict()
    assert predict.index.equals(data.index)

    # check that this raises an exception when date not on index
    with pytest.raises(KeyError):
        arma.predict(start="2010-1-9", end=10)

    with pytest.raises(KeyError):
        arma.predict(start="2010-1-9", end="2010-1-17")

    # raise because end not on index
    with pytest.raises(KeyError):
        arma.predict(start="2010-1-4", end="2010-1-10")

    # raise because end not on index
    with pytest.raises(KeyError):
        arma.predict(start=3, end="2010-1-10")

    predict = arma.predict(start="2010-1-7", end=10)  # should be of length 10
    assert len(predict) == 8
    assert predict.index.equals(data.index[3:10 + 1])

    predict = arma.predict(start="2010-1-7", end=14)
    assert predict.index.equals(pd.Index(range(3, 15)))

    predict = arma.predict(start=3, end=14)
    assert predict.index.equals(pd.Index(range(3, 15)))

    # end can be a date if it's in the sample and on the index
    # predict dates is just a slice of the dates index then
    predict = arma.predict(start="2010-1-6", end="2010-1-13")
    assert predict.index.equals(data.index[2:10])
    predict = arma.predict(start=2, end="2010-1-13")
    assert predict.index.equals(data.index[2:10])


@pytest.mark.not_vetted
def test_arima_predict_exog():
    # check GH#625 and GH#626
    # Note: upstream there is a bunch of commented-out code after this point;
    # I have not been able to get an explanation as to if/why it is worth
    # keeping.
    # TODO: At some point check back to see if it has been addressed.
    path = os.path.join(current_path, 'results',
                        'results_arima_exog_forecasts_mle.csv')
    arima_forecasts = pd.read_csv(path)
    y = arima_forecasts["y"].dropna()
    X = np.arange(len(y) + 25) / 20.
    predict_expected = arima_forecasts["predict"]
    arma_res = ARMA(y.values, order=(2, 1), exog=X[:100]).fit(trend="c",
                                                              disp=-1)
    # params from gretl
    params = np.array([2.786912485145725, -0.122650190196475,
                       0.533223846028938, -0.319344321763337,
                       0.132883233000064])
    assert_almost_equal(arma_res.params, params, 5)
    # no exog for in-sample
    predict = arma_res.predict()
    assert_almost_equal(predict, predict_expected.values[:100], 5)

    # check GH#626
    assert len(arma_res.model.exog_names) == 5

    # exog for out-of-sample and in-sample dynamic
    predict = arma_res.model.predict(params, end=124, exog=X[100:])
    assert_almost_equal(predict, predict_expected.values, 6)

    # conditional sum of squares
    #arima_forecasts = pd.read_csv(current_path + "/results/"
    #                              "results_arima_exog_forecasts_css.csv")
    #predict_expected = arima_forecasts["predict"].dropna()
    #arma_res = ARMA(y.values, order=(2, 1), exog=X[:100]).fit(trend="c",
    #                                                          method="css",
    #                                                          disp=-1)

    #params = np.array([2.152350033809826, -0.103602399018814,
    #                   0.566716580421188, -0.326208009247944,
    #                   0.102142932143421])
    #predict = arma_res.model.predict(params)
    # in-sample
    #assert_almost_equal(predict, predict_expected.values[:98], 6)

    #predict = arma_res.model.predict(params, end=124, exog=X[100:])
    # exog for out-of-sample and in-sample dynamic
    #assert_almost_equal(predict, predict_expected.values, 3)


@pytest.mark.not_vetted
def test_arimax():
    dta = datasets.macrodata.load_pandas().data
    dta.index = cpi_dates
    dta = dta[["realdpi", "m1", "realgdp"]]
    y = dta.pop("realdpi")

    # 1 exog
    #X = dta.iloc[1:]["m1"]
    #res = ARIMA(y, (2, 1, 1), X).fit(disp=-1)
    #params = [23.902305009084373, 0.024650911502790, -0.162140641341602,
    #          0.165262136028113, -0.066667022903974]
    #assert_almost_equal(res.params.values, params, 6)

    # 2 exog
    X = dta
    res = ARIMA(y, (2, 1, 1), X).fit(disp=False, solver="nm", maxiter=1000,
                                     ftol=1e-12, xtol=1e-12)

    # from gretl; we use the versions from stata below instead
    # params = [13.113976653926638, -0.003792125069387, 0.004123504809217,
    #          -0.199213760940898, 0.151563643588008, -0.033088661096699]
    # from stata using double
    stata_llf = -1076.108614859121
    params = [13.1259220104, -0.00376814509403812, 0.00411970083135622,
              -0.19921477896158524, 0.15154396192855729, -0.03308400760360837]
    # we can get close
    assert_almost_equal(res.params.values, params, 4)

    # This shows that it's an optimizer problem and not a problem in the code
    assert_almost_equal(res.model.loglike(np.array(params)), stata_llf, 6)

    X = dta.diff()
    X.iloc[0] = 0
    res = ARIMA(y, (2, 1, 1), X).fit(disp=False)

    # gretl won't estimate this - looks like maybe a bug on their part,
    # but we can just fine, we're close to Stata's answer
    # from Stata
    params = [19.5656863783347, 0.32653841355833396198,
              0.36286527042965188716, -1.01133792126884,
              -0.15722368379307766206, 0.69359822544092153418]

    assert_almost_equal(res.params.values, params, 3)


@pytest.mark.not_vetted
def test_bad_start_params():
    # TODO: what is bad about these params??
    # TODO: GH reference?
    endog = np.array([
        820.69093, 781.0103028, 785.8786988, 767.64282267,
        778.9837648, 824.6595702, 813.01877867, 751.65598567,
        753.431091, 746.920813, 795.6201904, 772.65732833,
        793.4486454, 868.8457766, 823.07226547, 783.09067747,
        791.50723847, 770.93086347, 835.34157333, 810.64147947,
        738.36071367, 776.49038513, 822.93272333, 815.26461227,
        773.70552987, 777.3726522, 811.83444853, 840.95489133,
        777.51031933, 745.90077307, 806.95113093, 805.77521973,
        756.70927733, 749.89091773, 1694.2266924, 2398.4802244,
        1434.6728516, 909.73940427, 929.01291907, 769.07561453,
        801.1112548, 796.16163313, 817.2496376, 857.73046447,
        838.849345, 761.92338873, 731.7842242, 770.4641844])
    mod = ARMA(endog, (15, 0))
    with pytest.raises(ValueError):
        mod.fit()

    inv = datasets.macrodata.load().data['realinv']
    arima_mod = ARIMA(np.log(inv), (1, 1, 2))
    with pytest.raises(ValueError):
        # TODO: Upstream this incorrectly re-tries `mod.fit()`
        arima_mod.fit()


@pytest.mark.not_vetted
def test_armax_predict_no_trend():
    # GH#1123 test ARMAX predict doesn't ignore exog when trend is none
    arparams = np.array([.75, -.25])
    maparams = np.array([.65, .35])

    nobs = 20

    np.random.seed(12345)
    y = arma_generate_sample(arparams, maparams, nobs)

    X = np.random.randn(nobs)
    y += 5 * X
    mod = ARMA(y[:-1], order=(1, 0), exog=X[:-1])
    res = mod.fit(trend='nc', disp=False)
    fc = res.forecast(exog=X[-1:])
    # results from gretl
    assert_almost_equal(fc[0], 2.200393, 6)
    assert_almost_equal(fc[1], 1.030743, 6)
    assert_almost_equal(fc[2][0, 0], 0.180175, 6)
    assert_almost_equal(fc[2][0, 1], 4.220611, 6)

    mod = ARMA(y[:-1], order=(1, 1), exog=X[:-1])
    res = mod.fit(trend='nc', disp=False)
    fc = res.forecast(exog=X[-1:])
    assert_almost_equal(fc[0], 2.765688, 6)
    assert_almost_equal(fc[1], 0.835048, 6)
    assert_almost_equal(fc[2][0, 0], 1.129023, 6)
    assert_almost_equal(fc[2][0, 1], 4.402353, 6)

    # make sure this works to. code looked fishy.
    mod = ARMA(y[:-1], order=(1, 0), exog=X[:-1])
    res = mod.fit(trend='c', disp=False)
    fc = res.forecast(exog=X[-1:])
    assert_almost_equal(fc[0], 2.481219, 6)
    assert_almost_equal(fc[1], 0.968759, 6)
    assert_almost_equal(fc[2][0], [0.582485, 4.379952], 6)


@pytest.mark.not_vetted
def test_small_data():
    # GH#1146
    y = [-1214.360173, -1848.209905, -2100.918158, -3647.483678, -4711.186773]

    # refuse to estimate these
    with pytest.raises(ValueError):
        ARIMA(y, (2, 0, 3))
    with pytest.raises(ValueError):
        ARIMA(y, (1, 1, 3))

    mod = ARIMA(y, (1, 0, 3))
    with pytest.raises(ValueError):
        mod.fit(trend="c")

    # TODO: mark these as smoke?
    # try to estimate these...leave it up to the user to check for garbage
    # and be clear, these are garbage parameters.
    # X-12 arima will estimate, gretl refuses to estimate likely a problem
    # in start params regression.
    mod.fit(trend="nc", disp=0, start_params=[.1, .1, .1, .1])

    mod = ARIMA(y, (1, 0, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod.fit(disp=0, start_params=[np.mean(y), .1, .1, .1])


@pytest.mark.not_vetted
class TestARMA00(object):

    @classmethod
    def setup_class(cls):
        sunspots = datasets.sunspots.load_pandas().data['SUNACTIVITY'].values
        cls.y = y = sunspots
        cls.arma_00_model = ARMA(y, order=(0, 0))
        cls.arma_00_res = cls.arma_00_model.fit(disp=-1)

    def test_parameters(self):
        params = self.arma_00_res.params
        assert_almost_equal(self.y.mean(), params)

    def test_predictions(self):
        predictions = self.arma_00_res.predict()
        assert_almost_equal(self.y.mean() * np.ones_like(predictions),
                            predictions)

    def test_arroots(self):  # TODO: Belongs in test_wold?
        # GH#4559
        # regression test; older implementation of arroots returned None
        # instead of en empty array
        roots = self.arma_00_res.arroots
        assert roots.size == 0

    def test_maroots(self):  # TODO: Belongs in test_wold?
        # GH#4559
        # regression test; older implementation of arroots returned None
        # instead of en empty array
        roots = self.arma_00_res.maroots
        assert roots.size == 0

    @pytest.mark.skip(reason=' This test is invalid since the ICs differ due '
                             'to df_model differences between OLS and ARIMA')
    def test_information_criteria(self):
        # This test is invalid since the ICs differ due to df_model differences
        # between OLS and ARIMA
        res = self.arma_00_res
        y = self.y
        ols_res = OLS(y, np.ones_like(y)).fit(disp=-1)
        ols_ic = np.array([ols_res.aic, ols_res.bic])
        arma_ic = np.array([res.aic, res.bic])
        assert_almost_equal(ols_ic, arma_ic, DECIMAL_4)

    def test_arma_00_nc(self):
        arma_00 = ARMA(self.y, order=(0, 0))
        with pytest.raises(ValueError):
            arma_00.fit(trend='nc', disp=-1)

    def test_css(self):
        arma = ARMA(self.y, order=(0, 0))
        fit = arma.fit(method='css', disp=-1)
        predictions = fit.predict()
        assert_almost_equal(self.y.mean() * np.ones_like(predictions),
                            predictions)

    def test_arima(self):
        yi = np.cumsum(self.y)
        arima = ARIMA(yi, order=(0, 1, 0))
        fit = arima.fit(disp=-1)
        assert_almost_equal(np.diff(yi).mean(),
                            fit.params,
                            DECIMAL_4)

    def test_arma_ols(self):
        y = self.y
        y_lead = y[1:]
        y_lag = y[:-1]
        T = y_lag.shape[0]
        X = np.hstack((np.ones((T, 1)), y_lag[:, None]))
        ols_res = OLS(y_lead, X).fit()
        arma_res = ARMA(y_lead, order=(0, 0), exog=y_lag).fit(trend='c',
                                                              disp=-1)
        assert_almost_equal(ols_res.params, arma_res.params)

    def test_arma_exog_no_constant(self):
        y = self.y
        y_lead = y[1:]
        y_lag = y[:-1]
        X = y_lag[:, None]
        ols_res = OLS(y_lead, X).fit()
        arma_res = ARMA(y_lead, order=(0, 0), exog=y_lag).fit(trend='nc',
                                                              disp=-1)
        assert_almost_equal(ols_res.params, arma_res.params)


@pytest.mark.not_vetted
def test_arima_dates_startatend():
    # TODO: GH reference?
    np.random.seed(18)
    x = pd.Series(np.random.random(36),
                  index=pd.DatetimeIndex(start='1/1/1990',
                                         periods=36, freq='M'))
    res = ARIMA(x, (1, 0, 0)).fit(disp=0)
    pred = res.predict(start=len(x), end=len(x))
    assert pred.index[0] == x.index.shift(1)[-1]
    fc = res.forecast()[0]
    assert_almost_equal(pred.values[0], fc)


@pytest.mark.not_vetted
def test_arima_diff2():
    dta = datasets.macrodata.load_pandas().data['cpi']
    dta.index = cpi_dates
    mod = ARIMA(dta, (3, 2, 1)).fit(disp=-1)
    fc, fcerr, conf_int = mod.forecast(10)
    # forecasts from gretl
    conf_int_res = [(216.139, 219.231),
                    (216.472, 221.520),
                    (217.064, 223.649),
                    (217.586, 225.727),
                    (218.119, 227.770),
                    (218.703, 229.784),
                    (219.306, 231.777),
                    (219.924, 233.759),
                    (220.559, 235.735),
                    (221.206, 237.709)]

    fc_res = [217.685, 218.996, 220.356, 221.656, 222.945,
              224.243, 225.541, 226.841, 228.147, 229.457]
    fcerr_res = [0.7888, 1.2878, 1.6798, 2.0768, 2.4620,
                 2.8269, 3.1816, 3.52950, 3.8715, 4.2099]

    assert_almost_equal(fc, fc_res, 3)
    assert_almost_equal(fcerr, fcerr_res, 3)
    assert_almost_equal(conf_int, conf_int_res, 3)

    predicted = mod.predict('2008Q1', '2012Q1', typ='levels')

    predicted_res = [214.464, 215.478, 221.277, 217.453, 212.419, 213.530,
                     215.087, 217.685, 218.996, 220.356, 221.656, 222.945,
                     224.243, 225.541, 226.841, 228.147, 229.457]
    assert_almost_equal(predicted, predicted_res, 3)


@pytest.mark.not_vetted
def test_arima111_predict_exog_2127():
    # regression test for issue GH#2127
    ef = [0.03005, 0.03917, 0.02828, 0.03644, 0.03379, 0.02744,
          0.03343, 0.02621, 0.03050, 0.02455, 0.03261, 0.03507,
          0.02734, 0.05373, 0.02677, 0.03443, 0.03331, 0.02741,
          0.03709, 0.02113, 0.03343, 0.02011, 0.03675, 0.03077,
          0.02201, 0.04844, 0.05518, 0.03765, 0.05433, 0.03049,
          0.04829, 0.02936, 0.04421, 0.02457, 0.04007, 0.03009,
          0.04504, 0.05041, 0.03651, 0.02719, 0.04383, 0.02887,
          0.03440, 0.03348, 0.02364, 0.03496, 0.02549, 0.03284,
          0.03523, 0.02579, 0.03080, 0.01784, 0.03237, 0.02078,
          0.03508, 0.03062, 0.02006, 0.02341, 0.02223, 0.03145,
          0.03081, 0.02520, 0.02683, 0.01720, 0.02225, 0.01579,
          0.02237, 0.02295, 0.01830, 0.02356, 0.02051, 0.02932,
          0.03025, 0.02390, 0.02635, 0.01863, 0.02994, 0.01762,
          0.02837, 0.02421, 0.01951, 0.02149, 0.02079, 0.02528,
          0.02575, 0.01634, 0.02563, 0.01719, 0.02915, 0.01724,
          0.02804, 0.02750, 0.02099, 0.02522, 0.02422, 0.03254,
          0.02095, 0.03241, 0.01867, 0.03998, 0.02212, 0.03034,
          0.03419, 0.01866, 0.02623, 0.02052]
    ue = [4.9, 5.0, 5.0, 5.0, 4.9, 4.7, 4.8, 4.7, 4.7,
          4.6, 4.6, 4.7, 4.7, 4.5, 4.4, 4.5, 4.4, 4.6,
          4.5, 4.4, 4.5, 4.4, 4.6, 4.7, 4.6, 4.7, 4.7,
          4.7, 5.0, 5.0, 4.9, 5.1, 5.0, 5.4, 5.6, 5.8,
          6.1, 6.1, 6.5, 6.8, 7.3, 7.8, 8.3, 8.7, 9.0,
          9.4, 9.5, 9.5, 9.6, 9.8, 10., 9.9, 9.9, 9.7,
          9.8, 9.9, 9.9, 9.6, 9.4, 9.5, 9.5, 9.5, 9.5,
          9.8, 9.4, 9.1, 9.0, 9.0, 9.1, 9.0, 9.1, 9.0,
          9.0, 9.0, 8.8, 8.6, 8.5, 8.2, 8.3, 8.2, 8.2,
          8.2, 8.2, 8.2, 8.1, 7.8, 7.8, 7.8, 7.9, 7.9,
          7.7, 7.5, 7.5, 7.5, 7.5, 7.3, 7.2, 7.2, 7.2,
          7.0, 6.7, 6.6, 6.7, 6.7, 6.3, 6.3]

    ue = np.array(ue) / 100
    model = ARIMA(ef, (1, 1, 1), exog=ue)
    res = model.fit(transparams=False, pgtol=1e-8, iprint=0, disp=0)

    assert res.mle_retvals['warnflag'] == 0

    predicts = res.predict(start=len(ef), end=len(ef) + 10,
                           exog=ue[-11:], typ='levels')

    # regression test, not verified numbers
    predicts_res = np.array([
        0.02591095, 0.02321325, 0.02436579, 0.02368759, 0.02389753,
        0.02372, 0.0237481, 0.0236738, 0.023644, 0.0236283,
        0.02362267])
    assert_allclose(predicts, predicts_res, atol=5e-6)


@pytest.mark.not_vetted
def test_arima_exog_predict():
    # TODO: break up giant test
    # test forecasting and dynamic prediction with exog against Stata
    dta = datasets.macrodata.load_pandas().data
    cpi_dates = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
    dta.index = cpi_dates

    data = dta
    data['loginv'] = np.log(data['realinv'])
    data['loggdp'] = np.log(data['realgdp'])
    data['logcons'] = np.log(data['realcons'])

    forecast_period = pd.PeriodIndex(start='2008Q2', end='2009Q4', freq='Q')
    end = forecast_period[0]
    data_sample = data.loc[dta.index < end]

    exog_full = data[['loggdp', 'logcons']]

    # pandas
    mod = ARIMA(data_sample['loginv'], (1, 0, 1),
                exog=data_sample[['loggdp', 'logcons']])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=0, solver='bfgs', maxiter=5000)

    predicted_arma_fp = res.predict(start=197, end=202,
                                    exog=exog_full.values[197:]).values
    predicted_arma_dp = res.predict(start=193, end=202,
                                    exog=exog_full[197:], dynamic=True)

    # numpy
    mod2 = ARIMA(np.asarray(data_sample['loginv']),
                 (1, 0, 1),
                 exog=np.asarray(data_sample[['loggdp', 'logcons']]))
    res2 = mod2.fit(start_params=res.params, disp=0,
                    solver='bfgs', maxiter=5000)

    exog_full = data[['loggdp', 'logcons']]
    predicted_arma_f = res2.predict(start=197, end=202,
                                    exog=exog_full.values[197:])
    predicted_arma_d = res2.predict(start=193, end=202,
                                    exog=exog_full[197:], dynamic=True)

    endog_scale = 100
    ex_scale = 1000
    # ARIMA(1, 1, 1)
    ex = ex_scale * np.asarray(data_sample[['loggdp', 'logcons']].diff())
    # The first obsevation is not (supposed to be) used, but I get
    # a Lapack problem
    # Intel MKL ERROR: Parameter 5 was incorrect on entry to DLASCL.
    ex[0] = 0
    mod111 = ARIMA(100 * np.asarray(data_sample['loginv']),
                   (1, 1, 1),
                   # Stata differences also the exog
                   exog=ex)

    res111 = mod111.fit(disp=0, solver='bfgs', maxiter=5000)
    exog_full_d = ex_scale * data[['loggdp', 'logcons']].diff()
    res111.predict(start=197, end=202, exog=exog_full_d.values[197:])

    predicted_arima_f = res111.predict(start=196, end=202,
                                       exog=exog_full_d.values[197:],
                                       typ='levels')
    predicted_arima_d = res111.predict(start=193, end=202,
                                       exog=exog_full_d.values[197:],
                                       typ='levels', dynamic=True)

    res_f101 = np.array([
        7.73975859954, 7.71660108543, 7.69808978329, 7.70872117504,
        7.6518392758, 7.69784279784, 7.70290907856, 7.69237782644,
        7.65017785174, 7.66061689028, 7.65980022857, 7.61505314129,
        7.51697158428, 7.5165760663, 7.5271053284])
    res_f111 = np.array([
        7.74460013693, 7.71958207517, 7.69629561172, 7.71208186737,
        7.65758850178, 7.69223472572, 7.70411775588, 7.68896109499,
        7.64016249001, 7.64871881901, 7.62550283402, 7.55814609462,
        7.44431310053, 7.42963968062, 7.43554675427])
    res_d111 = np.array([
        7.74460013693, 7.71958207517, 7.69629561172, 7.71208186737,
        7.65758850178, 7.69223472572, 7.71870821151, 7.7299430215,
        7.71439447355, 7.72544001101, 7.70521902623, 7.64020040524,
        7.5281927191, 7.5149442694, 7.52196378005])
    res_d101 = np.array([
        7.73975859954, 7.71660108543, 7.69808978329, 7.70872117504,
        7.6518392758, 7.69784279784, 7.72522142662, 7.73962377858,
        7.73245950636, 7.74935432862, 7.74449584691, 7.69589103679,
        7.5941274688, 7.59021764836, 7.59739267775])

    assert_allclose(predicted_arma_dp,
                    res_d101[-len(predicted_arma_d):],
                    atol=1e-4)
    assert_allclose(predicted_arma_fp,
                    res_f101[-len(predicted_arma_f):],
                    atol=1e-4)
    assert_allclose(predicted_arma_d,
                    res_d101[-len(predicted_arma_d):],
                    atol=1e-4)
    assert_allclose(predicted_arma_f,
                    res_f101[-len(predicted_arma_f):],
                    atol=1e-4)
    assert_allclose(predicted_arima_d / endog_scale,
                    res_d111[-len(predicted_arima_d):],
                    rtol=1e-4, atol=1e-4)
    assert_allclose(predicted_arima_f / endog_scale,
                    res_f111[-len(predicted_arima_f):],
                    rtol=1e-4, atol=1e-4)

    # test for forecast with 0 ar fix in GH#2457 numbers again from Stata

    res_f002 = np.array([
        7.70178181209, 7.67445481224, 7.6715373765, 7.6772915319,
        7.61173201163, 7.67913499878, 7.6727609212, 7.66275451925,
        7.65199799315, 7.65149983741, 7.65554131408, 7.62213286298,
        7.53795983357, 7.53626130154, 7.54539963934])
    res_d002 = np.array([
        7.70178181209, 7.67445481224, 7.6715373765, 7.6772915319,
        7.61173201163, 7.67913499878, 7.67306697759, 7.65287924998,
        7.64904451605, 7.66580449603, 7.66252081172, 7.62213286298,
        7.53795983357, 7.53626130154, 7.54539963934])

    mod_002 = ARIMA(np.asarray(data_sample['loginv']), (0, 0, 2),
                    exog=np.asarray(data_sample[['loggdp', 'logcons']]))

    # doesn't converge with default starting values
    start_params = np.concatenate((res.params[[0, 1, 2, 4]], [0]))
    res_002 = mod_002.fit(start_params=start_params,
                          disp=0, solver='bfgs', maxiter=5000)

    # forecast
    fpredict_002 = res_002.predict(start=197, end=202,
                                   exog=exog_full.values[197:])
    forecast_002 = res_002.forecast(steps=len(exog_full.values[197:]),
                                    exog=exog_full.values[197:])
    forecast_002 = forecast_002[0]
    # TODO: we are not checking the other results

    assert_allclose(fpredict_002,
                    res_f002[-len(fpredict_002):],
                    rtol=1e-4, atol=1e-6)
    assert_allclose(forecast_002,
                    res_f002[-len(forecast_002):],
                    rtol=1e-4, atol=1e-6)

    # dynamic predict
    dpredict_002 = res_002.predict(start=193, end=202,
                                   exog=exog_full.values[197:],
                                   dynamic=True)
    assert_allclose(dpredict_002,
                    res_d002[-len(dpredict_002):],
                    rtol=1e-4, atol=1e-6)

    # GH#4497
    # in-sample dynamic predict should not need exog, #2982
    predict_3a = res_002.predict(start=100, end=120, dynamic=True)
    predict_3b = res_002.predict(start=100, end=120,
                                 exog=exog_full.values[100:120], dynamic=True)
    assert_allclose(predict_3a, predict_3b, rtol=1e-10)

    # TODO: break this out into a specific non-smoke test
    # GH#4915 invalid exogs passed to forecast should raise
    h = len(exog_full.values[197:])
    with pytest.raises(ValueError):
        res_002.forecast(steps=h)
    with pytest.raises(ValueError):
        res_002.forecast(steps=h, exog=np.empty((h, 20)))
    with pytest.raises(ValueError):
        res_002.forecast(steps=h, exog=np.empty(20))


@pytest.mark.not_vetted
def test_arima_fit_multiple_calls():
    y = [-1214.360173, -1848.209905, -2100.918158, -3647.483678, -4711.186773]
    mod = ARIMA(y, (1, 0, 2))
    # Make multiple calls to fit
    with warnings.catch_warnings(record=True):
        mod.fit(disp=0, start_params=[np.mean(y), .1, .1, .1])
    assert mod.exog_names == ['const', 'ar.L1.y', 'ma.L1.y', 'ma.L2.y']

    with warnings.catch_warnings(record=True):
        mod.fit(disp=0, start_params=[np.mean(y), .1, .1, .1])
    assert mod.exog_names == ['const', 'ar.L1.y', 'ma.L1.y', 'ma.L2.y']

    # test multiple calls when there is only a constant term
    mod = ARIMA(y, (0, 0, 0))
    # Make multiple calls to fit
    with warnings.catch_warnings(record=True):
        mod.fit(disp=0, start_params=[np.mean(y)])
    assert mod.exog_names == ['const']

    with warnings.catch_warnings(record=True):
        mod.fit(disp=0, start_params=[np.mean(y)])
    assert mod.exog_names == ['const']


@pytest.mark.not_vetted
def test_long_ar_start_params():
    arparams = np.array([1, -.75, .25])
    maparams = np.array([1, .65, .35])

    nobs = 30

    np.random.seed(12345)
    y = arma_generate_sample(arparams, maparams, nobs)

    model = ARMA(y, order=(2, 2))

    model.fit(method='css', start_ar_lags=10, disp=0)
    model.fit(method='css-mle', start_ar_lags=10, disp=0)
    model.fit(method='mle', start_ar_lags=10, disp=0)
    with pytest.raises(ValueError):
        model.fit(start_ar_lags=nobs + 5, disp=0)


# ----------------------------------------------------------------
# `predict` tests

@pytest.mark.not_vetted
def test_arima_predict_mle():
    cpi = datasets.macrodata.load_pandas().data['cpi'].values
    res1 = ARIMA(cpi, (4, 1, 1)).fit(disp=-1)
    # fit the model so that we get correct endog length but use
    path = os.path.join(current_path, 'results',
                        'results_arima_forecasts_all_mle.csv')
    arima_forecasts = pd.read_csv(path).values
    fc = arima_forecasts[:, 0]
    fcdyn = arima_forecasts[:, 1]
    fcdyn2 = arima_forecasts[:, 2]
    fcdyn3 = arima_forecasts[:, 3]
    fcdyn4 = arima_forecasts[:, 4]

    # 0 indicates the first sample-observation below
    # ie., the index after the pre-sample, these are also differenced once
    # so the indices are moved back once from the cpi in levels
    # start < p, end <p 1959q2 - 1959q4
    start, end = 1, 3
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start < p, end 0 1959q3 - 1960q1
    start, end = 2, 4
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start < p, end >0 1959q3 - 1971q4
    start, end = 2, 51
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start < p, end nobs 1959q3 - 2009q3
    start, end = 2, 202
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start < p, end >nobs 1959q3 - 2015q4
    start, end = 2, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start 0, end >0 1960q1 - 1971q4
    start, end = 4, 51
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 4, 202
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 4, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    # NOTE: raises
    #start, end = 202, 202
    #fv = res1.predict(start, end, typ='levels')
    #assert_almost_equal(fv, [])
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_3)
    # start >nobs, end >nobs 2009q4 - 2015q4
    # NOTE: this raises but shouldn't, dynamic forecasts could start
    # one period out
    start, end = 203, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[1:203], DECIMAL_4)

    # Dynamic

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 0
        # start < p, end <p 1959q2 - 1959q4
        start, end = 1, 3
        fv = res1.predict(start, end, dynamic=True, typ='levels')
        #assert_almost_equal(fv, arima_forecasts[:, 15])

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end 0 1959q3 - 1960q1
        start, end = 2, 4
        fv = res1.predict(start, end, dynamic=True, typ='levels')
        #assert_almost_equal(fv, fcdyn[5:end + 1], DECIMAL_4)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >0 1959q3 - 1971q4
        start, end = 2, 51
        fv = res1.predict(start, end, dynamic=True, typ='levels')
        #assert_almost_equal(fv, fcdyn[5:end + 1], DECIMAL_4)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end nobs 1959q3 - 2009q3
        start, end = 2, 202
        fv = res1.predict(start, end, dynamic=True, typ='levels')
        #assert_almost_equal(fv, fcdyn[5:end + 1], DECIMAL_4)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >nobs 1959q3 - 2015q4
        start, end = 2, 227
        fv = res1.predict(start, end, dynamic=True, typ='levels')
        #assert_almost_equal(fv, fcdyn[5:end + 1], DECIMAL_4)

    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 5, 227
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn3[start:end + 1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn3[start:end + 1], DECIMAL_4)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn4[start:end + 1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[5:203], DECIMAL_4)


@pytest.mark.not_vetted
def test_arima_predict_css():
    cpi = datasets.macrodata.load_pandas().data['cpi'].values
    # NOTE: Doing no-constant for now to kick the conditional exogenous
    # GH#274 down the road
    # go ahead and git the model to set up necessary variables
    res1 = ARIMA(cpi, (4, 1, 1)).fit(disp=-1, method="css", trend="nc")
    # but use gretl parameters to predict to avoid precision problems
    params = np.array([1.231272508473910,
                       -0.282516097759915,
                       0.170052755782440,
                       -0.118203728504945,
                       -0.938783134717947])

    path = os.path.join(current_path, 'results',
                        'results_arima_forecasts_all_css.csv')
    arima_forecasts = pd.read_csv(path).values
    fc = arima_forecasts[:, 0]
    fcdyn = arima_forecasts[:, 1]
    fcdyn2 = arima_forecasts[:, 2]
    fcdyn3 = arima_forecasts[:, 3]
    fcdyn4 = arima_forecasts[:, 4]

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 0
        start, end = 1, 3
        fv = res1.model.predict(params, start, end)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end 0 1959q3 - 1960q1
        start, end = 2, 4
        fv = res1.model.predict(params, start, end)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >0 1959q3 - 1971q4
        start, end = 2, 51
        fv = res1.model.predict(params, start, end)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end nobs 1959q3 - 2009q3
        start, end = 2, 202
        fv = res1.model.predict(params, start, end)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >nobs 1959q3 - 2015q4
        start, end = 2, 227
        fv = res1.model.predict(params, start, end)

    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    # TODO: why detoriating precision?
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[5:203], DECIMAL_4)

    # Dynamic

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 0
        # start < p, end <p 1959q2 - 1959q4
        start, end = 1, 3
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end 0 1959q3 - 1960q1
        start, end = 2, 4
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >0 1959q3 - 1971q4
        start, end = 2, 51
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end nobs 1959q3 - 2009q3
        start, end = 2, 202
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >nobs 1959q3 - 2015q4
        start, end = 2, 227
        fv = res1.predict(start, end, dynamic=True)

    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 5, 227
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn3[start:end + 1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn4[start:end + 1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[5:203], DECIMAL_4)


@pytest.mark.not_vetted
def test_arima_predict_mle_diffs():
    cpi = datasets.macrodata.load_pandas().data['cpi'].values
    # NOTE: Doing no-constant for now to kick the conditional exogenous
    # GH#274 down the road
    # go ahead and git the model to set up necessary variables
    res1 = ARIMA(cpi, (4, 1, 1)).fit(disp=-1, trend="c")
    # but use gretl parameters to predict to avoid precision problems
    params = np.array([0.926875951549299,
                       -0.555862621524846,
                       0.320865492764400,
                       0.252253019082800,
                       0.113624958031799,
                       0.939144026934634])

    path = os.path.join(current_path, 'results',
                        'results_arima_forecasts_all_mle_diff.csv')
    arima_forecasts = pd.read_csv(path).values
    fc = arima_forecasts[:, 0]
    fcdyn = arima_forecasts[:, 1]
    fcdyn2 = arima_forecasts[:, 2]
    fcdyn3 = arima_forecasts[:, 3]
    fcdyn4 = arima_forecasts[:, 4]

    # NOTE: should raise
    # TODO: The above comment appears wrong.  See GH#4358
    start, end = 1, 3
    fv = res1.model.predict(params, start, end)
    # start < p, end 0 1959q3 - 1960q1
    start, end = 2, 4
    fv = res1.model.predict(params, start, end)
    # start < p, end >0 1959q3 - 1971q4
    start, end = 2, 51
    fv = res1.model.predict(params, start, end)
    # start < p, end nobs 1959q3 - 2009q3
    start, end = 2, 202
    fv = res1.model.predict(params, start, end)
    # start < p, end >nobs 1959q3 - 2015q4
    start, end = 2, 227
    fv = res1.model.predict(params, start, end)
    # -----------------------------------------
    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    # TODO: why detoriating precision?
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[1:203], DECIMAL_4)

    # Dynamic

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. got 0
        # start < p, end <p 1959q2 - 1959q4
        start, end = 1, 3
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. got 1
        # start < p, end 0 1959q3 - 1960q1
        start, end = 2, 4
        fv = res1.predict(start, end, dynamic=True)
    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >0 1959q3 - 1971q4
        start, end = 2, 51
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end nobs 1959q3 - 2009q3
        start, end = 2, 202
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >nobs 1959q3 - 2015q4
        start, end = 2, 227
        fv = res1.predict(start, end, dynamic=True)

    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 5, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn3[start:end + 1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn4[start:end + 1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[5:203], DECIMAL_4)


@pytest.mark.not_vetted
def test_arima_predict_css_diffs():
    cpi = datasets.macrodata.load_pandas().data['cpi'].values
    # NOTE: Doing no-constant for now to kick the conditional exogenous
    # issue GH#274 down the road
    # go ahead and git the model to set up necessary variables
    res1 = ARIMA(cpi, (4, 1, 1)).fit(disp=-1, method="css", trend="c")
    # but use gretl parameters to predict to avoid precision problems
    params = np.array([0.78349893861244,
                       -0.533444105973324,
                       0.321103691668809,
                       0.264012463189186,
                       0.107888256920655,
                       0.920132542916995])
    # we report mean, should we report constant?
    params[0] = params[0] / (1 - params[1:5].sum())

    path = os.path.join(current_path, 'results',
                        'results_arima_forecasts_all_css_diff.csv')
    arima_forecasts = pd.read_csv(path).values
    fc = arima_forecasts[:, 0]
    fcdyn = arima_forecasts[:, 1]
    fcdyn2 = arima_forecasts[:, 2]
    fcdyn3 = arima_forecasts[:, 3]
    fcdyn4 = arima_forecasts[:, 4]

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 0
        start, end = 1, 3
        fv = res1.model.predict(params, start, end)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end 0 1959q3 - 1960q1
        start, end = 2, 4
        fv = res1.model.predict(params, start, end)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >0 1959q3 - 1971q4
        start, end = 2, 51
        fv = res1.model.predict(params, start, end)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end nobs 1959q3 - 2009q3
        start, end = 2, 202
        fv = res1.model.predict(params, start, end)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >nobs 1959q3 - 2015q4
        start, end = 2, 227
        fv = res1.model.predict(params, start, end)

    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    # TODO: why detoriating precision?
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end + 1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[5:203], DECIMAL_4)

    # Dynamic

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 0
        # start < p, end <p 1959q2 - 1959q4
        start, end = 1, 3
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end 0 1959q3 - 1960q1
        start, end = 2, 4
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end >0 1959q3 - 1971q4
        start, end = 2, 51
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # Start must be >= k_ar for conditional MLE or dynamic forecast. Got 1
        # start < p, end nobs 1959q3 - 2009q3
        start, end = 2, 202
        fv = res1.predict(start, end, dynamic=True)

    with pytest.raises(ValueError):
        # start < p, end >nobs 1959q3 - 2015q4
        start, end = 2, 227
        fv = res1.predict(start, end, dynamic=True)

    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 5, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end + 1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end + 1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn3[start:end + 1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn4[start:end + 1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[5:203], DECIMAL_4)


def _check_start(model, given, expected, dynamic):
    start, _, _, _ = model._get_prediction_index(given, None, dynamic)
    assert start == expected


def _check_end(model, given, end_expect, out_of_sample_expect):
    _, end, out_of_sample, _ = model._get_prediction_index(None, given, False)
    assert end == end_expect
    assert out_of_sample == out_of_sample_expect


@pytest.mark.not_vetted
def test_arma_predict_indices():
    sunspots = datasets.sunspots.load_pandas().data['SUNACTIVITY'].values
    model = ARMA(sunspots, (9, 0), dates=sun_dates, freq='A')
    model.method = 'mle'

    # raises - pre-sample + dynamic
    with pytest.raises(ValueError):
        model._get_prediction_index(0, None, True)
    with pytest.raises(ValueError):
        model._get_prediction_index(8, None, True)
    with pytest.raises(ValueError):
        model._get_prediction_index('1700', None, True)
    with pytest.raises(ValueError):
        model._get_prediction_index('1708', None, True)

    # works - in-sample
    # None
    # given, expected, dynamic
    start_test_cases = [(None, 9, True),
                        # all start get moved back by k_diff
                        (9, 9, True),
                        (10, 10, True),
                        # what about end of sample start - last value is first
                        # forecast
                        (309, 309, True),
                        (308, 308, True),
                        (0, 0, False),
                        (1, 1, False),
                        (4, 4, False),

                        # all start get moved back by k_diff
                        ('1709', 9, True),
                        ('1710', 10, True),
                        # what about end of sample start - last value is first
                        # forecast
                        ('2008', 308, True),
                        ('2009', 309, True),
                        ('1700', 0, False),
                        ('1708', 8, False),
                        ('1709', 9, False)]

    for case in start_test_cases:
        _check_start(model, *case)

    # the length of sunspot is 309, so last index is 208
    end_test_cases = [(None, 308, 0),
                      (307, 307, 0),
                      (308, 308, 0),
                      (309, 308, 1),
                      (312, 308, 4),
                      (51, 51, 0),
                      (333, 308, 25),

                      ('2007', 307, 0),
                      ('2008', 308, 0),
                      ('2009', 308, 1),
                      ('2012', 308, 4),
                      ('1815', 115, 0),
                      ('2033', 308, 25)]

    for case in end_test_cases:
        _check_end(model, *case)


@pytest.mark.not_vetted
def test_arima_predict_indices():
    cpi = datasets.macrodata.load_pandas().data['cpi'].values
    model = ARIMA(cpi, (4, 1, 1), dates=cpi_dates, freq='Q')
    model.method = 'mle'

    # starting indices

    # raises - pre-sample + dynamic
    with pytest.raises(ValueError):
        model._get_prediction_index(0, None, True)
    with pytest.raises(ValueError):
        model._get_prediction_index(4, None, True)
    with pytest.raises(KeyError):
        model._get_prediction_index('1959Q1', None, True)
    with pytest.raises(ValueError):
        model._get_prediction_index('1960Q1', None, True)

    # raises - index differenced away
    with pytest.raises(ValueError):
        model._get_prediction_index(0, None, False)
    with pytest.raises(KeyError):
        model._get_prediction_index('1959Q1', None, False)

    # works - in-sample
    # None
    # given, expected, dynamic
    start_test_cases = [(None, 4, True),
                        # all start get moved back by k_diff
                        (5, 4, True),
                        (6, 5, True),
                        # what about end of sample start - last value is first
                        # forecast
                        (203, 202, True),
                        (1, 0, False),
                        (4, 3, False),
                        (5, 4, False),
                        # all start get moved back by k_diff
                        ('1960Q2', 4, True),
                        ('1960Q3', 5, True),
                        # what about end of sample start - last value is first
                        # forecast
                        ('2009Q4', 202, True),
                        ('1959Q2', 0, False),
                        ('1960Q1', 3, False),
                        ('1960Q2', 4, False)]

    for case in start_test_cases:
        _check_start(model, *case)

    # check raises
    # TODO: make sure dates are passing through unmolested
    #assert_raises(ValueError, model._get_predict_end, ("2001-1-1",))

    # the length of diff(cpi) is 202, so last index is 201
    end_test_cases = [(None, 201, 0),
                      (201, 200, 0),
                      (202, 201, 0),
                      (203, 201, 1),
                      (204, 201, 2),
                      (51, 50, 0),
                      (164 + 63, 201, 25),

                      ('2009Q2', 200, 0),
                      ('2009Q3', 201, 0),
                      ('2009Q4', 201, 1),
                      ('2010Q1', 201, 2),
                      ('1971Q4', 50, 0),
                      ('2015Q4', 201, 25)]

    for case in end_test_cases:
        _check_end(model, *case)

    # check higher k_diff
    # model.k_diff = 2
    model = ARIMA(cpi, (4, 2, 1), dates=cpi_dates, freq='Q')
    model.method = 'mle'

    # raises - pre-sample + dynamic
    with pytest.raises(ValueError):
        model._get_prediction_index(0, None, True)
    with pytest.raises(ValueError):
        model._get_prediction_index(5, None, True)
    with pytest.raises(KeyError):
        model._get_prediction_index('1959Q1', None, True)
    with pytest.raises(ValueError):
        model._get_prediction_index('1960Q1', None, True)

    # raises - index differenced away
    with pytest.raises(ValueError):
        model._get_prediction_index(1, None, False)
    with pytest.raises(KeyError):
        model._get_prediction_index('1959Q2', None, False)

    start_test_cases = [(None, 4, True),
                        # all start get moved back by k_diff
                        (6, 4, True),
                        # what about end of sample start - last value is first
                        # forecast
                        (203, 201, True),
                        (2, 0, False),
                        (4, 2, False),
                        (5, 3, False),
                        ('1960Q3', 4, True),
                        # what about end of sample start - last value is first
                        # forecast
                        ('2009Q4', 201, True),
                        ('2009Q4', 201, True),
                        ('1959Q3', 0, False),
                        ('1960Q1', 2, False),
                        ('1960Q2', 3, False)]

    for case in start_test_cases:
        _check_start(model, *case)

    end_test_cases = [(None, 200, 0),
                      (201, 199, 0),
                      (202, 200, 0),
                      (203, 200, 1),
                      (204, 200, 2),
                      (51, 49, 0),
                      (164 + 63, 200, 25),

                      ('2009Q2', 199, 0),
                      ('2009Q3', 200, 0),
                      ('2009Q4', 200, 1),
                      ('2010Q1', 200, 2),
                      ('1971Q4', 49, 0),
                      ('2015Q4', 200, 25)]

    for case in end_test_cases:
        _check_end(model, *case)


def test_arima_predict_indices_css_invalid():
    # TODO: GH Reference?
    cpi = datasets.macrodata.load_pandas().data['cpi'].values
    # NOTE: Doing no-constant for now to kick the conditional exogenous
    # GH#274 down the road
    model = ARIMA(cpi, (4, 1, 1))
    model.method = 'css'

    with pytest.raises(ValueError):
        model._get_prediction_index(0, None, False)
    with pytest.raises(ValueError):
        model._get_prediction_index(0, None, True)
    with pytest.raises(ValueError):
        model._get_prediction_index(2, None, False)
    with pytest.raises(ValueError):
        model._get_prediction_index(2, None, True)


# ----------------------------------------------------------------
# Smoke Tests Aimed at a specific issue/method

@pytest.mark.smoke
@pytest.mark.matplotlib
def test_plot_predict(close_figures):
    dta = datasets.sunspots.load_pandas().data[['SUNACTIVITY']]
    dta.index = pd.DatetimeIndex(start='1700', end='2009', freq='A')[:309]

    # TODO: parametrize?
    res = ARMA(dta, (3, 0)).fit(disp=-1)
    for dynamic in [True, False]:
        for plot_insample in [True, False]:
            res.plot_predict('1990', '2012', dynamic=dynamic,
                             plot_insample=plot_insample)

    res = ARIMA(dta, (3, 1, 0)).fit(disp=-1)
    for dynamic in [True, False]:
        for plot_insample in [True, False]:
            res.plot_predict('1990', '2012', dynamic=dynamic,
                             plot_insample=plot_insample)


def test_arima_forecast_exog_incorrect_size():
    # GH#4915; upstream bashtage rolled this in to test_arima_exog_predict_1d
    np.random.seed(12345)
    y = np.random.random(100)
    x = np.random.random(100)
    mod = ARMA(y, (2, 1), x).fit(disp=-1)
    newx = np.random.random(10)

    with pytest.raises(ValueError):
        mod.forecast(steps=10, alpha=0.05, exog=newx[:5])
    with pytest.raises(ValueError):
        mod.forecast(steps=10, alpha=0.05)

    too_many = pd.DataFrame(np.zeros((10, 2)),
                            columns=['x1', 'x2'])
    with pytest.raises(ValueError):
        mod.forecast(steps=10, alpha=0.05, exog=too_many)


def test_arma_forecast_exog_incorrect_size():
    # GH#4915; upstream bashtage rolled this in
    #   to test_arima111_predict_exog2127

    # TODO: de-duplicate data input with test_arima111_predict_exog2127
    ef = [0.03005, 0.03917, 0.02828, 0.03644, 0.03379, 0.02744,
          0.03343, 0.02621, 0.0305, 0.02455, 0.03261, 0.03507,
          0.02734, 0.05373, 0.02677, 0.03443, 0.03331, 0.02741,
          0.03709, 0.02113, 0.03343, 0.02011, 0.03675, 0.03077,
          0.02201, 0.04844, 0.05518, 0.03765, 0.05433, 0.03049,
          0.04829, 0.02936, 0.04421, 0.02457, 0.04007, 0.03009,
          0.04504, 0.05041, 0.03651, 0.02719, 0.04383, 0.02887,
          0.0344, 0.03348, 0.02364, 0.03496, 0.02549, 0.03284,
          0.03523, 0.02579, 0.0308, 0.01784, 0.03237, 0.02078,
          0.03508, 0.03062, 0.02006, 0.02341, 0.02223, 0.03145,
          0.03081, 0.0252, 0.02683, 0.0172, 0.02225, 0.01579,
          0.02237, 0.02295, 0.0183, 0.02356, 0.02051, 0.02932,
          0.03025, 0.0239, 0.02635, 0.01863, 0.02994, 0.01762,
          0.02837, 0.02421, 0.01951, 0.02149, 0.02079, 0.02528,
          0.02575, 0.01634, 0.02563, 0.01719, 0.02915, 0.01724,
          0.02804, 0.0275, 0.02099, 0.02522, 0.02422, 0.03254,
          0.02095, 0.03241, 0.01867, 0.03998, 0.02212, 0.03034,
          0.03419, 0.01866, 0.02623, 0.02052]
    ue = [4.9, 5., 5., 5., 4.9, 4.7, 4.8, 4.7, 4.7,
          4.6, 4.6, 4.7, 4.7, 4.5, 4.4, 4.5, 4.4, 4.6,
          4.5, 4.4, 4.5, 4.4, 4.6, 4.7, 4.6, 4.7, 4.7,
          4.7, 5., 5., 4.9, 5.1, 5., 5.4, 5.6, 5.8,
          6.1, 6.1, 6.5, 6.8, 7.3, 7.8, 8.3, 8.7, 9.,
          9.4, 9.5, 9.5, 9.6, 9.8, 10., 9.9, 9.9, 9.7,
          9.8, 9.9, 9.9, 9.6, 9.4, 9.5, 9.5, 9.5, 9.5,
          9.8, 9.4, 9.1, 9., 9., 9.1, 9., 9.1, 9.,
          9., 9., 8.8, 8.6, 8.5, 8.2, 8.3, 8.2, 8.2,
          8.2, 8.2, 8.2, 8.1, 7.8, 7.8, 7.8, 7.9, 7.9,
          7.7, 7.5, 7.5, 7.5, 7.5, 7.3, 7.2, 7.2, 7.2,
          7., 6.7, 6.6, 6.7, 6.7, 6.3, 6.3]

    model = ARIMA(np.array(ef), (1, 1, 1), exog=np.array(ue))
    res = model.fit(transparams=False, pgtol=1e-8, iprint=0, disp=0)

    # Smoke check of forecast with exog in ARIMA
    res.forecast(steps=10, exog=np.empty(10))

    with pytest.raises(ValueError):
        res.forecast(steps=10)
    with pytest.raises(ValueError):
        res.forecast(steps=10, exog=np.empty((10, 2)))
    with pytest.raises(ValueError):
        res.forecast(steps=10, exog=np.empty(100))


@pytest.mark.smoke
def test_arima_exog_predict_1d():
    # GH#1067
    np.random.seed(12345)
    y = np.random.random(100)
    x = np.random.random(100)
    mod = ARMA(y, (2, 1), x).fit(disp=-1)
    newx = np.random.random(10)
    mod.forecast(steps=10, alpha=0.05, exog=newx)


@pytest.mark.smoke
def test_arima_predict_noma():
    # GH#657
    ar = [1, .75]
    ma = [1]
    np.random.seed(12345)
    data = arma_generate_sample(ar, ma, 100)
    arma = ARMA(data, order=(0, 1))
    arma_res = arma.fit(disp=-1)
    arma_res.forecast(1)


@pytest.mark.smoke
def test_arima_dataframe_integer_name():
    # GH#1038
    vals = [96.2, 98.3, 99.1, 95.5, 94.0, 87.1, 87.9, 86.7402777504474,
            94.0, 96.5, 93.3, 97.5, 96.3, 92.]

    dr = pd.date_range("1990", periods=len(vals), freq='Q')
    ts = pd.Series(vals, index=dr)
    df = pd.DataFrame(ts)
    mod = ARIMA(df, (2, 0, 2))  # TODO: maybe _fit_ this model?


@pytest.mark.smoke
def test_arima_no_diff():
    # GH#736
    # smoke test, predict will break if we have ARIMAResults but
    # ARMA model, need ARIMA(p, 0, q) to return an ARMA in init.
    ar = [1, -.75, .15, .35]
    ma = [1, .25, .9]
    np.random.seed(12345)
    y = arma_generate_sample(ar, ma, 100)
    mod = ARIMA(y, (3, 0, 2))
    assert type(mod) is ARMA
    res = mod.fit(disp=-1)
    # smoke test just to be sure
    res.predict()


@pytest.mark.smoke
@pytest.mark.parametrize('order', [(1, 0, 2), (0, 0, 0)])
def test_arima_summary(order):
    # TODO: GH Reference?
    # same setup as test_arima_fit_multiple_calls
    y = [-1214.360173, -1848.209905, -2100.918158, -3647.483678, -4711.186773]
    model = ARIMA(y, order)
    start_params = [np.mean(y)] + [.1] * sum(order)
    res = model.fit(disp=False, start_params=start_params)

    # ensure summary() works (here defined as "doesnt raise")
    res.summary()


# ----------------------------------------------------------------
# Issue-specific tests -- Need to find GH references


@pytest.mark.skip(reason="fa not ported from upstream")
def test_arma_pickle():
    fa = None   # dummy to avoid flake8 complaints until ported

    np.random.seed(9876565)
    x = fa.ArmaFft([1, -0.5], [1., 0.4], 40).generate_sample(nsample=200,
                                                             burnin=1000)
    mod = ARMA(x, (1, 1))
    pkl_mod = cPickle.loads(cPickle.dumps(mod))

    res = mod.fit(trend="c", disp=-1, solver='newton')
    pkl_res = pkl_mod.fit(trend="c", disp=-1, solver='newton')

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.llf, pkl_res.llf)
    assert_almost_equal(res.resid, pkl_res.resid)
    assert_almost_equal(res.fittedvalues, pkl_res.fittedvalues)
    assert_almost_equal(res.pvalues, pkl_res.pvalues)
    # TODO: check __eq__ once that is implemented


def test_arima_pickle():
    # pickle+unpickle should give back an identical model
    # TODO: GH reference?
    endog = y_arma[:, 6]
    mod = ARIMA(endog, (1, 1, 1))
    pkl_mod = cPickle.loads(cPickle.dumps(mod))

    res = mod.fit(trend="c", disp=-1, solver='newton')
    pkl_res = pkl_mod.fit(trend="c", disp=-1, solver='newton')

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.llf, pkl_res.llf)
    assert_almost_equal(res.resid, pkl_res.resid)
    assert_almost_equal(res.fittedvalues, pkl_res.fittedvalues)
    assert_almost_equal(res.pvalues, pkl_res.pvalues)


def test_arima_not_implemented():
    # from_formula is not implemented
    # TODO: GH reference?
    formula = ' WUE ~ 1 + SFO3 '
    data = [-1214.360173, -1848.209905, -2100.918158]
    with pytest.raises(NotImplementedError):
        ARIMA.from_formula(formula, data)


def test_reset_trend():
    # re-calling `fit` with different trend should not change the trend
    # in the original result
    # TODO: GH Reference?
    endog = y_arma[:, 0]
    mod = ARMA(endog, order=(1, 1))
    res1 = mod.fit(trend="c", disp=-1)
    res2 = mod.fit(trend="nc", disp=-1)
    assert len(res1.params) == len(res2.params) + 1


# ----------------------------------------------------------------
# Issue-specific regression tests

def test_arima_too_few_observations_raises():
    # GH#1038, too few observations with given order should raise
    vals = [96.2, 98.3, 99.1, 95.5, 94.0, 87.1, 87.9, 86.7402777504474]

    dr = pd.date_range("1990", periods=len(vals), freq='Q')
    ts = pd.Series(vals, index=dr)
    df = pd.DataFrame(ts)
    mod = ARIMA(df, (2, 0, 2))
    with pytest.raises(ValueError):
        mod.fit()


def test_arma_missing():
    # GH#1343
    y = np.random.random(40)
    y[-1] = np.nan
    with pytest.raises(MissingDataError):
        ARMA(y, (1, 0), missing='raise')


def test_summary_roots_html():
    # regression test for html of roots table GH#4434
    # upstream this is a method in Test_Y_ARMA22_Const
    # we ignore whitespace in the assert
    res1 = ARMA(y_arma[:, 9], order=(2, 2)).fit(trend="c", disp=-1)
    summ = res1.summary()
    summ_roots = """\
    <tableclass="simpletable">
    <caption>Roots</caption>
    <tr>
    <td></td><th>Real</th><th>Imaginary</th><th>Modulus</th><th>Frequency</th>
    </tr>
    <tr>
    <th>AR.1</th><td>1.0991</td><td>-1.2571j</td><td>1.6698</td><td>-0.1357</td>
    </tr>
    <tr>
    <th>AR.2</th><td>1.0991</td><td>+1.2571j</td><td>1.6698</td><td>0.1357</td>
    </tr>
    <tr>
    <th>MA.1</th><td>-1.1702</td><td>+0.0000j</td><td>1.1702</td><td>0.5000</td>
    </tr>
    <tr>
    <th>MA.2</th><td>1.2215</td><td>+0.0000j</td><td>1.2215</td><td>0.0000</td>
    </tr>
    </table>"""
    table = summ.tables[2]
    assert table._repr_html_().replace(' ', '') == summ_roots.replace(' ', '')


def test_endog_int():
    # int endog should produce same result as float, GH#3504, GH#4512

    np.random.seed(123987)
    y = np.random.randint(0, 15, size=100)
    yf = y.astype(np.float64)

    res = AR(y).fit(5)
    resf = AR(yf).fit(5)
    assert_allclose(res.params, resf.params, atol=1e-6)
    assert_allclose(res.bse, resf.bse, atol=1e-6)

    res = ARMA(y, order=(2, 1)).fit(disp=0)
    resf = ARMA(yf, order=(2, 1)).fit(disp=0)
    assert_allclose(res.params, resf.params, atol=1e-6)
    assert_allclose(res.bse, resf.bse, atol=1e-6)

    res = ARIMA(y.cumsum(), order=(1, 1, 1)).fit(disp=0)
    resf = ARIMA(yf.cumsum(), order=(1, 1, 1)).fit(disp=0)
    assert_allclose(res.params, resf.params,
                    rtol=1e-6, atol=1e-5)
    assert_allclose(res.bse, resf.bse,
                    rtol=1e-6, atol=1e-5)
