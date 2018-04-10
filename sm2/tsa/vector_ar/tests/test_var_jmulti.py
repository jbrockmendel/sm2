from __future__ import absolute_import, print_function
import warnings

from six.moves import range

import numpy as np
from numpy.testing import assert_allclose
import pytest

import sm2.datasets.macrodata.data as macro
from sm2.tsa.vector_ar.var_model import VAR

from .JMulTi_results.parse_jmulti_vecm_output import sublists
from .JMulTi_results.parse_jmulti_var_output import load_results_jmulti

atol = 0.001  # absolute tolerance
rtol = 0.01   # relative tolerance
datasets = []
datasets.append(macro)  # TODO: append more data sets for more test cases.

dont_test_se_t_p = False
deterministic_terms_list = ["nc", "c", "ct"]
seasonal_list = [0, 4]
dt_s_list = [(trend, seasonal) for trend in deterministic_terms_list
             for seasonal in seasonal_list]
all_tests = ["coefs", "det", "Sigma_u", "log_like", "fc", "causality",
             "impulse-response", "lag order", "test normality", "whiteness",
             "exceptions"]
to_test = all_tests  # ["coefs","det","Sigma_u","log_like","fc","causality"]


def reorder_jmultis_det_terms(jmulti_output, constant, seasons):
    """
    In case of seasonal terms and a trend term we have to reorder them to make
    the outputs from JMulTi and sm2 comparable.
    JMulTi's ordering is: [constant], [seasonal terms], [trend term] while
    in sm2 it is: [constant], [trend term], [seasonal terms]

    Parameters
    ----------
    jmulti_output : ndarray (neqs x number_of_deterministic_terms)

    constant : bool
        Indicates whether there is a constant term or not in jmulti_output.
    seasons : int
        Number of seasons in the model. That means there are seasons-1
        columns for seasonal terms in jmulti_output

    Returns
    -------
    reordered : ndarray (neqs x number_of_deterministic_terms)
        jmulti_output reordered such that the order of deterministic terms
        matches that of sm2.
    """
    if seasons == 0:
        return jmulti_output
    constant = int(constant)
    const_column = jmulti_output[:, :constant]
    season_columns = jmulti_output[:, constant:constant + seasons - 1].copy()
    trend_columns = jmulti_output[:, constant + seasons - 1:].copy()
    return np.hstack((const_column,
                      trend_columns,
                      season_columns))


def generate_exog_from_season(seasons, endog_len):
    """
    Translate seasons to exog matrix.

    Parameters
    ----------
    seasons : int
        Number of seasons.
    endog_len : int
        Number of observations.

    Returns
    -------
    exog : ndarray or None
        If seasonal deterministic terms exist, the corresponding exog-matrix is
        returned.
        Otherwise, None is returned.
    """
    exog_stack = []
    if seasons > 0:
        season_exog = np.zeros((seasons - 1, endog_len))
        for i in range(seasons - 1):
            season_exog[i, i::seasons] = 1
        # season_exog = season_exog[:, ::-1]
        # season_exog = np.hstack((season_exog[:, 3:4],
        #   season_exog[:, :-1]))
        # season_exog = np.hstack((season_exog[:, 2:4],
        #                          season_exog[:, :-2]))
        # season_exog = np.hstack((season_exog[:, 1:4], season_exog[:, :-3]))
        # season_exog[1] = -season_exog[1]
        # the following line is commented out because seasonal terms are
        # *not* centered in JMulTi's VAR-framework (in contrast to VECM)
        # season_exog -= 1 / seasons
        season_exog = season_exog.T
        exog_stack.append(season_exog)
    if exog_stack != []:
        exog = np.column_stack(exog_stack)
    else:
        exog = None
    return exog


@pytest.fixture(params=[(dataset, trend, seasonal)
                        for dataset in datasets
                        for trend in deterministic_terms_list
                        for seasonal in seasonal_list])
def case(request):
    dataset, trend, seasonal = request.param

    dtset = dataset.load_pandas()
    variables = dataset.variable_names
    loaded = dtset.data[variables].astype(float).values
    endog = loaded.reshape((-1, len(variables)))
    exog = generate_exog_from_season(seasonal, len(endog))

    with warnings.catch_warnings():
        # `rcond` parameter will change to the default of machine
        # precision times ``max(M, N)`` where M and N are the input
        # matrix dimensions.v
        warnings.simplefilter("ignore")
        model = VAR(endog, exog)
        res = model.fit(maxlags=4,
                        trend=trend,
                        method="ols")

    expected = load_results_jmulti(dataset, dt_s_list)[(trend, seasonal)]
    return res, expected, trend, seasonal, dataset, endog, exog


@pytest.mark.not_vetted
def test_lag_order_selection(case):
    res, expected, trend = case[:3]
    endog_tot, exog = case[-2:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VAR(endog_tot, exog)
        obtained_all = model.select_order(10, trend=trend)

    for ic in ["aic", "fpe", "hqic", "bic"]:
        obtained = getattr(obtained_all, ic)
        desired = expected["lagorder"][ic]
        assert_allclose(obtained, desired,
                        rtol, atol, False)


@pytest.mark.not_vetted
def test_causality(case):
    # test Granger- and instantaneous causality
    res, expected = case[:2]
    dataset = case[4]

    v_ind = range(len(dataset.variable_names))
    for causing_ind in sublists(v_ind, 1, len(v_ind) - 1):
        causing_names = ["y" + str(i + 1) for i in causing_ind]
        causing_key = tuple(dataset.variable_names[i] for i in causing_ind)

        caused_ind = [i for i in v_ind if i not in causing_ind]
        caused_names = ["y" + str(i + 1) for i in caused_ind]
        caused_key = tuple(dataset.variable_names[i] for i in caused_ind)

        key = (causing_key, caused_key)

        # test Granger-causality
        granger_sm_ind = res.test_causality(caused_ind, causing_ind)
        granger_sm_str = res.test_causality(caused_names, causing_names)

        # test test-statistic for Granger non-causality:
        g_t_obt = granger_sm_ind.test_statistic
        g_t_des = expected["granger_caus"]["test_stat"][key]
        assert_allclose(g_t_obt, g_t_des,
                        rtol, atol, False)
        # check whether string sequences as args work in the same way:
        g_t_obt_str = granger_sm_str.test_statistic
        assert_allclose(g_t_obt_str, g_t_obt,
                        1e-07, 0, False)
        # check if int (e.g. 0) as index and list of int ([0]) yield
        # the same result:
        if len(causing_ind) == 1 or len(caused_ind) == 1:
            ci = causing_ind[0] if len(causing_ind) == 1 else causing_ind
            ce = caused_ind[0] if len(caused_ind) == 1 else caused_ind
            granger_sm_single_ind = res.test_causality(ce, ci)
            g_t_obt_single = granger_sm_single_ind.test_statistic
            assert_allclose(g_t_obt_single, g_t_obt,
                            1e-07, 0, False)

        # test p-value for Granger non-causality:
        g_p_obt = granger_sm_ind.pvalue
        g_p_des = expected["granger_caus"]["p"][key]
        assert_allclose(g_p_obt, g_p_des,
                        rtol, atol, False)
        # check whether string sequences as args work in the same way:
        g_p_obt_str = granger_sm_str.pvalue
        assert_allclose(g_p_obt_str, g_p_obt,
                        1e-07, 0, False)
        # check if int (e.g. 0) as index and list of int ([0]) yield
        # the same result:
        if len(causing_ind) == 1:
            g_p_obt_single = granger_sm_single_ind.pvalue
            assert_allclose(g_p_obt_single, g_p_obt,
                            1e-07, 0, False)

        # test instantaneous causality
        inst_sm_ind = res.test_inst_causality(causing_ind)
        inst_sm_str = res.test_inst_causality(causing_names)
        # test test-statistic for instantaneous non-causality
        t_obt = inst_sm_ind.test_statistic
        t_des = expected["inst_caus"]["test_stat"][key]
        assert_allclose(t_obt, t_des,
                        rtol, atol, False)
        # check whether string sequences as args work in the same way:
        t_obt_str = inst_sm_str.test_statistic
        assert_allclose(t_obt_str, t_obt,
                        1e-07, 0, False)
        # check if int (e.g. 0) as index and list of int ([0]) yield
        # the same result:
        if len(causing_ind) == 1:
            inst_sm_single_ind = res.test_inst_causality(causing_ind[0])
            t_obt_single = inst_sm_single_ind.test_statistic
            assert_allclose(t_obt_single, t_obt,
                            1e-07, 0, False)

        # test p-value for instantaneous non-causality
        p_obt = res.test_inst_causality(causing_ind).pvalue
        p_des = expected["inst_caus"]["p"][key]
        assert_allclose(p_obt, p_des,
                        rtol, atol, False)
        # check whether string sequences as args work in the same way:
        p_obt_str = inst_sm_str.pvalue
        assert_allclose(p_obt_str, p_obt,
                        1e-07, 0, False)
        # check if int (e.g. 0) as index and list of int ([0]) yield
        # the same result:
        if len(causing_ind) == 1:
            inst_sm_single_ind = res.test_inst_causality(causing_ind[0])
            p_obt_single = inst_sm_single_ind.pvalue
            assert_allclose(p_obt_single, p_obt,
                            1e-07, 0, False)


@pytest.mark.not_vetted
def test_ols_coefs(case):
    res, expected = case[:2]

    # estimated parameter vector
    obtained = np.hstack(res.coefs)
    desired = expected["est"]["Lagged endogenous term"]
    assert_allclose(obtained, desired,
                    rtol, atol, False)
    # standard errors
    obt = res.stderr_endog_lagged
    des = expected["se"]["Lagged endogenous term"].T
    assert_allclose(obt, des,
                    rtol, atol, False)
    # t-values
    obt = res.tvalues_endog_lagged
    des = expected["t"]["Lagged endogenous term"].T
    assert_allclose(obt, des,
                    rtol, atol, False)
    # p-values
    obt = res.pvalues_endog_lagged
    des = expected["p"]["Lagged endogenous term"].T
    assert_allclose(obt, des,
                    rtol, atol, False)


@pytest.mark.not_vetted
def test_ols_sigma(case):
    res, expected = case[:2]

    obtained = res.sigma_u
    desired = expected["est"]["Sigma_u"]
    assert_allclose(obtained, desired,
                    rtol, atol, False)


@pytest.mark.not_vetted
def test_ols_det_terms(case):
    res, expected, trend, seasonal = case[:4]

    det_key_ref = "Deterministic term"
    # If there are no det. terms, just make sure we don't compute any:
    if det_key_ref not in expected["est"].keys():
        assert res.coefs_exog.size == 0
        assert res.stderr_dt.size == 0
        assert res.tvalues_dt.size == 0
        assert res.pvalues_dt.size == 0
        return
    obtained = res.coefs_exog
    desired = expected["est"][det_key_ref]
    desired = reorder_jmultis_det_terms(desired,
                                        trend.startswith("c"), seasonal)
    assert_allclose(obtained, desired,
                    rtol, atol, False)
    # standard errors
    obt = res.stderr_dt
    des = expected["se"][det_key_ref]
    des = reorder_jmultis_det_terms(des, trend.startswith("c"),
                                    seasonal).T
    assert_allclose(obt, des,
                    rtol, atol, False)
    # t-values
    obt = res.tvalues_dt
    des = expected["t"][det_key_ref]
    des = reorder_jmultis_det_terms(des, trend.startswith("c"),
                                    seasonal).T
    assert_allclose(obt, des,
                    rtol, atol, False)
    # p-values
    obt = res.pvalues_dt
    des = expected["p"][det_key_ref]
    des = reorder_jmultis_det_terms(des, trend.startswith("c"),
                                    seasonal).T
    assert_allclose(obt, des,
                    rtol, atol, False)


@pytest.mark.not_vetted
def test_log_like(case):
    res, expected = case[:2]

    obtained = res.llf
    desired = expected["log_like"]
    assert_allclose(obtained, desired,
                    rtol, atol, False)


@pytest.mark.not_vetted
def test_fc(case):
    res, expected, trend, seasonal = case[:4]

    steps = 5  # parsed JMulTi output comprises 5 steps
    last_observations = res.endog[-res.k_ar:]
    seasons = seasonal
    if seasons == 0:
        exog_future = None
    else:
        exog_future = np.zeros((steps, seasons - 1))
        # the following line is appropriate only if the last
        # observation was in the next to last season (this is the case
        # for macrodata)
        exog_future[1:seasons] = np.identity(seasons - 1)
    # test point forecast functionality of forecast method
    obtained = res.forecast(y=last_observations, steps=steps,
                            exog_future=exog_future)
    desired = expected["fc"]["fc"]
    assert_allclose(obtained, desired,
                    rtol, atol, False)

    # test forecast method with confidence interval calculation
    obtained = res.forecast_interval(y=last_observations, steps=steps,
                                     alpha=0.05, exog_future=exog_future)
    obt = obtained[0]  # forecast
    obt_l = obtained[1]  # lower bound
    obt_u = obtained[2]  # upper bound
    des = expected["fc"]["fc"]
    des_l = expected["fc"]["lower"]
    des_u = expected["fc"]["upper"]
    assert_allclose(obt, des, rtol, atol, False)
    assert_allclose(obt_l, des_l, rtol, atol, False)
    assert_allclose(obt_u, des_u, rtol, atol, False)


@pytest.mark.not_vetted
def test_impulse_response(case):
    res, expected = case[:2]

    periods = 20
    obtained_all = res.irf(periods=periods).irfs
    # flatten inner arrays to make them comparable to parsed results:
    obtained_all = obtained_all.reshape(periods + 1, -1)
    desired_all = expected["ir"]
    assert_allclose(obtained_all, desired_all,
                    rtol, atol, False)


@pytest.mark.not_vetted
def test_normality(case):
    res, expected = case[:2]

    obtained = res.test_normality(signif=0.05)
    obt_statistic = obtained.test_statistic
    des_statistic = expected["test_norm"][
        "joint_test_statistic"]
    assert_allclose(obt_statistic, des_statistic,
                    rtol, atol, False)
    obt_pvalue = obtained.pvalue
    des_pvalue = expected["test_norm"]["joint_pvalue"]
    assert_allclose(obt_pvalue, des_pvalue,
                    rtol, atol, False)
    # call methods to assure they don't raise exceptions
    obtained.summary()
    str(obtained)  # __str__()


@pytest.mark.not_vetted
def test_whiteness(case):
    res, expected = case[:2]

    lags = expected["whiteness"]["tested order"]

    with warnings.catch_warnings():
        # `rcond` parameter will change to the default of machine
        # precision times ``max(M, N)`` where M and N are the input
        # matrix dimensions.v
        warnings.simplefilter("ignore")
        obtained = res.test_whiteness(nlags=lags)

    # test statistic
    desired = expected["whiteness"]["test statistic"]
    assert_allclose(obtained.test_statistic, desired,
                    rtol, atol, False)
    # p-value
    desired = expected["whiteness"]["p-value"]
    assert_allclose(obtained.pvalue, desired,
                    rtol, atol, False)

    with warnings.catch_warnings():
        # `rcond` parameter will change to the default of machine
        # precision times ``max(M, N)`` where M and N are the input
        # matrix dimensions.v
        warnings.simplefilter("ignore")
        obtained = res.test_whiteness(nlags=lags, adjusted=True)

    # test statistic (adjusted Portmanteau test)
    desired = expected["whiteness"]["test statistic adj."]
    assert_allclose(obtained.test_statistic, desired,
                    rtol, atol, False)
    # p-value (adjusted Portmanteau test)
    desired = expected["whiteness"]["p-value adjusted"]
    assert_allclose(obtained.pvalue, desired,
                    rtol, atol, False)


@pytest.mark.not_vetted
def test_exceptions(case):
    res = case[0]
    # instant causality:
    # 0<signif<1
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.test_inst_causality(0, 0)
            # this means signif=0

    # causing must be int, str or iterable of int or str
    with pytest.raises(TypeError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.test_inst_causality([0.5])
            # 0.5 not an int
