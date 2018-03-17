# -*- coding: utf-8 -*-
"""Tests for sandwich robust covariance estimation

see also in regression for cov_hac compared to Gretl and
sandbox.panel test_random_panel for comparing cov_cluster, cov_hac_panel and
cov_white

Created on Sat Dec 17 08:39:16 2011

Author: Josef Perktold
"""
import os

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd

from sm2.regression.linear_model import OLS
from sm2.tools.tools import add_constant
import sm2.stats.sandwich_covariance as sw
from sm2.datasets import macrodata

cur_dir = os.path.abspath(os.path.dirname(__file__))
# Petersen's test_data from:
# www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt
fpath = os.path.join(cur_dir, "test_data.txt")
pet_data = pd.read_csv(fpath, delimiter='\s+', header=None).values


@pytest.mark.not_vetted
def test_cov_cluster_2groups():
    # comparing cluster robust standard errors to Peterson
    endog = pet_data[:, -1]
    group = pet_data[:, 0].astype(int)
    time = pet_data[:, 1].astype(int)
    exog = add_constant(pet_data[:, 2])
    res = OLS(endog, exog).fit()

    cov01, covg, covt = sw.cov_cluster_2groups(res, group, group2=time)

    # Reference number from Petersen
    # www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.htm
    bse_petw = [0.0284, 0.0284]
    bse_pet0 = [0.0670, 0.0506]
    bse_pet1 = [0.0234, 0.0334]   # year
    bse_pet01 = [0.0651, 0.0536]  # firm and year
    bse_0 = sw.se_cov(covg)
    bse_1 = sw.se_cov(covt)
    bse_01 = sw.se_cov(cov01)
    assert_almost_equal(bse_petw, res.HC0_se, decimal=4)
    assert_almost_equal(bse_0, bse_pet0, decimal=4)
    assert_almost_equal(bse_1, bse_pet1, decimal=4)
    assert_almost_equal(bse_01, bse_pet01, decimal=4)


@pytest.mark.not_vetted
def test_hac_simple():
    d2 = macrodata.load().data
    g_gdp = 400 * np.diff(np.log(d2['realgdp']))
    g_inv = 400 * np.diff(np.log(d2['realinv']))
    exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1]])
    res_olsg = OLS(g_inv, exogg).fit()

    # > NeweyWest(fm, lag = 4, prewhite = FALSE, sandwich = TRUE,
    #              verbose=TRUE, adjust=TRUE)
    # Lag truncation parameter chosen: 4
    #                  (Intercept)                   ggdp                  lint
    cov1_r = [
        [1.40643899878678802, -0.3180328707083329709, -0.060621111216488610],
        [-0.31803287070833292, 0.1097308348999818661, 0.000395311760301478],
        [-0.06062111121648865, 0.0003953117603014895, 0.087511528912470993]]

    # > NeweyWest(fm, lag = 4, prewhite = FALSE, sandwich = TRUE,
    #              verbose=TRUE, adjust=FALSE)
    # Lag truncation parameter chosen: 4
    #                (Intercept)                  ggdp                  lint
    cov2_r = [
        [1.3855512908840137, -0.313309610252268500, -0.059720797683570477],
        [-0.3133096102522685, 0.108101169035130618, 0.000389440793564339],
        [-0.0597207976835705, 0.000389440793564336, 0.086211852740503622]]

    cov1 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=True)
    sw.se_cov(cov1)  # smoke?

    cov2 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    sw.se_cov(cov2)  # smoke?

    assert_almost_equal(cov1, cov1_r, decimal=14)
    assert_almost_equal(cov2, cov2_r, decimal=14)

    # compare default for nlags
    cov3 = sw.cov_hac_simple(res_olsg, use_correction=False)
    cov4 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    assert_almost_equal(cov3, cov4, decimal=14)
