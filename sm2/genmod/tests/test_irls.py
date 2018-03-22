"""
Tests for iteratively weighted least squares

Upstream this is part of test_glm
"""
import warnings

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sm2.api as sm
from sm2.genmod.generalized_linear_model import GLM
from sm2.genmod.families import links


@pytest.mark.not_vetted
def gen_endog(lin_pred, family_class, link, binom_version=0):
    np.random.seed(872)
    mu = link().inverse(lin_pred)

    if family_class == sm.families.Binomial:
        if binom_version == 0:
            endog = 1 * (np.random.uniform(size=len(lin_pred)) < mu)
        else:
            endog = np.empty((len(lin_pred), 2))
            n = 10
            uni = np.random.uniform(size=(len(lin_pred), n))
            endog[:, 0] = (uni < mu[:, None]).sum(1)
            endog[:, 1] = n - endog[:, 0]
    elif family_class == sm.families.Poisson:
        endog = np.random.poisson(mu)
    elif family_class == sm.families.Gamma:
        endog = np.random.gamma(2, mu)
    elif family_class == sm.families.Gaussian:
        endog = mu + np.random.normal(size=len(lin_pred))
    elif family_class == sm.families.NegativeBinomial:
        from scipy.stats.distributions import nbinom
        endog = nbinom.rvs(mu, 0.5)
    elif family_class == sm.families.InverseGaussian:
        from scipy.stats.distributions import invgauss
        endog = invgauss.rvs(mu)
    else:
        raise ValueError

    return endog


@pytest.mark.not_vetted
def test_gradient_irls():
    # Compare the results when using gradient optimization and IRLS.
    # TODO: Find working examples for inverse_squared link
    np.random.seed(87342)

    fams = [(sm.families.Binomial, [links.logit, links.probit, links.cloglog,
                                    links.log, links.cauchy]),
            (sm.families.Poisson, [links.log, links.identity, links.sqrt]),
            (sm.families.Gamma, [links.log, links.identity,
                                 links.inverse_power]),
            (sm.families.Gaussian, [links.identity, links.log,
                                    links.inverse_power]),
            (sm.families.InverseGaussian, [links.log, links.identity,
                                           links.inverse_power,
                                           links.inverse_squared]),
            (sm.families.NegativeBinomial, [links.log, links.inverse_power,
                                            links.inverse_squared,
                                            links.identity])]

    n = 100
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1

    skip_one = False
    for family_class, family_links in fams:
        for link in family_links:
            for binom_version in [0, 1]:

                if family_class != sm.families.Binomial and binom_version == 1:
                    continue

                if (family_class, link) == (sm.families.Poisson,
                                            links.identity):
                    lin_pred = 20 + exog.sum(1)
                elif (family_class, link) == (sm.families.Binomial, links.log):
                    lin_pred = -1 + exog.sum(1) / 8
                elif (family_class, link) == (sm.families.Poisson, links.sqrt):
                    lin_pred = 2 + exog.sum(1)
                elif (family_class, link) == (sm.families.InverseGaussian,
                                              links.log):
                    #skip_zero = True
                    lin_pred = -1 + exog.sum(1)
                elif (family_class, link) == (sm.families.InverseGaussian,
                                              links.identity):
                    lin_pred = 20 + 5 * exog.sum(1)
                    lin_pred = np.clip(lin_pred, 1e-4, np.inf)
                elif (family_class, link) == (sm.families.InverseGaussian,
                                              links.inverse_squared):
                    lin_pred = 0.5 + exog.sum(1) / 5
                    continue  # skip due to non-convergence
                elif (family_class, link) == (sm.families.InverseGaussian,
                                              links.inverse_power):
                    lin_pred = 1 + exog.sum(1) / 5
                elif (family_class, link) == (sm.families.NegativeBinomial,
                                              links.identity):
                    lin_pred = 20 + 5 * exog.sum(1)
                    lin_pred = np.clip(lin_pred, 1e-4, np.inf)
                elif (family_class, link) == (sm.families.NegativeBinomial,
                                              links.inverse_squared):
                    lin_pred = 0.1 + np.random.uniform(size=exog.shape[0])
                    continue  # skip due to non-convergence
                elif (family_class, link) == (sm.families.NegativeBinomial,
                                              links.inverse_power):
                    lin_pred = 1 + exog.sum(1) / 5

                elif (family_class, link) == (sm.families.Gaussian,
                                              links.inverse_power):
                    # adding skip because of convergence failure
                    skip_one = True
                else:
                    lin_pred = np.random.uniform(size=exog.shape[0])

                endog = gen_endog(lin_pred, family_class, link, binom_version)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod_irls = sm.GLM(endog, exog,
                                      family=family_class(link=link()))
                rslt_irls = mod_irls.fit(method="IRLS")

                # Try with and without starting values.
                for max_start_irls, start_params in [(0, rslt_irls.params),
                                                     (3, None)]:
                    # TODO: skip convergence failures for now
                    if max_start_irls > 0 and skip_one:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod_gradient = sm.GLM(endog, exog,
                                              family=family_class(link=link()))
                    rslt_gradient = mod_gradient.fit(
                        max_start_irls=max_start_irls,
                        start_params=start_params,
                        method="newton")

                    assert_allclose(rslt_gradient.params,
                                    rslt_irls.params, rtol=1e-6, atol=5e-5)

                    assert_allclose(rslt_gradient.llf, rslt_irls.llf,
                                    rtol=1e-6, atol=1e-6)

                    assert_allclose(rslt_gradient.scale, rslt_irls.scale,
                                    rtol=1e-6, atol=1e-6)

                    # Get the standard errors using expected information.
                    gradient_bse = rslt_gradient.bse
                    ehess = mod_gradient.hessian(rslt_gradient.params,
                                                 observed=False)
                    gradient_bse = np.sqrt(-np.diag(np.linalg.inv(ehess)))
                    assert_allclose(gradient_bse, rslt_irls.bse,
                                    rtol=1e-6, atol=5e-5)


@pytest.mark.not_vetted
def test_gradient_irls_eim():
    # Compare the results when using eime gradient optimization and IRLS.
    # TODO: Find working examples for inverse_squared link
    np.random.seed(87342)

    fams = [(sm.families.Binomial, [links.logit, links.probit, links.cloglog,
                                    links.log, links.cauchy]),
            (sm.families.Poisson, [links.log, links.identity, links.sqrt]),
            (sm.families.Gamma, [links.log, links.identity,
                                 links.inverse_power]),
            (sm.families.Gaussian, [links.identity, links.log,
                                    links.inverse_power]),
            (sm.families.InverseGaussian, [links.log, links.identity,
                                           links.inverse_power,
                                           links.inverse_squared]),
            (sm.families.NegativeBinomial, [links.log, links.inverse_power,
                                            links.inverse_squared,
                                            links.identity])]

    n = 100
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1

    skip_one = False
    for family_class, family_links in fams:
        for link in family_links:
            for binom_version in [0, 1]:

                if family_class != sm.families.Binomial and binom_version == 1:
                    continue

                if (family_class, link) == (sm.families.Poisson,
                                            links.identity):
                    lin_pred = 20 + exog.sum(1)
                elif (family_class, link) == (sm.families.Binomial, links.log):
                    lin_pred = -1 + exog.sum(1) / 8
                elif (family_class, link) == (sm.families.Poisson, links.sqrt):
                    lin_pred = 2 + exog.sum(1)
                elif (family_class, link) == (sm.families.InverseGaussian,
                                              links.log):
                    # skip_zero = True
                    lin_pred = -1 + exog.sum(1)
                elif (family_class, link) == (sm.families.InverseGaussian,
                                              links.identity):
                    lin_pred = 20 + 5 * exog.sum(1)
                    lin_pred = np.clip(lin_pred, 1e-4, np.inf)
                elif (family_class, link) == (sm.families.InverseGaussian,
                                              links.inverse_squared):
                    lin_pred = 0.5 + exog.sum(1) / 5
                    continue  # skip due to non-convergence
                elif (family_class, link) == (sm.families.InverseGaussian,
                                              links.inverse_power):
                    lin_pred = 1 + exog.sum(1) / 5
                elif (family_class, link) == (sm.families.NegativeBinomial,
                                              links.identity):
                    lin_pred = 20 + 5 * exog.sum(1)
                    lin_pred = np.clip(lin_pred, 1e-4, np.inf)
                elif (family_class, link) == (sm.families.NegativeBinomial,
                                              links.inverse_squared):
                    lin_pred = 0.1 + np.random.uniform(size=exog.shape[0])
                    continue  # skip due to non-convergence
                elif (family_class, link) == (sm.families.NegativeBinomial,
                                              links.inverse_power):
                    lin_pred = 1 + exog.sum(1) / 5

                elif (family_class, link) == (sm.families.Gaussian,
                                              links.inverse_power):
                    # adding skip because of convergence failure
                    skip_one = True
                else:
                    lin_pred = np.random.uniform(size=exog.shape[0])

                endog = gen_endog(lin_pred, family_class, link, binom_version)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod_irls = sm.GLM(endog, exog,
                                      family=family_class(link=link()))
                rslt_irls = mod_irls.fit(method="IRLS")

                # Try with and without starting values.
                for max_start_irls, start_params in ((0, rslt_irls.params),
                                                     (3, None)):
                    # TODO: skip convergence failures for now
                    if max_start_irls > 0 and skip_one:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod_gradient = sm.GLM(endog, exog,
                                              family=family_class(link=link()))
                    rslt_gradient = mod_gradient.fit(
                        max_start_irls=max_start_irls,
                        start_params=start_params,
                        method="newton",
                        optim_hessian='eim')

                    assert_allclose(rslt_gradient.params, rslt_irls.params,
                                    rtol=1e-6, atol=5e-5)

                    assert_allclose(rslt_gradient.llf, rslt_irls.llf,
                                    rtol=1e-6, atol=1e-6)

                    assert_allclose(rslt_gradient.scale, rslt_irls.scale,
                                    rtol=1e-6, atol=1e-6)

                    # Get the standard errors using expected information.
                    ehess = mod_gradient.hessian(rslt_gradient.params,
                                                 observed=False)
                    gradient_bse = np.sqrt(-np.diag(np.linalg.inv(ehess)))

                    assert_allclose(gradient_bse, rslt_irls.bse,
                                    rtol=1e-6, atol=5e-5)
