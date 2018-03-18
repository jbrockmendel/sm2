"""Testing helper functions

Warning: current status experimental, mostly copy paste

Warning: these functions will be changed without warning as the need
during refactoring arises.

The first group of functions provide consistency checks
"""
import numpy as np
from numpy.testing import assert_allclose

# the following are copied from
# sm2.base.tests.test_generic_methods.CheckGenericMixin
# and only adjusted to work as standalone functions


def check_ttest_tvalues(results):
    # test that t_test has same results a params, bse, tvalues, ...
    res = results
    mat = np.eye(len(res.params))
    tt = res.t_test(mat)

    assert_allclose(tt.effect, res.params, rtol=1e-12)
    # TODO: tt.sd and tt.tvalue are 2d also for single regressor, squeeze
    assert_allclose(np.squeeze(tt.sd), res.bse, rtol=1e-10)
    assert_allclose(np.squeeze(tt.tvalue), res.tvalues, rtol=1e-12)
    assert_allclose(tt.pvalue, res.pvalues, rtol=5e-10)
    assert_allclose(tt.conf_int(), res.conf_int(), rtol=1e-10)

    # test params table frame returned by t_test
    table_res = np.column_stack((res.params, res.bse, res.tvalues,
                                 res.pvalues, res.conf_int()))
    table2 = tt.summary_frame().values
    assert_allclose(table2, table_res, rtol=1e-12)

    # move this to test_attributes ?
    assert hasattr(res, 'use_t')

    tt = res.t_test(mat[0])
    tt.summary()   # smoke test for GH#1323
    assert_allclose(tt.pvalue, res.pvalues[0], rtol=5e-10)


def check_ftest_pvalues(results):
    res = results
    use_t = res.use_t
    k_vars = len(res.params)
    # check default use_t
    pvals = [res.wald_test(np.eye(k_vars)[k], use_f=use_t).pvalue
             for k in range(k_vars)]
    assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

    # automatic use_f based on results class use_t
    pvals = [res.wald_test(np.eye(k_vars)[k]).pvalue
             for k in range(k_vars)]
    assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

    # label for pvalues in summary
    string_use_t = 'P>|z|' if use_t is False else 'P>|t|'
    summ = str(res.summary())
    assert string_use_t in summ

    '''
    # summary2 not ported as of 2018-03-05
    # try except for models that don't have summary2
    try:
        summ2 = str(res.summary2())
    except AttributeError:
        summ2 = None
    if summ2 is not None:
        assert string_use_t in summ2
    '''


def check_fitted(results):
    raise NotImplementedError("check_fitted not ported from upstream, "
                              "as it is not used (or tested) there")


def check_predict_types(results):  # pragma: no cover
    raise NotImplementedError("check_predict_types not ported from upstream, "
                              "as it is not used (or tested) there")
