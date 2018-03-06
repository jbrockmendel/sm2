"""examples to check summary, not converted to tests yet
"""

from __future__ import print_function

if __name__ == '__main__':

    from sm2.regression.tests.test_regression  import TestOLS

    #def mytest():
    aregression = TestOLS()
    TestOLS.setup_class()
    results = aregression.res1
    r_summary = str(results.summary_old())
    print(r_summary)
    olsres = results

    print('\n\n')

    r_summary = str(results.summary())
    print(r_summary)
    print('\n\n')


    from sm2.discrete.tests.test_discrete  import TestProbitNewton

    aregression = TestProbitNewton()
    TestProbitNewton.setup_class()
    results = aregression.res1
    r_summary = str(results.summary())
    print(r_summary)
    print('\n\n')

    probres = results

    from sm2.robust.tests.test_rlm  import TestHampel

    aregression = TestHampel()
    #TestHampel.setup_class()
    results = aregression.res1
    r_summary = str(results.summary())
    print(r_summary)
    rlmres = results

    print('\n\n')

    from sm2.genmod.tests.test_glm  import TestGlmBinomial

    aregression = TestGlmBinomial()
    #TestGlmBinomial.setup_class()
    results = aregression.res1
    r_summary = str(results.summary())
    print(r_summary)

    #print(results.summary2(return_fmt='latex'))
    #print(results.summary2(return_fmt='csv'))

    smry = olsres.summary()
    print(smry.as_csv())

#    import matplotlib.pyplot as plt
#    plt.plot(rlmres.model.endog,'o')
#    plt.plot(rlmres.fittedvalues,'-')
#
#    plt.show()