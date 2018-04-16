sm2
---
[statsmodels](https://github.com/statsmodels/statsmodels) is an excellent
project and important part of the python scientific stack.  But due to resource
constraints, they cannot push out bugfixes often enough for my needs.  sm2
is a fork focused on bugfixes and addressing technical debt.

Ideally sm2 will be a drop-in replacement for statsmodels.  In places where
this fails, feel free to open an issue.

With luck, fixes made here will eventually be ported upstream.


<table>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://travis-ci.org/jbrockmendel/sm2">
    <img src="https://travis-ci.org/jbrockmendel/sm2.svg?branch=master" alt="travis build status" />
    </a>
  </td>
</tr>
<tr>
  <td></td>
  <td>
    <a href="https://ci.appveyor.com/project/jbrockmendel/sm2">
    <img src="https://ci.appveyor.com/api/projects/status/gw9cui82oc1lnyqi/branch/master?svg=true" alt="appveyor build status" />
    </a>
  </td>
</tr>
<tr>
  <td>Coverage</td>
  <td>
    <a href="https://codecov.io/gh/jbrockmendel/sm2">
    <img src="https://codecov.io/gh/jbrockmendel/sm2/branch/master/graph/badge.svg" />
    </a>
</td>
</tr>
</table>


Changes vs Statsmodels
----------------------
- sm2 contains a subset of the functionality of statsmodels.  The first big
difference is that statsmodels is more feature-complete.

- Test coverage statistics reported for sm2 are meaningful (:issue:`4331`)

- An enormous amount of code-cleanup has been done in sm2.  Thousands of lines
of unused, untested, or deprecated code have been removed.  _Many_ thousands
of flake8 formatting issues have been cleaned up.

- `VARResults.cov_params` will correctly return a `DataFrame` instead
of raising `ValueError`.

- `tsa.stattools.acf` will always return `(acf, confint, qstat, pvalue)` here
instead of a different subset of these depending on the inputs.

- stats.diagnostic.acorr_ljungbox will always return
`(qljungbox, pval, qboxpierce, pvalbp)` here instead of a different subset
of these depending on the inputs.

- `summary2` methods have not been ported from upstream, will
raise `NotImplementedError`.

- `VARResults.test_whiteness` has been superceeded upstream by
`test_whiteness_new` as the older method was not an actual statistical
test (:issue:`4036`).  `sm2` replaces the older version entirely and keeps
only the name `test_whiteness`.

- `ARModel.fit` incorrectly sets `model.df_resid` upstream.  That has been
fixed here.

- `GenericLikelihoodModelResults.__init__` incorrectly sets `model.df_resid`
and `model.df_model`.  That has been fixed here.

- `GeneralizedLinearModel.fit` incorrect sets `self.mu` and `self.scale`.
This has been fixed here.  (:issue:`4032`)

- `LikelihoodModelResults._get_robustcov_results` incorrectly ignores
`use_self` argument.  This has been fixed here.  (:issue:`4401`)

Contributing
------------
Issues and Pull Requests are welcome.  If you are looking a place to start,
here are some suggestions:

- Search for comments starting with `# TODO:` or `# FIXME:`
     - Some comments are copied from upstream and _should_ have these labels
       but are missing them.  If you find a comment that should have one of
       these labels (or is just unclear), add the label.

- Many tests from upstream are marked with `pytest.mark.not_vetted` to reflect
  the fact that they haven't been reviewed since being ported from statsmodels.
  To "vet" a test, try to determine:
    - Is this a "smoke test"?  If so, it should be marked with
      `pytest.mark.smoke`.
    - Is this a test for a specific bug?  Can an Issue reference
      (e.g. `# GH#1234`) be included?
    - Is there something specific being tested?  If so, the test name should
      be made informative and often a comment should be added
      (e.g. `# test function foo.bar in case where baz argument is
      near-singular`)
    - Is this testing results produced by statsmodels/sm2 against results
      produced by another package?  If so, it should be clear how those results
      were produced.  The original authors put a lot of effort into producing
      these comparisons; they should be reproducible.

- There are some spots where tests are meager and could use some attention:
    - `tsa.vector_ar.irf`
    - `regression._prediction`
    - `stats.sandwich_covariance`

- As of 2018-03-19 there are still 390 flake8 warnings/errors.  For many of
  these, fixing them requires figuring out what the writer's attention was
  upstream.

- As of 2018-03-19 about 20% of statsmodels has been ported to sm2 (though a
  much larger percentage of the usable, non-redundant, non-deprecated code).
  If there are portions of statsmodels that you want or need, don't be shy.

- If there is a change you parrticularly like, make a Pull Request upstream
  to get it implemented directly in statsmodels.
