#!/usr/bin/env python
# -*- coding: utf-8 -*-

from six.moves import range
import numpy as np
from scipy import stats

from sm2.tools.tools import not_ported


# TODO: should this be public if it's just a container?
class ContrastResults(object):
    """
    Class for results of tests of linear restrictions on coefficients
    in a model.

    This class functions mainly as a container for `t_test`, `f_test` and
    `wald_test` for the parameters of a model.

    The attributes depend on the statistical test and are either based on the
    normal, the t, the F or the chisquare distribution.
    """
    __array__ = not_ported("__array__")  # GH#4470

    def __init__(self, t=None, F=None, sd=None, effect=None, df_denom=None,
                 df_num=None, alpha=0.05, **kwds):

        self.effect = effect  # Let it be None for F
        if F is not None:
            self.distribution = 'F'
            self.fvalue = F
            self.statistic = self.fvalue
            self.df_denom = df_denom
            self.df_num = df_num
            self.dist = stats.f
            self.dist_args = (df_num, df_denom)
            self.pvalue = stats.f.sf(F, df_num, df_denom)
        elif t is not None:
            self.distribution = 't'
            self.tvalue = t
            self.statistic = t  # generic alias
            self.sd = sd
            self.df_denom = df_denom
            self.dist = stats.t
            self.dist_args = (df_denom,)
            self.pvalue = self.dist.sf(np.abs(t), df_denom) * 2
        elif 'statistic' in kwds:
            # TODO: currently targeted to normal distribution, and chi2
            self.distribution = kwds['distribution']
            self.statistic = kwds['statistic']
            self.tvalue = value = kwds['statistic']  # keep alias
            # TODO: for results instance we decided to use tvalues
            #       also for normal
            self.sd = sd
            self.dist = getattr(stats, self.distribution)
            self.dist_args = ()
            if self.distribution is 'chi2':
                self.pvalue = self.dist.sf(self.statistic, df_denom)
            else:
                # "normal"
                self.pvalue = np.full_like(value, np.nan)
                not_nan = ~np.isnan(value)
                self.pvalue[not_nan] = self.dist.sf(np.abs(value[not_nan])) * 2

        # cleanup
        # TODO: should we return python scalar?
        self.pvalue = np.squeeze(self.pvalue)

    # TODO: De-duplicate docstring identical
    # to _prediction.PredictionResults.conf_int
    def conf_int(self, alpha=0.05):
        """
        Returns the confidence interval of the value, `effect` of
        the constraint.

        This is currently only available for t and z tests.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        if self.effect is not None:
            # confidence intervals
            q = self.dist.ppf(1 - alpha / 2., *self.dist_args)
            lower = self.effect - q * self.sd
            upper = self.effect + q * self.sd
            return np.column_stack((lower, upper))
        else:  # pragma: no cover
            # TODO: Should this be a ValueError?
            raise NotImplementedError('Confidence Interval not available')

    def __str__(self):
        return self.summary().__str__()

    def __repr__(self):
        return str(self.__class__) + '\n' + self.__str__()

    def summary(self, xname=None, alpha=0.05, title=None):
        """Summarize the Results of the hypothesis test

        Parameters
        -----------

        xname : list of strings, optional
            Default is `c_##` for ## in p the number of regressors
        alpha : float
            significance level for the confidence intervals. Default is
            alpha = 0.05 which implies a confidence level of 95%.
        title : string, optional
            Title for the params table. If not None, then this replaces the
            default title

        Returns
        -------
        smry : string or Summary instance
            This contains a parameter results table in the case of t or z test
            in the same form as the parameter results table in the model
            results summary.
            For F or Wald test, the return is a string.
        """
        if self.effect is not None:
            # TODO: should also add some extra information, e.g. robust cov ?
            # TODO: can we infer names for constraints, xname in __init__ ?
            if title is None:
                title = 'Test for Constraints'
            elif title == '':
                # don't add any title,
                # I think SimpleTable skips on None - check
                title = None
            # we have everything for a params table
            use_t = (self.distribution == 't')
            yname = 'constraints'  # Not used in params_frame
            if xname is None:
                xname = ['c%d' % ii for ii in range(len(self.effect))]
            from sm2.iolib.summary import summary_params
            pvalues = np.atleast_1d(self.pvalue)
            summ = summary_params((self, self.effect, self.sd, self.statistic,
                                   pvalues, self.conf_int(alpha)),
                                  yname=yname, xname=xname, use_t=use_t,
                                  title=title, alpha=alpha)
            return summ
        elif hasattr(self, 'fvalue'):
            # TODO: create something nicer for these casee
            return ('<F test: F=%s, p=%s, df_denom=%d, df_num=%d>' %
                    (repr(self.fvalue), self.pvalue,
                     self.df_denom, self.df_num))
        else:
            # generic
            return ('<Wald test: statistic=%s, p-value=%s>' %
                    (self.statistic, self.pvalue))

    def summary_frame(self, xname=None, alpha=0.05):
        """Return the parameter table as a pandas DataFrame

        This is only available for t and normal tests
        """
        if self.effect is not None:
            # we have everything for a params table
            use_t = (self.distribution == 't')
            yname = 'constraints'  # Not used in params_frame
            if xname is None:
                xname = ['c%d' % ii for ii in range(len(self.effect))]
            from sm2.iolib.summary import summary_params_frame
            summ = summary_params_frame((self, self.effect, self.sd,
                                         self.statistic, self.pvalue,
                                         self.conf_int(alpha)),
                                        yname=yname, xname=xname,
                                        use_t=use_t, alpha=alpha)
            return summ
        else:  # pragma: no cover
            # TODO: Should this be a ValueError?
            # TODO: create something nicer
            raise NotImplementedError('only available for t and z')


def Contrast(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Contrast not ported from upstream, as it "
                              "is not used outside of tests "
                              "(and one sandbox example file)")


def contrastfromcols(L, D, pseudo=None):  # pragma: no cover
    raise NotImplementedError("contrastfromcols not ported from upstream, "
                              "as it is only used by Contrast, "
                              "which is itself not ported")


# TODO: this is currently a minimal version, stub
class WaldTestResults(object):
    # for F and chi2 tests of joint hypothesis, mainly for vectorized

    def __init__(self, statistic, distribution, dist_args, table=None,
                 pvalues=None):
        self.table = table

        self.distribution = distribution
        self.statistic = statistic
        self.dist_args = dist_args

        # The following is because I don't know which we want
        if table is not None:
            self.statistic = table['statistic'].values
            self.pvalues = table['pvalue'].values
            self.df_constraints = table['df_constraint'].values
            if self.distribution == 'F':
                self.df_denom = table['df_denom'].values

        else:  # TODO: not hit in tests
            if self.distribution == 'chi2':
                self.dist = stats.chi2
                self.df_constraints = self.dist_args[0]  # assumes tuple
                # using dist_args[0] is a bit dangerous,
            elif self.distribution == 'F':
                self.dist = stats.f
                self.df_constraints, self.df_denom = self.dist_args
            else:
                raise ValueError('only F and chi2 are possible distribution')

            if pvalues is None:
                self.pvalues = self.dist.sf(np.abs(statistic), *dist_args)
            else:
                self.pvalues = pvalues

    @property
    def col_names(self):
        """column names for summary table"""
        pr_test = "P>%s" % self.distribution
        col_names = [self.distribution, pr_test, 'df constraint']
        if self.distribution == 'F':
            col_names.append('df denom')
        return col_names

    def summary_frame(self):  # TODO: not hit in tests
        # needs to be a method for consistency
        if hasattr(self, '_dframe'):
            return self._dframe
        # rename the column names, but don't copy data
        renaming = dict(zip(self.table.columns, self.col_names))
        self.dframe = self.table.rename(columns=renaming)
        return self.dframe

    def __str__(self):  # TODO: not hit in tests
        return self.summary_frame().to_string()

    def __repr__(self):
        return str(self.__class__) + '\n' + self.__str__()
