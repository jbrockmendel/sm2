# -*- coding: utf-8 -*-
"""
Created on Mon Aug 04 08:00:16 2014

Author: Josef Perktold
License: BSD-3
"""
import numpy as np

import sm2.stats.sandwich_covariance as sw


def normalize_cov_type(cov_type):
    # normalize names
    if cov_type == 'nw-panel':
        cov_type = 'hac-panel'
    elif cov_type == 'nw-groupsum':
        cov_type = 'hac-groupsum'
    return cov_type


def set_df_adjustment(kwds, cov_type):
    adjust_df = False
    if cov_type in ['cluster', 'hac-panel', 'hac-groupsum']:
        df_correction = kwds.get('df_correction', None)
        # TODO: check also use_correction, do I need all combinations?
        if df_correction is not False:  # i.e. in [None, True]:
            # user didn't explicitly set it to False
            adjust_df = True

    return adjust_df


# FIXME: in places where `self` is used, should it be `res` instead?
def get_robustcov_results(self, cov_type='HC1', use_t=None, **kwds):
    """create new results instance with robust covariance as default

    Parameters
    ----------
    cov_type : string
        the type of robust sandwich estimator to use. see Notes below
    use_t : bool
        If true, then the t distribution is used for inference.
        If false, then the normal distribution is used.
        If `use_t` is None, then an appropriate default is used, which is
        `true` if the cov_type is nonrobust, and `false` in all other
        cases.
    kwds : depends on cov_type
        Required or optional arguments for robust covariance calculation.
        see Notes below

    Returns
    -------
    results : results instance
        This method creates a new results instance with the requested
        robust covariance as the default covariance of the parameters.
        Inferential statistics like p-values and hypothesis tests will be
        based on this covariance matrix.

    Notes
    -----
    Warning: Some of the options and defaults in cov_kwds may be changed in a
    future version.

    The covariance keywords provide an option 'scaling_factor' to adjust the
    scaling of the covariance matrix, that is the covariance is multiplied by
    this factor if it is given and is not `None`. This allows the user to
    adjust the scaling of the covariance matrix to match other statistical
    packages.
    For example, `scaling_factor=(nobs - 1.) / (nobs - k_params)` provides a
    correction so that the robust covariance matrices match those of Stata in
    some models like GLM and discrete Models.

    The following covariance types and required or optional arguments are
    currently available:

    - 'fixed scale' and optional keyword argument 'scale' which uses
        a predefined scale estimate with default equal to one.
    - 'HC0', 'HC1', 'HC2', 'HC3' and no keyword arguments:
        heteroscedasticity robust covariance
    - 'HAC' and keywords

        - `maxlag` integer (required) : number of lags to use
        - `kernel` string (optional) : kernel, default is Bartlett
        - `use_correction` bool (optional) : If true, use small sample
              correction

    - 'cluster' and required keyword `groups`, integer group indicator

        - `groups` array_like, integer (required) :
              index of clusters or groups
        - `use_correction` bool (optional) :
              If True the sandwich covariance is calculated with a small
              sample correction.
              If False the sandwich covariance is calculated without
              small sample correction.
        - `df_correction` bool (optional)
              If True (default), then the degrees of freedom for the
              inferential statistics and hypothesis tests, such as
              pvalues, f_pvalue, conf_int, and t_test and f_test, are
              based on the number of groups minus one instead of the
              total number of observations minus the number of explanatory
              variables. `df_resid` of the results instance is adjusted.
              If False, then `df_resid` of the results instance is not
              adjusted.

    - 'hac-groupsum' Driscoll and Kraay, heteroscedasticity and
        autocorrelation robust standard errors in panel data
        keywords

        - `time` array_like (required) : index of time periods
        - `maxlag` integer (required) : number of lags to use
        - `kernel` string (optional) : kernel, default is Bartlett
        - `use_correction` {False, 'hac', 'cluster'} (optional) :
              If False the sandwich covariance is calculated without
              small sample correction.
              If `use_correction = 'cluster'` (default), then the same
              small sample correction as in the case of 'covtype='cluster''
              is used.
        - `df_correction` bool (optional)
              adjustment to df_resid, see cov_type 'cluster' above
              # TODO: we need more options here

    - 'hac-panel' heteroscedasticity and autocorrelation robust standard
        errors in panel data.
        The data needs to be sorted in this case, the time series for
        each panel unit or cluster need to be stacked. The membership to
        a timeseries of an individual or group can be either specified by
        group indicators or by increasing time periods.

        keywords

        - either `groups` or `time` : array_like (required)
          `groups` : indicator for groups
          `time` : index of time periods
        - `maxlag` integer (required) : number of lags to use
        - `kernel` string (optional) : kernel, default is Bartlett
        - `use_correction` {False, 'hac', 'cluster'} (optional) :
              If False the sandwich covariance is calculated without
              small sample correction.
        - `df_correction` bool (optional)
              adjustment to df_resid, see cov_type 'cluster' above
              # TODO: we need more options here

    Reminder:
    `use_correction` in "hac-groupsum" and "hac-panel" is not bool,
    needs to be in [False, 'hac', 'cluster']

    TODO: Currently there is no check for extra or misspelled keywords,
    except in the case of cov_type `HCx`
    """
    cov_type = normalize_cov_type(cov_type)

    if 'kernel' in kwds:
        kwds['weights_func'] = kwds.pop('kernel')

    # pop because HCx raises if any kwds
    sc_factor = kwds.pop('scaling_factor', None)

    # TODO: make separate function that returns a robust cov plus info
    use_self = kwds.pop('use_self', False)
    if use_self:
        res = self
    else:
        # this doesn't work for most models, use raw instance instead from fit
        res = self.__class__(self.model, self.params,
                             normalized_cov_params=self.normalized_cov_params,
                             scale=self.scale)

    res.cov_type = cov_type
    # use_t might already be defined by the class, and already set
    if use_t is None:
        use_t = self.use_t
    res.cov_kwds = {'use_t': use_t}  # store for information
    res.use_t = use_t
    # TODO: use_t not used?

    adjust_df = set_df_adjustment(kwds, cov_type)
    res.cov_kwds['adjust_df'] = adjust_df

    n_groups = None  # not present upstream
    # verify and set kwds, and calculate cov
    # TODO: this should be outsourced in a function so we can reuse it in
    #       other models
    # TODO: make it DRYer   repeated code for checking kwds
    if cov_type == 'nonrobust':
        res.cov_type = 'nonrobust'
        res.cov_kwds = {'description': 'Standard Errors assume that the '
                                       'covariance matrix of the errors is '
                                       'correctly specified.'}
        return res  # return early to avoid pinning scaling_factor
    elif cov_type in ['fixed scale', 'fixed_scale']:
        res.cov_kwds['description'] = ('Standard Errors are based on '
                                       'fixed scale')

        res.cov_kwds['scale'] = scale = kwds.get('scale', 1.)
        res.cov_params_default = scale * res.normalized_cov_params
        # FIXME: n_groups is not defined in this branch.  This behavior
        # is consistent with upstream
    elif cov_type.upper() in ['HC0', 'HC1', 'HC2', 'HC3']:
        robust_stderrs(self, res, kwds, cov_type)
    elif cov_type.lower() == 'hac':
        n_groups = hac_stderrs(self, res, kwds)
    elif cov_type.lower() == 'cluster':
        # cluster robust standard errors, one- or two-way
        n_groups = clustered_stderrs(self, res, kwds, cov_type, adjust_df)
    elif cov_type.lower() == 'hac-panel':
        # cluster robust standard errors
        n_groups = hac_panel(self, res, kwds, cov_type)
    elif cov_type.lower() == 'hac-groupsum':
        # Driscoll-Kraay standard errors
        n_groups = hac_groupsum(self, res, kwds, adjust_df, cov_type)
    else:  # pragma: no cover
        raise ValueError('cov_type not recognized. See docstring for '
                         'available options and spelling')

    # TODO: this isn't done in the linear_model version.
    #       Should it always be done, even if sc_factor is None?
    # generic optional factor to scale covariance
    res.cov_kwds['scaling_factor'] = sc_factor
    if sc_factor is not None:
        res.cov_params_default *= sc_factor

    if adjust_df:
        # Note: df_resid is used for scale and others, add new attribute
        res.df_resid_inference = n_groups - 1

    return res


def robust_stderrs(self, res, kwds, cov_type):
    if kwds:
        raise ValueError('heteroscedasticity robust covariance '
                         'does not use keywords')
    res.cov_kwds['description'] = ('Standard Errors are '
                                   'heteroscedasticity robust '
                                   '(' + cov_type + ')')

    res.cov_params_default = getattr(self, 'cov_' + cov_type.upper(), None)
    if res.cov_params_default is None:
        # results classes that don't have cov_HCx attribute
        res.cov_params_default = sw.cov_white_simple(self,
                                                     use_correction=False)


def hac_stderrs(self, res, kwds):
    n_groups = None
    maxlags = kwds['maxlags']   # required?, default in cov_hac_simple
    res.cov_kwds['maxlags'] = maxlags
    weights_func = kwds.get('weights_func', sw.weights_bartlett)
    res.cov_kwds['weights_func'] = weights_func
    use_correction = kwds.get('use_correction', False)
    res.cov_kwds['use_correction'] = use_correction
    maybe_with = ['without', 'with'][use_correction]
    res.cov_kwds['description'] = ('Standard Errors are heteroscedasticity '
                                   'and autocorrelation robust (HAC) '
                                   'using %d lags and %s small sample '
                                   'correction') % (maxlags, maybe_with)

    cpd = sw.cov_hac_simple(self, nlags=maxlags,
                            weights_func=weights_func,
                            use_correction=use_correction)
    res.cov_params_default = cpd
    return n_groups


def clustered_stderrs(self, res, kwds, cov_type, adjust_df):
    # cluster robust standard errors, one- or two-way
    n_groups = None

    groups = kwds['groups']
    if not hasattr(groups, 'shape'):
        groups = np.asarray(groups).T

    if groups.ndim >= 2:
        groups = groups.squeeze()

    res.cov_kwds['groups'] = groups
    use_correction = kwds.get('use_correction', True)
    res.cov_kwds['use_correction'] = use_correction
    if groups.ndim == 1:
        if adjust_df:
            # need to find number of groups
            # duplicate work
            self.n_groups = n_groups = len(np.unique(groups))
        cpd = sw.cov_cluster(self, groups, use_correction=use_correction)
        res.cov_params_default = cpd

    elif groups.ndim == 2:
        if hasattr(groups, 'values'):
            groups = groups.values

        if adjust_df:
            # need to find number of groups
            # duplicate work
            n_groups0 = len(np.unique(groups[:, 0]))
            n_groups1 = len(np.unique(groups[:, 1]))
            self.n_groups = (n_groups0, n_groups1)
            n_groups = min(n_groups0, n_groups1)  # use for adjust_df

        # Note: sw.cov_cluster_2groups has 3 returns
        cpd = sw.cov_cluster_2groups(self, groups,
                                     use_correction=use_correction)[0]
        res.cov_params_default = cpd
    else:  # pragma: no cover
        raise ValueError('only two groups are supported')

    res.cov_kwds['description'] = ('Standard Errors are robust to '
                                   'cluster correlation ({cov_type})'
                                   .format(cov_type=cov_type))
    return n_groups


def hac_panel(self, res, kwds, cov_type):
    # cluster robust standard errors
    res.cov_kwds['time'] = time = kwds.get('time', None)
    res.cov_kwds['groups'] = groups = kwds.get('groups', None)
    # TODO: nlags is currently required
    # nlags = kwds.get('nlags', True)
    # res.cov_kwds['nlags'] = nlags
    # TODO: `nlags` or `maxlags`
    res.cov_kwds['maxlags'] = maxlags = kwds['maxlags']
    use_correction = kwds.get('use_correction', 'hac')
    res.cov_kwds['use_correction'] = use_correction
    weights_func = kwds.get('weights_func', sw.weights_bartlett)
    res.cov_kwds['weights_func'] = weights_func
    # TODO: clumsy time index in cov_nw_panel
    if groups is not None:
        groups = np.asarray(groups)
        tt = (np.nonzero(groups[:-1] != groups[1:])[0] + 1).tolist()
        nobs_ = len(groups)
    elif time is not None:
        # TODO: clumsy time index in cov_nw_panel
        time = np.asarray(time)
        tt = (np.nonzero(time[1:] < time[:-1])[0] + 1).tolist()
        nobs_ = len(time)
    else:
        raise ValueError('either time or groups needs to be given')
    groupidx = list(zip([0] + tt, tt + [nobs_]))
    self.n_groups = n_groups = len(groupidx)
    ncp = sw.cov_nw_panel(self, maxlags, groupidx,
                          weights_func=weights_func,
                          use_correction=use_correction)
    res.cov_params_default = ncp
    res.cov_kwds['description'] = ('Standard Errors are robust to'
                                   'cluster correlation ({cov_type})'
                                   .format(cov_type=cov_type))
    return n_groups


# FIXME: in places where `self` is used, should it be `res` instead?
# TODO: clean this up.  Splitting it out into its own function to whittle
# down repeated code in linear_model
def hac_groupsum(self, res, kwds, adjust_df, cov_type):
    # Driscoll-Kraay standard errors
    n_groups = None
    res.cov_kwds['time'] = time = kwds['time']
    # TODO: nlags is currently required
    # nlags = kwds.get('nlags', True)
    # res.cov_kwds['nlags'] = nlags
    # TODO: `nlags` or `maxlags`
    res.cov_kwds['maxlags'] = maxlags = kwds['maxlags']
    use_correction = kwds.get('use_correction', 'cluster')
    res.cov_kwds['use_correction'] = use_correction
    weights_func = kwds.get('weights_func', sw.weights_bartlett)
    res.cov_kwds['weights_func'] = weights_func
    if adjust_df:
        # need to find number of groups
        tt = (np.nonzero(time[1:] < time[:-1])[0] + 1)
        self.n_groups = n_groups = len(tt) + 1

    cpd = sw.cov_nw_groupsum(self, maxlags, time,
                             weights_func=weights_func,
                             use_correction=use_correction)
    res.cov_params_default = cpd
    res.cov_kwds['description'] = ('Driscoll and Kraay Standard Errors '
                                   'are robust to cluster correlation '
                                   '({cov_type})'
                                   .format(cov_type=cov_type))
    return n_groups
