import copy
import time

from six.moves import range, zip_longest
import numpy as np
import pandas as pd

from sm2.iolib.table import SimpleTable
from sm2.iolib.tableformatting import fmt_params, fmt_2cols


def forg(x, prec=3):
    if prec == 3:
        # for 3 decimals
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%9.3g' % x
        else:
            return '%9.3f' % x
    elif prec == 4:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%10.4g' % x
        else:
            return '%10.4f' % x
    else:  # pragma: no cover
        raise NotImplementedError  # TODO: Should this be ValueError?


# TODO: not hit in tests?  how is that possible?
def summary(self, yname=None, xname=None, title=0, alpha=.05,
            returns='text', model_info=None):  # pragma: no cover
    raise NotImplementedError("summary not ported from upstream, "
                              "as it is neither used nor tested there.")


def _getnames(self, yname=None, xname=None):
    """extract names from model or construct names
    """
    if yname is None:
        if getattr(self.model, 'endog_names', None) is not None:
            yname = self.model.endog_names
        else:
            yname = 'y'

    if xname is None:
        if getattr(self.model, 'exog_names', None) is not None:
            xname = self.model.exog_names
        else:
            xname = ['var_%d' % i for i in range(len(self.params))]

    return yname, xname


def _build_gen_tuples(gen, default_items):
    # replace missing (None) values with default values
    gen_tuples = []
    for item, value in gen:
        if value is None:
            value = default_items[item]()  # let KeyErrors raise exception
        gen_tuples.append((item, value))

    return gen_tuples


def summary_top(results, title=None, gleft=None, gright=None,
                yname=None, xname=None):
    """generate top table(s)

    TODO: this still uses predefined model_methods
    ? allow gleft, gright to be 1 element tuples instead of filling with None?
    """
    # change of names ?
    gen_left, gen_right = gleft, gright

    # time and names are always included
    time_now = time.localtime()
    time_of_day = [time.strftime("%H:%M:%S", time_now)]
    date = time.strftime("%a, %d %b %Y", time_now)

    yname, xname = _getnames(results, yname=yname, xname=xname)

    # create dictionary with default
    # use lambdas because some values raise exception if they are not available
    # alternate spellings are commented out to force unique labels
    default_items = dict([
        ('Dependent Variable:', lambda: [yname]),
        ('Dep. Variable:', lambda: [yname]),
        ('Model:', lambda: [results.model.__class__.__name__]),
        ('Date:', lambda: [date]),
        ('Time:', lambda: time_of_day),
        ('Number of Obs:', lambda: [results.nobs]),
        ('No. Observations:', lambda: ["%#6d" % results.nobs]),
        ('Df Model:', lambda: ["%#6d" % results.df_model]),
        # TODO: check when we have non-integer df
        ('Df Residuals:', lambda: ["%#6d" % results.df_resid]),
        ('Log-Likelihood:', lambda: ["%#8.5g" % results.llf])  # doesn't exist for RLM - exception
        #('Method:', lambda: [???]),  # no default for this
    ])

    if title is None:
        title = results.model.__class__.__name__ + 'Regression Results'

    if gen_left is None:  # pragma: no cover
        raise NotImplementedError("Case with gen_left being None "
                                  "is not ported from upstream, as it is not "
                                  "used or tested.  "
                                  "It is not clear what the intended use "
                                  "case is.  If you have such a use case, "
                                  "please file a bug report.  "
                                  "See GH#4450 (upstream)")

    gen_title = title
    gen_header = None

    gen_left = _build_gen_tuples(gen_left, default_items)

    if gen_right:
        gen_right = _build_gen_tuples(gen_right, default_items)

    missing_values = [k for k, v in gen_left + gen_right if v is None]
    assert missing_values == [], missing_values

    # pad both tables to equal number of rows
    if gen_right:
        if len(gen_right) < len(gen_left):
            # fill up with blank lines to same length
            gen_right += [(' ', ' ')] * (len(gen_left) - len(gen_right))
        elif len(gen_right) > len(gen_left):
            # fill up with blank lines to same length,
            # just to keep it symmetric
            gen_left += [(' ', ' ')] * (len(gen_right) - len(gen_left))

        # padding in SimpleTable doesn't work like I want
        # force extra spacing and exact string length in right table
        gen_right = [('%-21s' % ('  ' + k), v) for k, v in gen_right]
        # transpose row col
        gen_stubs_right, gen_data_right = zip_longest(*gen_right)

        gen_table_right = SimpleTable(gen_data_right,
                                      gen_header,
                                      gen_stubs_right,
                                      title=gen_title,
                                      txt_fmt=fmt_2cols)
    else:
        gen_table_right = []  # because .extend_right seems works with []

    # moved below so that we can pad if needed to match length of gen_right
    # transpose rows and columns, `unzip`
    gen_stubs_left, gen_data_left = zip_longest(*gen_left)  # transpose row col

    gen_table_left = SimpleTable(gen_data_left,
                                 gen_header,
                                 gen_stubs_left,
                                 title=gen_title,
                                 txt_fmt=fmt_2cols)

    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    return general_table


def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True,
                   skip_header=False, title=None):
    """create a summary table for the parameters

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : string or None
        optional name for the endogenous variable, default is "y"
    xname : list of strings or None
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    params_table : SimpleTable instance
    """

    # Parameters part of the summary table
    # ------------------------------------
    # Note: this is not necessary since we standardized names,
    # only t versus normal

    if isinstance(results, tuple):
        # for multivariate endog
        # TODO: check whether I don't want to refactor this
        # we need to give parameter alpha to conf_int
        results, params, std_err, tvalues, pvalues, conf_int = results
    else:
        params = results.params
        std_err = results.bse
        tvalues = results.tvalues  # is this sometimes called zvalues
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha)

    # Dictionary to store the header names for the parameter part of the
    # summary table. look up by modeltype
    if use_t:
        param_header = ['coef', 'std err', 't', 'P>|t|',
                        '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
    else:
        param_header = ['coef', 'std err', 'z', 'P>|z|',
                        '[' + str(alpha / 2), str(1 - alpha / 2) + ']']

    if skip_header:
        param_header = None

    _, xname = _getnames(results, yname=yname, xname=xname)

    params_stubs = xname

    exog_idx = list(range(len(xname)))

    params_data = list(zip([forg(params[i], prec=4) for i in exog_idx],
                           [forg(std_err[i]) for i in exog_idx],
                           [forg(tvalues[i]) for i in exog_idx],
                           ["%#6.3f" % (pvalues[i]) for i in exog_idx],
                           [forg(conf_int[i, 0]) for i in exog_idx],
                           [forg(conf_int[i, 1]) for i in exog_idx]))
    parameter_table = SimpleTable(params_data,
                                  param_header,
                                  params_stubs,
                                  title=title,
                                  txt_fmt=fmt_params)

    return parameter_table


def summary_params_frame(results, yname=None, xname=None, alpha=.05,
                         use_t=True):
    """create a summary table for the parameters

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : string or None
        optional name for the endogenous variable, default is "y"
    xname : list of strings or None
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    params_table : SimpleTable instance
    """

    # Parameters part of the summary table
    # ------------------------------------
    # Note: this is not necessary since we standardized names,
    # only t versus normal

    if isinstance(results, tuple):
        # for multivariate endog
        # TODO: check whether I don't want to refactor this
        # we need to give parameter alpha to conf_int
        results, params, std_err, tvalues, pvalues, conf_int = results
    else:  # TODO: not hit in tests
        params = results.params
        std_err = results.bse
        tvalues = results.tvalues  # is this sometimes called zvalues
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha)

    # Dictionary to store the header names for the parameter part of the
    # summary table. look up by modeltype
    alp = str((1 - alpha) * 100) + '%'  # TODO: Should this be used?
    if use_t:
        param_header = ['coef', 'std err', 't', 'P>|t|',
                        'Conf. Int. Low', 'Conf. Int. Upp.']
    else:
        param_header = ['coef', 'std err', 'z', 'P>|z|',
                        'Conf. Int. Low', 'Conf. Int. Upp.']

    _, xname = _getnames(results, yname=yname, xname=xname)

    table = np.column_stack((params, std_err, tvalues, pvalues, conf_int))
    return pd.DataFrame(table,
                        columns=param_header,
                        index=xname)


def summary_params_2d(result, extras=None, endog_names=None, exog_names=None,
                      title=None):  # pragma: no cover
    raise NotImplementedError("summary_params_2d not ported from upstream, "
                              "as it is only used in one example file there, "
                              "and never tested.")


def summary_params_2dflat(result, endog_names=None, exog_names=None, alpha=0.05,
                          use_t=True, keep_headers=True, endog_cols=False):
    """summary table for parameters that are 2d, e.g. multi-equation models

    Parameters
    ----------
    result : result instance
        the result instance with params, bse, tvalues and conf_int
    endog_names : None or list of strings
        names for rows of the parameter array (multivariate endog)
    exog_names : None or list of strings
        names for columns of the parameter array (exog)
    alpha : float
        level for confidence intervals, default 0.95
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    keep_headers : bool
        If true (default), then sub-tables keep their headers. If false, then
        only the first headers are kept, the other headerse are blanked out
    endog_cols : bool
        If false (default) then params and other result statistics have
        equations by rows. If true, then equations are assumed to be in
        columns.  Not implemented yet.

    Returns
    -------
    tables : list of SimpleTable
        this contains a list of all seperate Subtables
    table_all : SimpleTable
        the merged table with results concatenated for each row of the
        parameter array
    """
    res = result
    params = res.params
    if params.ndim == 2:  # we've got multiple equations
        n_equ = params.shape[1]
        if not len(endog_names) == params.shape[1]:
            raise ValueError('endog_names has wrong length')
    else:
        if not len(endog_names) == len(params):
            raise ValueError('endog_names has wrong length')
        n_equ = 1

    if not isinstance(endog_names, list):  # TODO: not hit in tests
        # this might be specific to multinomial logit type, move?
        if endog_names is None:
            endog_basename = 'endog'
        else:
            endog_basename = endog_names
        # TODO: note, the [1:] is specific to current MNLogit
        endog_names = res.model.endog_names[1:]

    # check if we have the right length of names

    tables = []
    for eq in range(n_equ):
        restup = (res, res.params[:, eq], res.bse[:, eq], res.tvalues[:, eq],
                  res.pvalues[:, eq], res.conf_int(alpha)[eq])

        skiph = False
        tble = summary_params(restup, yname=endog_names[eq],
                              xname=exog_names, alpha=alpha, use_t=use_t,
                              skip_header=skiph)

        tables.append(tble)

    # add titles, they will be moved to header lines in table_extend
    for i in range(len(endog_names)):
        tables[i].title = endog_names[i]

    table_all = table_extend(tables, keep_headers=keep_headers)

    return tables, table_all


def table_extend(tables, keep_headers=True):
    """extend a list of SimpleTables, adding titles to header of subtables

    This function returns the merged table as a deepcopy, in contrast to the
    SimpleTable extend method.

    Parameters
    ----------
    tables : list of SimpleTable instances
    keep_headers : bool
        If true, then all headers are kept. If falls, then the headers of
        subtables are blanked out.

    Returns
    -------
    table_all : SimpleTable
        merged tables as a single SimpleTable instance
    """
    for ii, t in enumerate(tables[:]):
        t = copy.deepcopy(t)
        # move title to first cell of header
        # TODO: check if we have multiline headers
        if t[0].datatype == 'header':
            t[0][0].data = t.title
            t[0][0]._datatype = None
            t[0][0].row = t[0][1].row
            if not keep_headers and (ii > 0):
                for c in t[0][1:]:
                    c.data = ''

        # add separating line and extend tables
        if ii == 0:
            table_all = t
        else:
            r1 = table_all[-1]
            r1.add_format('txt', row_dec_below='-')
            table_all.extend(t)

    table_all.title = None
    return table_all


def summary_return(tables, return_fmt='text'):
    # join table parts then print
    if return_fmt == 'text':
        # convert to string drop last line
        return '\n'.join([str(x).rsplit('\n', 1)[0] for x in tables[:-1]] +
                         [str(tables[-1])])
    elif return_fmt == 'tables':
        return tables
    elif return_fmt == 'csv':
        return '\n'.join(map(lambda x: x.as_csv(), tables))
    elif return_fmt == 'latex':
        # TODO: insert \hline after updating SimpleTable
        table = copy.deepcopy(tables[0])
        del table[-1]
        for part in tables[1:]:
            table.extend(part)
        return table.as_latex_tabular()
    elif return_fmt == 'html':
        return "\n".join(table.as_html() for table in tables)
    else:  # pragma: no cover
        raise ValueError('available output formats are text, csv, latex, html')


class Summary(object):
    """class to hold tables for result summary presentation

    Construction does not take any parameters. Tables and text can be added
    with the `add_` methods.

    Attributes
    ----------
    tables : list of tables
        Contains the list of SimpleTable instances, horizontally
        concatenated tables are not saved separately.
    extra_txt : string
        extra lines that are added to the text output, used for
        warnings and explanations.
    """
    def __init__(self):
        self.tables = []
        self.extra_txt = None

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        """Display as HTML in IPython notebook."""
        return self.as_html()

    def add_table_2cols(self, res, title=None, gleft=None, gright=None,
                        yname=None, xname=None):
        """add a double table, 2 tables with one column merged horizontally

        Parameters
        ----------
        res : results instance
            some required information is directly taken from the result
            instance
        title : string or None
            if None, then a default title is used.
        gleft : list of tuples
            elements for the left table, tuples are (name, value) pairs
            If gleft is None, then a default table is created
        gright : list of tuples or None
            elements for the right table, tuples are (name, value) pairs
        yname : string or None
            optional name for the endogenous variable, default is "y"
        xname : list of strings or None
            optional names for the exogenous variables, default is "var_xx"
        """
        table = summary_top(res, title=title, gleft=gleft, gright=gright,
                            yname=yname, xname=xname)
        self.tables.append(table)

    def add_table_params(self, res, yname=None, xname=None, alpha=.05,
                         use_t=True):
        """create and add a table for the parameter estimates

        Parameters
        ----------
        res : results instance
            some required information is directly taken from the result
            instance
        yname : string or None
            optional name for the endogenous variable, default is "y"
        xname : list of strings or None
            optional names for the exogenous variables, default is "var_xx"
        alpha : float
            significance level for the confidence intervals
        use_t : bool
            indicator whether the p-values are based on the Student-t
            distribution (if True) or on the normal distribution (if False)
        """
        if res.params.ndim == 1:
            table = summary_params(res, yname=yname, xname=xname, alpha=alpha,
                                   use_t=use_t)
        elif res.params.ndim == 2:
            _, table = summary_params_2dflat(res, endog_names=yname,
                                             exog_names=xname,
                                             alpha=alpha, use_t=use_t)
        else:  # pragma: no cover
            raise ValueError('params has to be 1d or 2d')
        self.tables.append(table)

    def add_extra_txt(self, etext):
        """add additional text that will be added at the end in text format

        Parameters
        ----------
        etext : list[str]
            string with lines that are added to the text output.
        """
        self.extra_txt = '\n'.join(etext)

    def as_text(self):
        """return tables as string

        Returns
        -------
        txt : string
            summary tables and extra text as one string
        """
        txt = summary_return(self.tables, return_fmt='text')
        if self.extra_txt is not None:
            txt = txt + '\n\n' + self.extra_txt
        return txt

    def as_latex(self):
        """return tables as string

        Returns
        -------
        latex : string
            summary tables and extra text as string of Latex

        Notes
        -----
        This currently merges tables with different number of columns.
        It is recommended to use `as_latex_tabular` directly on the individual
        tables.
        """
        latex = summary_return(self.tables, return_fmt='latex')
        if self.extra_txt is not None:
            latex += '\n\n' + self.extra_txt.replace('\n', ' \\newline\n ')
        return latex

    def as_csv(self):
        """return tables as string

        Returns
        -------
        csv : string
            concatenated summary tables in comma delimited format
        """
        csv = summary_return(self.tables, return_fmt='csv')
        if self.extra_txt is not None:
            csv = csv + '\n\n' + self.extra_txt
        return csv

    def as_html(self):
        """return tables as string

        Returns
        -------
        html : string
            concatenated summary tables in HTML format
        """
        html = summary_return(self.tables, return_fmt='html')
        if self.extra_txt is not None:
            html = html + '<br/><br/>' + self.extra_txt.replace('\n', '<br/>')
        return html
