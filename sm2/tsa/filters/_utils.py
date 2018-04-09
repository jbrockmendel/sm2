
from sm2.tools.data import _is_using_pandas


def _get_pandas_wrapper(X, trim_head=None, trim_tail=None, names=None):
    index = X.index
    # TODO: allow use index labels
    if trim_head is None and trim_tail is None:
        index = index
    elif trim_tail is None:
        index = index[trim_head:]
    elif trim_head is None:
        index = index[:-trim_tail]
    else:
        index = index[trim_head:-trim_tail]

    if hasattr(X, "columns"):
        if names is None:
            names = X.columns
        return lambda x: X.__class__(x, index=index, columns=names)
    else:
        if names is None:
            names = X.name
        return lambda x: X.__class__(x, index=index, name=names)


def _maybe_get_pandas_wrapper(X, trim_head=None, trim_tail=None):
    """
    If using pandas returns a function to wrap the results, e.g., wrapper(X)
    trim is an integer for the symmetric truncation of the series in some
    filters.
    otherwise returns None
    """
    if _is_using_pandas(X, None):
        return _get_pandas_wrapper(X, trim_head, trim_tail)
    else:
        return lambda x: x


# only used in tsa.seasonal, which is not yet ported.  TODO: un-port?
def _maybe_get_pandas_wrapper_freq(X, trim=None):  # pragma: no cover
    raise NotImplementedError("_maybe_get_pandas_wrapper not ported from "
                              "upstream, as it is only used in tsa.seasonal, "
                              "which is not (yet) ported.")


def pandas_wrapper(func, trim_head=None, trim_tail=None, names=None, *args,
                   **kwargs):  # pragma: no cover
    raise NotImplementedError("pandas_wrapper not ported from upstream "
                              "as it is neither used nor tested there.")


def pandas_wrapper_bunch(func, trim_head=None, trim_tail=None,
                         names=None, *args, **kwargs):  # pragma: no cover
    raise NotImplementedError("pandas_wrapper_bunch not ported from upstream "
                              "as it is neither used nor tested there.")


def pandas_wrapper_predict(func, trim_head=None, trim_tail=None,
                           columns=None, *args, **kwargs):  # pragma: no cover
    raise NotImplementedError("pandas_wrapper_predict not ported from upstream "
                              "as it is neither used nor tested there.")


def pandas_wrapper_freq(func, trim_head=None, trim_tail=None,
                        freq_kw='freq', columns=None, *args, **kwargs):
    raise NotImplementedError("pandas_wrapper_freq not ported from upstream "
                              "as it is neither used nor tested there.")
