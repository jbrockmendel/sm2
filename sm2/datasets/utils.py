import os
import shutil

from six import PY3, BytesIO, StringIO, integer_types
from six.moves import range, cPickle
from six.moves.urllib.error import URLError, HTTPError
from six.moves.urllib.request import urlopen
from six.moves.urllib_parse import urljoin

import numpy as np
import pandas as pd

from sm2.compat.numpy import recarray_select


def webuse(data, baseurl='http://www.stata-press.com/data/r11/', as_df=True):
    """
    Download and return an example dataset from Stata.

    Parameters
    ----------
    data : str
        Name of dataset to fetch.
    baseurl : str
        The base URL to the stata datasets.
    as_df : bool
        If True, returns a `pandas.DataFrame`

    Returns
    -------
    dta : Record Array
        A record array containing the Stata dataset.

    Examples
    --------
    >>> dta = webuse('auto')

    Notes
    -----
    Make sure baseurl has trailing forward slash. Doesn't do any
    error checking in response URLs.
    """
    url = urljoin(baseurl, data + '.dta')
    dta = urlopen(url)
    content = dta.read()
    dta.close()
    dta = BytesIO(content)  # make it truly file-like
    df = pd.read_stata(dta)
    if as_df:
        return df
    else:
        return df.to_records(index=False)


class Dataset(dict):
    def __init__(self, **kw):
        # define some default attributes, so pylint can find them
        self.endog = None
        self.exog = None
        self.data = None
        self.names = None

        dict.__init__(self, kw)
        self.__dict__ = self
        # Some datasets have string variables. If you want a raw_data
        # attribute you must create this in the dataset's load function.
        try:  # some datasets have string variables
            self.raw_data = self.data.view((float, len(self.names)))
        except (TypeError, ValueError, AttributeError):
            # AttributeError is for DataFrame.view
            pass

    def __repr__(self):
        return str(self.__class__)


def process_recarray(data, endog_idx=0, exog_idx=None, stack=True, dtype=None):
    names = list(data.dtype.names)

    if isinstance(endog_idx, integer_types):
        endog = np.array(data[names[endog_idx]], dtype=dtype)
        endog_name = names[endog_idx]
        endog_idx = [endog_idx]
    else:
        endog_name = [names[i] for i in endog_idx]

        if stack:
            endog = np.column_stack(data[field] for field in endog_name)
        else:
            endog = data[endog_name]

    if exog_idx is None:
        exog_name = [names[i] for i in range(len(names))
                     if i not in endog_idx]
    else:
        exog_name = [names[i] for i in exog_idx]

    if stack:
        exog = np.column_stack(data[field] for field in exog_name)
    else:
        exog = recarray_select(data, exog_name)

    if dtype:
        endog = endog.astype(dtype)
        exog = exog.astype(dtype)

    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
                      endog_name=endog_name, exog_name=exog_name)

    return dataset


def process_recarray_pandas(data, endog_idx=0, exog_idx=None, dtype=None,
                            index_idx=None):

    data = pd.DataFrame(data, dtype=dtype)
    names = data.columns

    if isinstance(endog_idx, integer_types):
        endog_name = names[endog_idx]
        endog = data[endog_name]
        if exog_idx is None:
            exog = data.drop([endog_name], axis=1)
        else:
            exog = data.filter(names[exog_idx])
    else:
        endog = data.loc[:, endog_idx]
        endog_name = list(endog.columns)
        if exog_idx is None:
            exog = data.drop(endog_name, axis=1)
        elif isinstance(exog_idx, integer_types):
            exog = data.filter([names[exog_idx]])
        else:
            exog = data.filter(names[exog_idx])

    if index_idx is not None:  # NOTE: will have to be improved for dates
        endog.index = pd.Index(data.iloc[:, index_idx])
        exog.index = pd.Index(data.iloc[:, index_idx])
        data = data.set_index(names[index_idx])

    exog_name = list(exog.columns)
    dataset = Dataset(data=data, names=list(names), endog=endog, exog=exog,
                      endog_name=endog_name, exog_name=exog_name)
    return dataset


def _maybe_reset_index(data):
    """
    All the Rdatasets have the integer row.labels from R if there is no
    real index. Strip this for a zero-based index
    """
    if data.index.equals(pd.Index(list(range(1, len(data) + 1)))):
        data = data.reset_index(drop=True)
    return data


def _get_cache(cache):
    if cache is False:
        # do not do any caching or load from cache
        cache = None
    elif cache is True:  # use default dir for cache
        cache = get_data_home(None)
    else:
        cache = get_data_home(cache)
    return cache


def _cache_it(data, cache_path):
    if PY3:
        # for some reason encode("zip") won't work for me in Python 3?
        import zlib
        # use protocol 2 so can open with python 2.x if cached in 3.x
        dumped = zlib.compress(cPickle.dumps(data, protocol=2))
    else:
        dumped = cPickle.dumps(data).encode("zip")

    with open(cache_path, "wb") as fd:
        fd.write(dumped)


def _open_cache(cache_path):
    if PY3:
        # NOTE: don't know why but decode('zip') doesn't work on my
        # Python 3 build
        import zlib
        data = zlib.decompress(open(cache_path, 'rb').read())
        # return as bytes object encoded in utf-8 for cross-compat of cached
        data = cPickle.loads(data).encode('utf-8')
    else:
        data = open(cache_path, 'rb').read().decode('zip')
        data = cPickle.loads(data)
    return data


def _urlopen_cached(url, cache):
    """
    Tries to load data from cache location otherwise downloads it. If it
    downloads the data and cache is not None then it will put the downloaded
    data in the cache path.
    """
    from_cache = False
    if cache is not None:
        fname = url.split("://")[-1].replace('/', ',') + ".zip"
        cache_path = os.path.join(cache, fname)
        try:
            data = _open_cache(cache_path)
            from_cache = True
        except OSError:  # TODO: Are there other errors to catch here?
            pass

    # not using the cache or didn't find it in cache
    if not from_cache:
        data = urlopen(url).read()
        if cache is not None:  # then put it in the cache
            _cache_it(data, cache_path)
    return data, from_cache


def _get_data(base_url, dataname, cache, extension="csv"):
    url = base_url + (dataname + ".%s") % extension
    try:
        data, from_cache = _urlopen_cached(url, cache)
    except HTTPError as err:
        if '404' in str(err):
            raise ValueError("Dataset %s was not found." % dataname)
        else:
            raise err

    data = data.decode('utf-8', 'strict')
    return StringIO(data), from_cache


def _get_dataset_meta(dataname, package, cache):
    # get the index, you'll probably want this cached because you have
    # to download info about all the data to get info about any of the data...
    index_url = ("https://raw.github.com/vincentarelbundock/Rdatasets/master/"
                 "datasets.csv")
    data, _ = _urlopen_cached(index_url, cache)
    if PY3:  # pragma: no cover
        data = data.decode('utf-8', 'strict')
    index = pd.read_csv(StringIO(data))
    idx = np.logical_and(index.Item == dataname, index.Package == package)
    dataset_meta = index.loc[idx]
    return dataset_meta["Title"].item()


def get_rdataset(dataname, package="datasets", cache=False):
    """download and return R dataset

    Parameters
    ----------
    dataname : str
        The name of the dataset you want to download
    package : str
        The package in which the dataset is found. The default is the core
        'datasets' package.
    cache : bool or str
        If True, will download this data into the STATSMODELS_DATA folder.
        The default location is a folder called sm2_data in the
        user home folder. Otherwise, you can specify a path to a folder to
        use for caching the data. If False, the data will not be cached.

    Returns
    -------
    dataset : Dataset instance
        A `sm2.data.utils.Dataset` instance. This objects has
        attributes:

        * data - A pandas DataFrame containing the data
        * title - The dataset title
        * package - The package from which the data came
        * from_cache - Whether not cached data was retrieved
        * __doc__ - The verbatim R documentation.

    Notes
    -----
    If the R dataset has an integer index. This is reset to be zero-based.
    Otherwise the index is preserved. The caching facilities are dumb. That
    is, no download dates, e-tags, or otherwise identifying information
    is checked to see if the data should be downloaded again or not. If the
    dataset is in the cache, it's used.
    """
    # NOTE: use raw github bc html site might not be most up to date
    data_base_url = ("https://raw.github.com/vincentarelbundock/Rdatasets/"
                     "master/csv/" + package + "/")
    docs_base_url = ("https://raw.github.com/vincentarelbundock/Rdatasets/"
                     "master/doc/" + package + "/rst/")
    cache = _get_cache(cache)
    data, from_cache = _get_data(data_base_url, dataname, cache)
    data = pd.read_csv(data, index_col=0)
    data = _maybe_reset_index(data)

    title = _get_dataset_meta(dataname, package, cache)
    doc, _ = _get_data(docs_base_url, dataname, cache, "rst")

    return Dataset(data=data, __doc__=doc.read(),
                   package=package, title=title,
                   from_cache=from_cache)


# The below function were taken from sklearn


def get_data_home(data_home=None):
    """Return the path of the sm2 data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'sm2_data'
    in the user home folder.

    Alternatively, it can be set by the 'STATSMODELS_DATA' environment
    variable or programatically by giving an explit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = os.environ.get('STATSMODELS_DATA',
                                   os.path.join('~', 'sm2_data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache."""
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def check_internet(url=None):
    """Check if internet is available"""
    url = "https://github.com" if url is None else url
    try:
        urlopen(url)
    except URLError:
        return False
    return True
