"""
Handle file opening for read/write
"""
import gzip

from numpy.lib._iotools import _is_string_like
from six import PY3


class EmptyContextManager(object):
    """
    This class is needed to allow file-like object to be used as
    context manager, but without getting closed.
    """
    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        """When entering, return the embedded object"""
        return self._obj

    def __exit__(self, *args):
        """Don't hide anything"""
        return False

    def __getattr__(self, name):  # pragma: no cover
        raise NotImplementedError("__getattr__ not ported from upstream, "
                                  "as it is neither used nor tested.")


def _open(fname, mode, encoding):
    kwargs = {'encoding': encoding} if PY3 else {}
    if fname.endswith('.gz'):
        return gzip.open(fname, mode, **kwargs)
    else:
        return open(fname, mode, **kwargs)


def get_file_obj(fname, mode='r', encoding=None):
    """
    Light wrapper to handle strings and let files (anything else) pass through.

    It also handle '.gz' files.

    Parameters
    ----------
    fname: string or file-like object
        File to open / forward
    mode: string
        Argument passed to the 'open' or 'gzip.open' function
    encoding: string
        For Python 3 only, specify the encoding of the file

    Returns
    -------
    A file-like object that is always a context-manager.  If the `fname`
    was already a file-like object, the returned context manager
    *will not close the file*.
    """
    if _is_string_like(fname):
        return _open(fname, mode, encoding)
    try:
        # Make sure the object has the write methods
        if 'r' in mode:
            assert hasattr(fname, 'read')
        if 'w' in mode or 'a' in mode:
            assert hasattr(fname, 'write')
    except AssertionError:  # pragma: no cover
        raise ValueError('fname must be a string or a file-like object')
    return EmptyContextManager(fname)
