"""
Compatibility tools for differences between Python 2 and 3
"""
import functools
import itertools
import sys
import urllib

PY3 = (sys.version_info[0] >= 3)
PY3_2 = sys.version_info[:2] == (3, 2)


combinations = itertools.combinations

from six.moves import (range, map, zip, reduce, filter,
                       builtins, zip_longest, cStringIO, cPickle)
from six import (callable,
                 iteritems, iterkeys, itervalues,
                 next, advance_iterator,
                 BytesIO, StringIO,
                 string_types)

from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError, HTTPError
from six.moves.urllib.parse import urlencode, urljoin
from six.moves.urllib.request import urlretrieve

pickle = cPickle

from inspect import getargspec  # TODO: get rid of upstream py32? shim?

if PY3:
    from collections import namedtuple

    import io
    str = str
    asunicode = lambda x, _ : str(x)

    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('latin1')

    def asstr(s):
        if isinstance(s, str):
            return s
        return s.decode('latin1')

    def asstr2(s):  #added JP, not in numpy version
        if isinstance(s, str):
            return s
        elif isinstance(s, bytes):
            return s.decode('latin1')
        else:
            return str(s)

    def isfileobj(f):
        return isinstance(f, io.FileIO)

    def open_latin1(filename, mode='r'):
        return open(filename, mode=mode, encoding='iso-8859-1')

    strchar = 'U'

    # have to explicitly put builtins into the namespace
    long = int
    unichr = chr

    # list-producing versions of the major Python iterating functions
    def lrange(*args, **kwargs):
        return list(range(*args, **kwargs))

    def lzip(*args, **kwargs):
        return list(zip(*args, **kwargs))

    def lmap(*args, **kwargs):
        return list(map(*args, **kwargs))

    def lfilter(*args, **kwargs):
        return list(filter(*args, **kwargs))

    input = input

else:

    pickle = cPickle

    str = str
    asbytes = str
    asstr = str
    asstr2 = str
    strchar = 'S'

    def isfileobj(f):
        return isinstance(f, file)

    def asunicode(s, encoding='ascii'):
        if isinstance(s, unicode):
            return s
        return s.decode(encoding)

    def open_latin1(filename, mode='r'):
        return open(filename, mode=mode)

    # import iterator versions of these functions
    long = long
    unichr = unichr

    # Python 2-builtin ranges produce lists
    lrange = builtins.range
    lzip = builtins.zip
    lmap = builtins.map
    lfilter = builtins.filter

    input = raw_input


def getexception():
    return sys.exc_info()[1]


def asbytes_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, str)):
        return [asbytes_nested(y) for y in x]
    else:
        return asbytes(x)


def asunicode_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, str)):
        return [asunicode_nested(y) for y in x]
    else:
        return asunicode(x)


def get_function_name(func):
    try:
        return func.im_func.func_name
    except AttributeError:
        # Python 3
        return func.__name__
