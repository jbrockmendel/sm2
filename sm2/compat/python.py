"""
Compatibility tools for differences between Python 2 and 3
"""
import sys

PY3 = (sys.version_info[0] >= 3)
PY3_2 = sys.version_info[:2] == (3, 2)

if PY3:
    def asstr(s):
        if isinstance(s, str):
            return s
        return s.decode('latin1')

    def asstr2(s):  # added JP, not in numpy version
        if isinstance(s, str):
            return s
        elif isinstance(s, bytes):
            return s.decode('latin1')
        else:
            return str(s)

else:
    asstr = str
    asstr2 = str
