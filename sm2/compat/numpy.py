"""Compatibility functions for numpy versions in lib

Copied from upstream, unused functions deleted.  The remainder of this
docstring is from the upstream file.

Copied from Numpy source, under license:

Copyright (c) 2005-2015, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

* Neither the name of the NumPy Developers nor the names of any
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import
from distutils.version import LooseVersion

import numpy as np
from six import PY3


NP_LT_114 = LooseVersion(np.__version__) < LooseVersion('1.14')


def lstsq(a, b, rcond=None):
    """
    Shim that allows modern rcond setting with backward compat for NumPY
    earlier than 1.14
    """
    # GH#5153
    if NP_LT_114 and rcond is None:
        rcond = -1
    return np.linalg.lstsq(a, b, rcond=rcond)


def _bytelike_dtype_names(arr):
    # See GH#3658
    if not PY3:
        dtype = arr.dtype
        names = dtype.names
        names = [bytes(name)
                 if isinstance(name, unicode) else name  # noqa:F821
                 for name in names]
        dtype.names = names
