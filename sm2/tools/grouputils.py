# -*- coding: utf-8 -*-
"""Tools for working with groups

This provides several functions to work with groups and a Group class that
keeps track of the different representations and has methods to work more
easily with groups.


Author: Josef Perktold,
Author: Nathaniel Smith, recipe for sparse_dummies on scipy user mailing list

Created on Tue Nov 29 15:44:53 2011 : sparse_dummies
Created on Wed Nov 30 14:28:24 2011 : combine_indices
changes: add Group class

Notes
~~~~~
This reverses the class I used before, where the class was for the data and
the group was auxiliary. Here, it is only the group, no data is kept.

sparse_dummies needs checking for corner cases, e.g.
what if a category level has zero elements? This can happen with subset
    selection even if the original groups where defined as arange.

Not all methods and options have been tried out yet after refactoring

need more efficient loop if groups are sorted -> see GroupSorted.group_iter
"""
import numpy as np


# Aside from a usage in a half-written sandbox panel_short file,
# combine_indices is the only part of grouputils that is needed upstream,
# though there the call goes through the Group constructor
def combine_indices(groups, prefix='', sep='.', return_labels=False):
    """use np.unique to get integer group indices for product, intersection"""
    if return_labels:  # pragma: no cover
        raise NotImplementedError("Option `return_labels` from upstream is "
                                  "deprecated.  Only `False` is "
                                  "accepted here.")
    if isinstance(groups, tuple):
        groups = np.column_stack(groups)
    else:
        groups = np.asarray(groups)

    dt = groups.dtype

    is2d = (groups.ndim == 2)  # need to store

    if is2d:
        ncols = groups.shape[1]
        if not groups.flags.c_contiguous:
            groups = np.array(groups, order='C')

        groups_ = groups.view([('', groups.dtype)] * groups.shape[1])
    else:
        groups_ = groups

    uni, uni_idx, uni_inv = np.unique(groups_, return_index=True,
                                      return_inverse=True)

    if is2d:
        uni = uni.view(dt).reshape(-1, ncols)
        # avoiding a view would be
        # for t in uni.dtype.fields.values():
        #     assert (t[0] == dt)
        #
        # uni.dtype = dt
        # uni.shape = (uni.size//ncols, ncols)

    return uni_inv, uni_idx, uni


import sys
module = sys.modules['sm2.tools.grouputils']
for name in ['Group', 'Grouping', 'GroupSorted',
             'group_sums', 'group_sums_dummy', 'dummy_sparse',
             '_is_hierarchical', '_make_hierarchical_index',
             '_make_generic_names']:
    def func(*args, **kwargs):  # pragma: no cover
        """placeholder for function not ported from upstream"""
        raise NotImplementedError("{name} not ported from upstream"
                                  .format(name=name))

    func.__name__ = name
    setattr(module, name, func)
