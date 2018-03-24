#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
Kim smoother

Author: Chad Fulton  
License: Simplified-BSD
"""
import warnings

cimport cython

import numpy as np
cimport numpy as cnp

cdef int FORTRAN = 1

cpdef ckim_smoother(int nobs, int k_regimes, int order,
                             cnp.complex64_t [:, :, :] regime_transition,
                             cnp.complex64_t [:, :] predicted_joint_probabilities,
                             cnp.complex64_t [:, :] filtered_joint_probabilities,
                             cnp.complex64_t [:, :] smoothed_joint_probabilities):
    cdef int t, i, j, k, ix, regime_transition_t = 0, time_varying_regime_transition
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        int k_regimes_order_p2 = k_regimes**(order + 2)
        cnp.complex64_t [:] tmp_joint_probabilities, tmp_probabilities_fraction

    time_varying_regime_transition = regime_transition.shape[2] > 1
    tmp_joint_probabilities = np.zeros(k_regimes_order_p2, dtype=np.complex64)
    tmp_probabilities_fraction = np.zeros(k_regimes_order_p1, dtype=np.complex64)

    # S_T, S_{T-1}, ..., S_{T-r} | T
    smoothed_joint_probabilities[:, nobs - 1] = filtered_joint_probabilities[:, nobs - 1]

    for t in range(nobs - 2, -1, -1):
        if time_varying_regime_transition:
            regime_transition_t = t + 1

        ckim_smoother_iteration(k_regimes, order,
                                         tmp_joint_probabilities,
                                         tmp_probabilities_fraction,
                                         regime_transition[:, :, regime_transition_t],
                                         predicted_joint_probabilities[:, t + 1],
                                         filtered_joint_probabilities[:, t],
                                         smoothed_joint_probabilities[:, t + 1],
                                         smoothed_joint_probabilities[:, t])


cdef ckim_smoother_iteration(int k_regimes, int order,
                             cnp.complex64_t [:] tmp_joint_probabilities,
                             cnp.complex64_t [:] tmp_probabilities_fraction,
                             cnp.complex64_t [:, :] regime_transition,
                             cnp.complex64_t [:] predicted_joint_probabilities,
                             cnp.complex64_t [:] filtered_joint_probabilities,
                             cnp.complex64_t [:] prev_smoothed_joint_probabilities,
                             cnp.complex64_t [:] next_smoothed_joint_probabilities):
    cdef int t, i, j, k, ix
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        int k_regimes_order_p2 = k_regimes**(order + 2)

    # Pr[S_{t+1}, S_t, ..., S_{t-r+1} | t] = Pr[S_{t+1} | S_t] * Pr[S_t, ..., S_{t-r+1} | t]
    ix = 0
    for i in range(k_regimes):
        for j in range(k_regimes):
            for k in range(k_regimes_order):
                tmp_joint_probabilities[ix] = (
                    filtered_joint_probabilities[j * k_regimes_order + k] *
                    regime_transition[i, j])
                ix += 1

    # S_{t+1}, S_t, ..., S_{t-r+2} | T / S_{t+1}, S_t, ..., S_{t-r+2} | t
    for i in range(k_regimes_order_p1):
        if predicted_joint_probabilities[i] == 0:
            tmp_probabilities_fraction[i] = np.inf
        else:
            tmp_probabilities_fraction[i] = (
                prev_smoothed_joint_probabilities[i] /
                predicted_joint_probabilities[i])

    # S_{t+1}, S_t, ..., S_{t-r+1} | T
    ix = 0
    for i in range(k_regimes_order_p1):
        for j in range(k_regimes):
            tmp_joint_probabilities[ix] = (
                tmp_probabilities_fraction[i] *
                tmp_joint_probabilities[ix])
            ix = ix + 1

    for i in range(k_regimes_order_p1):
        for j in range(k_regimes):
            ix = j * k_regimes_order_p1 + i
            next_smoothed_joint_probabilities[i] = (
                next_smoothed_joint_probabilities[i] +
                tmp_joint_probabilities[ix])

cpdef skim_smoother(int nobs, int k_regimes, int order,
                             cnp.float32_t [:, :, :] regime_transition,
                             cnp.float32_t [:, :] predicted_joint_probabilities,
                             cnp.float32_t [:, :] filtered_joint_probabilities,
                             cnp.float32_t [:, :] smoothed_joint_probabilities):
    cdef int t, i, j, k, ix, regime_transition_t = 0, time_varying_regime_transition
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        int k_regimes_order_p2 = k_regimes**(order + 2)
        cnp.float32_t [:] tmp_joint_probabilities, tmp_probabilities_fraction

    time_varying_regime_transition = regime_transition.shape[2] > 1
    tmp_joint_probabilities = np.zeros(k_regimes_order_p2, dtype=np.float32)
    tmp_probabilities_fraction = np.zeros(k_regimes_order_p1, dtype=np.float32)

    # S_T, S_{T-1}, ..., S_{T-r} | T
    smoothed_joint_probabilities[:, nobs - 1] = filtered_joint_probabilities[:, nobs - 1]

    for t in range(nobs - 2, -1, -1):
        if time_varying_regime_transition:
            regime_transition_t = t + 1

        skim_smoother_iteration(k_regimes, order,
                                         tmp_joint_probabilities,
                                         tmp_probabilities_fraction,
                                         regime_transition[:, :, regime_transition_t],
                                         predicted_joint_probabilities[:, t + 1],
                                         filtered_joint_probabilities[:, t],
                                         smoothed_joint_probabilities[:, t + 1],
                                         smoothed_joint_probabilities[:, t])


cdef skim_smoother_iteration(int k_regimes, int order,
                             cnp.float32_t [:] tmp_joint_probabilities,
                             cnp.float32_t [:] tmp_probabilities_fraction,
                             cnp.float32_t [:, :] regime_transition,
                             cnp.float32_t [:] predicted_joint_probabilities,
                             cnp.float32_t [:] filtered_joint_probabilities,
                             cnp.float32_t [:] prev_smoothed_joint_probabilities,
                             cnp.float32_t [:] next_smoothed_joint_probabilities):
    cdef int t, i, j, k, ix
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        int k_regimes_order_p2 = k_regimes**(order + 2)

    # Pr[S_{t+1}, S_t, ..., S_{t-r+1} | t] = Pr[S_{t+1} | S_t] * Pr[S_t, ..., S_{t-r+1} | t]
    ix = 0
    for i in range(k_regimes):
        for j in range(k_regimes):
            for k in range(k_regimes_order):
                tmp_joint_probabilities[ix] = (
                    filtered_joint_probabilities[j * k_regimes_order + k] *
                    regime_transition[i, j])
                ix += 1

    # S_{t+1}, S_t, ..., S_{t-r+2} | T / S_{t+1}, S_t, ..., S_{t-r+2} | t
    for i in range(k_regimes_order_p1):
        if predicted_joint_probabilities[i] == 0:
            tmp_probabilities_fraction[i] = np.inf
        else:
            tmp_probabilities_fraction[i] = (
                prev_smoothed_joint_probabilities[i] /
                predicted_joint_probabilities[i])

    # S_{t+1}, S_t, ..., S_{t-r+1} | T
    ix = 0
    for i in range(k_regimes_order_p1):
        for j in range(k_regimes):
            tmp_joint_probabilities[ix] = (
                tmp_probabilities_fraction[i] *
                tmp_joint_probabilities[ix])
            ix = ix + 1

    for i in range(k_regimes_order_p1):
        for j in range(k_regimes):
            ix = j * k_regimes_order_p1 + i
            next_smoothed_joint_probabilities[i] = (
                next_smoothed_joint_probabilities[i] +
                tmp_joint_probabilities[ix])

cpdef zkim_smoother(int nobs, int k_regimes, int order,
                             cnp.complex128_t [:, :, :] regime_transition,
                             cnp.complex128_t [:, :] predicted_joint_probabilities,
                             cnp.complex128_t [:, :] filtered_joint_probabilities,
                             cnp.complex128_t [:, :] smoothed_joint_probabilities):
    cdef int t, i, j, k, ix, regime_transition_t = 0, time_varying_regime_transition
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        int k_regimes_order_p2 = k_regimes**(order + 2)
        cnp.complex128_t [:] tmp_joint_probabilities, tmp_probabilities_fraction

    time_varying_regime_transition = regime_transition.shape[2] > 1
    tmp_joint_probabilities = np.zeros(k_regimes_order_p2, dtype=complex)
    tmp_probabilities_fraction = np.zeros(k_regimes_order_p1, dtype=complex)

    # S_T, S_{T-1}, ..., S_{T-r} | T
    smoothed_joint_probabilities[:, nobs - 1] = filtered_joint_probabilities[:, nobs - 1]

    for t in range(nobs - 2, -1, -1):
        if time_varying_regime_transition:
            regime_transition_t = t + 1

        zkim_smoother_iteration(k_regimes, order,
                                         tmp_joint_probabilities,
                                         tmp_probabilities_fraction,
                                         regime_transition[:, :, regime_transition_t],
                                         predicted_joint_probabilities[:, t + 1],
                                         filtered_joint_probabilities[:, t],
                                         smoothed_joint_probabilities[:, t + 1],
                                         smoothed_joint_probabilities[:, t])


cdef zkim_smoother_iteration(int k_regimes, int order,
                             cnp.complex128_t [:] tmp_joint_probabilities,
                             cnp.complex128_t [:] tmp_probabilities_fraction,
                             cnp.complex128_t [:, :] regime_transition,
                             cnp.complex128_t [:] predicted_joint_probabilities,
                             cnp.complex128_t [:] filtered_joint_probabilities,
                             cnp.complex128_t [:] prev_smoothed_joint_probabilities,
                             cnp.complex128_t [:] next_smoothed_joint_probabilities):
    cdef int t, i, j, k, ix
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        int k_regimes_order_p2 = k_regimes**(order + 2)

    # Pr[S_{t+1}, S_t, ..., S_{t-r+1} | t] = Pr[S_{t+1} | S_t] * Pr[S_t, ..., S_{t-r+1} | t]
    ix = 0
    for i in range(k_regimes):
        for j in range(k_regimes):
            for k in range(k_regimes_order):
                tmp_joint_probabilities[ix] = (
                    filtered_joint_probabilities[j * k_regimes_order + k] *
                    regime_transition[i, j])
                ix += 1

    # S_{t+1}, S_t, ..., S_{t-r+2} | T / S_{t+1}, S_t, ..., S_{t-r+2} | t
    for i in range(k_regimes_order_p1):
        if predicted_joint_probabilities[i] == 0:
            tmp_probabilities_fraction[i] = np.inf
        else:
            tmp_probabilities_fraction[i] = (
                prev_smoothed_joint_probabilities[i] /
                predicted_joint_probabilities[i])

    # S_{t+1}, S_t, ..., S_{t-r+1} | T
    ix = 0
    for i in range(k_regimes_order_p1):
        for j in range(k_regimes):
            tmp_joint_probabilities[ix] = (
                tmp_probabilities_fraction[i] *
                tmp_joint_probabilities[ix])
            ix = ix + 1

    for i in range(k_regimes_order_p1):
        for j in range(k_regimes):
            ix = j * k_regimes_order_p1 + i
            next_smoothed_joint_probabilities[i] = (
                next_smoothed_joint_probabilities[i] +
                tmp_joint_probabilities[ix])

cpdef dkim_smoother(int nobs, int k_regimes, int order,
                             cnp.float64_t [:, :, :] regime_transition,
                             cnp.float64_t [:, :] predicted_joint_probabilities,
                             cnp.float64_t [:, :] filtered_joint_probabilities,
                             cnp.float64_t [:, :] smoothed_joint_probabilities):
    cdef int t, i, j, k, ix, regime_transition_t = 0, time_varying_regime_transition
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        int k_regimes_order_p2 = k_regimes**(order + 2)
        cnp.float64_t [:] tmp_joint_probabilities, tmp_probabilities_fraction

    time_varying_regime_transition = regime_transition.shape[2] > 1
    tmp_joint_probabilities = np.zeros(k_regimes_order_p2, dtype=float)
    tmp_probabilities_fraction = np.zeros(k_regimes_order_p1, dtype=float)

    # S_T, S_{T-1}, ..., S_{T-r} | T
    smoothed_joint_probabilities[:, nobs - 1] = filtered_joint_probabilities[:, nobs - 1]

    for t in range(nobs - 2, -1, -1):
        if time_varying_regime_transition:
            regime_transition_t = t + 1

        dkim_smoother_iteration(k_regimes, order,
                                         tmp_joint_probabilities,
                                         tmp_probabilities_fraction,
                                         regime_transition[:, :, regime_transition_t],
                                         predicted_joint_probabilities[:, t + 1],
                                         filtered_joint_probabilities[:, t],
                                         smoothed_joint_probabilities[:, t + 1],
                                         smoothed_joint_probabilities[:, t])


cdef dkim_smoother_iteration(int k_regimes, int order,
                             cnp.float64_t [:] tmp_joint_probabilities,
                             cnp.float64_t [:] tmp_probabilities_fraction,
                             cnp.float64_t [:, :] regime_transition,
                             cnp.float64_t [:] predicted_joint_probabilities,
                             cnp.float64_t [:] filtered_joint_probabilities,
                             cnp.float64_t [:] prev_smoothed_joint_probabilities,
                             cnp.float64_t [:] next_smoothed_joint_probabilities):
    cdef int t, i, j, k, ix
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        int k_regimes_order_p2 = k_regimes**(order + 2)

    # Pr[S_{t+1}, S_t, ..., S_{t-r+1} | t] = Pr[S_{t+1} | S_t] * Pr[S_t, ..., S_{t-r+1} | t]
    ix = 0
    for i in range(k_regimes):
        for j in range(k_regimes):
            for k in range(k_regimes_order):
                tmp_joint_probabilities[ix] = (
                    filtered_joint_probabilities[j * k_regimes_order + k] *
                    regime_transition[i, j])
                ix += 1

    # S_{t+1}, S_t, ..., S_{t-r+2} | T / S_{t+1}, S_t, ..., S_{t-r+2} | t
    for i in range(k_regimes_order_p1):
        if predicted_joint_probabilities[i] == 0:
            tmp_probabilities_fraction[i] = np.inf
        else:
            tmp_probabilities_fraction[i] = (
                prev_smoothed_joint_probabilities[i] /
                predicted_joint_probabilities[i])

    # S_{t+1}, S_t, ..., S_{t-r+1} | T
    ix = 0
    for i in range(k_regimes_order_p1):
        for j in range(k_regimes):
            tmp_joint_probabilities[ix] = (
                tmp_probabilities_fraction[i] *
                tmp_joint_probabilities[ix])
            ix = ix + 1

    for i in range(k_regimes_order_p1):
        for j in range(k_regimes):
            ix = j * k_regimes_order_p1 + i
            next_smoothed_joint_probabilities[i] = (
                next_smoothed_joint_probabilities[i] +
                tmp_joint_probabilities[ix])
