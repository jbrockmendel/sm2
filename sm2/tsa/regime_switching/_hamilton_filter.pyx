#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
Hamilton filter

Author: Chad Fulton  
License: Simplified-BSD
"""
import warnings

cimport cython

import numpy as np
cimport numpy as cnp

cdef int FORTRAN = 1

def chamilton_filter(int nobs, int k_regimes, int order,
                              cnp.complex64_t [:, :, :] regime_transition,
                              cnp.complex64_t [:, :] conditional_likelihoods,
                              cnp.complex64_t [:] joint_likelihoods,
                              cnp.complex64_t [:, :] predicted_joint_probabilities,
                              cnp.complex64_t [:, :] filtered_joint_probabilities):
    cdef int t, i, j, k, ix, regime_transition_t = 0, time_varying_regime_transition
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        cnp.complex64_t [:] weighted_likelihoods, tmp_filtered_marginalized_probabilities

    time_varying_regime_transition = regime_transition.shape[2] > 1
    weighted_likelihoods = np.zeros(k_regimes_order_p1, dtype=np.complex64)
    tmp_filtered_marginalized_probabilities = np.zeros(k_regimes_order, dtype=np.complex64)

    for t in range(nobs):
        if time_varying_regime_transition:
            regime_transition_t = t

        # Collapse filtered joint probabilities over the last dimension
        # Pr[S_{t-1}, ..., S_{t-r} | t-1] = \sum_{ S_{t-r-1} } Pr[S_{t-1}, ..., S_{t-r}, S_{t-r-1} | t-1]
        ix = 0
        tmp_filtered_marginalized_probabilities[:] = 0
        for j in range(k_regimes_order):
            for i in range(k_regimes):
                tmp_filtered_marginalized_probabilities[j] = (
                    tmp_filtered_marginalized_probabilities[j] +
                    filtered_joint_probabilities[ix, t])
                ix = ix + 1

        chamilton_filter_iteration(t, k_regimes, order,
                                  regime_transition[:, :, regime_transition_t],
                                  weighted_likelihoods,
                                  tmp_filtered_marginalized_probabilities,
                                  conditional_likelihoods[:, t],
                                  joint_likelihoods,
                                  predicted_joint_probabilities[:, t],
                                  filtered_joint_probabilities[:, t],
                                  filtered_joint_probabilities[:, t + 1])


cdef chamilton_filter_iteration(int t, int k_regimes, int order,
                              cnp.complex64_t [:,:] regime_transition,
                              cnp.complex64_t [:] weighted_likelihoods,
                              cnp.complex64_t [:] prev_filtered_marginalized_probabilities,
                              cnp.complex64_t [:] conditional_likelihoods,
                              cnp.complex64_t [:] joint_likelihoods,
                              cnp.complex64_t [:] curr_predicted_joint_probabilities,
                              cnp.complex64_t [:] prev_filtered_joint_probabilities,
                              cnp.complex64_t [:] curr_filtered_joint_probabilities):
    cdef int i, j, k, ix
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)

    # Compute predicted joint probabilities
    # Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1] = Pr[S_t | S_{t-1}] * Pr[S_{t-1}, ..., S_{t-r} | t-1]
    ix = 0
    for i in range(k_regimes):
        for j in range(k_regimes):
            for k in range(k_regimes_order_m1):
                curr_predicted_joint_probabilities[ix] = (
                    prev_filtered_marginalized_probabilities[j * k_regimes_order_m1 + k] *
                    regime_transition[i, j])
                ix += 1

    # Compute weighted likelihoods f(y_t | S_t, S_{t-1}, ..., S_{t-r}, t-1) * Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1]
    # and the joint likelihood f(y_t | t-1)
    for i in range(k_regimes_order_p1):
        weighted_likelihoods[i] = (
            curr_predicted_joint_probabilities[i] *
            conditional_likelihoods[i])
        joint_likelihoods[t] = joint_likelihoods[t] + weighted_likelihoods[i]

    # Compute filtered joint probabilities
    # Pr[S_t, S_{t-1}, ..., S_{t-r} | t] = (
    #     f(y_t | S_t, S_{t-1}, ..., S_{t-r}, t-1) *
    #     Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1] /
    #     f(y_t | t-1))
    for i in range(k_regimes_order_p1):
        if joint_likelihoods[t] == 0:
            curr_filtered_joint_probabilities[i] = np.inf
        else:
            curr_filtered_joint_probabilities[i] = (
                weighted_likelihoods[i] / joint_likelihoods[t])

def shamilton_filter(int nobs, int k_regimes, int order,
                              cnp.float32_t [:, :, :] regime_transition,
                              cnp.float32_t [:, :] conditional_likelihoods,
                              cnp.float32_t [:] joint_likelihoods,
                              cnp.float32_t [:, :] predicted_joint_probabilities,
                              cnp.float32_t [:, :] filtered_joint_probabilities):
    cdef int t, i, j, k, ix, regime_transition_t = 0, time_varying_regime_transition
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        cnp.float32_t [:] weighted_likelihoods, tmp_filtered_marginalized_probabilities

    time_varying_regime_transition = regime_transition.shape[2] > 1
    weighted_likelihoods = np.zeros(k_regimes_order_p1, dtype=np.float32)
    tmp_filtered_marginalized_probabilities = np.zeros(k_regimes_order, dtype=np.float32)

    for t in range(nobs):
        if time_varying_regime_transition:
            regime_transition_t = t

        # Collapse filtered joint probabilities over the last dimension
        # Pr[S_{t-1}, ..., S_{t-r} | t-1] = \sum_{ S_{t-r-1} } Pr[S_{t-1}, ..., S_{t-r}, S_{t-r-1} | t-1]
        ix = 0
        tmp_filtered_marginalized_probabilities[:] = 0
        for j in range(k_regimes_order):
            for i in range(k_regimes):
                tmp_filtered_marginalized_probabilities[j] = (
                    tmp_filtered_marginalized_probabilities[j] +
                    filtered_joint_probabilities[ix, t])
                ix = ix + 1

        shamilton_filter_iteration(t, k_regimes, order,
                                  regime_transition[:, :, regime_transition_t],
                                  weighted_likelihoods,
                                  tmp_filtered_marginalized_probabilities,
                                  conditional_likelihoods[:, t],
                                  joint_likelihoods,
                                  predicted_joint_probabilities[:, t],
                                  filtered_joint_probabilities[:, t],
                                  filtered_joint_probabilities[:, t + 1])


cdef shamilton_filter_iteration(int t, int k_regimes, int order,
                              cnp.float32_t [:,:] regime_transition,
                              cnp.float32_t [:] weighted_likelihoods,
                              cnp.float32_t [:] prev_filtered_marginalized_probabilities,
                              cnp.float32_t [:] conditional_likelihoods,
                              cnp.float32_t [:] joint_likelihoods,
                              cnp.float32_t [:] curr_predicted_joint_probabilities,
                              cnp.float32_t [:] prev_filtered_joint_probabilities,
                              cnp.float32_t [:] curr_filtered_joint_probabilities):
    cdef int i, j, k, ix
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)

    # Compute predicted joint probabilities
    # Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1] = Pr[S_t | S_{t-1}] * Pr[S_{t-1}, ..., S_{t-r} | t-1]
    ix = 0
    for i in range(k_regimes):
        for j in range(k_regimes):
            for k in range(k_regimes_order_m1):
                curr_predicted_joint_probabilities[ix] = (
                    prev_filtered_marginalized_probabilities[j * k_regimes_order_m1 + k] *
                    regime_transition[i, j])
                ix += 1

    # Compute weighted likelihoods f(y_t | S_t, S_{t-1}, ..., S_{t-r}, t-1) * Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1]
    # and the joint likelihood f(y_t | t-1)
    for i in range(k_regimes_order_p1):
        weighted_likelihoods[i] = (
            curr_predicted_joint_probabilities[i] *
            conditional_likelihoods[i])
        joint_likelihoods[t] = joint_likelihoods[t] + weighted_likelihoods[i]

    # Compute filtered joint probabilities
    # Pr[S_t, S_{t-1}, ..., S_{t-r} | t] = (
    #     f(y_t | S_t, S_{t-1}, ..., S_{t-r}, t-1) *
    #     Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1] /
    #     f(y_t | t-1))
    for i in range(k_regimes_order_p1):
        if joint_likelihoods[t] == 0:
            curr_filtered_joint_probabilities[i] = np.inf
        else:
            curr_filtered_joint_probabilities[i] = (
                weighted_likelihoods[i] / joint_likelihoods[t])

def zhamilton_filter(int nobs, int k_regimes, int order,
                              cnp.complex128_t [:, :, :] regime_transition,
                              cnp.complex128_t [:, :] conditional_likelihoods,
                              cnp.complex128_t [:] joint_likelihoods,
                              cnp.complex128_t [:, :] predicted_joint_probabilities,
                              cnp.complex128_t [:, :] filtered_joint_probabilities):
    cdef int t, i, j, k, ix, regime_transition_t = 0, time_varying_regime_transition
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        cnp.complex128_t [:] weighted_likelihoods, tmp_filtered_marginalized_probabilities

    time_varying_regime_transition = regime_transition.shape[2] > 1
    weighted_likelihoods = np.zeros(k_regimes_order_p1, dtype=complex)
    tmp_filtered_marginalized_probabilities = np.zeros(k_regimes_order, dtype=complex)

    for t in range(nobs):
        if time_varying_regime_transition:
            regime_transition_t = t

        # Collapse filtered joint probabilities over the last dimension
        # Pr[S_{t-1}, ..., S_{t-r} | t-1] = \sum_{ S_{t-r-1} } Pr[S_{t-1}, ..., S_{t-r}, S_{t-r-1} | t-1]
        ix = 0
        tmp_filtered_marginalized_probabilities[:] = 0
        for j in range(k_regimes_order):
            for i in range(k_regimes):
                tmp_filtered_marginalized_probabilities[j] = (
                    tmp_filtered_marginalized_probabilities[j] +
                    filtered_joint_probabilities[ix, t])
                ix = ix + 1

        zhamilton_filter_iteration(t, k_regimes, order,
                                  regime_transition[:, :, regime_transition_t],
                                  weighted_likelihoods,
                                  tmp_filtered_marginalized_probabilities,
                                  conditional_likelihoods[:, t],
                                  joint_likelihoods,
                                  predicted_joint_probabilities[:, t],
                                  filtered_joint_probabilities[:, t],
                                  filtered_joint_probabilities[:, t + 1])


cdef zhamilton_filter_iteration(int t, int k_regimes, int order,
                              cnp.complex128_t [:,:] regime_transition,
                              cnp.complex128_t [:] weighted_likelihoods,
                              cnp.complex128_t [:] prev_filtered_marginalized_probabilities,
                              cnp.complex128_t [:] conditional_likelihoods,
                              cnp.complex128_t [:] joint_likelihoods,
                              cnp.complex128_t [:] curr_predicted_joint_probabilities,
                              cnp.complex128_t [:] prev_filtered_joint_probabilities,
                              cnp.complex128_t [:] curr_filtered_joint_probabilities):
    cdef int i, j, k, ix
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)

    # Compute predicted joint probabilities
    # Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1] = Pr[S_t | S_{t-1}] * Pr[S_{t-1}, ..., S_{t-r} | t-1]
    ix = 0
    for i in range(k_regimes):
        for j in range(k_regimes):
            for k in range(k_regimes_order_m1):
                curr_predicted_joint_probabilities[ix] = (
                    prev_filtered_marginalized_probabilities[j * k_regimes_order_m1 + k] *
                    regime_transition[i, j])
                ix += 1

    # Compute weighted likelihoods f(y_t | S_t, S_{t-1}, ..., S_{t-r}, t-1) * Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1]
    # and the joint likelihood f(y_t | t-1)
    for i in range(k_regimes_order_p1):
        weighted_likelihoods[i] = (
            curr_predicted_joint_probabilities[i] *
            conditional_likelihoods[i])
        joint_likelihoods[t] = joint_likelihoods[t] + weighted_likelihoods[i]

    # Compute filtered joint probabilities
    # Pr[S_t, S_{t-1}, ..., S_{t-r} | t] = (
    #     f(y_t | S_t, S_{t-1}, ..., S_{t-r}, t-1) *
    #     Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1] /
    #     f(y_t | t-1))
    for i in range(k_regimes_order_p1):
        if joint_likelihoods[t] == 0:
            curr_filtered_joint_probabilities[i] = np.inf
        else:
            curr_filtered_joint_probabilities[i] = (
                weighted_likelihoods[i] / joint_likelihoods[t])

def dhamilton_filter(int nobs, int k_regimes, int order,
                              cnp.float64_t [:, :, :] regime_transition,
                              cnp.float64_t [:, :] conditional_likelihoods,
                              cnp.float64_t [:] joint_likelihoods,
                              cnp.float64_t [:, :] predicted_joint_probabilities,
                              cnp.float64_t [:, :] filtered_joint_probabilities):
    cdef int t, i, j, k, ix, regime_transition_t = 0, time_varying_regime_transition
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)
        cnp.float64_t [:] weighted_likelihoods, tmp_filtered_marginalized_probabilities

    time_varying_regime_transition = regime_transition.shape[2] > 1
    weighted_likelihoods = np.zeros(k_regimes_order_p1, dtype=float)
    tmp_filtered_marginalized_probabilities = np.zeros(k_regimes_order, dtype=float)

    for t in range(nobs):
        if time_varying_regime_transition:
            regime_transition_t = t

        # Collapse filtered joint probabilities over the last dimension
        # Pr[S_{t-1}, ..., S_{t-r} | t-1] = \sum_{ S_{t-r-1} } Pr[S_{t-1}, ..., S_{t-r}, S_{t-r-1} | t-1]
        ix = 0
        tmp_filtered_marginalized_probabilities[:] = 0
        for j in range(k_regimes_order):
            for i in range(k_regimes):
                tmp_filtered_marginalized_probabilities[j] = (
                    tmp_filtered_marginalized_probabilities[j] +
                    filtered_joint_probabilities[ix, t])
                ix = ix + 1

        dhamilton_filter_iteration(t, k_regimes, order,
                                  regime_transition[:, :, regime_transition_t],
                                  weighted_likelihoods,
                                  tmp_filtered_marginalized_probabilities,
                                  conditional_likelihoods[:, t],
                                  joint_likelihoods,
                                  predicted_joint_probabilities[:, t],
                                  filtered_joint_probabilities[:, t],
                                  filtered_joint_probabilities[:, t + 1])


cdef dhamilton_filter_iteration(int t, int k_regimes, int order,
                              cnp.float64_t [:,:] regime_transition,
                              cnp.float64_t [:] weighted_likelihoods,
                              cnp.float64_t [:] prev_filtered_marginalized_probabilities,
                              cnp.float64_t [:] conditional_likelihoods,
                              cnp.float64_t [:] joint_likelihoods,
                              cnp.float64_t [:] curr_predicted_joint_probabilities,
                              cnp.float64_t [:] prev_filtered_joint_probabilities,
                              cnp.float64_t [:] curr_filtered_joint_probabilities):
    cdef int i, j, k, ix
    cdef:
        int k_regimes_order_m1 = k_regimes**(order - 1)
        int k_regimes_order = k_regimes**order
        int k_regimes_order_p1 = k_regimes**(order + 1)

    # Compute predicted joint probabilities
    # Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1] = Pr[S_t | S_{t-1}] * Pr[S_{t-1}, ..., S_{t-r} | t-1]
    ix = 0
    for i in range(k_regimes):
        for j in range(k_regimes):
            for k in range(k_regimes_order_m1):
                curr_predicted_joint_probabilities[ix] = (
                    prev_filtered_marginalized_probabilities[j * k_regimes_order_m1 + k] *
                    regime_transition[i, j])
                ix += 1

    # Compute weighted likelihoods f(y_t | S_t, S_{t-1}, ..., S_{t-r}, t-1) * Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1]
    # and the joint likelihood f(y_t | t-1)
    for i in range(k_regimes_order_p1):
        weighted_likelihoods[i] = (
            curr_predicted_joint_probabilities[i] *
            conditional_likelihoods[i])
        joint_likelihoods[t] = joint_likelihoods[t] + weighted_likelihoods[i]

    # Compute filtered joint probabilities
    # Pr[S_t, S_{t-1}, ..., S_{t-r} | t] = (
    #     f(y_t | S_t, S_{t-1}, ..., S_{t-r}, t-1) *
    #     Pr[S_t, S_{t-1}, ..., S_{t-r} | t-1] /
    #     f(y_t | t-1))
    for i in range(k_regimes_order_p1):
        if joint_likelihoods[t] == 0:
            curr_filtered_joint_probabilities[i] = np.inf
        else:
            curr_filtered_joint_probabilities[i] = (
                weighted_likelihoods[i] / joint_likelihoods[t])
