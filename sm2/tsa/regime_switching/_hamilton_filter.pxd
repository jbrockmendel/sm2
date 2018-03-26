#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
Hamilton filter declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as cnp

cdef shamilton_filter_iteration(int t, int k_regimes, int order,
                                cnp.float32_t[:, :] transition,
                                cnp.float32_t[:] weighted_likelihoods,
                                cnp.float32_t[:] prev_filtered_marginalized_probabilities,
                                cnp.float32_t[:] conditional_likelihoods,
                                cnp.float32_t[:] joint_likelihoods,
                                cnp.float32_t[:] curr_predicted_joint_probabilities,
                                cnp.float32_t[:] prev_filtered_joint_probabilities,
                                cnp.float32_t[:] curr_filtered_joint_probabilities)
cdef dhamilton_filter_iteration(int t, int k_regimes, int order,
                                cnp.float64_t[:, :] transition,
                                cnp.float64_t[:] weighted_likelihoods,
                                cnp.float64_t[:] prev_filtered_marginalized_probabilities,
                                cnp.float64_t[:] conditional_likelihoods,
                                cnp.float64_t[:] joint_likelihoods,
                                cnp.float64_t[:] curr_predicted_joint_probabilities,
                                cnp.float64_t[:] prev_filtered_joint_probabilities,
                                cnp.float64_t[:] curr_filtered_joint_probabilities)
cdef chamilton_filter_iteration(int t, int k_regimes, int order,
                                cnp.complex64_t[:, :] transition,
                                cnp.complex64_t[:] weighted_likelihoods,
                                cnp.complex64_t[:] prev_filtered_marginalized_probabilities,
                                cnp.complex64_t[:] conditional_likelihoods,
                                cnp.complex64_t[:] joint_likelihoods,
                                cnp.complex64_t[:] curr_predicted_joint_probabilities,
                                cnp.complex64_t[:] prev_filtered_joint_probabilities,
                                cnp.complex64_t[:] curr_filtered_joint_probabilities)
cdef zhamilton_filter_iteration(int t, int k_regimes, int order,
                                cnp.complex128_t[:, :] transition,
                                cnp.complex128_t[:] weighted_likelihoods,
                                cnp.complex128_t[:] prev_filtered_marginalized_probabilities,
                                cnp.complex128_t[:] conditional_likelihoods,
                                cnp.complex128_t[:] joint_likelihoods,
                                cnp.complex128_t[:] curr_predicted_joint_probabilities,
                                cnp.complex128_t[:] prev_filtered_joint_probabilities,
                                cnp.complex128_t[:] curr_filtered_joint_probabilities)
