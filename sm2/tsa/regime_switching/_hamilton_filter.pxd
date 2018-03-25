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
                                cnp.float32_t[:,:] transition,
                                cnp.float32t[:] weighted_likelihoods,
                                cnp.float32t[:] prev_filtered_marginalized_probabilities,
                                cnp.float32t[:] conditional_likelihoods,
                                cnp.float32t[:] joint_likelihoods,
                                cnp.float32t[:] curr_predicted_joint_probabilities,
                                cnp.float32t[:] prev_filtered_joint_probabilities,
                                cnp.float32t[:] curr_filtered_joint_probabilities)
cdef dhamilton_filter_iteration(int t, int k_regimes, int order,
                                cnp.float64t[:,:] transition,
                                cnp.float64t[:] weighted_likelihoods,
                                cnp.float64t[:] prev_filtered_marginalized_probabilities,
                                cnp.float64t[:] conditional_likelihoods,
                                cnp.float64t[:] joint_likelihoods,
                                cnp.float64t[:] curr_predicted_joint_probabilities,
                                cnp.float64t[:] prev_filtered_joint_probabilities,
                                cnp.float64t[:] curr_filtered_joint_probabilities)
cdef chamilton_filter_iteration(int t, int k_regimes, int order,
                                cnp.complex64t[:,:] transition,
                                cnp.complex64t[:] weighted_likelihoods,
                                cnp.complex64t[:] prev_filtered_marginalized_probabilities,
                                cnp.complex64t[:] conditional_likelihoods,
                                cnp.complex64t[:] joint_likelihoods,
                                cnp.complex64t[:] curr_predicted_joint_probabilities,
                                cnp.complex64t[:] prev_filtered_joint_probabilities,
                                cnp.complex64t[:] curr_filtered_joint_probabilities)
cdef zhamilton_filter_iteration(int t, int k_regimes, int order,
                                cnp.complex128t[:,:] transition,
                                cnp.complex128t[:] weighted_likelihoods,
                                cnp.complex128t[:] prev_filtered_marginalized_probabilities,
                                cnp.complex128t[:] conditional_likelihoods,
                                cnp.complex128t[:] joint_likelihoods,
                                cnp.complex128t[:] curr_predicted_joint_probabilities,
                                cnp.complex128t[:] prev_filtered_joint_probabilities,
                                cnp.complex128t[:] curr_filtered_joint_probabilities)
