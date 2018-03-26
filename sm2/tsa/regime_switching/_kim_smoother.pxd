#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
Kim smoother declarations

Author: Chad Fulton  
License: Simplified-BSD
"""
from cython cimport Py_ssize_t
cimport numpy as cnp

cpdef skim_smoother(Py_ssize_t nobs, int k_regimes, int order,
                    cnp.float32_t[:, :, :] regime_transition,
                    cnp.float32_t[:, :] predicted_joint_probabilities,
                    cnp.float32_t[:, :] filtered_joint_probabilities,
                    cnp.float32_t[:, :] smoothed_joint_probabilities)

cpdef dkim_smoother(Py_ssize_t nobs, int k_regimes, int order,
                    cnp.float64_t[:, :, :] regime_transition,
                    cnp.float64_t[:, :] predicted_joint_probabilities,
                    cnp.float64_t[:, :] filtered_joint_probabilities,
                    cnp.float64_t[:, :] smoothed_joint_probabilities)

cpdef ckim_smoother(Py_ssize_t nobs, int k_regimes, int order,
                    cnp.complex64_t[:, :, :] regime_transition,
                    cnp.complex64_t[:, :] predicted_joint_probabilities,
                    cnp.complex64_t[:, :] filtered_joint_probabilities,
                    cnp.complex64_t[:, :] smoothed_joint_probabilities)

cpdef zkim_smoother(Py_ssize_t nobs, int k_regimes, int order,
                    cnp.complex128_t[:, :, :] regime_transition,
                    cnp.complex128_t[:, :] predicted_joint_probabilities,
                    cnp.complex128_t[:, :] filtered_joint_probabilities,
                    cnp.complex128_t[:, :] smoothed_joint_probabilities)

cdef skim_smoother_iteration(int k_regimes, int order,
                    cnp.float32_t[:] tmp_joint_probabilities,
                    cnp.float32_t[:] tmp_probabilities_fraction,
                    cnp.float32_t[:, :] regime_transition,
                    cnp.float32_t[:] predicted_joint_probabilities,
                    cnp.float32_t[:] filtered_joint_probabilities,
                    cnp.float32_t[:] prev_smoothed_joint_probabilities,
                    cnp.float32_t[:] next_smoothed_joint_probabilities)

cdef dkim_smoother_iteration(int k_regimes, int order,
                    cnp.float64_t[:] tmp_joint_probabilities,
                    cnp.float64_t[:] tmp_probabilities_fraction,
                    cnp.float64_t[:, :] regime_transition,
                    cnp.float64_t[:] predicted_joint_probabilities,
                    cnp.float64_t[:] filtered_joint_probabilities,
                    cnp.float64_t[:] prev_smoothed_joint_probabilities,
                    cnp.float64_t[:] next_smoothed_joint_probabilities)

cdef ckim_smoother_iteration(int k_regimes, int order,
                    cnp.complex64_t[:] tmp_joint_probabilities,
                    cnp.complex64_t[:] tmp_probabilities_fraction,
                    cnp.complex64_t[:, :] regime_transition,
                    cnp.complex64_t[:] predicted_joint_probabilities,
                    cnp.complex64_t[:] filtered_joint_probabilities,
                    cnp.complex64_t[:] prev_smoothed_joint_probabilities,
                    cnp.complex64_t[:] next_smoothed_joint_probabilities)

cdef zkim_smoother_iteration(int k_regimes, int order,
                    cnp.complex128_t[:] tmp_joint_probabilities,
                    cnp.complex128_t[:] tmp_probabilities_fraction,
                    cnp.complex128_t[:, :] regime_transition,
                    cnp.complex128_t[:] predicted_joint_probabilities,
                    cnp.complex128_t[:] filtered_joint_probabilities,
                    cnp.complex128_t[:] prev_smoothed_joint_probabilities,
                    cnp.complex128_t[:] next_smoothed_joint_probabilities)
