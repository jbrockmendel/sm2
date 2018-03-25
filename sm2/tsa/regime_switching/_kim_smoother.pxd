#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
Kim smoother declarations

Author: Chad Fulton  
License: Simplified-BSD
"""
cimport numpy as cnp

cpdef skim_smoother(int nobs, int k_regimes, int order,
                    cnp.float32t[:, :, :] regime_transition,
                    cnp.float32t[:, :] predicted_joint_probabilities,
                    cnp.float32t[:, :] filtered_joint_probabilities,
                    cnp.float32t[:, :] smoothed_joint_probabilities)

cpdef dkim_smoother(int nobs, int k_regimes, int order,
                    cnp.float64t[:, :, :] regime_transition,
                    cnp.float64t[:, :] predicted_joint_probabilities,
                    cnp.float64t[:, :] filtered_joint_probabilities,
                    cnp.float64t[:, :] smoothed_joint_probabilities)

cpdef ckim_smoother(int nobs, int k_regimes, int order,
                    cnp.complex64t[:, :, :] regime_transition,
                    cnp.complex64t[:, :] predicted_joint_probabilities,
                    cnp.complex64t[:, :] filtered_joint_probabilities,
                    cnp.complex64t[:, :] smoothed_joint_probabilities)

cpdef zkim_smoother(int nobs, int k_regimes, int order,
                    cnp.complex128t[:, :, :] regime_transition,
                    cnp.complex128t[:, :] predicted_joint_probabilities,
                    cnp.complex128t[:, :] filtered_joint_probabilities,
                    cnp.complex128t[:, :] smoothed_joint_probabilities)

cdef skim_smoother_iteration(int k_regimes, int order,
                    cnp.float32t[:] tmp_joint_probabilities,
                    cnp.float32t[:] tmp_probabilities_fraction,
                    cnp.float32t[:, :] regime_transition,
                    cnp.float32t[:] predicted_joint_probabilities,
                    cnp.float32t[:] filtered_joint_probabilities,
                    cnp.float32t[:] prev_smoothed_joint_probabilities,
                    cnp.float32t[:] next_smoothed_joint_probabilities)

cdef dkim_smoother_iteration(int k_regimes, int order,
                    cnp.float64t[:] tmp_joint_probabilities,
                    cnp.float64t[:] tmp_probabilities_fraction,
                    cnp.float64t[:, :] regime_transition,
                    cnp.float64t[:] predicted_joint_probabilities,
                    cnp.float64t[:] filtered_joint_probabilities,
                    cnp.float64t[:] prev_smoothed_joint_probabilities,
                    cnp.float64t[:] next_smoothed_joint_probabilities)

cdef ckim_smoother_iteration(int k_regimes, int order,
                    cnp.complex64t[:] tmp_joint_probabilities,
                    cnp.complex64t[:] tmp_probabilities_fraction,
                    cnp.complex64t[:, :] regime_transition,
                    cnp.complex64t[:] predicted_joint_probabilities,
                    cnp.complex64t[:] filtered_joint_probabilities,
                    cnp.complex64t[:] prev_smoothed_joint_probabilities,
                    cnp.complex64t[:] next_smoothed_joint_probabilities)

cdef zkim_smoother_iteration(int k_regimes, int order,
                    cnp.complex128t[:] tmp_joint_probabilities,
                    cnp.complex128t[:] tmp_probabilities_fraction,
                    cnp.complex128t[:, :] regime_transition,
                    cnp.complex128t[:] predicted_joint_probabilities,
                    cnp.complex128t[:] filtered_joint_probabilities,
                    cnp.complex128t[:] prev_smoothed_joint_probabilities,
                    cnp.complex128t[:] next_smoothed_joint_probabilities)
