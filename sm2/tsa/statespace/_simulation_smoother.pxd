#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Model Smoother declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cdef int SMOOTHER_STATE           # Durbin and Koopman (2012), Chapter 4.4.2
cdef int SMOOTHER_STATE_COV       # Durbin and Koopman (2012), Chapter 4.4.3
cdef int SMOOTHER_DISTURBANCE     # Durbin and Koopman (2012), Chapter 4.5
cdef int SMOOTHER_DISTURBANCE_COV # Durbin and Koopman (2012), Chapter 4.5
cdef int SMOOTHER_ALL

# Typical imports
cimport numpy as cnp

from sm2.tsa.statespace._representation cimport (
    sStatespace, dStatespace, cStatespace, zStatespace)

from sm2.tsa.statespace._kalman_filter cimport (
    sKalmanFilter, dKalmanFilter, cKalmanFilter, zKalmanFilter)

from sm2.tsa.statespace._kalman_smoother cimport (
    sKalmanSmoother, dKalmanSmoother, cKalmanSmoother, zKalmanSmoother)


# Single precision
cdef class sSimulationSmoother(object):
    # ### Statespace model
    cdef readonly sStatespace model
    # ### Kalman filter
    cdef readonly sKalmanFilter kfilter
    # ### Kalman smoother
    cdef readonly sKalmanSmoother smoother

    # ### Simulated Statespace model
    cdef readonly sStatespace simulated_model
    # ### Simulated Kalman filter
    cdef readonly sKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly sKalmanSmoother simulated_smoother

    # ### Simulated Statespace model
    cdef readonly sStatespace secondary_simulated_model
    # ### Simulated Kalman filter
    cdef readonly sKalmanFilter secondary_simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly sKalmanSmoother secondary_simulated_smoother

    # ### Simulation parameters
    cdef public int simulation_output
    cdef public int has_missing

    # ### Random variates
    cdef int n_disturbance_variates
    cdef readonly cnp.float32_t [:] disturbance_variates
    cdef int n_initial_state_variates
    cdef readonly cnp.float32_t [:] initial_state_variates

    # ### Simulated Data
    cdef readonly cnp.float32_t [::1,:] simulated_measurement_disturbance
    cdef readonly cnp.float32_t [::1,:] simulated_state_disturbance
    cdef readonly cnp.float32_t [::1,:] simulated_state

    # ### Generated Data
    cdef readonly cnp.float32_t [::1,:] generated_obs
    cdef readonly cnp.float32_t [::1,:] generated_state

    # ### Temporary arrays
    cdef readonly cnp.float32_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    cdef cnp.float32_t * _tmp0
    cdef cnp.float32_t * _tmp1
    cdef cnp.float32_t * _tmp2

    # ### Parameters
    cdef readonly int nobs
    cdef readonly int pretransformed_disturbance_variates
    cdef readonly int pretransformed_initial_state_variates
    cdef readonly int fixed_initial_state

    cpdef draw_disturbance_variates(self)
    cpdef draw_initial_state_variates(self)
    cpdef set_disturbance_variates(self, cnp.float32_t [:] variates, int pretransformed=*)
    cpdef set_initial_state_variates(self, cnp.float32_t [:] variates, int pretransformed=*)
    cpdef set_initial_state(self, cnp.float32_t [:] initial_state)
    cpdef simulate(self, int simulation_output=*)

    cdef cnp.float32_t generate_obs(self, int t, cnp.float32_t * obs, cnp.float32_t * state, cnp.float32_t * variates)
    cdef cnp.float32_t generate_state(self, int t, cnp.float32_t * state, cnp.float32_t * input_state, cnp.float32_t * variates)
    cdef void cholesky(self, cnp.float32_t * source, cnp.float32_t * destination, int n)
    cdef void transform_variates(self, cnp.float32_t * variates, cnp.float32_t * cholesky_factor, int n)
    cdef void _reinitialize_temp_pointers(self) except *

# Double precision
cdef class dSimulationSmoother(object):
    # ### Statespace model
    cdef readonly dStatespace model
    # ### Kalman filter
    cdef readonly dKalmanFilter kfilter
    # ### Kalman smoother
    cdef readonly dKalmanSmoother smoother

    # ### Simulated Statespace model
    cdef readonly dStatespace simulated_model
    # ### Simulated Kalman filter
    cdef readonly dKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly dKalmanSmoother simulated_smoother

    # ### Simulated Statespace model
    cdef readonly dStatespace secondary_simulated_model
    # ### Simulated Kalman filter
    cdef readonly dKalmanFilter secondary_simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly dKalmanSmoother secondary_simulated_smoother

    # ### Simulation parameters
    cdef public int simulation_output
    cdef public int has_missing

    # ### Random variates
    cdef int n_disturbance_variates
    cdef readonly cnp.float64_t [:] disturbance_variates
    cdef int n_initial_state_variates
    cdef readonly cnp.float64_t [:] initial_state_variates

    # ### Simulated Data
    cdef readonly cnp.float64_t [::1,:] simulated_measurement_disturbance
    cdef readonly cnp.float64_t [::1,:] simulated_state_disturbance
    cdef readonly cnp.float64_t [::1,:] simulated_state

    # ### Generated Data
    cdef readonly cnp.float64_t [::1,:] generated_obs
    cdef readonly cnp.float64_t [::1,:] generated_state

    # ### Temporary arrays
    cdef readonly cnp.float64_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    cdef cnp.float64_t * _tmp0
    cdef cnp.float64_t * _tmp1
    cdef cnp.float64_t * _tmp2

    # ### Parameters
    cdef readonly int nobs
    cdef readonly int pretransformed_disturbance_variates
    cdef readonly int pretransformed_initial_state_variates
    cdef readonly int fixed_initial_state

    cpdef draw_disturbance_variates(self)
    cpdef draw_initial_state_variates(self)
    cpdef set_disturbance_variates(self, cnp.float64_t [:] variates, int pretransformed=*)
    cpdef set_initial_state_variates(self, cnp.float64_t [:] variates, int pretransformed=*)
    cpdef set_initial_state(self, cnp.float64_t [:] initial_state)
    cpdef simulate(self, int simulation_output=*)

    cdef cnp.float64_t generate_obs(self, int t, cnp.float64_t * obs, cnp.float64_t * state, cnp.float64_t * variates)
    cdef cnp.float64_t generate_state(self, int t, cnp.float64_t * state, cnp.float64_t * input_state, cnp.float64_t * variates)
    cdef void cholesky(self, cnp.float64_t * source, cnp.float64_t * destination, int n)
    cdef void transform_variates(self, cnp.float64_t * variates, cnp.float64_t * cholesky_factor, int n)
    cdef void _reinitialize_temp_pointers(self) except *

# Single precision complex
cdef class cSimulationSmoother(object):
    # ### Statespace model
    cdef readonly cStatespace model
    # ### Kalman filter
    cdef readonly cKalmanFilter kfilter
    # ### Kalman smoother
    cdef readonly cKalmanSmoother smoother

    # ### Simulated Statespace model
    cdef readonly cStatespace simulated_model
    # ### Simulated Kalman filter
    cdef readonly cKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly cKalmanSmoother simulated_smoother

    # ### Simulated Statespace model
    cdef readonly cStatespace secondary_simulated_model
    # ### Simulated Kalman filter
    cdef readonly cKalmanFilter secondary_simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly cKalmanSmoother secondary_simulated_smoother

    # ### Simulation parameters
    cdef public int simulation_output
    cdef public int has_missing

    # ### Random variates
    cdef int n_disturbance_variates
    cdef readonly cnp.complex64_t [:] disturbance_variates
    cdef int n_initial_state_variates
    cdef readonly cnp.complex64_t [:] initial_state_variates

    # ### Simulated Data
    cdef readonly cnp.complex64_t [::1,:] simulated_measurement_disturbance
    cdef readonly cnp.complex64_t [::1,:] simulated_state_disturbance
    cdef readonly cnp.complex64_t [::1,:] simulated_state

    # ### Generated Data
    cdef readonly cnp.complex64_t [::1,:] generated_obs
    cdef readonly cnp.complex64_t [::1,:] generated_state

    # ### Temporary arrays
    cdef readonly cnp.complex64_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    cdef cnp.complex64_t * _tmp0
    cdef cnp.complex64_t * _tmp1
    cdef cnp.complex64_t * _tmp2

    # ### Parameters
    cdef readonly int nobs
    cdef readonly int pretransformed_disturbance_variates
    cdef readonly int pretransformed_initial_state_variates
    cdef readonly int fixed_initial_state

    cpdef draw_disturbance_variates(self)
    cpdef draw_initial_state_variates(self)
    cpdef set_disturbance_variates(self, cnp.complex64_t [:] variates, int pretransformed=*)
    cpdef set_initial_state_variates(self, cnp.complex64_t [:] variates, int pretransformed=*)
    cpdef set_initial_state(self, cnp.complex64_t [:] initial_state)
    cpdef simulate(self, int simulation_output=*)

    cdef cnp.complex64_t generate_obs(self, int t, cnp.complex64_t * obs, cnp.complex64_t * state, cnp.complex64_t * variates)
    cdef cnp.complex64_t generate_state(self, int t, cnp.complex64_t * state, cnp.complex64_t * input_state, cnp.complex64_t * variates)
    cdef void cholesky(self, cnp.complex64_t * source, cnp.complex64_t * destination, int n)
    cdef void transform_variates(self, cnp.complex64_t * variates, cnp.complex64_t * cholesky_factor, int n)
    cdef void _reinitialize_temp_pointers(self) except *

# Double precision complex
cdef class zSimulationSmoother(object):
    # ### Statespace model
    cdef readonly zStatespace model
    # ### Kalman filter
    cdef readonly zKalmanFilter kfilter
    # ### Kalman smoother
    cdef readonly zKalmanSmoother smoother

    # ### Simulated Statespace model
    cdef readonly zStatespace simulated_model
    # ### Simulated Kalman filter
    cdef readonly zKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly zKalmanSmoother simulated_smoother

    # ### Simulated Statespace model
    cdef readonly zStatespace secondary_simulated_model
    # ### Simulated Kalman filter
    cdef readonly zKalmanFilter secondary_simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly zKalmanSmoother secondary_simulated_smoother

    # ### Simulation parameters
    cdef public int simulation_output
    cdef public int has_missing

    # ### Random variates
    cdef int n_disturbance_variates
    cdef readonly cnp.complex128_t [:] disturbance_variates
    cdef int n_initial_state_variates
    cdef readonly cnp.complex128_t [:] initial_state_variates

    # ### Simulated Data
    cdef readonly cnp.complex128_t [::1,:] simulated_measurement_disturbance
    cdef readonly cnp.complex128_t [::1,:] simulated_state_disturbance
    cdef readonly cnp.complex128_t [::1,:] simulated_state

    # ### Generated Data
    cdef readonly cnp.complex128_t [::1,:] generated_obs
    cdef readonly cnp.complex128_t [::1,:] generated_state

    # ### Temporary arrays
    cdef readonly cnp.complex128_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    cdef cnp.complex128_t * _tmp0
    cdef cnp.complex128_t * _tmp1
    cdef cnp.complex128_t * _tmp2

    # ### Parameters
    cdef readonly int nobs
    cdef readonly int pretransformed_disturbance_variates
    cdef readonly int pretransformed_initial_state_variates
    cdef readonly int fixed_initial_state

    cpdef draw_disturbance_variates(self)
    cpdef draw_initial_state_variates(self)
    cpdef set_disturbance_variates(self, cnp.complex128_t [:] variates, int pretransformed=*)
    cpdef set_initial_state_variates(self, cnp.complex128_t [:] variates, int pretransformed=*)
    cpdef set_initial_state(self, cnp.complex128_t [:] initial_state)
    cpdef simulate(self, int simulation_output=*)

    cdef cnp.complex128_t generate_obs(self, int t, cnp.complex128_t * obs, cnp.complex128_t * state, cnp.complex128_t * variates)
    cdef cnp.complex128_t generate_state(self, int t, cnp.complex128_t * state, cnp.complex128_t * input_state, cnp.complex128_t * variates)
    cdef void cholesky(self, cnp.complex128_t * source, cnp.complex128_t * destination, int n)
    cdef void transform_variates(self, cnp.complex128_t * variates, cnp.complex128_t * cholesky_factor, int n)
    cdef void _reinitialize_temp_pointers(self) except *