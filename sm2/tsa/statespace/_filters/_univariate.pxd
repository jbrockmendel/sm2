#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models - Conventional Kalman Filter declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as cnp
from sm2.tsa.statespace._representation cimport (
    sStatespace, dStatespace, cStatespace, zStatespace)

from sm2.tsa.statespace._kalman_filter cimport (
    sKalmanFilter, dKalmanFilter, cKalmanFilter, zKalmanFilter)

# Single precision
cdef int sforecast_univariate(sKalmanFilter kfilter, sStatespace model)
cdef int supdating_univariate(sKalmanFilter kfilter, sStatespace model)
cdef int sprediction_univariate(sKalmanFilter kfilter, sStatespace model)
cdef cnp.float32_t sinverse_noop_univariate(sKalmanFilter kfilter, sStatespace model, cnp.float32_t determinant) except *
cdef cnp.float32_t sloglikelihood_univariate(sKalmanFilter kfilter, sStatespace model, cnp.float32_t determinant)

cdef void sforecast_error(sKalmanFilter kfilter, sStatespace model, int i)
cdef cnp.float32_t sforecast_error_cov(sKalmanFilter kfilter, sStatespace model, int i)
cdef void stemp_arrays(sKalmanFilter kfilter, sStatespace model, int i, cnp.float32_t forecast_error_cov_inv)
cdef void sfiltered_state(sKalmanFilter kfilter, sStatespace model, int i, cnp.float32_t forecast_error_cov_inv)
cdef void sfiltered_state_cov(sKalmanFilter kfilter, sStatespace model, int i, cnp.float32_t forecast_error_cov_inv)
cdef void sloglikelihood(sKalmanFilter kfilter, sStatespace model, int i, cnp.float32_t forecast_error_cov, cnp.float32_t forecast_error_cov_inv)

# Double precision
cdef int dforecast_univariate(dKalmanFilter kfilter, dStatespace model)
cdef int dupdating_univariate(dKalmanFilter kfilter, dStatespace model)
cdef int dprediction_univariate(dKalmanFilter kfilter, dStatespace model)
cdef cnp.float64_t dinverse_noop_univariate(dKalmanFilter kfilter, dStatespace model, cnp.float64_t determinant) except *
cdef cnp.float64_t dloglikelihood_univariate(dKalmanFilter kfilter, dStatespace model, cnp.float64_t determinant)

cdef void dforecast_error(dKalmanFilter kfilter, dStatespace model, int i)
cdef cnp.float64_t dforecast_error_cov(dKalmanFilter kfilter, dStatespace model, int i)
cdef void dtemp_arrays(dKalmanFilter kfilter, dStatespace model, int i, cnp.float64_t forecast_error_cov_inv)
cdef void dfiltered_state(dKalmanFilter kfilter, dStatespace model, int i, cnp.float64_t forecast_error_cov_inv)
cdef void dfiltered_state_cov(dKalmanFilter kfilter, dStatespace model, int i, cnp.float64_t forecast_error_cov_inv)
cdef void dloglikelihood(dKalmanFilter kfilter, dStatespace model, int i, cnp.float64_t forecast_error_cov, cnp.float64_t forecast_error_cov_inv)

# Single precision complex
cdef int cforecast_univariate(cKalmanFilter kfilter, cStatespace model)
cdef int cupdating_univariate(cKalmanFilter kfilter, cStatespace model)
cdef int cprediction_univariate(cKalmanFilter kfilter, cStatespace model)
cdef cnp.complex64_t cinverse_noop_univariate(cKalmanFilter kfilter, cStatespace model, cnp.complex64_t determinant) except *
cdef cnp.complex64_t cloglikelihood_univariate(cKalmanFilter kfilter, cStatespace model, cnp.complex64_t determinant)

cdef void cforecast_error(cKalmanFilter kfilter, cStatespace model, int i)
cdef cnp.complex64_t cforecast_error_cov(cKalmanFilter kfilter, cStatespace model, int i)
cdef void ctemp_arrays(cKalmanFilter kfilter, cStatespace model, int i, cnp.complex64_t forecast_error_cov_inv)
cdef void cfiltered_state(cKalmanFilter kfilter, cStatespace model, int i, cnp.complex64_t forecast_error_cov_inv)
cdef void cfiltered_state_cov(cKalmanFilter kfilter, cStatespace model, int i, cnp.complex64_t forecast_error_cov_inv)
cdef void cloglikelihood(cKalmanFilter kfilter, cStatespace model, int i, cnp.complex64_t forecast_error_cov, cnp.complex64_t forecast_error_cov_inv)

# Double precision complex
cdef int zforecast_univariate(zKalmanFilter kfilter, zStatespace model)
cdef int zupdating_univariate(zKalmanFilter kfilter, zStatespace model)
cdef int zprediction_univariate(zKalmanFilter kfilter, zStatespace model)
cdef cnp.complex128_t zinverse_noop_univariate(zKalmanFilter kfilter, zStatespace model, cnp.complex128_t determinant) except *
cdef cnp.complex128_t zloglikelihood_univariate(zKalmanFilter kfilter, zStatespace model, cnp.complex128_t determinant)

cdef void zforecast_error(zKalmanFilter kfilter, zStatespace model, int i)
cdef cnp.complex128_t zforecast_error_cov(zKalmanFilter kfilter, zStatespace model, int i)
cdef void ztemp_arrays(zKalmanFilter kfilter, zStatespace model, int i, cnp.complex128_t forecast_error_cov_inv)
cdef void zfiltered_state(zKalmanFilter kfilter, zStatespace model, int i, cnp.complex128_t forecast_error_cov_inv)
cdef void zfiltered_state_cov(zKalmanFilter kfilter, zStatespace model, int i, cnp.complex128_t forecast_error_cov_inv)
cdef void zloglikelihood(zKalmanFilter kfilter, zStatespace model, int i, cnp.complex128_t forecast_error_cov, cnp.complex128_t forecast_error_cov_inv)