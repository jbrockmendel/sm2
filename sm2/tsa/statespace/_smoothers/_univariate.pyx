#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models

Note: there is a typo in Durbin and Koopman (2012) in the equations for the
univariate smoothed measurement disturbances and smoothed measurement
disturbance covariances. In each equation (p157), the Kalman gain vector
K_{t,i} is used, but in fact these should be multiplied by the forecast error
covariance F_{t,i}. The original paper on the univariate approach, Koopman and
Durbin (2000) has the correct form. The typo arose because the original paper
defines the Kalman gain as K_{t,i} = P_{t,i} Z_{t,i}' but the book defines it
as K_{t,i} = P_{t,i} Z_{t,i}' F_{t,i}^{-1}, and the book does not correct the
disturbances formulas for this change.

Furthermore, in analogy to the disturbance smoother from chapter 4, the
formula for the univariate covariance ought to be subtracted from the
observation covariance.

So, what we ought to have is:

\hat \varepsilon_{t,i} = \sigma_{t,i}^2 F_{t,i}^{-1} (v_{t,i} - F_{t,i} K_{t,i}' r_{t,i})
Var(\hat \varepsilon_{t,i}) = \sigma_{t,i}^2 - \sigma_{t,i}^4 F_{t,i}^{-2} (v_{t,i} - F_{t,i} K_{t,i}' r_{t,i})

Author: Chad Fulton  
License: Simplified-BSD
"""
import numpy as np
cimport numpy as cnp

cimport scipy.linalg.cython_blas as blas

from sm2.src.math cimport *
from sm2.tsa.statespace._kalman_smoother cimport (
    SMOOTHER_STATE, SMOOTHER_STATE_COV, SMOOTHER_STATE_AUTOCOV,
    SMOOTHER_DISTURBANCE, SMOOTHER_DISTURBANCE_COV
)

# ### Univariate Kalman smoother
#
# The following are the routines as defined in the univariate Kalman filter.
# 
# The only modification to the conventional Kalman smoother is the recursive
# definition of the scaled smoothing error and the scaled smoothing error
# covariance matrix.
#
# See Durbin and Koopman (2012) Chapter 6.4

cdef int csmoothed_estimators_measurement_univariate(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model) except *:
    cdef:
        int i, j, inc = 1
        cnp.complex64_t alpha = 1.0
        cnp.complex64_t beta = 0.0
        cnp.complex64_t gamma = -1.0
        cnp.complex64_t scalar
        int k_states = model._k_states
        # cnp.complex64_t [::1, :] tmpL
        # cnp.complex64_t * _tmpL

    # dim2[0] = self.kfilter.k_states; dim2[1] = self.kfilter.k_states;
    # self.tmpL = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)

    # Adjust for a VAR transition (i.e. design = [#, 0], where the zeros
    # correspond to all states except the first k_posdef states)
    if model.subset_design:
        k_states = model._k_posdef

    # Need to clear out the scaled_smoothed_estimator and
    # scaled_smoothed_estimator_cov in case we're re-running the filter
    if smoother.t == model.nobs - 1:
        smoother.scaled_smoothed_estimator[:, model.nobs-1] = 0
        smoother.scaled_smoothed_estimator_cov[:, :, model.nobs-1] = 0

    # Smoothing error  
    # (not used in the univariate approach)  

    # Given r_{t,0}:
    # calculate r_{t-1,p}, ..., r_{t-1, 0} and N_{t-1,p}, ..., N_{t-1,0}

    # Iterate
    for i in range(kfilter.k_endog-1,-1,-1):
        # If we want smoothed disturbances, then we need to calculate
        # and store K_{t,i}' r_{t,i} for later (otherwise r_{t,i} will not be
        # available)
        if smoother.smoother_output & SMOOTHER_DISTURBANCE:
            # Note: zdot and cdot are broken, so have to use gemv for those
            blas.cgemv("N", &inc, &model._k_states,
                           &alpha, smoother._scaled_smoothed_estimator, &inc,
                                   &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                           &beta, &smoother._smoothed_measurement_disturbance[i], &inc)

        # If we want smoothed disturbance covs, then we need to calculate
        # and store K_{t,i}' N_{t,i} K_{t,i} for later (otherwise N_{t,i} will not be
        # available)
        if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
            blas.cgemv("N", &model._k_states, &model._k_states,
                                     &alpha, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                                             &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                                     &beta, smoother._tmpL, &inc)
            # Note: zdot and cdot are broken, so have to use gemv for those
            blas.cgemv("N", &inc, &model._k_states,
                           &alpha, smoother._tmpL, &inc,
                                   &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                           &beta, &smoother._smoothed_measurement_disturbance_cov[i*kfilter.k_endog + i], &inc)

        # $L_{t,i} = (I_m - K_{t,i} Z_{t,i})$  
        # $(m \times m) = (m \times m) - (m \times 1) (1 \times m)$
        # blas.cgemm("N", "N", &model._k_states, &model._k_states, &inc,
        #           &gamma, &kfilter._kalman_gain[i*kfilter.k_states], &kfilter.k_states,
        #                   &model._design[i], &model._k_endog,
        #           &beta, smoother._tmpL, &kfilter.k_states)
        # Zero the temporary matrix
        # blas.cscal(&kfilter.k_states2, &beta, smoother._tmpL, &inc)
        smoother.tmpL[:,:k_states] = 0
        # Create the K_{t,i} Z_{t,i} component
        # (m x p) (m x 1) x (1 x p)
        blas.cgeru(&model._k_states, &k_states,
                  &gamma, &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                          &model._design[i], &model._k_endog,
                          smoother._tmpL, &kfilter.k_states
        )
        # Add the identity matrix
        for j in range(k_states):
            smoother._tmpL[j + j*kfilter.k_states] = smoother._tmpL[j + j*kfilter.k_states] + 1

        # Accumulate L_{t,i} into L_{t} = L_{t,n} L_{t,n-1} ... L_{t,1}
        if i == kfilter.k_endog-1:
            blas.ccopy(&kfilter.k_states2, smoother._tmpL, &inc, smoother._tmpL2, &inc)
        else:
            blas.ccopy(&kfilter.k_states2, smoother._tmpL2, &inc, smoother._tmp0, &inc)
            blas.cgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._tmpL, &kfilter.k_states,
                      &beta, smoother._tmpL2, &kfilter.k_states)

        # Scaled smoothed estimator  
        # $r_{t,i-1} = Z_{t,i}' v_{t,i} / F_{t,i} + L_{t,i}' r_{t,i}$  
        # $(m \times 1) = (m \times 1) (1 \times 1) + (m \times m) (m \times 1)$
        # Note: save $r_{t-1}$ as scaled_smoothed_estimator[t] rather than
        # as scaled_smoothed_estimator[t-1] because we actually need to store
        # T+1 of them (r_{T-1} to r_{-1} -> r_T to r_0)
        if smoother.smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
            #blas.cscal(&kfilter.k_states, &beta, smoother._tmp0, &inc)

            blas.cgemv("T", &model._k_states, &k_states,
                      &alpha, smoother._tmpL, &kfilter.k_states,
                              smoother._scaled_smoothed_estimator, &inc,
                      &beta, smoother._tmp0, &inc)
            blas.cswap(&k_states, smoother._tmp0, &inc,
                                           smoother._scaled_smoothed_estimator, &inc)
            blas.caxpy(&k_states, &kfilter._tmp2[i], &model._design[i], &model._k_endog,
                                                              smoother._scaled_smoothed_estimator, &inc)

        if smoother.smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
            # Scaled smoothed estimator covariance matrix  
            # $N_{t,i-1} = Z_{t,i}' Z_{t,i} / F_{t,i} + L_{t,i}' N_{t,i} L_{t,i}$  
            # $(m \times m) = (m \times p) (p \times m) + (m \times m) (m \times m) (m \times m)$  
            # Note: save $N_{t-1}$ as scaled_smoothed_estimator_cov[t] rather
            # than as scaled_smoothed_estimator_cov[t-1] because we actually
            # need to store T+1 of them (N_{T-1} to N_{-1} -> N_T to N_0)
            blas.cgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                              smoother._tmpL, &kfilter.k_states,
                      &beta, smoother._tmp0, &kfilter.k_states)
            blas.cgemm("T", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._tmpL, &kfilter.k_states,
                              smoother._tmp0, &kfilter.k_states,
                      &beta, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states)
            blas.cgeru(&model._k_states, &model._k_states,
                &alpha, &model._design[i], &model._k_endog,
                        &kfilter._tmp3[i], &kfilter.k_endog,
                smoother._scaled_smoothed_estimator_cov, &kfilter.k_states
            )

    # Replace L with accumulated version
    # blas.ccopy(&kfilter.k_states2, smoother._tmpL2, &inc, smoother._tmpL, &inc)
    blas.cgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, model._transition, &kfilter.k_states,
                              smoother._tmpL2, &kfilter.k_states,
                      &beta, smoother._tmpL, &kfilter.k_states)

cdef int csmoothed_estimators_time_univariate(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model):
    cdef:
        int i, j, inc = 1
        cnp.complex64_t alpha = 1.0
        cnp.complex64_t beta = 0.0
        cnp.complex64_t gamma = -1.0
        cnp.complex64_t scalar
        int k_states = model._k_states

    if smoother.t == 0:
        return 1

    # TODO check that this is the right transition matrix to use in the case
    # of time-varying matrices
    # r_{t-1,p} = T_{t-1}' r_{t,0}
    if smoother.smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
        blas.cgemv("T", &model._k_states, &model._k_states,
                                 &alpha, model._transition, &model._k_states,
                                         smoother._scaled_smoothed_estimator, &inc,
                                 &beta, &smoother.scaled_smoothed_estimator[0, smoother.t-1], &inc)
    # N_{t-1,p} = T_{t-1}' N_{t,0} T_{t-1}
    if smoother.smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
        blas.ccopy(&kfilter.k_states2, smoother._scaled_smoothed_estimator_cov, &inc,
                                                 &smoother.scaled_smoothed_estimator_cov[0, 0, smoother.t-1], &inc)
        blas.cgemm("T", "N", &model._k_states, &model._k_states, &model._k_states,
                                      &alpha, model._transition, &model._k_states,
                                              smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                                      &beta, smoother._tmp0, &kfilter.k_states)
        blas.cgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                                      &alpha, smoother._tmp0, &kfilter.k_states,
                                              model._transition, &model._k_states,
                                      &beta, &smoother.scaled_smoothed_estimator_cov[0, 0, smoother.t-1], &kfilter.k_states)

cdef int csmoothed_disturbances_univariate(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model):
    # Note: this only differs from the conventional version in the
    # definition of the smoothed measurement disturbance and cov
    cdef int i, j
    cdef:
        int inc = 1
        cnp.complex64_t alpha = 1.0
        cnp.complex64_t beta = 0.0
        cnp.complex64_t gamma = -1.0

    # Temporary arrays

    # $\\#_0 = R_t Q_t$  
    # $(m \times r) = (m \times r) (r \times r)$
    blas.cgemm("N", "N", &model._k_states, &model._k_posdef, &model._k_posdef,
              &alpha, model._selection, &model._k_states,
                      model._state_cov, &model._k_posdef,
              &beta, smoother._tmp0, &kfilter.k_states)

    if smoother.smoother_output & SMOOTHER_DISTURBANCE:
        for i in range(model._k_endog):
            # Smoothed measurement disturbance  
            # $\hat \varepsilon_t = (H_{t,i} / F_{t,i}) (v_{t,i} - F_{t,i} K_{t,i}' r_{t,i})$  
            # Note: K_{t,i}' r_{t,i} was stored in _smoothed_measurement_disturbance[i]
            # in smoothed_estimators_univariate, above, so we just need to implement
            # $\hat \varepsilon_t = (H_{t,i} / F_{t,i}) (v_{t,i} - \\#)$ here
            # (this is because we do not otherwise store the r_{t,i} values)  
            # $(p \times 1) = (p \times p) (p \times 1)$  
            smoother._smoothed_measurement_disturbance[i] = (
                kfilter._tmp4[i + i*kfilter.k_endog] * (
                    kfilter._forecast_error[i] -
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog] * smoother._smoothed_measurement_disturbance[i]
                )
            )

        # Smoothed state disturbance  
        # $\hat \eta_t = \\#_0' r_t$  
        # $(r \times 1) = (r \times m) (m \times 1)$  
        blas.cgemv("T", &model._k_states, &model._k_posdef,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._input_scaled_smoothed_estimator, &inc,
                      &beta, smoother._smoothed_state_disturbance, &inc)

    if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
        for i in range(model._k_endog):
            # Smoothed measurement disturbance covariance matrix  
            # $Var(\varepsilon_{t,i} | Y_n) = H_{t,i} - (H_{t,i} / F_{t,i})^2 * (F_{t,i} + F_{t,i}^2 K_{t,i}' N_{t,i} K_{t,i})$  
            # Note: K_{t,i}' N_{t,i} K_{t,i} was stored in _smoothed_measurement_disturbance_cov[i,i]
            # in smoothed_estimators_univariate, above, so we just need to implement
            # $Var(\varepsilon_{t,i} | Y_n) = H_{t,i} - (H_{t,i} / F_{t,i})^2 * (F_{t,i} + F_{t,i}^2 * \\#)$ here
            # (this is because we do not otherwise store the N_{t,i} values)  
            # $(1 \times 1) = (p \times p) - (p \times p) (p \times p) - (p \times m) (m \times m) (m \times p)$  
            smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog] = model._obs_cov[i + i*model._k_endog] - (
                (kfilter._tmp4[i + i*kfilter.k_endog]**2) * (
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog] +
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog]**2 * smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog]
                )
            )
        
        # Smoothed state disturbance covariance matrix  
        # $Var(\eta_t | Y_n) = Q_t - \\#_0' N_t \\#_0$  
        # $(r \times r) = (r \times r) - (r \times m) (m \times m) (m \times r)$  
        blas.cgemm("N", "N", &model._k_states, &model._k_posdef, &model._k_states,
                  &alpha, smoother._input_scaled_smoothed_estimator_cov, &kfilter.k_states,
                          smoother._tmp0, &kfilter.k_states,
                  &beta, smoother._tmpL, &kfilter.k_states)
        blas.ccopy(&model._k_posdef2, model._state_cov, &inc, smoother._smoothed_state_disturbance_cov, &inc)
        blas.cgemm("T", "N", &kfilter.k_posdef, &kfilter.k_posdef, &kfilter.k_states,
                  &gamma, smoother._tmp0, &kfilter.k_states,
                          smoother._tmpL, &kfilter.k_states,
                  &alpha, smoother._smoothed_state_disturbance_cov, &kfilter.k_posdef)

# ### Univariate Kalman smoother
#
# The following are the routines as defined in the univariate Kalman filter.
# 
# The only modification to the conventional Kalman smoother is the recursive
# definition of the scaled smoothing error and the scaled smoothing error
# covariance matrix.
#
# See Durbin and Koopman (2012) Chapter 6.4

cdef int ssmoothed_estimators_measurement_univariate(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model) except *:
    cdef:
        int i, j, inc = 1
        cnp.float32_t alpha = 1.0
        cnp.float32_t beta = 0.0
        cnp.float32_t gamma = -1.0
        cnp.float32_t scalar
        int k_states = model._k_states
        # cnp.float32_t [::1, :] tmpL
        # cnp.float32_t * _tmpL

    # dim2[0] = self.kfilter.k_states; dim2[1] = self.kfilter.k_states;
    # self.tmpL = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)

    # Adjust for a VAR transition (i.e. design = [#, 0], where the zeros
    # correspond to all states except the first k_posdef states)
    if model.subset_design:
        k_states = model._k_posdef

    # Need to clear out the scaled_smoothed_estimator and
    # scaled_smoothed_estimator_cov in case we're re-running the filter
    if smoother.t == model.nobs - 1:
        smoother.scaled_smoothed_estimator[:, model.nobs-1] = 0
        smoother.scaled_smoothed_estimator_cov[:, :, model.nobs-1] = 0

    # Smoothing error  
    # (not used in the univariate approach)  

    # Given r_{t,0}:
    # calculate r_{t-1,p}, ..., r_{t-1, 0} and N_{t-1,p}, ..., N_{t-1,0}

    # Iterate
    for i in range(kfilter.k_endog-1,-1,-1):
        # If we want smoothed disturbances, then we need to calculate
        # and store K_{t,i}' r_{t,i} for later (otherwise r_{t,i} will not be
        # available)
        if smoother.smoother_output & SMOOTHER_DISTURBANCE:
            # Note: zdot and cdot are broken, so have to use gemv for those
            smoother._smoothed_measurement_disturbance[i] = (
                blas.sdot(&model._k_states, &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                                                       smoother._scaled_smoothed_estimator, &inc)
            )

        # If we want smoothed disturbance covs, then we need to calculate
        # and store K_{t,i}' N_{t,i} K_{t,i} for later (otherwise N_{t,i} will not be
        # available)
        if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
            blas.sgemv("N", &model._k_states, &model._k_states,
                                     &alpha, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                                             &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                                     &beta, smoother._tmpL, &inc)
            # Note: zdot and cdot are broken, so have to use gemv for those
            smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog] = (
                blas.sdot(&model._k_states, &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                                                      smoother._tmpL, &inc)
            )

        # $L_{t,i} = (I_m - K_{t,i} Z_{t,i})$  
        # $(m \times m) = (m \times m) - (m \times 1) (1 \times m)$
        # blas.sgemm("N", "N", &model._k_states, &model._k_states, &inc,
        #           &gamma, &kfilter._kalman_gain[i*kfilter.k_states], &kfilter.k_states,
        #                   &model._design[i], &model._k_endog,
        #           &beta, smoother._tmpL, &kfilter.k_states)
        # Zero the temporary matrix
        # blas.sscal(&kfilter.k_states2, &beta, smoother._tmpL, &inc)
        smoother.tmpL[:,:k_states] = 0
        # Create the K_{t,i} Z_{t,i} component
        # (m x p) (m x 1) x (1 x p)
        blas.sger(&model._k_states, &k_states,
                  &gamma, &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                          &model._design[i], &model._k_endog,
                          smoother._tmpL, &kfilter.k_states
        )
        # Add the identity matrix
        for j in range(k_states):
            smoother._tmpL[j + j*kfilter.k_states] = smoother._tmpL[j + j*kfilter.k_states] + 1

        # Accumulate L_{t,i} into L_{t} = L_{t,n} L_{t,n-1} ... L_{t,1}
        if i == kfilter.k_endog-1:
            blas.scopy(&kfilter.k_states2, smoother._tmpL, &inc, smoother._tmpL2, &inc)
        else:
            blas.scopy(&kfilter.k_states2, smoother._tmpL2, &inc, smoother._tmp0, &inc)
            blas.sgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._tmpL, &kfilter.k_states,
                      &beta, smoother._tmpL2, &kfilter.k_states)

        # Scaled smoothed estimator  
        # $r_{t,i-1} = Z_{t,i}' v_{t,i} / F_{t,i} + L_{t,i}' r_{t,i}$  
        # $(m \times 1) = (m \times 1) (1 \times 1) + (m \times m) (m \times 1)$
        # Note: save $r_{t-1}$ as scaled_smoothed_estimator[t] rather than
        # as scaled_smoothed_estimator[t-1] because we actually need to store
        # T+1 of them (r_{T-1} to r_{-1} -> r_T to r_0)
        if smoother.smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
            #blas.sscal(&kfilter.k_states, &beta, smoother._tmp0, &inc)

            blas.sgemv("T", &model._k_states, &k_states,
                      &alpha, smoother._tmpL, &kfilter.k_states,
                              smoother._scaled_smoothed_estimator, &inc,
                      &beta, smoother._tmp0, &inc)
            blas.sswap(&k_states, smoother._tmp0, &inc,
                                           smoother._scaled_smoothed_estimator, &inc)
            blas.saxpy(&k_states, &kfilter._tmp2[i], &model._design[i], &model._k_endog,
                                                              smoother._scaled_smoothed_estimator, &inc)

        if smoother.smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
            # Scaled smoothed estimator covariance matrix  
            # $N_{t,i-1} = Z_{t,i}' Z_{t,i} / F_{t,i} + L_{t,i}' N_{t,i} L_{t,i}$  
            # $(m \times m) = (m \times p) (p \times m) + (m \times m) (m \times m) (m \times m)$  
            # Note: save $N_{t-1}$ as scaled_smoothed_estimator_cov[t] rather
            # than as scaled_smoothed_estimator_cov[t-1] because we actually
            # need to store T+1 of them (N_{T-1} to N_{-1} -> N_T to N_0)
            blas.sgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                              smoother._tmpL, &kfilter.k_states,
                      &beta, smoother._tmp0, &kfilter.k_states)
            blas.sgemm("T", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._tmpL, &kfilter.k_states,
                              smoother._tmp0, &kfilter.k_states,
                      &beta, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states)
            blas.sger(&model._k_states, &model._k_states,
                &alpha, &model._design[i], &model._k_endog,
                        &kfilter._tmp3[i], &kfilter.k_endog,
                smoother._scaled_smoothed_estimator_cov, &kfilter.k_states
            )

    # Replace L with accumulated version
    # blas.scopy(&kfilter.k_states2, smoother._tmpL2, &inc, smoother._tmpL, &inc)
    blas.sgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, model._transition, &kfilter.k_states,
                              smoother._tmpL2, &kfilter.k_states,
                      &beta, smoother._tmpL, &kfilter.k_states)

cdef int ssmoothed_estimators_time_univariate(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model):
    cdef:
        int i, j, inc = 1
        cnp.float32_t alpha = 1.0
        cnp.float32_t beta = 0.0
        cnp.float32_t gamma = -1.0
        cnp.float32_t scalar
        int k_states = model._k_states

    if smoother.t == 0:
        return 1

    # TODO check that this is the right transition matrix to use in the case
    # of time-varying matrices
    # r_{t-1,p} = T_{t-1}' r_{t,0}
    if smoother.smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
        blas.sgemv("T", &model._k_states, &model._k_states,
                                 &alpha, model._transition, &model._k_states,
                                         smoother._scaled_smoothed_estimator, &inc,
                                 &beta, &smoother.scaled_smoothed_estimator[0, smoother.t-1], &inc)
    # N_{t-1,p} = T_{t-1}' N_{t,0} T_{t-1}
    if smoother.smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
        blas.scopy(&kfilter.k_states2, smoother._scaled_smoothed_estimator_cov, &inc,
                                                 &smoother.scaled_smoothed_estimator_cov[0, 0, smoother.t-1], &inc)
        blas.sgemm("T", "N", &model._k_states, &model._k_states, &model._k_states,
                                      &alpha, model._transition, &model._k_states,
                                              smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                                      &beta, smoother._tmp0, &kfilter.k_states)
        blas.sgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                                      &alpha, smoother._tmp0, &kfilter.k_states,
                                              model._transition, &model._k_states,
                                      &beta, &smoother.scaled_smoothed_estimator_cov[0, 0, smoother.t-1], &kfilter.k_states)

cdef int ssmoothed_disturbances_univariate(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model):
    # Note: this only differs from the conventional version in the
    # definition of the smoothed measurement disturbance and cov
    cdef int i, j
    cdef:
        int inc = 1
        cnp.float32_t alpha = 1.0
        cnp.float32_t beta = 0.0
        cnp.float32_t gamma = -1.0

    # Temporary arrays

    # $\\#_0 = R_t Q_t$  
    # $(m \times r) = (m \times r) (r \times r)$
    blas.sgemm("N", "N", &model._k_states, &model._k_posdef, &model._k_posdef,
              &alpha, model._selection, &model._k_states,
                      model._state_cov, &model._k_posdef,
              &beta, smoother._tmp0, &kfilter.k_states)

    if smoother.smoother_output & SMOOTHER_DISTURBANCE:
        for i in range(model._k_endog):
            # Smoothed measurement disturbance  
            # $\hat \varepsilon_t = (H_{t,i} / F_{t,i}) (v_{t,i} - F_{t,i} K_{t,i}' r_{t,i})$  
            # Note: K_{t,i}' r_{t,i} was stored in _smoothed_measurement_disturbance[i]
            # in smoothed_estimators_univariate, above, so we just need to implement
            # $\hat \varepsilon_t = (H_{t,i} / F_{t,i}) (v_{t,i} - \\#)$ here
            # (this is because we do not otherwise store the r_{t,i} values)  
            # $(p \times 1) = (p \times p) (p \times 1)$  
            smoother._smoothed_measurement_disturbance[i] = (
                kfilter._tmp4[i + i*kfilter.k_endog] * (
                    kfilter._forecast_error[i] -
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog] * smoother._smoothed_measurement_disturbance[i]
                )
            )

        # Smoothed state disturbance  
        # $\hat \eta_t = \\#_0' r_t$  
        # $(r \times 1) = (r \times m) (m \times 1)$  
        blas.sgemv("T", &model._k_states, &model._k_posdef,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._input_scaled_smoothed_estimator, &inc,
                      &beta, smoother._smoothed_state_disturbance, &inc)

    if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
        for i in range(model._k_endog):
            # Smoothed measurement disturbance covariance matrix  
            # $Var(\varepsilon_{t,i} | Y_n) = H_{t,i} - (H_{t,i} / F_{t,i})^2 * (F_{t,i} + F_{t,i}^2 K_{t,i}' N_{t,i} K_{t,i})$  
            # Note: K_{t,i}' N_{t,i} K_{t,i} was stored in _smoothed_measurement_disturbance_cov[i,i]
            # in smoothed_estimators_univariate, above, so we just need to implement
            # $Var(\varepsilon_{t,i} | Y_n) = H_{t,i} - (H_{t,i} / F_{t,i})^2 * (F_{t,i} + F_{t,i}^2 * \\#)$ here
            # (this is because we do not otherwise store the N_{t,i} values)  
            # $(1 \times 1) = (p \times p) - (p \times p) (p \times p) - (p \times m) (m \times m) (m \times p)$  
            smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog] = model._obs_cov[i + i*model._k_endog] - (
                (kfilter._tmp4[i + i*kfilter.k_endog]**2) * (
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog] +
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog]**2 * smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog]
                )
            )
        
        # Smoothed state disturbance covariance matrix  
        # $Var(\eta_t | Y_n) = Q_t - \\#_0' N_t \\#_0$  
        # $(r \times r) = (r \times r) - (r \times m) (m \times m) (m \times r)$  
        blas.sgemm("N", "N", &model._k_states, &model._k_posdef, &model._k_states,
                  &alpha, smoother._input_scaled_smoothed_estimator_cov, &kfilter.k_states,
                          smoother._tmp0, &kfilter.k_states,
                  &beta, smoother._tmpL, &kfilter.k_states)
        blas.scopy(&model._k_posdef2, model._state_cov, &inc, smoother._smoothed_state_disturbance_cov, &inc)
        blas.sgemm("T", "N", &kfilter.k_posdef, &kfilter.k_posdef, &kfilter.k_states,
                  &gamma, smoother._tmp0, &kfilter.k_states,
                          smoother._tmpL, &kfilter.k_states,
                  &alpha, smoother._smoothed_state_disturbance_cov, &kfilter.k_posdef)

# ### Univariate Kalman smoother
#
# The following are the routines as defined in the univariate Kalman filter.
# 
# The only modification to the conventional Kalman smoother is the recursive
# definition of the scaled smoothing error and the scaled smoothing error
# covariance matrix.
#
# See Durbin and Koopman (2012) Chapter 6.4

cdef int zsmoothed_estimators_measurement_univariate(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model) except *:
    cdef:
        int i, j, inc = 1
        cnp.complex128_t alpha = 1.0
        cnp.complex128_t beta = 0.0
        cnp.complex128_t gamma = -1.0
        cnp.complex128_t scalar
        int k_states = model._k_states
        # cnp.complex128_t [::1, :] tmpL
        # cnp.complex128_t * _tmpL

    # dim2[0] = self.kfilter.k_states; dim2[1] = self.kfilter.k_states;
    # self.tmpL = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)

    # Adjust for a VAR transition (i.e. design = [#, 0], where the zeros
    # correspond to all states except the first k_posdef states)
    if model.subset_design:
        k_states = model._k_posdef

    # Need to clear out the scaled_smoothed_estimator and
    # scaled_smoothed_estimator_cov in case we're re-running the filter
    if smoother.t == model.nobs - 1:
        smoother.scaled_smoothed_estimator[:, model.nobs-1] = 0
        smoother.scaled_smoothed_estimator_cov[:, :, model.nobs-1] = 0

    # Smoothing error  
    # (not used in the univariate approach)  

    # Given r_{t,0}:
    # calculate r_{t-1,p}, ..., r_{t-1, 0} and N_{t-1,p}, ..., N_{t-1,0}

    # Iterate
    for i in range(kfilter.k_endog-1,-1,-1):
        # If we want smoothed disturbances, then we need to calculate
        # and store K_{t,i}' r_{t,i} for later (otherwise r_{t,i} will not be
        # available)
        if smoother.smoother_output & SMOOTHER_DISTURBANCE:
            # Note: zdot and cdot are broken, so have to use gemv for those
            blas.zgemv("N", &inc, &model._k_states,
                           &alpha, smoother._scaled_smoothed_estimator, &inc,
                                   &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                           &beta, &smoother._smoothed_measurement_disturbance[i], &inc)

        # If we want smoothed disturbance covs, then we need to calculate
        # and store K_{t,i}' N_{t,i} K_{t,i} for later (otherwise N_{t,i} will not be
        # available)
        if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
            blas.zgemv("N", &model._k_states, &model._k_states,
                                     &alpha, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                                             &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                                     &beta, smoother._tmpL, &inc)
            # Note: zdot and cdot are broken, so have to use gemv for those
            blas.zgemv("N", &inc, &model._k_states,
                           &alpha, smoother._tmpL, &inc,
                                   &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                           &beta, &smoother._smoothed_measurement_disturbance_cov[i*kfilter.k_endog + i], &inc)

        # $L_{t,i} = (I_m - K_{t,i} Z_{t,i})$  
        # $(m \times m) = (m \times m) - (m \times 1) (1 \times m)$
        # blas.zgemm("N", "N", &model._k_states, &model._k_states, &inc,
        #           &gamma, &kfilter._kalman_gain[i*kfilter.k_states], &kfilter.k_states,
        #                   &model._design[i], &model._k_endog,
        #           &beta, smoother._tmpL, &kfilter.k_states)
        # Zero the temporary matrix
        # blas.zscal(&kfilter.k_states2, &beta, smoother._tmpL, &inc)
        smoother.tmpL[:,:k_states] = 0
        # Create the K_{t,i} Z_{t,i} component
        # (m x p) (m x 1) x (1 x p)
        blas.zgeru(&model._k_states, &k_states,
                  &gamma, &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                          &model._design[i], &model._k_endog,
                          smoother._tmpL, &kfilter.k_states
        )
        # Add the identity matrix
        for j in range(k_states):
            smoother._tmpL[j + j*kfilter.k_states] = smoother._tmpL[j + j*kfilter.k_states] + 1

        # Accumulate L_{t,i} into L_{t} = L_{t,n} L_{t,n-1} ... L_{t,1}
        if i == kfilter.k_endog-1:
            blas.zcopy(&kfilter.k_states2, smoother._tmpL, &inc, smoother._tmpL2, &inc)
        else:
            blas.zcopy(&kfilter.k_states2, smoother._tmpL2, &inc, smoother._tmp0, &inc)
            blas.zgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._tmpL, &kfilter.k_states,
                      &beta, smoother._tmpL2, &kfilter.k_states)

        # Scaled smoothed estimator  
        # $r_{t,i-1} = Z_{t,i}' v_{t,i} / F_{t,i} + L_{t,i}' r_{t,i}$  
        # $(m \times 1) = (m \times 1) (1 \times 1) + (m \times m) (m \times 1)$
        # Note: save $r_{t-1}$ as scaled_smoothed_estimator[t] rather than
        # as scaled_smoothed_estimator[t-1] because we actually need to store
        # T+1 of them (r_{T-1} to r_{-1} -> r_T to r_0)
        if smoother.smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
            #blas.zscal(&kfilter.k_states, &beta, smoother._tmp0, &inc)

            blas.zgemv("T", &model._k_states, &k_states,
                      &alpha, smoother._tmpL, &kfilter.k_states,
                              smoother._scaled_smoothed_estimator, &inc,
                      &beta, smoother._tmp0, &inc)
            blas.zswap(&k_states, smoother._tmp0, &inc,
                                           smoother._scaled_smoothed_estimator, &inc)
            blas.zaxpy(&k_states, &kfilter._tmp2[i], &model._design[i], &model._k_endog,
                                                              smoother._scaled_smoothed_estimator, &inc)

        if smoother.smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
            # Scaled smoothed estimator covariance matrix  
            # $N_{t,i-1} = Z_{t,i}' Z_{t,i} / F_{t,i} + L_{t,i}' N_{t,i} L_{t,i}$  
            # $(m \times m) = (m \times p) (p \times m) + (m \times m) (m \times m) (m \times m)$  
            # Note: save $N_{t-1}$ as scaled_smoothed_estimator_cov[t] rather
            # than as scaled_smoothed_estimator_cov[t-1] because we actually
            # need to store T+1 of them (N_{T-1} to N_{-1} -> N_T to N_0)
            blas.zgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                              smoother._tmpL, &kfilter.k_states,
                      &beta, smoother._tmp0, &kfilter.k_states)
            blas.zgemm("T", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._tmpL, &kfilter.k_states,
                              smoother._tmp0, &kfilter.k_states,
                      &beta, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states)
            blas.zgeru(&model._k_states, &model._k_states,
                &alpha, &model._design[i], &model._k_endog,
                        &kfilter._tmp3[i], &kfilter.k_endog,
                smoother._scaled_smoothed_estimator_cov, &kfilter.k_states
            )

    # Replace L with accumulated version
    # blas.zcopy(&kfilter.k_states2, smoother._tmpL2, &inc, smoother._tmpL, &inc)
    blas.zgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, model._transition, &kfilter.k_states,
                              smoother._tmpL2, &kfilter.k_states,
                      &beta, smoother._tmpL, &kfilter.k_states)

cdef int zsmoothed_estimators_time_univariate(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model):
    cdef:
        int i, j, inc = 1
        cnp.complex128_t alpha = 1.0
        cnp.complex128_t beta = 0.0
        cnp.complex128_t gamma = -1.0
        cnp.complex128_t scalar
        int k_states = model._k_states

    if smoother.t == 0:
        return 1

    # TODO check that this is the right transition matrix to use in the case
    # of time-varying matrices
    # r_{t-1,p} = T_{t-1}' r_{t,0}
    if smoother.smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
        blas.zgemv("T", &model._k_states, &model._k_states,
                                 &alpha, model._transition, &model._k_states,
                                         smoother._scaled_smoothed_estimator, &inc,
                                 &beta, &smoother.scaled_smoothed_estimator[0, smoother.t-1], &inc)
    # N_{t-1,p} = T_{t-1}' N_{t,0} T_{t-1}
    if smoother.smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
        blas.zcopy(&kfilter.k_states2, smoother._scaled_smoothed_estimator_cov, &inc,
                                                 &smoother.scaled_smoothed_estimator_cov[0, 0, smoother.t-1], &inc)
        blas.zgemm("T", "N", &model._k_states, &model._k_states, &model._k_states,
                                      &alpha, model._transition, &model._k_states,
                                              smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                                      &beta, smoother._tmp0, &kfilter.k_states)
        blas.zgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                                      &alpha, smoother._tmp0, &kfilter.k_states,
                                              model._transition, &model._k_states,
                                      &beta, &smoother.scaled_smoothed_estimator_cov[0, 0, smoother.t-1], &kfilter.k_states)

cdef int zsmoothed_disturbances_univariate(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model):
    # Note: this only differs from the conventional version in the
    # definition of the smoothed measurement disturbance and cov
    cdef int i, j
    cdef:
        int inc = 1
        cnp.complex128_t alpha = 1.0
        cnp.complex128_t beta = 0.0
        cnp.complex128_t gamma = -1.0

    # Temporary arrays

    # $\\#_0 = R_t Q_t$  
    # $(m \times r) = (m \times r) (r \times r)$
    blas.zgemm("N", "N", &model._k_states, &model._k_posdef, &model._k_posdef,
              &alpha, model._selection, &model._k_states,
                      model._state_cov, &model._k_posdef,
              &beta, smoother._tmp0, &kfilter.k_states)

    if smoother.smoother_output & SMOOTHER_DISTURBANCE:
        for i in range(model._k_endog):
            # Smoothed measurement disturbance  
            # $\hat \varepsilon_t = (H_{t,i} / F_{t,i}) (v_{t,i} - F_{t,i} K_{t,i}' r_{t,i})$  
            # Note: K_{t,i}' r_{t,i} was stored in _smoothed_measurement_disturbance[i]
            # in smoothed_estimators_univariate, above, so we just need to implement
            # $\hat \varepsilon_t = (H_{t,i} / F_{t,i}) (v_{t,i} - \\#)$ here
            # (this is because we do not otherwise store the r_{t,i} values)  
            # $(p \times 1) = (p \times p) (p \times 1)$  
            smoother._smoothed_measurement_disturbance[i] = (
                kfilter._tmp4[i + i*kfilter.k_endog] * (
                    kfilter._forecast_error[i] -
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog] * smoother._smoothed_measurement_disturbance[i]
                )
            )

        # Smoothed state disturbance  
        # $\hat \eta_t = \\#_0' r_t$  
        # $(r \times 1) = (r \times m) (m \times 1)$  
        blas.zgemv("T", &model._k_states, &model._k_posdef,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._input_scaled_smoothed_estimator, &inc,
                      &beta, smoother._smoothed_state_disturbance, &inc)

    if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
        for i in range(model._k_endog):
            # Smoothed measurement disturbance covariance matrix  
            # $Var(\varepsilon_{t,i} | Y_n) = H_{t,i} - (H_{t,i} / F_{t,i})^2 * (F_{t,i} + F_{t,i}^2 K_{t,i}' N_{t,i} K_{t,i})$  
            # Note: K_{t,i}' N_{t,i} K_{t,i} was stored in _smoothed_measurement_disturbance_cov[i,i]
            # in smoothed_estimators_univariate, above, so we just need to implement
            # $Var(\varepsilon_{t,i} | Y_n) = H_{t,i} - (H_{t,i} / F_{t,i})^2 * (F_{t,i} + F_{t,i}^2 * \\#)$ here
            # (this is because we do not otherwise store the N_{t,i} values)  
            # $(1 \times 1) = (p \times p) - (p \times p) (p \times p) - (p \times m) (m \times m) (m \times p)$  
            smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog] = model._obs_cov[i + i*model._k_endog] - (
                (kfilter._tmp4[i + i*kfilter.k_endog]**2) * (
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog] +
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog]**2 * smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog]
                )
            )
        
        # Smoothed state disturbance covariance matrix  
        # $Var(\eta_t | Y_n) = Q_t - \\#_0' N_t \\#_0$  
        # $(r \times r) = (r \times r) - (r \times m) (m \times m) (m \times r)$  
        blas.zgemm("N", "N", &model._k_states, &model._k_posdef, &model._k_states,
                  &alpha, smoother._input_scaled_smoothed_estimator_cov, &kfilter.k_states,
                          smoother._tmp0, &kfilter.k_states,
                  &beta, smoother._tmpL, &kfilter.k_states)
        blas.zcopy(&model._k_posdef2, model._state_cov, &inc, smoother._smoothed_state_disturbance_cov, &inc)
        blas.zgemm("T", "N", &kfilter.k_posdef, &kfilter.k_posdef, &kfilter.k_states,
                  &gamma, smoother._tmp0, &kfilter.k_states,
                          smoother._tmpL, &kfilter.k_states,
                  &alpha, smoother._smoothed_state_disturbance_cov, &kfilter.k_posdef)

# ### Univariate Kalman smoother
#
# The following are the routines as defined in the univariate Kalman filter.
# 
# The only modification to the conventional Kalman smoother is the recursive
# definition of the scaled smoothing error and the scaled smoothing error
# covariance matrix.
#
# See Durbin and Koopman (2012) Chapter 6.4

cdef int dsmoothed_estimators_measurement_univariate(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model) except *:
    cdef:
        int i, j, inc = 1
        cnp.float64_t alpha = 1.0
        cnp.float64_t beta = 0.0
        cnp.float64_t gamma = -1.0
        cnp.float64_t scalar
        int k_states = model._k_states
        # cnp.float64_t [::1, :] tmpL
        # cnp.float64_t * _tmpL

    # dim2[0] = self.kfilter.k_states; dim2[1] = self.kfilter.k_states;
    # self.tmpL = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)

    # Adjust for a VAR transition (i.e. design = [#, 0], where the zeros
    # correspond to all states except the first k_posdef states)
    if model.subset_design:
        k_states = model._k_posdef

    # Need to clear out the scaled_smoothed_estimator and
    # scaled_smoothed_estimator_cov in case we're re-running the filter
    if smoother.t == model.nobs - 1:
        smoother.scaled_smoothed_estimator[:, model.nobs-1] = 0
        smoother.scaled_smoothed_estimator_cov[:, :, model.nobs-1] = 0

    # Smoothing error  
    # (not used in the univariate approach)  

    # Given r_{t,0}:
    # calculate r_{t-1,p}, ..., r_{t-1, 0} and N_{t-1,p}, ..., N_{t-1,0}

    # Iterate
    for i in range(kfilter.k_endog-1,-1,-1):
        # If we want smoothed disturbances, then we need to calculate
        # and store K_{t,i}' r_{t,i} for later (otherwise r_{t,i} will not be
        # available)
        if smoother.smoother_output & SMOOTHER_DISTURBANCE:
            # Note: zdot and cdot are broken, so have to use gemv for those
            smoother._smoothed_measurement_disturbance[i] = (
                blas.ddot(&model._k_states, &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                                                       smoother._scaled_smoothed_estimator, &inc)
            )

        # If we want smoothed disturbance covs, then we need to calculate
        # and store K_{t,i}' N_{t,i} K_{t,i} for later (otherwise N_{t,i} will not be
        # available)
        if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
            blas.dgemv("N", &model._k_states, &model._k_states,
                                     &alpha, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                                             &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                                     &beta, smoother._tmpL, &inc)
            # Note: zdot and cdot are broken, so have to use gemv for those
            smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog] = (
                blas.ddot(&model._k_states, &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                                                      smoother._tmpL, &inc)
            )

        # $L_{t,i} = (I_m - K_{t,i} Z_{t,i})$  
        # $(m \times m) = (m \times m) - (m \times 1) (1 \times m)$
        # blas.dgemm("N", "N", &model._k_states, &model._k_states, &inc,
        #           &gamma, &kfilter._kalman_gain[i*kfilter.k_states], &kfilter.k_states,
        #                   &model._design[i], &model._k_endog,
        #           &beta, smoother._tmpL, &kfilter.k_states)
        # Zero the temporary matrix
        # blas.dscal(&kfilter.k_states2, &beta, smoother._tmpL, &inc)
        smoother.tmpL[:,:k_states] = 0
        # Create the K_{t,i} Z_{t,i} component
        # (m x p) (m x 1) x (1 x p)
        blas.dger(&model._k_states, &k_states,
                  &gamma, &kfilter._kalman_gain[i*kfilter.k_states], &inc,
                          &model._design[i], &model._k_endog,
                          smoother._tmpL, &kfilter.k_states
        )
        # Add the identity matrix
        for j in range(k_states):
            smoother._tmpL[j + j*kfilter.k_states] = smoother._tmpL[j + j*kfilter.k_states] + 1

        # Accumulate L_{t,i} into L_{t} = L_{t,n} L_{t,n-1} ... L_{t,1}
        if i == kfilter.k_endog-1:
            blas.dcopy(&kfilter.k_states2, smoother._tmpL, &inc, smoother._tmpL2, &inc)
        else:
            blas.dcopy(&kfilter.k_states2, smoother._tmpL2, &inc, smoother._tmp0, &inc)
            blas.dgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._tmpL, &kfilter.k_states,
                      &beta, smoother._tmpL2, &kfilter.k_states)

        # Scaled smoothed estimator  
        # $r_{t,i-1} = Z_{t,i}' v_{t,i} / F_{t,i} + L_{t,i}' r_{t,i}$  
        # $(m \times 1) = (m \times 1) (1 \times 1) + (m \times m) (m \times 1)$
        # Note: save $r_{t-1}$ as scaled_smoothed_estimator[t] rather than
        # as scaled_smoothed_estimator[t-1] because we actually need to store
        # T+1 of them (r_{T-1} to r_{-1} -> r_T to r_0)
        if smoother.smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
            #blas.dscal(&kfilter.k_states, &beta, smoother._tmp0, &inc)

            blas.dgemv("T", &model._k_states, &k_states,
                      &alpha, smoother._tmpL, &kfilter.k_states,
                              smoother._scaled_smoothed_estimator, &inc,
                      &beta, smoother._tmp0, &inc)
            blas.dswap(&k_states, smoother._tmp0, &inc,
                                           smoother._scaled_smoothed_estimator, &inc)
            blas.daxpy(&k_states, &kfilter._tmp2[i], &model._design[i], &model._k_endog,
                                                              smoother._scaled_smoothed_estimator, &inc)

        if smoother.smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
            # Scaled smoothed estimator covariance matrix  
            # $N_{t,i-1} = Z_{t,i}' Z_{t,i} / F_{t,i} + L_{t,i}' N_{t,i} L_{t,i}$  
            # $(m \times m) = (m \times p) (p \times m) + (m \times m) (m \times m) (m \times m)$  
            # Note: save $N_{t-1}$ as scaled_smoothed_estimator_cov[t] rather
            # than as scaled_smoothed_estimator_cov[t-1] because we actually
            # need to store T+1 of them (N_{T-1} to N_{-1} -> N_T to N_0)
            blas.dgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                              smoother._tmpL, &kfilter.k_states,
                      &beta, smoother._tmp0, &kfilter.k_states)
            blas.dgemm("T", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, smoother._tmpL, &kfilter.k_states,
                              smoother._tmp0, &kfilter.k_states,
                      &beta, smoother._scaled_smoothed_estimator_cov, &kfilter.k_states)
            blas.dger(&model._k_states, &model._k_states,
                &alpha, &model._design[i], &model._k_endog,
                        &kfilter._tmp3[i], &kfilter.k_endog,
                smoother._scaled_smoothed_estimator_cov, &kfilter.k_states
            )

    # Replace L with accumulated version
    # blas.dcopy(&kfilter.k_states2, smoother._tmpL2, &inc, smoother._tmpL, &inc)
    blas.dgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                      &alpha, model._transition, &kfilter.k_states,
                              smoother._tmpL2, &kfilter.k_states,
                      &beta, smoother._tmpL, &kfilter.k_states)

cdef int dsmoothed_estimators_time_univariate(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model):
    cdef:
        int i, j, inc = 1
        cnp.float64_t alpha = 1.0
        cnp.float64_t beta = 0.0
        cnp.float64_t gamma = -1.0
        cnp.float64_t scalar
        int k_states = model._k_states

    if smoother.t == 0:
        return 1

    # TODO check that this is the right transition matrix to use in the case
    # of time-varying matrices
    # r_{t-1,p} = T_{t-1}' r_{t,0}
    if smoother.smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
        blas.dgemv("T", &model._k_states, &model._k_states,
                                 &alpha, model._transition, &model._k_states,
                                         smoother._scaled_smoothed_estimator, &inc,
                                 &beta, &smoother.scaled_smoothed_estimator[0, smoother.t-1], &inc)
    # N_{t-1,p} = T_{t-1}' N_{t,0} T_{t-1}
    if smoother.smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
        blas.dcopy(&kfilter.k_states2, smoother._scaled_smoothed_estimator_cov, &inc,
                                                 &smoother.scaled_smoothed_estimator_cov[0, 0, smoother.t-1], &inc)
        blas.dgemm("T", "N", &model._k_states, &model._k_states, &model._k_states,
                                      &alpha, model._transition, &model._k_states,
                                              smoother._scaled_smoothed_estimator_cov, &kfilter.k_states,
                                      &beta, smoother._tmp0, &kfilter.k_states)
        blas.dgemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                                      &alpha, smoother._tmp0, &kfilter.k_states,
                                              model._transition, &model._k_states,
                                      &beta, &smoother.scaled_smoothed_estimator_cov[0, 0, smoother.t-1], &kfilter.k_states)

cdef int dsmoothed_disturbances_univariate(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model):
    # Note: this only differs from the conventional version in the
    # definition of the smoothed measurement disturbance and cov
    cdef int i, j
    cdef:
        int inc = 1
        cnp.float64_t alpha = 1.0
        cnp.float64_t beta = 0.0
        cnp.float64_t gamma = -1.0

    # Temporary arrays

    # $\\#_0 = R_t Q_t$  
    # $(m \times r) = (m \times r) (r \times r)$
    blas.dgemm("N", "N", &model._k_states, &model._k_posdef, &model._k_posdef,
              &alpha, model._selection, &model._k_states,
                      model._state_cov, &model._k_posdef,
              &beta, smoother._tmp0, &kfilter.k_states)

    if smoother.smoother_output & SMOOTHER_DISTURBANCE:
        for i in range(model._k_endog):
            # Smoothed measurement disturbance  
            # $\hat \varepsilon_t = (H_{t,i} / F_{t,i}) (v_{t,i} - F_{t,i} K_{t,i}' r_{t,i})$  
            # Note: K_{t,i}' r_{t,i} was stored in _smoothed_measurement_disturbance[i]
            # in smoothed_estimators_univariate, above, so we just need to implement
            # $\hat \varepsilon_t = (H_{t,i} / F_{t,i}) (v_{t,i} - \\#)$ here
            # (this is because we do not otherwise store the r_{t,i} values)  
            # $(p \times 1) = (p \times p) (p \times 1)$  
            smoother._smoothed_measurement_disturbance[i] = (
                kfilter._tmp4[i + i*kfilter.k_endog] * (
                    kfilter._forecast_error[i] -
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog] * smoother._smoothed_measurement_disturbance[i]
                )
            )

        # Smoothed state disturbance  
        # $\hat \eta_t = \\#_0' r_t$  
        # $(r \times 1) = (r \times m) (m \times 1)$  
        blas.dgemv("T", &model._k_states, &model._k_posdef,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._input_scaled_smoothed_estimator, &inc,
                      &beta, smoother._smoothed_state_disturbance, &inc)

    if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
        for i in range(model._k_endog):
            # Smoothed measurement disturbance covariance matrix  
            # $Var(\varepsilon_{t,i} | Y_n) = H_{t,i} - (H_{t,i} / F_{t,i})^2 * (F_{t,i} + F_{t,i}^2 K_{t,i}' N_{t,i} K_{t,i})$  
            # Note: K_{t,i}' N_{t,i} K_{t,i} was stored in _smoothed_measurement_disturbance_cov[i,i]
            # in smoothed_estimators_univariate, above, so we just need to implement
            # $Var(\varepsilon_{t,i} | Y_n) = H_{t,i} - (H_{t,i} / F_{t,i})^2 * (F_{t,i} + F_{t,i}^2 * \\#)$ here
            # (this is because we do not otherwise store the N_{t,i} values)  
            # $(1 \times 1) = (p \times p) - (p \times p) (p \times p) - (p \times m) (m \times m) (m \times p)$  
            smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog] = model._obs_cov[i + i*model._k_endog] - (
                (kfilter._tmp4[i + i*kfilter.k_endog]**2) * (
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog] +
                    kfilter._forecast_error_cov[i + i*kfilter.k_endog]**2 * smoother._smoothed_measurement_disturbance_cov[i + i*kfilter.k_endog]
                )
            )
        
        # Smoothed state disturbance covariance matrix  
        # $Var(\eta_t | Y_n) = Q_t - \\#_0' N_t \\#_0$  
        # $(r \times r) = (r \times r) - (r \times m) (m \times m) (m \times r)$  
        blas.dgemm("N", "N", &model._k_states, &model._k_posdef, &model._k_states,
                  &alpha, smoother._input_scaled_smoothed_estimator_cov, &kfilter.k_states,
                          smoother._tmp0, &kfilter.k_states,
                  &beta, smoother._tmpL, &kfilter.k_states)
        blas.dcopy(&model._k_posdef2, model._state_cov, &inc, smoother._smoothed_state_disturbance_cov, &inc)
        blas.dgemm("T", "N", &kfilter.k_posdef, &kfilter.k_posdef, &kfilter.k_states,
                  &gamma, smoother._tmp0, &kfilter.k_states,
                          smoother._tmpL, &kfilter.k_states,
                  &alpha, smoother._smoothed_state_disturbance_cov, &kfilter.k_posdef)