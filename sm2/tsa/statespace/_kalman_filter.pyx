#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
Kalman Filter

Author: Chad Fulton  
License: Simplified-BSD
"""
import warnings

cimport cython

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack

from sm2.src.math cimport *
cimport sm2.tsa.statespace._tools as tools

# ### Filters
# TODO note that only the conventional filter is implemented
cdef int FILTER_CONVENTIONAL = 0x01     # Durbin and Koopman (2012), Chapter 4
cdef int FILTER_EXACT_INITIAL = 0x02    # ibid., Chapter 5.6
cdef int FILTER_AUGMENTED = 0x04        # ibid., Chapter 5.7
cdef int FILTER_SQUARE_ROOT = 0x08      # ibid., Chapter 6.3
cdef int FILTER_UNIVARIATE = 0x10       # ibid., Chapter 6.4
cdef int FILTER_COLLAPSED = 0x20        # ibid., Chapter 6.5
cdef int FILTER_EXTENDED = 0x40         # ibid., Chapter 10.2
cdef int FILTER_UNSCENTED = 0x80        # ibid., Chapter 10.3
cdef int SMOOTHER_CLASSICAL = 0x100     # ibid., Chapter 4.6.1
cdef int SMOOTHER_ALTERNATIVE = 0x200   # ibid., Chapter 4.6.1

# ### Inversion methods
# Methods by which the terms using the inverse of the forecast error
# covariance matrix are solved.
cdef int INVERT_UNIVARIATE = 0x01
cdef int SOLVE_LU = 0x02
cdef int INVERT_LU = 0x04
cdef int SOLVE_CHOLESKY = 0x08
cdef int INVERT_CHOLESKY = 0x10

# ### Numerical Stability methods
# Methods to improve numerical stability
cdef int STABILITY_FORCE_SYMMETRY = 0x01

# ### Memory conservation options
cdef int MEMORY_STORE_ALL = 0
cdef int MEMORY_NO_FORECAST = 0x01
cdef int MEMORY_NO_PREDICTED = 0x02
cdef int MEMORY_NO_FILTERED = 0x04
cdef int MEMORY_NO_LIKELIHOOD = 0x08
cdef int MEMORY_NO_GAIN = 0x10
cdef int MEMORY_NO_SMOOTHING = 0x20
cdef int MEMORY_NO_STD_FORECAST = 0x40
cdef int MEMORY_CONSERVE = (
    MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED | MEMORY_NO_FILTERED |
    MEMORY_NO_LIKELIHOOD | MEMORY_NO_GAIN | MEMORY_NO_SMOOTHING |
    MEMORY_NO_STD_FORECAST
)

# ### Timing options
# By default, assume filter is initialized with predicted values (as in
# Durbin and Koopman 2012)
#
# Alternately, assume filter is initialized with filtered values (as in
# Kim and Nelson 1999)
cdef int TIMING_INIT_PREDICTED = 0
cdef int TIMING_INIT_FILTERED = 1
from sm2.tsa.statespace._representation cimport cStatespace
from sm2.tsa.statespace._filters._conventional cimport (
    cforecast_missing_conventional,
    cupdating_missing_conventional,
    cinverse_missing_conventional,
    cloglikelihood_missing_conventional,
    cforecast_conventional,
    cupdating_conventional,
    cprediction_conventional,
    cloglikelihood_conventional
)
from sm2.tsa.statespace._filters._univariate cimport (
    cforecast_univariate,
    cupdating_univariate,
    cinverse_noop_univariate,
    cprediction_univariate,
    cloglikelihood_univariate
)
from sm2.tsa.statespace._filters._inversions cimport (
    cinverse_univariate,
    cfactorize_cholesky,
    cfactorize_lu,
    cinverse_cholesky,
    cinverse_lu,
    csolve_cholesky,
    csolve_lu
)
from sm2.tsa.statespace._representation cimport sStatespace
from sm2.tsa.statespace._filters._conventional cimport (
    sforecast_missing_conventional,
    supdating_missing_conventional,
    sinverse_missing_conventional,
    sloglikelihood_missing_conventional,
    sforecast_conventional,
    supdating_conventional,
    sprediction_conventional,
    sloglikelihood_conventional
)
from sm2.tsa.statespace._filters._univariate cimport (
    sforecast_univariate,
    supdating_univariate,
    sinverse_noop_univariate,
    sprediction_univariate,
    sloglikelihood_univariate
)
from sm2.tsa.statespace._filters._inversions cimport (
    sinverse_univariate,
    sfactorize_cholesky,
    sfactorize_lu,
    sinverse_cholesky,
    sinverse_lu,
    ssolve_cholesky,
    ssolve_lu
)
from sm2.tsa.statespace._representation cimport zStatespace
from sm2.tsa.statespace._filters._conventional cimport (
    zforecast_missing_conventional,
    zupdating_missing_conventional,
    zinverse_missing_conventional,
    zloglikelihood_missing_conventional,
    zforecast_conventional,
    zupdating_conventional,
    zprediction_conventional,
    zloglikelihood_conventional
)
from sm2.tsa.statespace._filters._univariate cimport (
    zforecast_univariate,
    zupdating_univariate,
    zinverse_noop_univariate,
    zprediction_univariate,
    zloglikelihood_univariate
)
from sm2.tsa.statespace._filters._inversions cimport (
    zinverse_univariate,
    zfactorize_cholesky,
    zfactorize_lu,
    zinverse_cholesky,
    zinverse_lu,
    zsolve_cholesky,
    zsolve_lu
)
from sm2.tsa.statespace._representation cimport dStatespace
from sm2.tsa.statespace._filters._conventional cimport (
    dforecast_missing_conventional,
    dupdating_missing_conventional,
    dinverse_missing_conventional,
    dloglikelihood_missing_conventional,
    dforecast_conventional,
    dupdating_conventional,
    dprediction_conventional,
    dloglikelihood_conventional
)
from sm2.tsa.statespace._filters._univariate cimport (
    dforecast_univariate,
    dupdating_univariate,
    dinverse_noop_univariate,
    dprediction_univariate,
    dloglikelihood_univariate
)
from sm2.tsa.statespace._filters._inversions cimport (
    dinverse_univariate,
    dfactorize_cholesky,
    dfactorize_lu,
    dinverse_cholesky,
    dinverse_lu,
    dsolve_cholesky,
    dsolve_lu
)

cdef int FORTRAN = 1

# ## Kalman filter
cdef class cKalmanFilter(object):
    """
    cKalmanFilter(model, filter_method=FILTER_CONVENTIONAL, inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY, stability_method=STABILITY_FORCE_SYMMETRY, filter_timing=TIMING_INIT_PREDICTED, tolerance=1e-19)

    A representation of the Kalman filter recursions.

    While the filter is mathematically represented as a recursion, it is here
    translated into Python as a stateful iterator.

    Because there are actually several types of Kalman filter depending on the
    state space model of interest, this class only handles the *iteration*
    aspect of filtering, and delegates the actual operations to four general
    workhorse routines, which can be implemented separately for each type of
    Kalman filter.

    In order to maintain a consistent interface, and because these four general
    routines may be quite different across filter types, their argument is only
    the stateful ?KalmanFilter object. Furthermore, in order to allow the
    different types of filter to substitute alternate matrices, this class
    defines a set of pointers to the various state space arrays and the
    filtering output arrays.

    For example, handling missing observations requires not only substituting
    `obs`, `design`, and `obs_cov` matrices, but the new matrices actually have
    different dimensions than the originals. This can be flexibly accomodated
    simply by replacing e.g. the `obs` pointer to the substituted `obs` array
    and replacing `k_endog` for that iteration. Then in the next iteration, when
    the `obs` vector may be missing different elements (or none at all), it can
    again be redefined.

    Each iteration of the filter (see `__next__`) proceeds in a number of
    steps.

    `initialize_object_pointers` initializes pointers to current-iteration
    objects (i.e. the state space arrays and filter output arrays).  

    `initialize_function_pointers` initializes pointers to the appropriate
    Kalman filtering routines (i.e. `forecast_conventional` or
    `forecast_exact_initial`, etc.).  

    `select_arrays` converts the base arrays into "selected" arrays using
    selection matrices. In particular, it handles the state covariance matrix
    and redefined matrices based on missing values.  

    `post_convergence` handles copying arrays from time $t-1$ to time $t$ when
    the Kalman filter has converged and they don't need to be re-calculated.  

    `forecasting` calls the Kalman filter `forcasting_<filter type>` routine

    `inversion` calls the appropriate function to invert the forecast error
    covariance matrix.  

    `updating` calls the Kalman filter `updating_<filter type>` routine

    `loglikelihood` calls the Kalman filter `loglikelihood_<filter type>` routine

    `prediction` calls the Kalman filter `prediction_<filter type>` routine

    `numerical_stability` performs end-of-iteration tasks to improve the numerical
    stability of the filter 

    `check_convergence` checks for convergence of the filter to steady-state.
    """

    # ### Statespace model
    # cdef readonly cStatespace model

    # ### Filter parameters
    # Holds the time-iteration state of the filter  
    # *Note*: must be changed using the `seek` method
    # cdef readonly int t
    # Holds the tolerance parameter for convergence
    # cdef public cnp.float64_t tolerance
    # Holds the convergence to steady-state status of the filter
    # *Note*: is by default reset each time `seek` is called
    # cdef readonly int converged
    # cdef readonly int period_converged
    # Holds whether or not the model is time-invariant
    # *Note*: is by default reset each time `seek` is called
    # cdef readonly int time_invariant
    # The Kalman filter procedure to use  
    # cdef readonly int filter_method
    # The method by which the terms using the inverse of the forecast
    # error covariance matrix are solved.
    # cdef public int inversion_method
    # Methods to improve numerical stability
    # cdef public int stability_method
    # Whether or not to conserve memory
    # If True, only stores filtered states and covariance matrices
    # cdef readonly int conserve_memory
    # Whether or not to use alternate timing
    # If True, uses the Kim and Nelson (1999) timing
    # cdef readonly int filter_timing
    # If conserving loglikelihood, the number of periods to "burn"
    # before starting to record the loglikelihood
    # cdef readonly int loglikelihood_burn

    # ### Kalman filter properties

    # `loglikelihood` $\equiv \log p(y_t | Y_{t-1})$
    # cdef readonly cnp.complex64_t [:] loglikelihood

    # `filtered_state` $\equiv a_{t|t} = E(\alpha_t | Y_t)$ is the **filtered estimator** of the state $(m \times T)$  
    # `predicted_state` $\equiv a_{t+1} = E(\alpha_{t+1} | Y_t)$ is the **one-step ahead predictor** of the state $(m \times T-1)$  
    # `forecast` $\equiv E(y_t|Y_{t-1})$ is the **forecast** of the next observation $(p \times T)$   
    # `forecast_error` $\equiv v_t = y_t - E(y_t|Y_{t-1})$ is the **one-step ahead forecast error** of the next observation $(p \times T)$  
    # 
    # *Note*: Actual values in `filtered_state` will be from 1 to `nobs`+1. Actual
    # values in `predicted_state` will be from 0 to `nobs`+1 because the initialization
    # is copied over to the zeroth entry, and similar for the covariances, below.
    #
    # *Old notation: beta_tt, beta_tt1, y_tt1, eta_tt1*
    # cdef readonly cnp.complex64_t [::1,:] filtered_state, predicted_state, forecast, forecast_error

    # `filtered_state_cov` $\equiv P_{t|t} = Var(\alpha_t | Y_t)$ is the **filtered state covariance matrix** $(m \times m \times T)$  
    # `predicted_state_cov` $\equiv P_{t+1} = Var(\alpha_{t+1} | Y_t)$ is the **predicted state covariance matrix** $(m \times m \times T)$  
    # `forecast_error_cov` $\equiv F_t = Var(v_t | Y_{t-1})$ is the **forecast error covariance matrix** $(p \times p \times T)$  
    # 
    # *Old notation: P_tt, P_tt1, f_tt1*
    # cdef readonly cnp.complex64_t [::1,:,:] filtered_state_cov, predicted_state_cov, forecast_error_cov

    # `kalman_gain` $\equiv K_{t} = T_t P_t Z_t' F_t^{-1}$ is the **Kalman gain** $(m \times p \times T)$  
    # cdef readonly cnp.complex64_t [::1,:,:] kalman_gain

    # ### Steady State Values
    # These matrices are used to hold the converged matrices after the Kalman
    # filter has reached steady-state
    # cdef readonly cnp.complex64_t [::1,:] converged_forecast_error_cov
    # cdef readonly cnp.complex64_t [::1,:] converged_filtered_state_cov
    # cdef readonly cnp.complex64_t [::1,:] converged_predicted_state_cov
    # cdef readonly cnp.complex64_t [::1,:] converged_kalman_gain
    # cdef readonly cnp.complex64_t converged_determinant

    # ### Temporary arrays
    # These matrices are used to temporarily hold selected observation vectors,
    # design matrices, and observation covariance matrices in the case of
    # missing data.  
    # `forecast_error_fac` is a forecast error covariance matrix **factorization** $(p \times p)$.
    # Depending on the method for handling the inverse of the forecast error covariance matrix, it may be:
    # - a Cholesky factorization if `cholesky_solve` is used
    # - an inverse calculated via Cholesky factorization if `cholesky_inverse` is used
    # - an LU factorization if `lu_solve` is used
    # - an inverse calculated via LU factorization if `lu_inverse` is used
    # cdef readonly cnp.complex64_t [::1,:] forecast_error_fac
    # `forecast_error_ipiv` holds pivot indices if an LU decomposition is used
    # cdef readonly int [:] forecast_error_ipiv
    # `forecast_error_work` is a work array for matrix inversion if an LU
    # decomposition is used
    # cdef readonly cnp.complex64_t [::1,:] forecast_error_work
    # These hold the memory allocations of the anonymous temporary arrays
    # cdef readonly cnp.complex64_t [::1,:] tmp0, tmp00
    # These hold the memory allocations of the named temporary arrays  
    # (these are all time-varying in the last dimension)
    # cdef readonly cnp.complex64_t [::1,:] tmp2
    # cdef readonly cnp.complex64_t [::1,:,:] tmp1, tmp3

    # Holds the determinant across calculations (this is done because after
    # convergence, it doesn't need to be re-calculated anymore)
    # cdef readonly cnp.complex64_t determinant

    # ### Pointers to current-iteration arrays
    # cdef cnp.complex64_t * _obs
    # cdef cnp.complex64_t * _design
    # cdef cnp.complex64_t * _obs_intercept
    # cdef cnp.complex64_t * _obs_cov
    # cdef cnp.complex64_t * _transition
    # cdef cnp.complex64_t * _state_intercept
    # cdef cnp.complex64_t * _selection
    # cdef cnp.complex64_t * _state_cov
    # cdef cnp.complex64_t * _selected_state_cov
    # cdef cnp.complex64_t * _initial_state
    # cdef cnp.complex64_t * _initial_state_cov

    # cdef cnp.complex64_t * _input_state
    # cdef cnp.complex64_t * _input_state_cov

    # cdef cnp.complex64_t * _forecast
    # cdef cnp.complex64_t * _forecast_error
    # cdef cnp.complex64_t * _forecast_error_cov
    # cdef cnp.complex64_t * _filtered_state
    # cdef cnp.complex64_t * _filtered_state_cov
    # cdef cnp.complex64_t * _predicted_state
    # cdef cnp.complex64_t * _predicted_state_cov

    # cdef cnp.complex64_t * _kalman_gain
    # cdef cnp.complex64_t * _loglikelihood

    # cdef cnp.complex64_t * _converged_forecast_error_cov
    # cdef cnp.complex64_t * _converged_filtered_state_cov
    # cdef cnp.complex64_t * _converged_predicted_state_cov
    # cdef cnp.complex64_t * _converged_kalman_gain

    # cdef cnp.complex64_t * _forecast_error_fac
    # cdef int * _forecast_error_ipiv
    # cdef cnp.complex64_t * _forecast_error_work

    # cdef cnp.complex64_t * _tmp0
    # cdef cnp.complex64_t * _tmp00
    # cdef cnp.complex64_t * _tmp1
    # cdef cnp.complex64_t * _tmp2
    # cdef cnp.complex64_t * _tmp3

    # ### Pointers to current-iteration Kalman filtering functions
    # cdef int (*forecasting)(
    #     cKalmanFilter, cStatespace
    # )
    # cdef cnp.complex64_t (*inversion)(
    #     cKalmanFilter, cStatespace, cnp.complex64_t
    # ) except *
    # cdef int (*updating)(
    #     cKalmanFilter, cStatespace
    # )
    # cdef cnp.complex64_t (*calculate_loglikelihood)(
    #     cKalmanFilter, cStatespace, cnp.complex64_t
    # )
    # cdef int (*prediction)(
    #     cKalmanFilter, cStatespace
    # )

    # ### Define some constants
    # cdef readonly int k_endog, k_states, k_posdef, k_endog2, k_states2, k_endogstates
    # cdef readonly ldwork
    
    def __init__(self,
                 cStatespace model,
                 int filter_method=FILTER_CONVENTIONAL,
                 int inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY,
                 int stability_method=STABILITY_FORCE_SYMMETRY,
                 int conserve_memory=MEMORY_STORE_ALL,
                 int filter_timing=TIMING_INIT_PREDICTED,
                 cnp.float64_t tolerance=1e-19,
                 int loglikelihood_burn=0):

        # Save the model
        self.model = model

        # Initialize filter parameters
        self.tolerance = tolerance
        self.inversion_method = inversion_method
        self.stability_method = stability_method
        self.conserve_memory = conserve_memory
        self.filter_timing = filter_timing
        self.loglikelihood_burn = loglikelihood_burn

        # Initialize the constant values
        self.time_invariant = self.model.time_invariant

        # TODO replace with optimal work array size
        self.ldwork = self.model.k_endog

        # Set the filter method
        self.set_dimensions()
        self.set_filter_method(filter_method, True)

        # Initialize time and convergence status
        self.t = 0
        self.converged = 0
        self.period_converged = 0

    def __reduce__(self):
        args = (self.model, self.filter_method, self.inversion_method,
                self.stability_method,  self.conserve_memory, self.filter_timing,
                self.tolerance, self.loglikelihood_burn)
        state = {'t': self.t,
                 'converged' : self.converged ,
                 'converged_determinant' : self.converged_determinant,
                 'determinant' : self.determinant,
                 'period_converged' : self.period_converged,
                 'converged_filtered_state_cov': np.array(self.converged_filtered_state_cov, copy=True, order='F'),
                 'converged_forecast_error_cov': np.array(self.converged_forecast_error_cov, copy=True, order='F'),
                 'converged_kalman_gain': np.array(self.converged_kalman_gain, copy=True, order='F'),
                 'converged_predicted_state_cov': np.array(self.converged_predicted_state_cov, copy=True, order='F'),
                 'filtered_state': np.array(self.filtered_state, copy=True, order='F'),
                 'filtered_state_cov': np.array(self.filtered_state_cov, copy=True, order='F'),
                 'forecast': np.array(self.forecast, copy=True, order='F'),
                 'forecast_error': np.array(self.forecast_error, copy=True, order='F'),
                 'forecast_error_cov': np.array(self.forecast_error_cov, copy=True, order='F'),
                 'forecast_error_fac': np.array(self.forecast_error_fac, copy=True, order='F'),
                 'forecast_error_ipiv': np.array(self.forecast_error_ipiv, copy=True, order='F'),
                 'forecast_error_work': np.array(self.forecast_error_work, copy=True, order='F'),
                 'kalman_gain': np.array(self.kalman_gain, copy=True, order='F'),
                 'loglikelihood': np.array(self.loglikelihood, copy=True, order='F'),
                 'predicted_state': np.array(self.predicted_state, copy=True, order='F'),
                 'predicted_state_cov': np.array(self.predicted_state_cov, copy=True, order='F'),
                 'standardized_forecast_error': np.array(self.standardized_forecast_error, copy=True, order='F'),
                 'tmp0': np.array(self.tmp0, copy=True, order='F'),
                 'tmp00': np.array(self.tmp00, copy=True, order='F'),
                 'tmp1': np.array(self.tmp1, copy=True, order='F'),
                 'tmp2': np.array(self.tmp2, copy=True, order='F'),
                 'tmp3': np.array(self.tmp3, copy=True, order='F'),
                 'tmp4': np.array(self.tmp4, copy=True, order='F')
                 }

        return (self.__class__, args, state)

    def __setstate__(self, state):
        self.t = state['t']
        self.converged  = state['converged']
        self.converged_determinant = state['converged_determinant']
        self.determinant = state['determinant']
        self.period_converged = state['period_converged']
        self.converged_filtered_state_cov = state['converged_filtered_state_cov']
        self.converged_forecast_error_cov = state['converged_forecast_error_cov']
        self.converged_kalman_gain = state['converged_kalman_gain']
        self.converged_predicted_state_cov = state['converged_predicted_state_cov']
        self.filtered_state = state['filtered_state']
        self.filtered_state_cov = state['filtered_state_cov']
        self.forecast = state['forecast']
        self.forecast_error = state['forecast_error']
        self.forecast_error_cov = state['forecast_error_cov']
        self.forecast_error_fac = state['forecast_error_fac']
        self.forecast_error_ipiv = state['forecast_error_ipiv']
        self.forecast_error_work = state['forecast_error_work']
        self.kalman_gain = state['kalman_gain']
        self.loglikelihood = state['loglikelihood']
        self.predicted_state = state['predicted_state']
        self.predicted_state_cov = state['predicted_state_cov']
        self.standardized_forecast_error = state['standardized_forecast_error']
        self.tmp0 = state['tmp0']
        self.tmp00 = state['tmp00']
        self.tmp1 = state['tmp1']
        self.tmp2 = state['tmp2']
        self.tmp3 = state['tmp3']
        self.tmp4 = state['tmp4']
        self._reinitialize_pointers()

    cdef void _reinitialize_pointers(self) except *:
        self._converged_forecast_error_cov = &self.converged_forecast_error_cov[0,0]
        self._converged_filtered_state_cov = &self.converged_filtered_state_cov[0,0]
        self._converged_predicted_state_cov = &self.converged_predicted_state_cov[0,0]
        self._converged_kalman_gain = &self.converged_kalman_gain[0,0]
        self._forecast_error_fac = &self.forecast_error_fac[0,0]
        self._forecast_error_work = &self.forecast_error_work[0,0]
        self._forecast_error_ipiv = &self.forecast_error_ipiv[0]
        self._tmp0 = &self.tmp0[0, 0]
        self._tmp00 = &self.tmp00[0, 0]

    cdef allocate_arrays(self):
        # Local variables
        cdef:
            cnp.npy_intp dim1[1]
            cnp.npy_intp dim2[2]
            cnp.npy_intp dim3[3]
        cdef int storage
        # #### Allocate arrays for calculations

        # Arrays for Kalman filter output

        # Forecast
        if self.conserve_memory & MEMORY_NO_FORECAST:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.forecast = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self.forecast_error = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        dim3[0] = self.k_endog; dim3[1] = self.k_endog; dim3[2] = storage;
        self.forecast_error_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX64, FORTRAN)
        # Standardized forecast errors
        if self.conserve_memory & MEMORY_NO_STD_FORECAST > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.standardized_forecast_error = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)

        # Filtered
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_states; dim2[1] = storage;
        self.filtered_state = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = storage;
        self.filtered_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX64, FORTRAN)

        # Predicted
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_states; dim2[1] = storage+1;
        self.predicted_state = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = storage+1;
        self.predicted_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX64, FORTRAN)

        # Kalman Gain
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim3[0] = self.k_states; dim3[1] = self.k_endog; dim3[2] = storage;
        self.kalman_gain = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX64, FORTRAN)

        # Likelihood
        if self.conserve_memory & MEMORY_NO_LIKELIHOOD > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim1[0] = storage
        self.loglikelihood = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX64, FORTRAN)

        # Converged matrices
        dim2[0] = self.k_endog; dim2[1] = self.k_endog;
        self.converged_forecast_error_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self._converged_forecast_error_cov = &self.converged_forecast_error_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.converged_filtered_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self._converged_filtered_state_cov = &self.converged_filtered_state_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.converged_predicted_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self._converged_predicted_state_cov = &self.converged_predicted_state_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_endog;
        self.converged_kalman_gain = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self._converged_kalman_gain = &self.converged_kalman_gain[0,0]

        # #### Arrays for temporary calculations
        # *Note*: in math notation below, a $\\#$ will represent a generic
        # temporary array, and a $\\#_i$ will represent a named temporary array.

        # Arrays related to matrix factorizations / inverses
        dim2[0] = self.k_endog; dim2[1] = self.k_endog;
        self.forecast_error_fac = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self._forecast_error_fac = &self.forecast_error_fac[0,0]
        dim2[0] = self.ldwork; dim2[1] = self.ldwork;
        self.forecast_error_work = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self._forecast_error_work = &self.forecast_error_work[0,0]
        dim1[0] = self.k_endog;
        self.forecast_error_ipiv = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_INT, FORTRAN)
        self._forecast_error_ipiv = &self.forecast_error_ipiv[0]

        # Holds arrays of dimension $(m \times m)$ and $(m \times r)$
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.tmp0 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self._tmp0 = &self.tmp0[0, 0]

        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.tmp00 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self._tmp00 = &self.tmp00[0, 0]

        # Optionally we may not want to store temporary arrays required  
        # for smoothing
        if self.conserve_memory & MEMORY_NO_SMOOTHING > 0:
            storage = 1
        else:
            storage = self.model.nobs

        # Holds arrays of dimension $(m \times p \times T)$  
        # $\\#_1 = P_t Z_t'$
        dim3[0] = self.k_states; dim3[1] = self.k_endog; dim3[2] = storage;
        self.tmp1 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX64, FORTRAN)

        # Holds arrays of dimension $(p \times T)$  
        # $\\#_2 = F_t^{-1} v_t$
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.tmp2 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)

        # Holds arrays of dimension $(p \times m \times T)$  
        # $\\#_3 = F_t^{-1} Z_t$
        dim3[0] = self.k_endog; dim3[1] = self.k_states; dim3[2] = storage;
        self.tmp3 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX64, FORTRAN)

        # Holds arrays of dimension $(p \times p \times T)$  
        # $\\#_4 = F_t^{-1} H_t$
        dim3[0] = self.k_endog; dim3[1] = self.k_endog; dim3[2] = storage;
        self.tmp4 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX64, FORTRAN)

    @cython.cfunc
    @cython.returns(cython.void)
    def set_dimensions(self):
        """
        Set dimensions for the Kalman filter

        These are used *only* to define the shapes of the Kalman filter output
        and temporary arrays in memory. They will not change between iterations
        of the filter.

        They only differ from the cStatespace versions in the case
        that the FILTER_COLLAPSED flag is set, in which case model.k_endog
        and kfilter.k_endog will be different
        (since kfilter.k_endog = model.k_states).

        Across *iterations* of the Kalman filter, both model.k_* and
        kfilter.k_* are fixed, although model._k_* may be different from either
        when there is missing data in a given period's observations.

        The actual dimension of the *data* being considered at a given
        iteration is always given by model._k_* variables, which take into
        account both FILTER_COLLAPSED and missing data.

        But, the dimension *in memory* of the Kalman filter arrays will always
        be given by kfilter.k_*.

        The following relations will always hold:

        kfilter.k_endog = model.k_states if self.filter_method & FILTER_COLLAPSED else model.k_endog
        kfilter.k_endog = model._k_endog + model._nmissing
        """
        self.k_endog = self.model.k_states if self.filter_method & FILTER_COLLAPSED else self.model.k_endog
        self.k_states = self.model.k_states
        self.k_posdef = self.model.k_posdef
        self.k_endog2 = self.k_endog**2
        self.k_states2 = self.k_states**2
        self.k_posdef2 = self.k_posdef**2
        self.k_endogstates = self.k_endog * self.k_states
        self.k_statesposdef = self.k_states * self.k_posdef

    @cython.ccall
    def set_filter_method(self, int filter_method, int force_reset=True):
        """
        set_filter_method(self, filter_method, force_reset=True)

        Change the filter method.
        """
        if not filter_method == self.filter_method or force_reset:
            # Check for invalid filter methods
            if filter_method & FILTER_COLLAPSED and self.k_endog <= self.k_states:
                raise RuntimeError('Cannot collapse observation vector if the'
                                   ' state dimension is equal to or larger than the'
                                   ' dimension of the observation vector.')

            # Change the smoother output flag
            self.filter_method = filter_method

            # Reset dimensions
            self.set_dimensions()

            # Reset matrices
            self.allocate_arrays()

            # Seek to the beginning
            self.seek(0, True)

    @cython.ccall
    def seek(self, unsigned int t, int reset_convergence=True):
        """
        seek(self, t, reset_convergence = True)

        Change the time-state of the filter

        Is usually called to reset the filter to the beginning.
        """
        if not t == 0 and t >= self.model.nobs:
            raise IndexError("Observation index out of range")
        self.t = t

        if reset_convergence:
            self.converged = 0
            self.period_converged = 0

    def __iter__(self):
        return self

    def __call__(self, int filter_method=-1):
        """
        Iterate the filter across the entire set of observations.
        """
        cdef int i

        # Reset the filter method if necessary
        if not filter_method == -1:
            self.set_filter_method(filter_method)

        # Reset the filter
        self.seek(0, True)

        # Perform forward filtering iterations
        for i in range(self.model.nobs):
            next(self)

    def __next__(self):
        """
        Perform an iteration of the Kalman filter
        """
        cdef int filtered_t = self.t
        cdef int predicted_t = self.t
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1

        # Get time subscript, and stop the iterator if at the end
        if not self.t < self.model.nobs:
            raise StopIteration

        # Clear values
        if self.t == 0 or not (self.conserve_memory & MEMORY_NO_LIKELIHOOD):
            self.loglikelihood[self.t] = 0

        # Initialize pointers to current-iteration objects
        self.initialize_statespace_object_pointers()
        self.initialize_filter_object_pointers()

        # Initialize pointers to appropriate Kalman filtering functions
        self.initialize_function_pointers()

        # Convert base arrays into "selected" arrays  
        # - State covariance matrix? $Q_t \to R_t Q_t R_t`$
        # - Missing values: $y_t \to W_t y_t$, $Z_t \to W_t Z_t$, $H_t \to W_t H_t$
        # self.select_state_cov()
        # self.select_missing()
        # self.transform()

        # Post-convergence: copy previous iteration arrays
        self.post_convergence()

        # Prediction step (alternate timing)
        if self.filter_timing == TIMING_INIT_FILTERED:
            # We need to shift back to the previous filtered_* arrays, or to
            # the initial_* arrays if we're at time t==0
            if self.t == 0:
                self._filtered_state = self.model._initial_state
                self._filtered_state_cov = self.model._initial_state_cov
            else:
                self._filtered_state = &self.filtered_state[0, filtered_t-1]
                self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t-1]

            # Perform the prediction step
            self.prediction(self, self.model)
            # self._prediction()

            # Aids to numerical stability
            self.numerical_stability()

            # Now shift back to the current filtered_* arrays (so they can be
            # set in the updating step)
            self._filtered_state = &self.filtered_state[0, filtered_t]
            self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t]

        # Form forecasts
        self.forecasting(self, self.model)
        # self._forecasting()

        # Perform `forecast_error_cov` inversion (or decomposition)
        self.determinant = self.inversion(self, self.model, self.determinant)
        # self.determinant = self._inversion()

        # Updating step
        self.updating(self, self.model)
        # self._updating()

        # Retrieve the loglikelihood
        if not self.conserve_memory & MEMORY_NO_LIKELIHOOD or self.t >= self.loglikelihood_burn:
            self._loglikelihood[0] = (
                self._loglikelihood[0] +
                self.calculate_loglikelihood(self, self.model, self.determinant) +
                # self._calculate_loglikelihood() +
                self.model.collapse_loglikelihood
            )

        # Prediction step (default timing)
        if self.filter_timing == TIMING_INIT_PREDICTED:
            self.prediction(self, self.model)
            # self._prediction()

            # Aids to numerical stability
            self.numerical_stability()

        # Last prediction step (alternate timing)
        if self.filter_timing == TIMING_INIT_FILTERED and self.t == self.model.nobs-1:
            self._predicted_state = &self.predicted_state[0, predicted_t+1]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t+1]
            self.prediction(self, self.model)

        # Check for convergence
        self.check_convergence()

        # If conserving memory, migrate storage: t->t-1, t+1->t
        self.migrate_storage()

        # Advance the time
        self.t += 1

    @cython.cfunc
    @cython.returns(cython.void)
    def _forecasting(self):
        cforecast_univariate(self, self.model)

    @cython.cfunc
    @cython.returns(cnp.complex64_t)
    def _inversion(self):
        cinverse_noop_univariate(self, self.model, self.determinant)

    @cython.cfunc
    @cython.returns(cython.void)
    def _updating(self):
        cupdating_univariate(self, self.model)

    @cython.cfunc
    @cython.returns(cnp.complex64_t)
    def _calculate_loglikelihood(self):
        return cloglikelihood_univariate(self, self.model, self.determinant)

    @cython.cfunc
    @cython.returns(cython.void)
    def _prediction(self):
        cprediction_univariate(self, self.model)

    cdef void initialize_statespace_object_pointers(self) except *:
        cdef:
            int transform_diagonalize = 0
            int transform_generalized_collapse = 0

        # Determine which transformations need to be made
        transform_generalized_collapse = self.filter_method & FILTER_COLLAPSED
        transform_diagonalize = self.filter_method & FILTER_UNIVARIATE

        # Initialize object-level pointers to statespace arrays
        #self.model.initialize_object_pointers(self.t)
        self.model.seek(self.t, transform_diagonalize, transform_generalized_collapse)

        # Handle missing data
        if self.model._nmissing > 0 or (self.model.has_missing and self.filter_method & FILTER_UNIVARIATE):
            # TODO there is likely a way to allow convergence and the univariate filter, but it
            # doesn't work "out-of-the-box" right now
            self.converged = 0

    @cython.cfunc
    @cython.returns(cython.void)
    def initialize_filter_object_pointers(self):
        cdef:
            int t = self.t
            int inc = 1
        # Indices for arrays that may or may not be stored completely
        cdef:
            int forecast_t = t
            int filtered_t = t
            int predicted_t = t
            int gain_t = t
            int smoothing_t = t
            int loglikelihood_t = t
            int std_forecast_t = t
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            forecast_t = 1
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            gain_t = 0
        if self.conserve_memory & MEMORY_NO_SMOOTHING > 0:
            smoothing_t = 0
        if self.conserve_memory & MEMORY_NO_LIKELIHOOD > 0:
            loglikelihood_t = 0
        if self.conserve_memory & MEMORY_NO_STD_FORECAST > 0:
            std_forecast_t = 0

        # Initialize object-level pointers to input arrays
        self._input_state = &self.predicted_state[0, predicted_t]
        self._input_state_cov = &self.predicted_state_cov[0, 0, predicted_t]

        # Copy initialization arrays to input arrays if we're starting the
        # filter
        if t == 0 and self.filter_timing == TIMING_INIT_PREDICTED:
            # `predicted_state[:,0]` $= a_1 =$ `initial_state`  
            # `predicted_state_cov[:,:,0]` $= P_1 =$ `initial_state_cov`  
            # Under the default timing assumption (TIMING_INIT_PREDICTED), the
            # recursion takes $a_t, P_t$ as input, and as a last step computes
            # $a_{t+1}, P_{t+1}$, which can be input for the next recursion.
            # This means that the filter ends by computing $a_{T+1}, P_{T+1}$,
            # so that the predicted_* arrays have time-dimension T+1, rather than
            # T like all the other arrays.
            # Note that $a_{T+1}, P_{T+1}$ should not be in use anywhere.
            # TODO phase out any use of these, and eventually stop computing it
            # This means that the zeroth entry in the time-dimension can hold the
            # input array (even though it is no different than what is held in the
            # initial_state_* arrays).
            blas.ccopy(
                &self.model._k_states, self.model._initial_state, &inc,
                self._input_state, &inc)
            blas.ccopy(
                &self.model._k_states2, self.model._initial_state_cov, &inc,
                self._input_state_cov, &inc)

        # Initialize object-level pointers to output arrays
        self._forecast = &self.forecast[0, forecast_t]
        self._forecast_error = &self.forecast_error[0, forecast_t]
        self._forecast_error_cov = &self.forecast_error_cov[0, 0, forecast_t]
        self._standardized_forecast_error = &self.standardized_forecast_error[0, std_forecast_t]

        self._filtered_state = &self.filtered_state[0, filtered_t]
        self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t]
        
        if self.filter_timing == TIMING_INIT_PREDICTED:
            self._predicted_state = &self.predicted_state[0, predicted_t + 1]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t + 1]
        else:
            self._predicted_state = &self.predicted_state[0, predicted_t]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t]

        self._kalman_gain = &self.kalman_gain[0, 0, gain_t]

        self._loglikelihood = &self.loglikelihood[loglikelihood_t]

        # Initialize object-level pointers to named temporary arrays
        self._tmp1 = &self.tmp1[0, 0, smoothing_t]
        self._tmp2 = &self.tmp2[0, smoothing_t]
        self._tmp3 = &self.tmp3[0, 0, smoothing_t]
        self._tmp4 = &self.tmp4[0, 0, smoothing_t]

    cdef void initialize_function_pointers(self) except *:
        # Filtering method
        if self.filter_method & FILTER_UNIVARIATE:
            self.forecasting = cforecast_univariate
            self.updating = cupdating_univariate
            self.inversion = cinverse_noop_univariate
            self.calculate_loglikelihood = cloglikelihood_univariate
            self.prediction = cprediction_univariate

        # Conventional method
        elif self.filter_method & FILTER_CONVENTIONAL:
            self.forecasting = cforecast_conventional
            self.updating = cupdating_conventional
            self.calculate_loglikelihood = cloglikelihood_conventional
            self.prediction = cprediction_conventional

            # Inversion method
            if self.inversion_method & INVERT_UNIVARIATE and self.k_endog == 1:
                self.inversion = cinverse_univariate
            elif self.inversion_method & SOLVE_CHOLESKY:
                self.inversion = csolve_cholesky
            elif self.inversion_method & SOLVE_LU:
                self.inversion = csolve_lu
            elif self.inversion_method & INVERT_CHOLESKY:
                self.inversion = cinverse_cholesky
            elif self.inversion_method & INVERT_LU:
                self.inversion = cinverse_lu
            else:
                raise NotImplementedError("Invalid inversion method")
        else:
            raise NotImplementedError("Invalid filtering method")

        # Handle completely missing data, can always just use conventional 
        # methods
        if self.model._nmissing == self.model.k_endog:
            # Change the forecasting step to set the forecast at the intercept
            # $d_t$, so that the forecast error is $v_t = y_t - d_t$.
            self.forecasting = cforecast_missing_conventional

            # Change the updating step to just copy $a_{t|t} = a_t$ and
            # $P_{t|t} = P_t$
            self.updating = cupdating_missing_conventional

            # Change the inversion step to inverse to nans.
            self.inversion = cinverse_missing_conventional

            # Change the loglikelihood calculation to give zero.
            self.calculate_loglikelihood = cloglikelihood_missing_conventional

            # The prediction step is the same as the conventional Kalman
            # filter

    @cython.cfunc
    @cython.returns(cython.void)
    def post_convergence(self):
        cdef:
            int inc = 1

        if self.converged:
            # $F_t$
            blas.ccopy(
                &self.k_endog2, self._converged_forecast_error_cov, &inc,
                self._forecast_error_cov, &inc)
            # $P_{t|t}$
            blas.ccopy(
                &self.k_states2, self._converged_filtered_state_cov, &inc,
                self._filtered_state_cov, &inc)
            # $P_t$
            blas.ccopy(
                &self.k_states2, self._converged_predicted_state_cov, &inc,
                self._predicted_state_cov, &inc)
            # $K_t$
            blas.ccopy(
                &self.k_endogstates, self._converged_kalman_gain, &inc,
                self._kalman_gain, &inc)
            # $|F_t|$
            self.determinant = self.converged_determinant

    @cython.cfunc
    @cython.returns(cython.void)
    def numerical_stability(self):
        cdef int i, j
        cdef int predicted_t = self.t
        cdef cnp.complex64_t value

        if self.conserve_memory & MEMORY_NO_PREDICTED:
            predicted_t = 1

        if self.filter_timing == TIMING_INIT_PREDICTED:
            predicted_t += 1

        if self.stability_method & STABILITY_FORCE_SYMMETRY:
            # Enforce symmetry of predicted covariance matrix  
            # $P_{t+1} = 0.5 * (P_{t+1} + P_{t+1}')$  
            # See Grewal (2001), Section 6.3.1.1
            for i in range(self.k_states):
                for j in range(i, self.k_states):
                    value = 0.5 * (
                        self.predicted_state_cov[i, j, predicted_t] +
                        self.predicted_state_cov[j, i, predicted_t]
                    )
                    self.predicted_state_cov[i, j, predicted_t] = value
                    self.predicted_state_cov[j, i, predicted_t] = value

    @cython.cfunc
    @cython.returns(cython.void)
    def check_convergence(self):
        cdef:
            int inc = 1, missing_flag = 0
            cnp.complex64_t alpha = 1.0
            cnp.complex64_t beta = 0.0
            cnp.complex64_t gamma = -1.0
        # Indices for arrays that may or may not be stored completely
        cdef:
            int forecast_t = self.t
            int filtered_t = self.t
            int predicted_t = self.t
            int gain_t = self.t
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            forecast_t = 1
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            gain_t = 0

        # Figure out if there is a missing value
        if self.model.nmissing[self.t] > 0 or (not self.t == 0 and self.model.nmissing[self.t - 1] > 0):
            missing_flag = 1

        if self.time_invariant and not self.converged and not missing_flag:
            # #### Check for steady-state convergence
            # 
            # `tmp0` array used here, dimension $(m \times m)$  
            # `tmp00` array used here, dimension $(1 \times 1)$  
            if self.filter_timing == TIMING_INIT_PREDICTED:
                blas.ccopy(
                    &self.k_states2, self._input_state_cov, &inc,
                    self._tmp0, &inc)
                blas.caxpy(&self.k_states2, &gamma, self._predicted_state_cov, &inc, self._tmp0, &inc)
            elif self.t > 0:
                blas.ccopy(
                    &self.k_states2, &self.predicted_state_cov[0,0,predicted_t], &inc,
                    self._tmp0, &inc)
                blas.caxpy(&self.k_states2, &gamma, &self.predicted_state_cov[0,0,predicted_t-1], &inc, self._tmp0, &inc)
            else:
                return


            blas.cgemv(
                "N", &inc, &self.k_states2, &alpha, self._tmp0, &inc,
                self._tmp0, &inc,
                &beta, self._tmp00, &inc)
            if zabs(self._tmp00[0]) < self.tolerance:
                self.converged = 1
                self.period_converged = self.t

            # If we just converged, copy the current iteration matrices to the
            # converged storage
            if self.converged == 1:
                # $F_t$
                blas.ccopy(
                    &self.k_endog2, &self.forecast_error_cov[0, 0, forecast_t], &inc,
                    self._converged_forecast_error_cov, &inc)
                # $P_{t|t}$
                blas.ccopy(
                    &self.k_states2, &self.filtered_state_cov[0, 0, filtered_t], &inc,
                    self._converged_filtered_state_cov, &inc)
                # $P_t$
                blas.ccopy(
                    &self.k_states2, &self.predicted_state_cov[0, 0, predicted_t], &inc,
                    self._converged_predicted_state_cov, &inc)
                # $|F_t|$
                self.converged_determinant = self.determinant
                # $K_t$
                blas.ccopy(
                    &self.k_endogstates, &self.kalman_gain[0, 0, gain_t], &inc,
                    self._converged_kalman_gain, &inc)

    @cython.cfunc
    @cython.returns(cython.void)
    def migrate_storage(self):
        cdef:
            int inc = 1

        # Forecast: 1 -> 0
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            blas.ccopy(&self.k_endog, &self.forecast[0, 1], &inc, &self.forecast[0, 0], &inc)
            blas.ccopy(&self.k_endog, &self.forecast_error[0, 1], &inc, &self.forecast_error[0, 0], &inc)
            blas.ccopy(&self.k_endog2, &self.forecast_error_cov[0, 0, 1], &inc, &self.forecast_error_cov[0, 0, 0], &inc)

        # Filtered: 1 -> 0
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            blas.ccopy(&self.k_states, &self.filtered_state[0, 1], &inc, &self.filtered_state[0, 0], &inc)
            blas.ccopy(&self.k_states2, &self.filtered_state_cov[0, 0, 1], &inc, &self.filtered_state_cov[0, 0, 0], &inc)

        # Predicted: 1 -> 0
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            blas.ccopy(&self.k_states, &self.predicted_state[0, 1], &inc, &self.predicted_state[0, 0], &inc)
            blas.ccopy(&self.k_states2, &self.predicted_state_cov[0, 0, 1], &inc, &self.predicted_state_cov[0, 0, 0], &inc)

            # Predicted: 2 -> 1
            if self.filter_timing == TIMING_INIT_PREDICTED:
                blas.ccopy(&self.k_states, &self.predicted_state[0, 2], &inc, &self.predicted_state[0, 1], &inc)
                blas.ccopy(&self.k_states2, &self.predicted_state_cov[0, 0, 2], &inc, &self.predicted_state_cov[0, 0, 1], &inc)

# ## Kalman filter
cdef class sKalmanFilter(object):
    """
    sKalmanFilter(model, filter_method=FILTER_CONVENTIONAL, inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY, stability_method=STABILITY_FORCE_SYMMETRY, filter_timing=TIMING_INIT_PREDICTED, tolerance=1e-19)

    A representation of the Kalman filter recursions.

    While the filter is mathematically represented as a recursion, it is here
    translated into Python as a stateful iterator.

    Because there are actually several types of Kalman filter depending on the
    state space model of interest, this class only handles the *iteration*
    aspect of filtering, and delegates the actual operations to four general
    workhorse routines, which can be implemented separately for each type of
    Kalman filter.

    In order to maintain a consistent interface, and because these four general
    routines may be quite different across filter types, their argument is only
    the stateful ?KalmanFilter object. Furthermore, in order to allow the
    different types of filter to substitute alternate matrices, this class
    defines a set of pointers to the various state space arrays and the
    filtering output arrays.

    For example, handling missing observations requires not only substituting
    `obs`, `design`, and `obs_cov` matrices, but the new matrices actually have
    different dimensions than the originals. This can be flexibly accomodated
    simply by replacing e.g. the `obs` pointer to the substituted `obs` array
    and replacing `k_endog` for that iteration. Then in the next iteration, when
    the `obs` vector may be missing different elements (or none at all), it can
    again be redefined.

    Each iteration of the filter (see `__next__`) proceeds in a number of
    steps.

    `initialize_object_pointers` initializes pointers to current-iteration
    objects (i.e. the state space arrays and filter output arrays).  

    `initialize_function_pointers` initializes pointers to the appropriate
    Kalman filtering routines (i.e. `forecast_conventional` or
    `forecast_exact_initial`, etc.).  

    `select_arrays` converts the base arrays into "selected" arrays using
    selection matrices. In particular, it handles the state covariance matrix
    and redefined matrices based on missing values.  

    `post_convergence` handles copying arrays from time $t-1$ to time $t$ when
    the Kalman filter has converged and they don't need to be re-calculated.  

    `forecasting` calls the Kalman filter `forcasting_<filter type>` routine

    `inversion` calls the appropriate function to invert the forecast error
    covariance matrix.  

    `updating` calls the Kalman filter `updating_<filter type>` routine

    `loglikelihood` calls the Kalman filter `loglikelihood_<filter type>` routine

    `prediction` calls the Kalman filter `prediction_<filter type>` routine

    `numerical_stability` performs end-of-iteration tasks to improve the numerical
    stability of the filter 

    `check_convergence` checks for convergence of the filter to steady-state.
    """

    # ### Statespace model
    # cdef readonly sStatespace model

    # ### Filter parameters
    # Holds the time-iteration state of the filter  
    # *Note*: must be changed using the `seek` method
    # cdef readonly int t
    # Holds the tolerance parameter for convergence
    # cdef public cnp.float64_t tolerance
    # Holds the convergence to steady-state status of the filter
    # *Note*: is by default reset each time `seek` is called
    # cdef readonly int converged
    # cdef readonly int period_converged
    # Holds whether or not the model is time-invariant
    # *Note*: is by default reset each time `seek` is called
    # cdef readonly int time_invariant
    # The Kalman filter procedure to use  
    # cdef readonly int filter_method
    # The method by which the terms using the inverse of the forecast
    # error covariance matrix are solved.
    # cdef public int inversion_method
    # Methods to improve numerical stability
    # cdef public int stability_method
    # Whether or not to conserve memory
    # If True, only stores filtered states and covariance matrices
    # cdef readonly int conserve_memory
    # Whether or not to use alternate timing
    # If True, uses the Kim and Nelson (1999) timing
    # cdef readonly int filter_timing
    # If conserving loglikelihood, the number of periods to "burn"
    # before starting to record the loglikelihood
    # cdef readonly int loglikelihood_burn

    # ### Kalman filter properties

    # `loglikelihood` $\equiv \log p(y_t | Y_{t-1})$
    # cdef readonly cnp.float32_t [:] loglikelihood

    # `filtered_state` $\equiv a_{t|t} = E(\alpha_t | Y_t)$ is the **filtered estimator** of the state $(m \times T)$  
    # `predicted_state` $\equiv a_{t+1} = E(\alpha_{t+1} | Y_t)$ is the **one-step ahead predictor** of the state $(m \times T-1)$  
    # `forecast` $\equiv E(y_t|Y_{t-1})$ is the **forecast** of the next observation $(p \times T)$   
    # `forecast_error` $\equiv v_t = y_t - E(y_t|Y_{t-1})$ is the **one-step ahead forecast error** of the next observation $(p \times T)$  
    # 
    # *Note*: Actual values in `filtered_state` will be from 1 to `nobs`+1. Actual
    # values in `predicted_state` will be from 0 to `nobs`+1 because the initialization
    # is copied over to the zeroth entry, and similar for the covariances, below.
    #
    # *Old notation: beta_tt, beta_tt1, y_tt1, eta_tt1*
    # cdef readonly cnp.float32_t [::1,:] filtered_state, predicted_state, forecast, forecast_error

    # `filtered_state_cov` $\equiv P_{t|t} = Var(\alpha_t | Y_t)$ is the **filtered state covariance matrix** $(m \times m \times T)$  
    # `predicted_state_cov` $\equiv P_{t+1} = Var(\alpha_{t+1} | Y_t)$ is the **predicted state covariance matrix** $(m \times m \times T)$  
    # `forecast_error_cov` $\equiv F_t = Var(v_t | Y_{t-1})$ is the **forecast error covariance matrix** $(p \times p \times T)$  
    # 
    # *Old notation: P_tt, P_tt1, f_tt1*
    # cdef readonly cnp.float32_t [::1,:,:] filtered_state_cov, predicted_state_cov, forecast_error_cov

    # `kalman_gain` $\equiv K_{t} = T_t P_t Z_t' F_t^{-1}$ is the **Kalman gain** $(m \times p \times T)$  
    # cdef readonly cnp.float32_t [::1,:,:] kalman_gain

    # ### Steady State Values
    # These matrices are used to hold the converged matrices after the Kalman
    # filter has reached steady-state
    # cdef readonly cnp.float32_t [::1,:] converged_forecast_error_cov
    # cdef readonly cnp.float32_t [::1,:] converged_filtered_state_cov
    # cdef readonly cnp.float32_t [::1,:] converged_predicted_state_cov
    # cdef readonly cnp.float32_t [::1,:] converged_kalman_gain
    # cdef readonly cnp.float32_t converged_determinant

    # ### Temporary arrays
    # These matrices are used to temporarily hold selected observation vectors,
    # design matrices, and observation covariance matrices in the case of
    # missing data.  
    # `forecast_error_fac` is a forecast error covariance matrix **factorization** $(p \times p)$.
    # Depending on the method for handling the inverse of the forecast error covariance matrix, it may be:
    # - a Cholesky factorization if `cholesky_solve` is used
    # - an inverse calculated via Cholesky factorization if `cholesky_inverse` is used
    # - an LU factorization if `lu_solve` is used
    # - an inverse calculated via LU factorization if `lu_inverse` is used
    # cdef readonly cnp.float32_t [::1,:] forecast_error_fac
    # `forecast_error_ipiv` holds pivot indices if an LU decomposition is used
    # cdef readonly int [:] forecast_error_ipiv
    # `forecast_error_work` is a work array for matrix inversion if an LU
    # decomposition is used
    # cdef readonly cnp.float32_t [::1,:] forecast_error_work
    # These hold the memory allocations of the anonymous temporary arrays
    # cdef readonly cnp.float32_t [::1,:] tmp0, tmp00
    # These hold the memory allocations of the named temporary arrays  
    # (these are all time-varying in the last dimension)
    # cdef readonly cnp.float32_t [::1,:] tmp2
    # cdef readonly cnp.float32_t [::1,:,:] tmp1, tmp3

    # Holds the determinant across calculations (this is done because after
    # convergence, it doesn't need to be re-calculated anymore)
    # cdef readonly cnp.float32_t determinant

    # ### Pointers to current-iteration arrays
    # cdef cnp.float32_t * _obs
    # cdef cnp.float32_t * _design
    # cdef cnp.float32_t * _obs_intercept
    # cdef cnp.float32_t * _obs_cov
    # cdef cnp.float32_t * _transition
    # cdef cnp.float32_t * _state_intercept
    # cdef cnp.float32_t * _selection
    # cdef cnp.float32_t * _state_cov
    # cdef cnp.float32_t * _selected_state_cov
    # cdef cnp.float32_t * _initial_state
    # cdef cnp.float32_t * _initial_state_cov

    # cdef cnp.float32_t * _input_state
    # cdef cnp.float32_t * _input_state_cov

    # cdef cnp.float32_t * _forecast
    # cdef cnp.float32_t * _forecast_error
    # cdef cnp.float32_t * _forecast_error_cov
    # cdef cnp.float32_t * _filtered_state
    # cdef cnp.float32_t * _filtered_state_cov
    # cdef cnp.float32_t * _predicted_state
    # cdef cnp.float32_t * _predicted_state_cov

    # cdef cnp.float32_t * _kalman_gain
    # cdef cnp.float32_t * _loglikelihood

    # cdef cnp.float32_t * _converged_forecast_error_cov
    # cdef cnp.float32_t * _converged_filtered_state_cov
    # cdef cnp.float32_t * _converged_predicted_state_cov
    # cdef cnp.float32_t * _converged_kalman_gain

    # cdef cnp.float32_t * _forecast_error_fac
    # cdef int * _forecast_error_ipiv
    # cdef cnp.float32_t * _forecast_error_work

    # cdef cnp.float32_t * _tmp0
    # cdef cnp.float32_t * _tmp00
    # cdef cnp.float32_t * _tmp1
    # cdef cnp.float32_t * _tmp2
    # cdef cnp.float32_t * _tmp3

    # ### Pointers to current-iteration Kalman filtering functions
    # cdef int (*forecasting)(
    #     sKalmanFilter, sStatespace
    # )
    # cdef cnp.float32_t (*inversion)(
    #     sKalmanFilter, sStatespace, cnp.float32_t
    # ) except *
    # cdef int (*updating)(
    #     sKalmanFilter, sStatespace
    # )
    # cdef cnp.float32_t (*calculate_loglikelihood)(
    #     sKalmanFilter, sStatespace, cnp.float32_t
    # )
    # cdef int (*prediction)(
    #     sKalmanFilter, sStatespace
    # )

    # ### Define some constants
    # cdef readonly int k_endog, k_states, k_posdef, k_endog2, k_states2, k_endogstates
    # cdef readonly ldwork
    
    def __init__(self,
                 sStatespace model,
                 int filter_method=FILTER_CONVENTIONAL,
                 int inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY,
                 int stability_method=STABILITY_FORCE_SYMMETRY,
                 int conserve_memory=MEMORY_STORE_ALL,
                 int filter_timing=TIMING_INIT_PREDICTED,
                 cnp.float64_t tolerance=1e-19,
                 int loglikelihood_burn=0):

        # Save the model
        self.model = model

        # Initialize filter parameters
        self.tolerance = tolerance
        self.inversion_method = inversion_method
        self.stability_method = stability_method
        self.conserve_memory = conserve_memory
        self.filter_timing = filter_timing
        self.loglikelihood_burn = loglikelihood_burn

        # Initialize the constant values
        self.time_invariant = self.model.time_invariant

        # TODO replace with optimal work array size
        self.ldwork = self.model.k_endog

        # Set the filter method
        self.set_dimensions()
        self.set_filter_method(filter_method, True)

        # Initialize time and convergence status
        self.t = 0
        self.converged = 0
        self.period_converged = 0

    def __reduce__(self):
        args = (self.model, self.filter_method, self.inversion_method,
                self.stability_method,  self.conserve_memory, self.filter_timing,
                self.tolerance, self.loglikelihood_burn)
        state = {'t': self.t,
                 'converged' : self.converged ,
                 'converged_determinant' : self.converged_determinant,
                 'determinant' : self.determinant,
                 'period_converged' : self.period_converged,
                 'converged_filtered_state_cov': np.array(self.converged_filtered_state_cov, copy=True, order='F'),
                 'converged_forecast_error_cov': np.array(self.converged_forecast_error_cov, copy=True, order='F'),
                 'converged_kalman_gain': np.array(self.converged_kalman_gain, copy=True, order='F'),
                 'converged_predicted_state_cov': np.array(self.converged_predicted_state_cov, copy=True, order='F'),
                 'filtered_state': np.array(self.filtered_state, copy=True, order='F'),
                 'filtered_state_cov': np.array(self.filtered_state_cov, copy=True, order='F'),
                 'forecast': np.array(self.forecast, copy=True, order='F'),
                 'forecast_error': np.array(self.forecast_error, copy=True, order='F'),
                 'forecast_error_cov': np.array(self.forecast_error_cov, copy=True, order='F'),
                 'forecast_error_fac': np.array(self.forecast_error_fac, copy=True, order='F'),
                 'forecast_error_ipiv': np.array(self.forecast_error_ipiv, copy=True, order='F'),
                 'forecast_error_work': np.array(self.forecast_error_work, copy=True, order='F'),
                 'kalman_gain': np.array(self.kalman_gain, copy=True, order='F'),
                 'loglikelihood': np.array(self.loglikelihood, copy=True, order='F'),
                 'predicted_state': np.array(self.predicted_state, copy=True, order='F'),
                 'predicted_state_cov': np.array(self.predicted_state_cov, copy=True, order='F'),
                 'standardized_forecast_error': np.array(self.standardized_forecast_error, copy=True, order='F'),
                 'tmp0': np.array(self.tmp0, copy=True, order='F'),
                 'tmp00': np.array(self.tmp00, copy=True, order='F'),
                 'tmp1': np.array(self.tmp1, copy=True, order='F'),
                 'tmp2': np.array(self.tmp2, copy=True, order='F'),
                 'tmp3': np.array(self.tmp3, copy=True, order='F'),
                 'tmp4': np.array(self.tmp4, copy=True, order='F')
                 }

        return (self.__class__, args, state)

    def __setstate__(self, state):
        self.t = state['t']
        self.converged  = state['converged']
        self.converged_determinant = state['converged_determinant']
        self.determinant = state['determinant']
        self.period_converged = state['period_converged']
        self.converged_filtered_state_cov = state['converged_filtered_state_cov']
        self.converged_forecast_error_cov = state['converged_forecast_error_cov']
        self.converged_kalman_gain = state['converged_kalman_gain']
        self.converged_predicted_state_cov = state['converged_predicted_state_cov']
        self.filtered_state = state['filtered_state']
        self.filtered_state_cov = state['filtered_state_cov']
        self.forecast = state['forecast']
        self.forecast_error = state['forecast_error']
        self.forecast_error_cov = state['forecast_error_cov']
        self.forecast_error_fac = state['forecast_error_fac']
        self.forecast_error_ipiv = state['forecast_error_ipiv']
        self.forecast_error_work = state['forecast_error_work']
        self.kalman_gain = state['kalman_gain']
        self.loglikelihood = state['loglikelihood']
        self.predicted_state = state['predicted_state']
        self.predicted_state_cov = state['predicted_state_cov']
        self.standardized_forecast_error = state['standardized_forecast_error']
        self.tmp0 = state['tmp0']
        self.tmp00 = state['tmp00']
        self.tmp1 = state['tmp1']
        self.tmp2 = state['tmp2']
        self.tmp3 = state['tmp3']
        self.tmp4 = state['tmp4']
        self._reinitialize_pointers()

    cdef void _reinitialize_pointers(self) except *:
        self._converged_forecast_error_cov = &self.converged_forecast_error_cov[0,0]
        self._converged_filtered_state_cov = &self.converged_filtered_state_cov[0,0]
        self._converged_predicted_state_cov = &self.converged_predicted_state_cov[0,0]
        self._converged_kalman_gain = &self.converged_kalman_gain[0,0]
        self._forecast_error_fac = &self.forecast_error_fac[0,0]
        self._forecast_error_work = &self.forecast_error_work[0,0]
        self._forecast_error_ipiv = &self.forecast_error_ipiv[0]
        self._tmp0 = &self.tmp0[0, 0]
        self._tmp00 = &self.tmp00[0, 0]

    cdef allocate_arrays(self):
        # Local variables
        cdef:
            cnp.npy_intp dim1[1]
            cnp.npy_intp dim2[2]
            cnp.npy_intp dim3[3]
        cdef int storage
        # #### Allocate arrays for calculations

        # Arrays for Kalman filter output

        # Forecast
        if self.conserve_memory & MEMORY_NO_FORECAST:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.forecast = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self.forecast_error = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        dim3[0] = self.k_endog; dim3[1] = self.k_endog; dim3[2] = storage;
        self.forecast_error_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT32, FORTRAN)
        # Standardized forecast errors
        if self.conserve_memory & MEMORY_NO_STD_FORECAST > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.standardized_forecast_error = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)

        # Filtered
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_states; dim2[1] = storage;
        self.filtered_state = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = storage;
        self.filtered_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT32, FORTRAN)

        # Predicted
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_states; dim2[1] = storage+1;
        self.predicted_state = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = storage+1;
        self.predicted_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT32, FORTRAN)

        # Kalman Gain
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim3[0] = self.k_states; dim3[1] = self.k_endog; dim3[2] = storage;
        self.kalman_gain = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT32, FORTRAN)

        # Likelihood
        if self.conserve_memory & MEMORY_NO_LIKELIHOOD > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim1[0] = storage
        self.loglikelihood = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT32, FORTRAN)

        # Converged matrices
        dim2[0] = self.k_endog; dim2[1] = self.k_endog;
        self.converged_forecast_error_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self._converged_forecast_error_cov = &self.converged_forecast_error_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.converged_filtered_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self._converged_filtered_state_cov = &self.converged_filtered_state_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.converged_predicted_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self._converged_predicted_state_cov = &self.converged_predicted_state_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_endog;
        self.converged_kalman_gain = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self._converged_kalman_gain = &self.converged_kalman_gain[0,0]

        # #### Arrays for temporary calculations
        # *Note*: in math notation below, a $\\#$ will represent a generic
        # temporary array, and a $\\#_i$ will represent a named temporary array.

        # Arrays related to matrix factorizations / inverses
        dim2[0] = self.k_endog; dim2[1] = self.k_endog;
        self.forecast_error_fac = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self._forecast_error_fac = &self.forecast_error_fac[0,0]
        dim2[0] = self.ldwork; dim2[1] = self.ldwork;
        self.forecast_error_work = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self._forecast_error_work = &self.forecast_error_work[0,0]
        dim1[0] = self.k_endog;
        self.forecast_error_ipiv = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_INT, FORTRAN)
        self._forecast_error_ipiv = &self.forecast_error_ipiv[0]

        # Holds arrays of dimension $(m \times m)$ and $(m \times r)$
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.tmp0 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self._tmp0 = &self.tmp0[0, 0]

        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.tmp00 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self._tmp00 = &self.tmp00[0, 0]

        # Optionally we may not want to store temporary arrays required  
        # for smoothing
        if self.conserve_memory & MEMORY_NO_SMOOTHING > 0:
            storage = 1
        else:
            storage = self.model.nobs

        # Holds arrays of dimension $(m \times p \times T)$  
        # $\\#_1 = P_t Z_t'$
        dim3[0] = self.k_states; dim3[1] = self.k_endog; dim3[2] = storage;
        self.tmp1 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT32, FORTRAN)

        # Holds arrays of dimension $(p \times T)$  
        # $\\#_2 = F_t^{-1} v_t$
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.tmp2 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)

        # Holds arrays of dimension $(p \times m \times T)$  
        # $\\#_3 = F_t^{-1} Z_t$
        dim3[0] = self.k_endog; dim3[1] = self.k_states; dim3[2] = storage;
        self.tmp3 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT32, FORTRAN)

        # Holds arrays of dimension $(p \times p \times T)$  
        # $\\#_4 = F_t^{-1} H_t$
        dim3[0] = self.k_endog; dim3[1] = self.k_endog; dim3[2] = storage;
        self.tmp4 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT32, FORTRAN)

    @cython.cfunc
    @cython.returns(cython.void)
    def set_dimensions(self):
        """
        Set dimensions for the Kalman filter

        These are used *only* to define the shapes of the Kalman filter output
        and temporary arrays in memory. They will not change between iterations
        of the filter.

        They only differ from the sStatespace versions in the case
        that the FILTER_COLLAPSED flag is set, in which case model.k_endog
        and kfilter.k_endog will be different
        (since kfilter.k_endog = model.k_states).

        Across *iterations* of the Kalman filter, both model.k_* and
        kfilter.k_* are fixed, although model._k_* may be different from either
        when there is missing data in a given period's observations.

        The actual dimension of the *data* being considered at a given
        iteration is always given by model._k_* variables, which take into
        account both FILTER_COLLAPSED and missing data.

        But, the dimension *in memory* of the Kalman filter arrays will always
        be given by kfilter.k_*.

        The following relations will always hold:

        kfilter.k_endog = model.k_states if self.filter_method & FILTER_COLLAPSED else model.k_endog
        kfilter.k_endog = model._k_endog + model._nmissing
        """
        self.k_endog = self.model.k_states if self.filter_method & FILTER_COLLAPSED else self.model.k_endog
        self.k_states = self.model.k_states
        self.k_posdef = self.model.k_posdef
        self.k_endog2 = self.k_endog**2
        self.k_states2 = self.k_states**2
        self.k_posdef2 = self.k_posdef**2
        self.k_endogstates = self.k_endog * self.k_states
        self.k_statesposdef = self.k_states * self.k_posdef

    @cython.ccall
    def set_filter_method(self, int filter_method, int force_reset=True):
        """
        set_filter_method(self, filter_method, force_reset=True)

        Change the filter method.
        """
        if not filter_method == self.filter_method or force_reset:
            # Check for invalid filter methods
            if filter_method & FILTER_COLLAPSED and self.k_endog <= self.k_states:
                raise RuntimeError('Cannot collapse observation vector if the'
                                   ' state dimension is equal to or larger than the'
                                   ' dimension of the observation vector.')

            # Change the smoother output flag
            self.filter_method = filter_method

            # Reset dimensions
            self.set_dimensions()

            # Reset matrices
            self.allocate_arrays()

            # Seek to the beginning
            self.seek(0, True)

    @cython.ccall
    def seek(self, unsigned int t, int reset_convergence=True):
        """
        seek(self, t, reset_convergence = True)

        Change the time-state of the filter

        Is usually called to reset the filter to the beginning.
        """
        if not t == 0 and t >= self.model.nobs:
            raise IndexError("Observation index out of range")
        self.t = t

        if reset_convergence:
            self.converged = 0
            self.period_converged = 0

    def __iter__(self):
        return self

    def __call__(self, int filter_method=-1):
        """
        Iterate the filter across the entire set of observations.
        """
        cdef int i

        # Reset the filter method if necessary
        if not filter_method == -1:
            self.set_filter_method(filter_method)

        # Reset the filter
        self.seek(0, True)

        # Perform forward filtering iterations
        for i in range(self.model.nobs):
            next(self)

    def __next__(self):
        """
        Perform an iteration of the Kalman filter
        """
        cdef int filtered_t = self.t
        cdef int predicted_t = self.t
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1

        # Get time subscript, and stop the iterator if at the end
        if not self.t < self.model.nobs:
            raise StopIteration

        # Clear values
        if self.t == 0 or not (self.conserve_memory & MEMORY_NO_LIKELIHOOD):
            self.loglikelihood[self.t] = 0

        # Initialize pointers to current-iteration objects
        self.initialize_statespace_object_pointers()
        self.initialize_filter_object_pointers()

        # Initialize pointers to appropriate Kalman filtering functions
        self.initialize_function_pointers()

        # Convert base arrays into "selected" arrays  
        # - State covariance matrix? $Q_t \to R_t Q_t R_t`$
        # - Missing values: $y_t \to W_t y_t$, $Z_t \to W_t Z_t$, $H_t \to W_t H_t$
        # self.select_state_cov()
        # self.select_missing()
        # self.transform()

        # Post-convergence: copy previous iteration arrays
        self.post_convergence()

        # Prediction step (alternate timing)
        if self.filter_timing == TIMING_INIT_FILTERED:
            # We need to shift back to the previous filtered_* arrays, or to
            # the initial_* arrays if we're at time t==0
            if self.t == 0:
                self._filtered_state = self.model._initial_state
                self._filtered_state_cov = self.model._initial_state_cov
            else:
                self._filtered_state = &self.filtered_state[0, filtered_t-1]
                self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t-1]

            # Perform the prediction step
            self.prediction(self, self.model)
            # self._prediction()

            # Aids to numerical stability
            self.numerical_stability()

            # Now shift back to the current filtered_* arrays (so they can be
            # set in the updating step)
            self._filtered_state = &self.filtered_state[0, filtered_t]
            self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t]

        # Form forecasts
        self.forecasting(self, self.model)
        # self._forecasting()

        # Perform `forecast_error_cov` inversion (or decomposition)
        self.determinant = self.inversion(self, self.model, self.determinant)
        # self.determinant = self._inversion()

        # Updating step
        self.updating(self, self.model)
        # self._updating()

        # Retrieve the loglikelihood
        if not self.conserve_memory & MEMORY_NO_LIKELIHOOD or self.t >= self.loglikelihood_burn:
            self._loglikelihood[0] = (
                self._loglikelihood[0] +
                self.calculate_loglikelihood(self, self.model, self.determinant) +
                # self._calculate_loglikelihood() +
                self.model.collapse_loglikelihood
            )

        # Prediction step (default timing)
        if self.filter_timing == TIMING_INIT_PREDICTED:
            self.prediction(self, self.model)
            # self._prediction()

            # Aids to numerical stability
            self.numerical_stability()

        # Last prediction step (alternate timing)
        if self.filter_timing == TIMING_INIT_FILTERED and self.t == self.model.nobs-1:
            self._predicted_state = &self.predicted_state[0, predicted_t+1]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t+1]
            self.prediction(self, self.model)

        # Check for convergence
        self.check_convergence()

        # If conserving memory, migrate storage: t->t-1, t+1->t
        self.migrate_storage()

        # Advance the time
        self.t += 1

    @cython.cfunc
    @cython.returns(cython.void)
    def _forecasting(self):
        sforecast_univariate(self, self.model)

    @cython.cfunc
    @cython.returns(cnp.float32_t)
    def _inversion(self):
        sinverse_noop_univariate(self, self.model, self.determinant)

    @cython.cfunc
    @cython.returns(cython.void)
    def _updating(self):
        supdating_univariate(self, self.model)

    @cython.cfunc
    @cython.returns(cnp.float32_t)
    def _calculate_loglikelihood(self):
        return sloglikelihood_univariate(self, self.model, self.determinant)

    @cython.cfunc
    @cython.returns(cython.void)
    def _prediction(self):
        sprediction_univariate(self, self.model)

    cdef void initialize_statespace_object_pointers(self) except *:
        cdef:
            int transform_diagonalize = 0
            int transform_generalized_collapse = 0

        # Determine which transformations need to be made
        transform_generalized_collapse = self.filter_method & FILTER_COLLAPSED
        transform_diagonalize = self.filter_method & FILTER_UNIVARIATE

        # Initialize object-level pointers to statespace arrays
        #self.model.initialize_object_pointers(self.t)
        self.model.seek(self.t, transform_diagonalize, transform_generalized_collapse)

        # Handle missing data
        if self.model._nmissing > 0 or (self.model.has_missing and self.filter_method & FILTER_UNIVARIATE):
            # TODO there is likely a way to allow convergence and the univariate filter, but it
            # doesn't work "out-of-the-box" right now
            self.converged = 0

    @cython.cfunc
    @cython.returns(cython.void)
    def initialize_filter_object_pointers(self):
        cdef:
            int t = self.t
            int inc = 1
        # Indices for arrays that may or may not be stored completely
        cdef:
            int forecast_t = t
            int filtered_t = t
            int predicted_t = t
            int gain_t = t
            int smoothing_t = t
            int loglikelihood_t = t
            int std_forecast_t = t
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            forecast_t = 1
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            gain_t = 0
        if self.conserve_memory & MEMORY_NO_SMOOTHING > 0:
            smoothing_t = 0
        if self.conserve_memory & MEMORY_NO_LIKELIHOOD > 0:
            loglikelihood_t = 0
        if self.conserve_memory & MEMORY_NO_STD_FORECAST > 0:
            std_forecast_t = 0

        # Initialize object-level pointers to input arrays
        self._input_state = &self.predicted_state[0, predicted_t]
        self._input_state_cov = &self.predicted_state_cov[0, 0, predicted_t]

        # Copy initialization arrays to input arrays if we're starting the
        # filter
        if t == 0 and self.filter_timing == TIMING_INIT_PREDICTED:
            # `predicted_state[:,0]` $= a_1 =$ `initial_state`  
            # `predicted_state_cov[:,:,0]` $= P_1 =$ `initial_state_cov`  
            # Under the default timing assumption (TIMING_INIT_PREDICTED), the
            # recursion takes $a_t, P_t$ as input, and as a last step computes
            # $a_{t+1}, P_{t+1}$, which can be input for the next recursion.
            # This means that the filter ends by computing $a_{T+1}, P_{T+1}$,
            # so that the predicted_* arrays have time-dimension T+1, rather than
            # T like all the other arrays.
            # Note that $a_{T+1}, P_{T+1}$ should not be in use anywhere.
            # TODO phase out any use of these, and eventually stop computing it
            # This means that the zeroth entry in the time-dimension can hold the
            # input array (even though it is no different than what is held in the
            # initial_state_* arrays).
            blas.scopy(
                &self.model._k_states, self.model._initial_state, &inc,
                self._input_state, &inc)
            blas.scopy(
                &self.model._k_states2, self.model._initial_state_cov, &inc,
                self._input_state_cov, &inc)

        # Initialize object-level pointers to output arrays
        self._forecast = &self.forecast[0, forecast_t]
        self._forecast_error = &self.forecast_error[0, forecast_t]
        self._forecast_error_cov = &self.forecast_error_cov[0, 0, forecast_t]
        self._standardized_forecast_error = &self.standardized_forecast_error[0, std_forecast_t]

        self._filtered_state = &self.filtered_state[0, filtered_t]
        self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t]
        
        if self.filter_timing == TIMING_INIT_PREDICTED:
            self._predicted_state = &self.predicted_state[0, predicted_t + 1]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t + 1]
        else:
            self._predicted_state = &self.predicted_state[0, predicted_t]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t]

        self._kalman_gain = &self.kalman_gain[0, 0, gain_t]

        self._loglikelihood = &self.loglikelihood[loglikelihood_t]

        # Initialize object-level pointers to named temporary arrays
        self._tmp1 = &self.tmp1[0, 0, smoothing_t]
        self._tmp2 = &self.tmp2[0, smoothing_t]
        self._tmp3 = &self.tmp3[0, 0, smoothing_t]
        self._tmp4 = &self.tmp4[0, 0, smoothing_t]

    cdef void initialize_function_pointers(self) except *:
        # Filtering method
        if self.filter_method & FILTER_UNIVARIATE:
            self.forecasting = sforecast_univariate
            self.updating = supdating_univariate
            self.inversion = sinverse_noop_univariate
            self.calculate_loglikelihood = sloglikelihood_univariate
            self.prediction = sprediction_univariate

        # Conventional method
        elif self.filter_method & FILTER_CONVENTIONAL:
            self.forecasting = sforecast_conventional
            self.updating = supdating_conventional
            self.calculate_loglikelihood = sloglikelihood_conventional
            self.prediction = sprediction_conventional

            # Inversion method
            if self.inversion_method & INVERT_UNIVARIATE and self.k_endog == 1:
                self.inversion = sinverse_univariate
            elif self.inversion_method & SOLVE_CHOLESKY:
                self.inversion = ssolve_cholesky
            elif self.inversion_method & SOLVE_LU:
                self.inversion = ssolve_lu
            elif self.inversion_method & INVERT_CHOLESKY:
                self.inversion = sinverse_cholesky
            elif self.inversion_method & INVERT_LU:
                self.inversion = sinverse_lu
            else:
                raise NotImplementedError("Invalid inversion method")
        else:
            raise NotImplementedError("Invalid filtering method")

        # Handle completely missing data, can always just use conventional 
        # methods
        if self.model._nmissing == self.model.k_endog:
            # Change the forecasting step to set the forecast at the intercept
            # $d_t$, so that the forecast error is $v_t = y_t - d_t$.
            self.forecasting = sforecast_missing_conventional

            # Change the updating step to just copy $a_{t|t} = a_t$ and
            # $P_{t|t} = P_t$
            self.updating = supdating_missing_conventional

            # Change the inversion step to inverse to nans.
            self.inversion = sinverse_missing_conventional

            # Change the loglikelihood calculation to give zero.
            self.calculate_loglikelihood = sloglikelihood_missing_conventional

            # The prediction step is the same as the conventional Kalman
            # filter

    @cython.cfunc
    @cython.returns(cython.void)
    def post_convergence(self):
        cdef:
            int inc = 1

        if self.converged:
            # $F_t$
            blas.scopy(
                &self.k_endog2, self._converged_forecast_error_cov, &inc,
                self._forecast_error_cov, &inc)
            # $P_{t|t}$
            blas.scopy(
                &self.k_states2, self._converged_filtered_state_cov, &inc,
                self._filtered_state_cov, &inc)
            # $P_t$
            blas.scopy(
                &self.k_states2, self._converged_predicted_state_cov, &inc,
                self._predicted_state_cov, &inc)
            # $K_t$
            blas.scopy(
                &self.k_endogstates, self._converged_kalman_gain, &inc,
                self._kalman_gain, &inc)
            # $|F_t|$
            self.determinant = self.converged_determinant

    @cython.cfunc
    @cython.returns(cython.void)
    def numerical_stability(self):
        cdef int i, j
        cdef int predicted_t = self.t
        cdef cnp.float32_t value

        if self.conserve_memory & MEMORY_NO_PREDICTED:
            predicted_t = 1

        if self.filter_timing == TIMING_INIT_PREDICTED:
            predicted_t += 1

        if self.stability_method & STABILITY_FORCE_SYMMETRY:
            # Enforce symmetry of predicted covariance matrix  
            # $P_{t+1} = 0.5 * (P_{t+1} + P_{t+1}')$  
            # See Grewal (2001), Section 6.3.1.1
            for i in range(self.k_states):
                for j in range(i, self.k_states):
                    value = 0.5 * (
                        self.predicted_state_cov[i, j, predicted_t] +
                        self.predicted_state_cov[j, i, predicted_t]
                    )
                    self.predicted_state_cov[i, j, predicted_t] = value
                    self.predicted_state_cov[j, i, predicted_t] = value

    @cython.cfunc
    @cython.returns(cython.void)
    def check_convergence(self):
        cdef:
            int inc = 1, missing_flag = 0
            cnp.float32_t alpha = 1.0
            cnp.float32_t beta = 0.0
            cnp.float32_t gamma = -1.0
        # Indices for arrays that may or may not be stored completely
        cdef:
            int forecast_t = self.t
            int filtered_t = self.t
            int predicted_t = self.t
            int gain_t = self.t
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            forecast_t = 1
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            gain_t = 0

        # Figure out if there is a missing value
        if self.model.nmissing[self.t] > 0 or (not self.t == 0 and self.model.nmissing[self.t - 1] > 0):
            missing_flag = 1

        if self.time_invariant and not self.converged and not missing_flag:
            # #### Check for steady-state convergence
            # 
            # `tmp0` array used here, dimension $(m \times m)$  
            # `tmp00` array used here, dimension $(1 \times 1)$  
            if self.filter_timing == TIMING_INIT_PREDICTED:
                blas.scopy(
                    &self.k_states2, self._input_state_cov, &inc,
                    self._tmp0, &inc)
                blas.saxpy(&self.k_states2, &gamma, self._predicted_state_cov, &inc, self._tmp0, &inc)
            elif self.t > 0:
                blas.scopy(
                    &self.k_states2, &self.predicted_state_cov[0,0,predicted_t], &inc,
                    self._tmp0, &inc)
                blas.saxpy(&self.k_states2, &gamma, &self.predicted_state_cov[0,0,predicted_t-1], &inc, self._tmp0, &inc)
            else:
                return


            if blas.sdot(&self.k_states2, self._tmp0, &inc, self._tmp0, &inc) < self.tolerance:
                self.converged = 1
                self.period_converged = self.t


            # If we just converged, copy the current iteration matrices to the
            # converged storage
            if self.converged == 1:
                # $F_t$
                blas.scopy(
                    &self.k_endog2, &self.forecast_error_cov[0, 0, forecast_t], &inc,
                    self._converged_forecast_error_cov, &inc)
                # $P_{t|t}$
                blas.scopy(
                    &self.k_states2, &self.filtered_state_cov[0, 0, filtered_t], &inc,
                    self._converged_filtered_state_cov, &inc)
                # $P_t$
                blas.scopy(
                    &self.k_states2, &self.predicted_state_cov[0, 0, predicted_t], &inc,
                    self._converged_predicted_state_cov, &inc)
                # $|F_t|$
                self.converged_determinant = self.determinant
                # $K_t$
                blas.scopy(
                    &self.k_endogstates, &self.kalman_gain[0, 0, gain_t], &inc,
                    self._converged_kalman_gain, &inc)

    @cython.cfunc
    @cython.returns(cython.void)
    def migrate_storage(self):
        cdef:
            int inc = 1

        # Forecast: 1 -> 0
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            blas.scopy(&self.k_endog, &self.forecast[0, 1], &inc, &self.forecast[0, 0], &inc)
            blas.scopy(&self.k_endog, &self.forecast_error[0, 1], &inc, &self.forecast_error[0, 0], &inc)
            blas.scopy(&self.k_endog2, &self.forecast_error_cov[0, 0, 1], &inc, &self.forecast_error_cov[0, 0, 0], &inc)

        # Filtered: 1 -> 0
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            blas.scopy(&self.k_states, &self.filtered_state[0, 1], &inc, &self.filtered_state[0, 0], &inc)
            blas.scopy(&self.k_states2, &self.filtered_state_cov[0, 0, 1], &inc, &self.filtered_state_cov[0, 0, 0], &inc)

        # Predicted: 1 -> 0
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            blas.scopy(&self.k_states, &self.predicted_state[0, 1], &inc, &self.predicted_state[0, 0], &inc)
            blas.scopy(&self.k_states2, &self.predicted_state_cov[0, 0, 1], &inc, &self.predicted_state_cov[0, 0, 0], &inc)

            # Predicted: 2 -> 1
            if self.filter_timing == TIMING_INIT_PREDICTED:
                blas.scopy(&self.k_states, &self.predicted_state[0, 2], &inc, &self.predicted_state[0, 1], &inc)
                blas.scopy(&self.k_states2, &self.predicted_state_cov[0, 0, 2], &inc, &self.predicted_state_cov[0, 0, 1], &inc)

# ## Kalman filter
cdef class zKalmanFilter(object):
    """
    zKalmanFilter(model, filter_method=FILTER_CONVENTIONAL, inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY, stability_method=STABILITY_FORCE_SYMMETRY, filter_timing=TIMING_INIT_PREDICTED, tolerance=1e-19)

    A representation of the Kalman filter recursions.

    While the filter is mathematically represented as a recursion, it is here
    translated into Python as a stateful iterator.

    Because there are actually several types of Kalman filter depending on the
    state space model of interest, this class only handles the *iteration*
    aspect of filtering, and delegates the actual operations to four general
    workhorse routines, which can be implemented separately for each type of
    Kalman filter.

    In order to maintain a consistent interface, and because these four general
    routines may be quite different across filter types, their argument is only
    the stateful ?KalmanFilter object. Furthermore, in order to allow the
    different types of filter to substitute alternate matrices, this class
    defines a set of pointers to the various state space arrays and the
    filtering output arrays.

    For example, handling missing observations requires not only substituting
    `obs`, `design`, and `obs_cov` matrices, but the new matrices actually have
    different dimensions than the originals. This can be flexibly accomodated
    simply by replacing e.g. the `obs` pointer to the substituted `obs` array
    and replacing `k_endog` for that iteration. Then in the next iteration, when
    the `obs` vector may be missing different elements (or none at all), it can
    again be redefined.

    Each iteration of the filter (see `__next__`) proceeds in a number of
    steps.

    `initialize_object_pointers` initializes pointers to current-iteration
    objects (i.e. the state space arrays and filter output arrays).  

    `initialize_function_pointers` initializes pointers to the appropriate
    Kalman filtering routines (i.e. `forecast_conventional` or
    `forecast_exact_initial`, etc.).  

    `select_arrays` converts the base arrays into "selected" arrays using
    selection matrices. In particular, it handles the state covariance matrix
    and redefined matrices based on missing values.  

    `post_convergence` handles copying arrays from time $t-1$ to time $t$ when
    the Kalman filter has converged and they don't need to be re-calculated.  

    `forecasting` calls the Kalman filter `forcasting_<filter type>` routine

    `inversion` calls the appropriate function to invert the forecast error
    covariance matrix.  

    `updating` calls the Kalman filter `updating_<filter type>` routine

    `loglikelihood` calls the Kalman filter `loglikelihood_<filter type>` routine

    `prediction` calls the Kalman filter `prediction_<filter type>` routine

    `numerical_stability` performs end-of-iteration tasks to improve the numerical
    stability of the filter 

    `check_convergence` checks for convergence of the filter to steady-state.
    """

    # ### Statespace model
    # cdef readonly zStatespace model

    # ### Filter parameters
    # Holds the time-iteration state of the filter  
    # *Note*: must be changed using the `seek` method
    # cdef readonly int t
    # Holds the tolerance parameter for convergence
    # cdef public cnp.float64_t tolerance
    # Holds the convergence to steady-state status of the filter
    # *Note*: is by default reset each time `seek` is called
    # cdef readonly int converged
    # cdef readonly int period_converged
    # Holds whether or not the model is time-invariant
    # *Note*: is by default reset each time `seek` is called
    # cdef readonly int time_invariant
    # The Kalman filter procedure to use  
    # cdef readonly int filter_method
    # The method by which the terms using the inverse of the forecast
    # error covariance matrix are solved.
    # cdef public int inversion_method
    # Methods to improve numerical stability
    # cdef public int stability_method
    # Whether or not to conserve memory
    # If True, only stores filtered states and covariance matrices
    # cdef readonly int conserve_memory
    # Whether or not to use alternate timing
    # If True, uses the Kim and Nelson (1999) timing
    # cdef readonly int filter_timing
    # If conserving loglikelihood, the number of periods to "burn"
    # before starting to record the loglikelihood
    # cdef readonly int loglikelihood_burn

    # ### Kalman filter properties

    # `loglikelihood` $\equiv \log p(y_t | Y_{t-1})$
    # cdef readonly cnp.complex128_t [:] loglikelihood

    # `filtered_state` $\equiv a_{t|t} = E(\alpha_t | Y_t)$ is the **filtered estimator** of the state $(m \times T)$  
    # `predicted_state` $\equiv a_{t+1} = E(\alpha_{t+1} | Y_t)$ is the **one-step ahead predictor** of the state $(m \times T-1)$  
    # `forecast` $\equiv E(y_t|Y_{t-1})$ is the **forecast** of the next observation $(p \times T)$   
    # `forecast_error` $\equiv v_t = y_t - E(y_t|Y_{t-1})$ is the **one-step ahead forecast error** of the next observation $(p \times T)$  
    # 
    # *Note*: Actual values in `filtered_state` will be from 1 to `nobs`+1. Actual
    # values in `predicted_state` will be from 0 to `nobs`+1 because the initialization
    # is copied over to the zeroth entry, and similar for the covariances, below.
    #
    # *Old notation: beta_tt, beta_tt1, y_tt1, eta_tt1*
    # cdef readonly cnp.complex128_t [::1,:] filtered_state, predicted_state, forecast, forecast_error

    # `filtered_state_cov` $\equiv P_{t|t} = Var(\alpha_t | Y_t)$ is the **filtered state covariance matrix** $(m \times m \times T)$  
    # `predicted_state_cov` $\equiv P_{t+1} = Var(\alpha_{t+1} | Y_t)$ is the **predicted state covariance matrix** $(m \times m \times T)$  
    # `forecast_error_cov` $\equiv F_t = Var(v_t | Y_{t-1})$ is the **forecast error covariance matrix** $(p \times p \times T)$  
    # 
    # *Old notation: P_tt, P_tt1, f_tt1*
    # cdef readonly cnp.complex128_t [::1,:,:] filtered_state_cov, predicted_state_cov, forecast_error_cov

    # `kalman_gain` $\equiv K_{t} = T_t P_t Z_t' F_t^{-1}$ is the **Kalman gain** $(m \times p \times T)$  
    # cdef readonly cnp.complex128_t [::1,:,:] kalman_gain

    # ### Steady State Values
    # These matrices are used to hold the converged matrices after the Kalman
    # filter has reached steady-state
    # cdef readonly cnp.complex128_t [::1,:] converged_forecast_error_cov
    # cdef readonly cnp.complex128_t [::1,:] converged_filtered_state_cov
    # cdef readonly cnp.complex128_t [::1,:] converged_predicted_state_cov
    # cdef readonly cnp.complex128_t [::1,:] converged_kalman_gain
    # cdef readonly cnp.complex128_t converged_determinant

    # ### Temporary arrays
    # These matrices are used to temporarily hold selected observation vectors,
    # design matrices, and observation covariance matrices in the case of
    # missing data.  
    # `forecast_error_fac` is a forecast error covariance matrix **factorization** $(p \times p)$.
    # Depending on the method for handling the inverse of the forecast error covariance matrix, it may be:
    # - a Cholesky factorization if `cholesky_solve` is used
    # - an inverse calculated via Cholesky factorization if `cholesky_inverse` is used
    # - an LU factorization if `lu_solve` is used
    # - an inverse calculated via LU factorization if `lu_inverse` is used
    # cdef readonly cnp.complex128_t [::1,:] forecast_error_fac
    # `forecast_error_ipiv` holds pivot indices if an LU decomposition is used
    # cdef readonly int [:] forecast_error_ipiv
    # `forecast_error_work` is a work array for matrix inversion if an LU
    # decomposition is used
    # cdef readonly cnp.complex128_t [::1,:] forecast_error_work
    # These hold the memory allocations of the anonymous temporary arrays
    # cdef readonly cnp.complex128_t [::1,:] tmp0, tmp00
    # These hold the memory allocations of the named temporary arrays  
    # (these are all time-varying in the last dimension)
    # cdef readonly cnp.complex128_t [::1,:] tmp2
    # cdef readonly cnp.complex128_t [::1,:,:] tmp1, tmp3

    # Holds the determinant across calculations (this is done because after
    # convergence, it doesn't need to be re-calculated anymore)
    # cdef readonly cnp.complex128_t determinant

    # ### Pointers to current-iteration arrays
    # cdef cnp.complex128_t * _obs
    # cdef cnp.complex128_t * _design
    # cdef cnp.complex128_t * _obs_intercept
    # cdef cnp.complex128_t * _obs_cov
    # cdef cnp.complex128_t * _transition
    # cdef cnp.complex128_t * _state_intercept
    # cdef cnp.complex128_t * _selection
    # cdef cnp.complex128_t * _state_cov
    # cdef cnp.complex128_t * _selected_state_cov
    # cdef cnp.complex128_t * _initial_state
    # cdef cnp.complex128_t * _initial_state_cov

    # cdef cnp.complex128_t * _input_state
    # cdef cnp.complex128_t * _input_state_cov

    # cdef cnp.complex128_t * _forecast
    # cdef cnp.complex128_t * _forecast_error
    # cdef cnp.complex128_t * _forecast_error_cov
    # cdef cnp.complex128_t * _filtered_state
    # cdef cnp.complex128_t * _filtered_state_cov
    # cdef cnp.complex128_t * _predicted_state
    # cdef cnp.complex128_t * _predicted_state_cov

    # cdef cnp.complex128_t * _kalman_gain
    # cdef cnp.complex128_t * _loglikelihood

    # cdef cnp.complex128_t * _converged_forecast_error_cov
    # cdef cnp.complex128_t * _converged_filtered_state_cov
    # cdef cnp.complex128_t * _converged_predicted_state_cov
    # cdef cnp.complex128_t * _converged_kalman_gain

    # cdef cnp.complex128_t * _forecast_error_fac
    # cdef int * _forecast_error_ipiv
    # cdef cnp.complex128_t * _forecast_error_work

    # cdef cnp.complex128_t * _tmp0
    # cdef cnp.complex128_t * _tmp00
    # cdef cnp.complex128_t * _tmp1
    # cdef cnp.complex128_t * _tmp2
    # cdef cnp.complex128_t * _tmp3

    # ### Pointers to current-iteration Kalman filtering functions
    # cdef int (*forecasting)(
    #     zKalmanFilter, zStatespace
    # )
    # cdef cnp.complex128_t (*inversion)(
    #     zKalmanFilter, zStatespace, cnp.complex128_t
    # ) except *
    # cdef int (*updating)(
    #     zKalmanFilter, zStatespace
    # )
    # cdef cnp.complex128_t (*calculate_loglikelihood)(
    #     zKalmanFilter, zStatespace, cnp.complex128_t
    # )
    # cdef int (*prediction)(
    #     zKalmanFilter, zStatespace
    # )

    # ### Define some constants
    # cdef readonly int k_endog, k_states, k_posdef, k_endog2, k_states2, k_endogstates
    # cdef readonly ldwork
    
    def __init__(self,
                 zStatespace model,
                 int filter_method=FILTER_CONVENTIONAL,
                 int inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY,
                 int stability_method=STABILITY_FORCE_SYMMETRY,
                 int conserve_memory=MEMORY_STORE_ALL,
                 int filter_timing=TIMING_INIT_PREDICTED,
                 cnp.float64_t tolerance=1e-19,
                 int loglikelihood_burn=0):

        # Save the model
        self.model = model

        # Initialize filter parameters
        self.tolerance = tolerance
        self.inversion_method = inversion_method
        self.stability_method = stability_method
        self.conserve_memory = conserve_memory
        self.filter_timing = filter_timing
        self.loglikelihood_burn = loglikelihood_burn

        # Initialize the constant values
        self.time_invariant = self.model.time_invariant

        # TODO replace with optimal work array size
        self.ldwork = self.model.k_endog

        # Set the filter method
        self.set_dimensions()
        self.set_filter_method(filter_method, True)

        # Initialize time and convergence status
        self.t = 0
        self.converged = 0
        self.period_converged = 0

    def __reduce__(self):
        args = (self.model, self.filter_method, self.inversion_method,
                self.stability_method,  self.conserve_memory, self.filter_timing,
                self.tolerance, self.loglikelihood_burn)
        state = {'t': self.t,
                 'converged' : self.converged ,
                 'converged_determinant' : self.converged_determinant,
                 'determinant' : self.determinant,
                 'period_converged' : self.period_converged,
                 'converged_filtered_state_cov': np.array(self.converged_filtered_state_cov, copy=True, order='F'),
                 'converged_forecast_error_cov': np.array(self.converged_forecast_error_cov, copy=True, order='F'),
                 'converged_kalman_gain': np.array(self.converged_kalman_gain, copy=True, order='F'),
                 'converged_predicted_state_cov': np.array(self.converged_predicted_state_cov, copy=True, order='F'),
                 'filtered_state': np.array(self.filtered_state, copy=True, order='F'),
                 'filtered_state_cov': np.array(self.filtered_state_cov, copy=True, order='F'),
                 'forecast': np.array(self.forecast, copy=True, order='F'),
                 'forecast_error': np.array(self.forecast_error, copy=True, order='F'),
                 'forecast_error_cov': np.array(self.forecast_error_cov, copy=True, order='F'),
                 'forecast_error_fac': np.array(self.forecast_error_fac, copy=True, order='F'),
                 'forecast_error_ipiv': np.array(self.forecast_error_ipiv, copy=True, order='F'),
                 'forecast_error_work': np.array(self.forecast_error_work, copy=True, order='F'),
                 'kalman_gain': np.array(self.kalman_gain, copy=True, order='F'),
                 'loglikelihood': np.array(self.loglikelihood, copy=True, order='F'),
                 'predicted_state': np.array(self.predicted_state, copy=True, order='F'),
                 'predicted_state_cov': np.array(self.predicted_state_cov, copy=True, order='F'),
                 'standardized_forecast_error': np.array(self.standardized_forecast_error, copy=True, order='F'),
                 'tmp0': np.array(self.tmp0, copy=True, order='F'),
                 'tmp00': np.array(self.tmp00, copy=True, order='F'),
                 'tmp1': np.array(self.tmp1, copy=True, order='F'),
                 'tmp2': np.array(self.tmp2, copy=True, order='F'),
                 'tmp3': np.array(self.tmp3, copy=True, order='F'),
                 'tmp4': np.array(self.tmp4, copy=True, order='F')
                 }

        return (self.__class__, args, state)

    def __setstate__(self, state):
        self.t = state['t']
        self.converged  = state['converged']
        self.converged_determinant = state['converged_determinant']
        self.determinant = state['determinant']
        self.period_converged = state['period_converged']
        self.converged_filtered_state_cov = state['converged_filtered_state_cov']
        self.converged_forecast_error_cov = state['converged_forecast_error_cov']
        self.converged_kalman_gain = state['converged_kalman_gain']
        self.converged_predicted_state_cov = state['converged_predicted_state_cov']
        self.filtered_state = state['filtered_state']
        self.filtered_state_cov = state['filtered_state_cov']
        self.forecast = state['forecast']
        self.forecast_error = state['forecast_error']
        self.forecast_error_cov = state['forecast_error_cov']
        self.forecast_error_fac = state['forecast_error_fac']
        self.forecast_error_ipiv = state['forecast_error_ipiv']
        self.forecast_error_work = state['forecast_error_work']
        self.kalman_gain = state['kalman_gain']
        self.loglikelihood = state['loglikelihood']
        self.predicted_state = state['predicted_state']
        self.predicted_state_cov = state['predicted_state_cov']
        self.standardized_forecast_error = state['standardized_forecast_error']
        self.tmp0 = state['tmp0']
        self.tmp00 = state['tmp00']
        self.tmp1 = state['tmp1']
        self.tmp2 = state['tmp2']
        self.tmp3 = state['tmp3']
        self.tmp4 = state['tmp4']
        self._reinitialize_pointers()

    cdef void _reinitialize_pointers(self) except *:
        self._converged_forecast_error_cov = &self.converged_forecast_error_cov[0,0]
        self._converged_filtered_state_cov = &self.converged_filtered_state_cov[0,0]
        self._converged_predicted_state_cov = &self.converged_predicted_state_cov[0,0]
        self._converged_kalman_gain = &self.converged_kalman_gain[0,0]
        self._forecast_error_fac = &self.forecast_error_fac[0,0]
        self._forecast_error_work = &self.forecast_error_work[0,0]
        self._forecast_error_ipiv = &self.forecast_error_ipiv[0]
        self._tmp0 = &self.tmp0[0, 0]
        self._tmp00 = &self.tmp00[0, 0]

    cdef allocate_arrays(self):
        # Local variables
        cdef:
            cnp.npy_intp dim1[1]
            cnp.npy_intp dim2[2]
            cnp.npy_intp dim3[3]
        cdef int storage
        # #### Allocate arrays for calculations

        # Arrays for Kalman filter output

        # Forecast
        if self.conserve_memory & MEMORY_NO_FORECAST:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.forecast = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self.forecast_error = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        dim3[0] = self.k_endog; dim3[1] = self.k_endog; dim3[2] = storage;
        self.forecast_error_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX128, FORTRAN)
        # Standardized forecast errors
        if self.conserve_memory & MEMORY_NO_STD_FORECAST > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.standardized_forecast_error = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)

        # Filtered
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_states; dim2[1] = storage;
        self.filtered_state = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = storage;
        self.filtered_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX128, FORTRAN)

        # Predicted
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_states; dim2[1] = storage+1;
        self.predicted_state = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = storage+1;
        self.predicted_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX128, FORTRAN)

        # Kalman Gain
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim3[0] = self.k_states; dim3[1] = self.k_endog; dim3[2] = storage;
        self.kalman_gain = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX128, FORTRAN)

        # Likelihood
        if self.conserve_memory & MEMORY_NO_LIKELIHOOD > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim1[0] = storage
        self.loglikelihood = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX128, FORTRAN)

        # Converged matrices
        dim2[0] = self.k_endog; dim2[1] = self.k_endog;
        self.converged_forecast_error_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self._converged_forecast_error_cov = &self.converged_forecast_error_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.converged_filtered_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self._converged_filtered_state_cov = &self.converged_filtered_state_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.converged_predicted_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self._converged_predicted_state_cov = &self.converged_predicted_state_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_endog;
        self.converged_kalman_gain = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self._converged_kalman_gain = &self.converged_kalman_gain[0,0]

        # #### Arrays for temporary calculations
        # *Note*: in math notation below, a $\\#$ will represent a generic
        # temporary array, and a $\\#_i$ will represent a named temporary array.

        # Arrays related to matrix factorizations / inverses
        dim2[0] = self.k_endog; dim2[1] = self.k_endog;
        self.forecast_error_fac = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self._forecast_error_fac = &self.forecast_error_fac[0,0]
        dim2[0] = self.ldwork; dim2[1] = self.ldwork;
        self.forecast_error_work = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self._forecast_error_work = &self.forecast_error_work[0,0]
        dim1[0] = self.k_endog;
        self.forecast_error_ipiv = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_INT, FORTRAN)
        self._forecast_error_ipiv = &self.forecast_error_ipiv[0]

        # Holds arrays of dimension $(m \times m)$ and $(m \times r)$
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.tmp0 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self._tmp0 = &self.tmp0[0, 0]

        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.tmp00 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self._tmp00 = &self.tmp00[0, 0]

        # Optionally we may not want to store temporary arrays required  
        # for smoothing
        if self.conserve_memory & MEMORY_NO_SMOOTHING > 0:
            storage = 1
        else:
            storage = self.model.nobs

        # Holds arrays of dimension $(m \times p \times T)$  
        # $\\#_1 = P_t Z_t'$
        dim3[0] = self.k_states; dim3[1] = self.k_endog; dim3[2] = storage;
        self.tmp1 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX128, FORTRAN)

        # Holds arrays of dimension $(p \times T)$  
        # $\\#_2 = F_t^{-1} v_t$
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.tmp2 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)

        # Holds arrays of dimension $(p \times m \times T)$  
        # $\\#_3 = F_t^{-1} Z_t$
        dim3[0] = self.k_endog; dim3[1] = self.k_states; dim3[2] = storage;
        self.tmp3 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX128, FORTRAN)

        # Holds arrays of dimension $(p \times p \times T)$  
        # $\\#_4 = F_t^{-1} H_t$
        dim3[0] = self.k_endog; dim3[1] = self.k_endog; dim3[2] = storage;
        self.tmp4 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX128, FORTRAN)

    @cython.cfunc
    @cython.returns(cython.void)
    def set_dimensions(self):
        """
        Set dimensions for the Kalman filter

        These are used *only* to define the shapes of the Kalman filter output
        and temporary arrays in memory. They will not change between iterations
        of the filter.

        They only differ from the zStatespace versions in the case
        that the FILTER_COLLAPSED flag is set, in which case model.k_endog
        and kfilter.k_endog will be different
        (since kfilter.k_endog = model.k_states).

        Across *iterations* of the Kalman filter, both model.k_* and
        kfilter.k_* are fixed, although model._k_* may be different from either
        when there is missing data in a given period's observations.

        The actual dimension of the *data* being considered at a given
        iteration is always given by model._k_* variables, which take into
        account both FILTER_COLLAPSED and missing data.

        But, the dimension *in memory* of the Kalman filter arrays will always
        be given by kfilter.k_*.

        The following relations will always hold:

        kfilter.k_endog = model.k_states if self.filter_method & FILTER_COLLAPSED else model.k_endog
        kfilter.k_endog = model._k_endog + model._nmissing
        """
        self.k_endog = self.model.k_states if self.filter_method & FILTER_COLLAPSED else self.model.k_endog
        self.k_states = self.model.k_states
        self.k_posdef = self.model.k_posdef
        self.k_endog2 = self.k_endog**2
        self.k_states2 = self.k_states**2
        self.k_posdef2 = self.k_posdef**2
        self.k_endogstates = self.k_endog * self.k_states
        self.k_statesposdef = self.k_states * self.k_posdef

    @cython.ccall
    def set_filter_method(self, int filter_method, int force_reset=True):
        """
        set_filter_method(self, filter_method, force_reset=True)

        Change the filter method.
        """
        if not filter_method == self.filter_method or force_reset:
            # Check for invalid filter methods
            if filter_method & FILTER_COLLAPSED and self.k_endog <= self.k_states:
                raise RuntimeError('Cannot collapse observation vector if the'
                                   ' state dimension is equal to or larger than the'
                                   ' dimension of the observation vector.')

            # Change the smoother output flag
            self.filter_method = filter_method

            # Reset dimensions
            self.set_dimensions()

            # Reset matrices
            self.allocate_arrays()

            # Seek to the beginning
            self.seek(0, True)

    @cython.ccall
    def seek(self, unsigned int t, int reset_convergence=True):
        """
        seek(self, t, reset_convergence = True)

        Change the time-state of the filter

        Is usually called to reset the filter to the beginning.
        """
        if not t == 0 and t >= self.model.nobs:
            raise IndexError("Observation index out of range")
        self.t = t

        if reset_convergence:
            self.converged = 0
            self.period_converged = 0

    def __iter__(self):
        return self

    def __call__(self, int filter_method=-1):
        """
        Iterate the filter across the entire set of observations.
        """
        cdef int i

        # Reset the filter method if necessary
        if not filter_method == -1:
            self.set_filter_method(filter_method)

        # Reset the filter
        self.seek(0, True)

        # Perform forward filtering iterations
        for i in range(self.model.nobs):
            next(self)

    def __next__(self):
        """
        Perform an iteration of the Kalman filter
        """
        cdef int filtered_t = self.t
        cdef int predicted_t = self.t
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1

        # Get time subscript, and stop the iterator if at the end
        if not self.t < self.model.nobs:
            raise StopIteration

        # Clear values
        if self.t == 0 or not (self.conserve_memory & MEMORY_NO_LIKELIHOOD):
            self.loglikelihood[self.t] = 0

        # Initialize pointers to current-iteration objects
        self.initialize_statespace_object_pointers()
        self.initialize_filter_object_pointers()

        # Initialize pointers to appropriate Kalman filtering functions
        self.initialize_function_pointers()

        # Convert base arrays into "selected" arrays  
        # - State covariance matrix? $Q_t \to R_t Q_t R_t`$
        # - Missing values: $y_t \to W_t y_t$, $Z_t \to W_t Z_t$, $H_t \to W_t H_t$
        # self.select_state_cov()
        # self.select_missing()
        # self.transform()

        # Post-convergence: copy previous iteration arrays
        self.post_convergence()

        # Prediction step (alternate timing)
        if self.filter_timing == TIMING_INIT_FILTERED:
            # We need to shift back to the previous filtered_* arrays, or to
            # the initial_* arrays if we're at time t==0
            if self.t == 0:
                self._filtered_state = self.model._initial_state
                self._filtered_state_cov = self.model._initial_state_cov
            else:
                self._filtered_state = &self.filtered_state[0, filtered_t-1]
                self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t-1]

            # Perform the prediction step
            self.prediction(self, self.model)
            # self._prediction()

            # Aids to numerical stability
            self.numerical_stability()

            # Now shift back to the current filtered_* arrays (so they can be
            # set in the updating step)
            self._filtered_state = &self.filtered_state[0, filtered_t]
            self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t]

        # Form forecasts
        self.forecasting(self, self.model)
        # self._forecasting()

        # Perform `forecast_error_cov` inversion (or decomposition)
        self.determinant = self.inversion(self, self.model, self.determinant)
        # self.determinant = self._inversion()

        # Updating step
        self.updating(self, self.model)
        # self._updating()

        # Retrieve the loglikelihood
        if not self.conserve_memory & MEMORY_NO_LIKELIHOOD or self.t >= self.loglikelihood_burn:
            self._loglikelihood[0] = (
                self._loglikelihood[0] +
                self.calculate_loglikelihood(self, self.model, self.determinant) +
                # self._calculate_loglikelihood() +
                self.model.collapse_loglikelihood
            )

        # Prediction step (default timing)
        if self.filter_timing == TIMING_INIT_PREDICTED:
            self.prediction(self, self.model)
            # self._prediction()

            # Aids to numerical stability
            self.numerical_stability()

        # Last prediction step (alternate timing)
        if self.filter_timing == TIMING_INIT_FILTERED and self.t == self.model.nobs-1:
            self._predicted_state = &self.predicted_state[0, predicted_t+1]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t+1]
            self.prediction(self, self.model)

        # Check for convergence
        self.check_convergence()

        # If conserving memory, migrate storage: t->t-1, t+1->t
        self.migrate_storage()

        # Advance the time
        self.t += 1

    @cython.cfunc
    @cython.returns(cython.void)
    def _forecasting(self):
        zforecast_univariate(self, self.model)

    @cython.cfunc
    @cython.returns(cnp.complex128_t)
    def _inversion(self):
        zinverse_noop_univariate(self, self.model, self.determinant)

    @cython.cfunc
    @cython.returns(cython.void)
    def _updating(self):
        zupdating_univariate(self, self.model)

    @cython.cfunc
    @cython.returns(cnp.complex128_t)
    def _calculate_loglikelihood(self):
        return zloglikelihood_univariate(self, self.model, self.determinant)

    @cython.cfunc
    @cython.returns(cython.void)
    def _prediction(self):
        zprediction_univariate(self, self.model)

    cdef void initialize_statespace_object_pointers(self) except *:
        cdef:
            int transform_diagonalize = 0
            int transform_generalized_collapse = 0

        # Determine which transformations need to be made
        transform_generalized_collapse = self.filter_method & FILTER_COLLAPSED
        transform_diagonalize = self.filter_method & FILTER_UNIVARIATE

        # Initialize object-level pointers to statespace arrays
        #self.model.initialize_object_pointers(self.t)
        self.model.seek(self.t, transform_diagonalize, transform_generalized_collapse)

        # Handle missing data
        if self.model._nmissing > 0 or (self.model.has_missing and self.filter_method & FILTER_UNIVARIATE):
            # TODO there is likely a way to allow convergence and the univariate filter, but it
            # doesn't work "out-of-the-box" right now
            self.converged = 0

    @cython.cfunc
    @cython.returns(cython.void)
    def initialize_filter_object_pointers(self):
        cdef:
            int t = self.t
            int inc = 1
        # Indices for arrays that may or may not be stored completely
        cdef:
            int forecast_t = t
            int filtered_t = t
            int predicted_t = t
            int gain_t = t
            int smoothing_t = t
            int loglikelihood_t = t
            int std_forecast_t = t
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            forecast_t = 1
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            gain_t = 0
        if self.conserve_memory & MEMORY_NO_SMOOTHING > 0:
            smoothing_t = 0
        if self.conserve_memory & MEMORY_NO_LIKELIHOOD > 0:
            loglikelihood_t = 0
        if self.conserve_memory & MEMORY_NO_STD_FORECAST > 0:
            std_forecast_t = 0

        # Initialize object-level pointers to input arrays
        self._input_state = &self.predicted_state[0, predicted_t]
        self._input_state_cov = &self.predicted_state_cov[0, 0, predicted_t]

        # Copy initialization arrays to input arrays if we're starting the
        # filter
        if t == 0 and self.filter_timing == TIMING_INIT_PREDICTED:
            # `predicted_state[:,0]` $= a_1 =$ `initial_state`  
            # `predicted_state_cov[:,:,0]` $= P_1 =$ `initial_state_cov`  
            # Under the default timing assumption (TIMING_INIT_PREDICTED), the
            # recursion takes $a_t, P_t$ as input, and as a last step computes
            # $a_{t+1}, P_{t+1}$, which can be input for the next recursion.
            # This means that the filter ends by computing $a_{T+1}, P_{T+1}$,
            # so that the predicted_* arrays have time-dimension T+1, rather than
            # T like all the other arrays.
            # Note that $a_{T+1}, P_{T+1}$ should not be in use anywhere.
            # TODO phase out any use of these, and eventually stop computing it
            # This means that the zeroth entry in the time-dimension can hold the
            # input array (even though it is no different than what is held in the
            # initial_state_* arrays).
            blas.zcopy(
                &self.model._k_states, self.model._initial_state, &inc,
                self._input_state, &inc)
            blas.zcopy(
                &self.model._k_states2, self.model._initial_state_cov, &inc,
                self._input_state_cov, &inc)

        # Initialize object-level pointers to output arrays
        self._forecast = &self.forecast[0, forecast_t]
        self._forecast_error = &self.forecast_error[0, forecast_t]
        self._forecast_error_cov = &self.forecast_error_cov[0, 0, forecast_t]
        self._standardized_forecast_error = &self.standardized_forecast_error[0, std_forecast_t]

        self._filtered_state = &self.filtered_state[0, filtered_t]
        self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t]
        
        if self.filter_timing == TIMING_INIT_PREDICTED:
            self._predicted_state = &self.predicted_state[0, predicted_t + 1]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t + 1]
        else:
            self._predicted_state = &self.predicted_state[0, predicted_t]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t]

        self._kalman_gain = &self.kalman_gain[0, 0, gain_t]

        self._loglikelihood = &self.loglikelihood[loglikelihood_t]

        # Initialize object-level pointers to named temporary arrays
        self._tmp1 = &self.tmp1[0, 0, smoothing_t]
        self._tmp2 = &self.tmp2[0, smoothing_t]
        self._tmp3 = &self.tmp3[0, 0, smoothing_t]
        self._tmp4 = &self.tmp4[0, 0, smoothing_t]

    cdef void initialize_function_pointers(self) except *:
        # Filtering method
        if self.filter_method & FILTER_UNIVARIATE:
            self.forecasting = zforecast_univariate
            self.updating = zupdating_univariate
            self.inversion = zinverse_noop_univariate
            self.calculate_loglikelihood = zloglikelihood_univariate
            self.prediction = zprediction_univariate

        # Conventional method
        elif self.filter_method & FILTER_CONVENTIONAL:
            self.forecasting = zforecast_conventional
            self.updating = zupdating_conventional
            self.calculate_loglikelihood = zloglikelihood_conventional
            self.prediction = zprediction_conventional

            # Inversion method
            if self.inversion_method & INVERT_UNIVARIATE and self.k_endog == 1:
                self.inversion = zinverse_univariate
            elif self.inversion_method & SOLVE_CHOLESKY:
                self.inversion = zsolve_cholesky
            elif self.inversion_method & SOLVE_LU:
                self.inversion = zsolve_lu
            elif self.inversion_method & INVERT_CHOLESKY:
                self.inversion = zinverse_cholesky
            elif self.inversion_method & INVERT_LU:
                self.inversion = zinverse_lu
            else:
                raise NotImplementedError("Invalid inversion method")
        else:
            raise NotImplementedError("Invalid filtering method")

        # Handle completely missing data, can always just use conventional 
        # methods
        if self.model._nmissing == self.model.k_endog:
            # Change the forecasting step to set the forecast at the intercept
            # $d_t$, so that the forecast error is $v_t = y_t - d_t$.
            self.forecasting = zforecast_missing_conventional

            # Change the updating step to just copy $a_{t|t} = a_t$ and
            # $P_{t|t} = P_t$
            self.updating = zupdating_missing_conventional

            # Change the inversion step to inverse to nans.
            self.inversion = zinverse_missing_conventional

            # Change the loglikelihood calculation to give zero.
            self.calculate_loglikelihood = zloglikelihood_missing_conventional

            # The prediction step is the same as the conventional Kalman
            # filter

    @cython.cfunc
    @cython.returns(cython.void)
    def post_convergence(self):
        cdef:
            int inc = 1

        if self.converged:
            # $F_t$
            blas.zcopy(
                &self.k_endog2, self._converged_forecast_error_cov, &inc,
                self._forecast_error_cov, &inc)
            # $P_{t|t}$
            blas.zcopy(
                &self.k_states2, self._converged_filtered_state_cov, &inc,
                self._filtered_state_cov, &inc)
            # $P_t$
            blas.zcopy(
                &self.k_states2, self._converged_predicted_state_cov, &inc,
                self._predicted_state_cov, &inc)
            # $K_t$
            blas.zcopy(
                &self.k_endogstates, self._converged_kalman_gain, &inc,
                self._kalman_gain, &inc)
            # $|F_t|$
            self.determinant = self.converged_determinant

    @cython.cfunc
    @cython.returns(cython.void)
    def numerical_stability(self):
        cdef int i, j
        cdef int predicted_t = self.t
        cdef cnp.complex128_t value

        if self.conserve_memory & MEMORY_NO_PREDICTED:
            predicted_t = 1

        if self.filter_timing == TIMING_INIT_PREDICTED:
            predicted_t += 1

        if self.stability_method & STABILITY_FORCE_SYMMETRY:
            # Enforce symmetry of predicted covariance matrix  
            # $P_{t+1} = 0.5 * (P_{t+1} + P_{t+1}')$  
            # See Grewal (2001), Section 6.3.1.1
            for i in range(self.k_states):
                for j in range(i, self.k_states):
                    value = 0.5 * (
                        self.predicted_state_cov[i, j, predicted_t] +
                        self.predicted_state_cov[j, i, predicted_t]
                    )
                    self.predicted_state_cov[i, j, predicted_t] = value
                    self.predicted_state_cov[j, i, predicted_t] = value

    @cython.cfunc
    @cython.returns(cython.void)
    def check_convergence(self):
        cdef:
            int inc = 1, missing_flag = 0
            cnp.complex128_t alpha = 1.0
            cnp.complex128_t beta = 0.0
            cnp.complex128_t gamma = -1.0
        # Indices for arrays that may or may not be stored completely
        cdef:
            int forecast_t = self.t
            int filtered_t = self.t
            int predicted_t = self.t
            int gain_t = self.t
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            forecast_t = 1
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            gain_t = 0

        # Figure out if there is a missing value
        if self.model.nmissing[self.t] > 0 or (not self.t == 0 and self.model.nmissing[self.t - 1] > 0):
            missing_flag = 1

        if self.time_invariant and not self.converged and not missing_flag:
            # #### Check for steady-state convergence
            # 
            # `tmp0` array used here, dimension $(m \times m)$  
            # `tmp00` array used here, dimension $(1 \times 1)$  
            if self.filter_timing == TIMING_INIT_PREDICTED:
                blas.zcopy(
                    &self.k_states2, self._input_state_cov, &inc,
                    self._tmp0, &inc)
                blas.zaxpy(&self.k_states2, &gamma, self._predicted_state_cov, &inc, self._tmp0, &inc)
            elif self.t > 0:
                blas.zcopy(
                    &self.k_states2, &self.predicted_state_cov[0,0,predicted_t], &inc,
                    self._tmp0, &inc)
                blas.zaxpy(&self.k_states2, &gamma, &self.predicted_state_cov[0,0,predicted_t-1], &inc, self._tmp0, &inc)
            else:
                return


            blas.zgemv(
                "N", &inc, &self.k_states2, &alpha, self._tmp0, &inc,
                self._tmp0, &inc,
                &beta, self._tmp00, &inc)
            if zabs(self._tmp00[0]) < self.tolerance:
                self.converged = 1
                self.period_converged = self.t

            # If we just converged, copy the current iteration matrices to the
            # converged storage
            if self.converged == 1:
                # $F_t$
                blas.zcopy(
                    &self.k_endog2, &self.forecast_error_cov[0, 0, forecast_t], &inc,
                    self._converged_forecast_error_cov, &inc)
                # $P_{t|t}$
                blas.zcopy(
                    &self.k_states2, &self.filtered_state_cov[0, 0, filtered_t], &inc,
                    self._converged_filtered_state_cov, &inc)
                # $P_t$
                blas.zcopy(
                    &self.k_states2, &self.predicted_state_cov[0, 0, predicted_t], &inc,
                    self._converged_predicted_state_cov, &inc)
                # $|F_t|$
                self.converged_determinant = self.determinant
                # $K_t$
                blas.zcopy(
                    &self.k_endogstates, &self.kalman_gain[0, 0, gain_t], &inc,
                    self._converged_kalman_gain, &inc)

    @cython.cfunc
    @cython.returns(cython.void)
    def migrate_storage(self):
        cdef:
            int inc = 1

        # Forecast: 1 -> 0
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            blas.zcopy(&self.k_endog, &self.forecast[0, 1], &inc, &self.forecast[0, 0], &inc)
            blas.zcopy(&self.k_endog, &self.forecast_error[0, 1], &inc, &self.forecast_error[0, 0], &inc)
            blas.zcopy(&self.k_endog2, &self.forecast_error_cov[0, 0, 1], &inc, &self.forecast_error_cov[0, 0, 0], &inc)

        # Filtered: 1 -> 0
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            blas.zcopy(&self.k_states, &self.filtered_state[0, 1], &inc, &self.filtered_state[0, 0], &inc)
            blas.zcopy(&self.k_states2, &self.filtered_state_cov[0, 0, 1], &inc, &self.filtered_state_cov[0, 0, 0], &inc)

        # Predicted: 1 -> 0
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            blas.zcopy(&self.k_states, &self.predicted_state[0, 1], &inc, &self.predicted_state[0, 0], &inc)
            blas.zcopy(&self.k_states2, &self.predicted_state_cov[0, 0, 1], &inc, &self.predicted_state_cov[0, 0, 0], &inc)

            # Predicted: 2 -> 1
            if self.filter_timing == TIMING_INIT_PREDICTED:
                blas.zcopy(&self.k_states, &self.predicted_state[0, 2], &inc, &self.predicted_state[0, 1], &inc)
                blas.zcopy(&self.k_states2, &self.predicted_state_cov[0, 0, 2], &inc, &self.predicted_state_cov[0, 0, 1], &inc)

# ## Kalman filter
cdef class dKalmanFilter(object):
    """
    dKalmanFilter(model, filter_method=FILTER_CONVENTIONAL, inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY, stability_method=STABILITY_FORCE_SYMMETRY, filter_timing=TIMING_INIT_PREDICTED, tolerance=1e-19)

    A representation of the Kalman filter recursions.

    While the filter is mathematically represented as a recursion, it is here
    translated into Python as a stateful iterator.

    Because there are actually several types of Kalman filter depending on the
    state space model of interest, this class only handles the *iteration*
    aspect of filtering, and delegates the actual operations to four general
    workhorse routines, which can be implemented separately for each type of
    Kalman filter.

    In order to maintain a consistent interface, and because these four general
    routines may be quite different across filter types, their argument is only
    the stateful ?KalmanFilter object. Furthermore, in order to allow the
    different types of filter to substitute alternate matrices, this class
    defines a set of pointers to the various state space arrays and the
    filtering output arrays.

    For example, handling missing observations requires not only substituting
    `obs`, `design`, and `obs_cov` matrices, but the new matrices actually have
    different dimensions than the originals. This can be flexibly accomodated
    simply by replacing e.g. the `obs` pointer to the substituted `obs` array
    and replacing `k_endog` for that iteration. Then in the next iteration, when
    the `obs` vector may be missing different elements (or none at all), it can
    again be redefined.

    Each iteration of the filter (see `__next__`) proceeds in a number of
    steps.

    `initialize_object_pointers` initializes pointers to current-iteration
    objects (i.e. the state space arrays and filter output arrays).  

    `initialize_function_pointers` initializes pointers to the appropriate
    Kalman filtering routines (i.e. `forecast_conventional` or
    `forecast_exact_initial`, etc.).  

    `select_arrays` converts the base arrays into "selected" arrays using
    selection matrices. In particular, it handles the state covariance matrix
    and redefined matrices based on missing values.  

    `post_convergence` handles copying arrays from time $t-1$ to time $t$ when
    the Kalman filter has converged and they don't need to be re-calculated.  

    `forecasting` calls the Kalman filter `forcasting_<filter type>` routine

    `inversion` calls the appropriate function to invert the forecast error
    covariance matrix.  

    `updating` calls the Kalman filter `updating_<filter type>` routine

    `loglikelihood` calls the Kalman filter `loglikelihood_<filter type>` routine

    `prediction` calls the Kalman filter `prediction_<filter type>` routine

    `numerical_stability` performs end-of-iteration tasks to improve the numerical
    stability of the filter 

    `check_convergence` checks for convergence of the filter to steady-state.
    """

    # ### Statespace model
    # cdef readonly dStatespace model

    # ### Filter parameters
    # Holds the time-iteration state of the filter  
    # *Note*: must be changed using the `seek` method
    # cdef readonly int t
    # Holds the tolerance parameter for convergence
    # cdef public cnp.float64_t tolerance
    # Holds the convergence to steady-state status of the filter
    # *Note*: is by default reset each time `seek` is called
    # cdef readonly int converged
    # cdef readonly int period_converged
    # Holds whether or not the model is time-invariant
    # *Note*: is by default reset each time `seek` is called
    # cdef readonly int time_invariant
    # The Kalman filter procedure to use  
    # cdef readonly int filter_method
    # The method by which the terms using the inverse of the forecast
    # error covariance matrix are solved.
    # cdef public int inversion_method
    # Methods to improve numerical stability
    # cdef public int stability_method
    # Whether or not to conserve memory
    # If True, only stores filtered states and covariance matrices
    # cdef readonly int conserve_memory
    # Whether or not to use alternate timing
    # If True, uses the Kim and Nelson (1999) timing
    # cdef readonly int filter_timing
    # If conserving loglikelihood, the number of periods to "burn"
    # before starting to record the loglikelihood
    # cdef readonly int loglikelihood_burn

    # ### Kalman filter properties

    # `loglikelihood` $\equiv \log p(y_t | Y_{t-1})$
    # cdef readonly cnp.float64_t [:] loglikelihood

    # `filtered_state` $\equiv a_{t|t} = E(\alpha_t | Y_t)$ is the **filtered estimator** of the state $(m \times T)$  
    # `predicted_state` $\equiv a_{t+1} = E(\alpha_{t+1} | Y_t)$ is the **one-step ahead predictor** of the state $(m \times T-1)$  
    # `forecast` $\equiv E(y_t|Y_{t-1})$ is the **forecast** of the next observation $(p \times T)$   
    # `forecast_error` $\equiv v_t = y_t - E(y_t|Y_{t-1})$ is the **one-step ahead forecast error** of the next observation $(p \times T)$  
    # 
    # *Note*: Actual values in `filtered_state` will be from 1 to `nobs`+1. Actual
    # values in `predicted_state` will be from 0 to `nobs`+1 because the initialization
    # is copied over to the zeroth entry, and similar for the covariances, below.
    #
    # *Old notation: beta_tt, beta_tt1, y_tt1, eta_tt1*
    # cdef readonly cnp.float64_t [::1,:] filtered_state, predicted_state, forecast, forecast_error

    # `filtered_state_cov` $\equiv P_{t|t} = Var(\alpha_t | Y_t)$ is the **filtered state covariance matrix** $(m \times m \times T)$  
    # `predicted_state_cov` $\equiv P_{t+1} = Var(\alpha_{t+1} | Y_t)$ is the **predicted state covariance matrix** $(m \times m \times T)$  
    # `forecast_error_cov` $\equiv F_t = Var(v_t | Y_{t-1})$ is the **forecast error covariance matrix** $(p \times p \times T)$  
    # 
    # *Old notation: P_tt, P_tt1, f_tt1*
    # cdef readonly cnp.float64_t [::1,:,:] filtered_state_cov, predicted_state_cov, forecast_error_cov

    # `kalman_gain` $\equiv K_{t} = T_t P_t Z_t' F_t^{-1}$ is the **Kalman gain** $(m \times p \times T)$  
    # cdef readonly cnp.float64_t [::1,:,:] kalman_gain

    # ### Steady State Values
    # These matrices are used to hold the converged matrices after the Kalman
    # filter has reached steady-state
    # cdef readonly cnp.float64_t [::1,:] converged_forecast_error_cov
    # cdef readonly cnp.float64_t [::1,:] converged_filtered_state_cov
    # cdef readonly cnp.float64_t [::1,:] converged_predicted_state_cov
    # cdef readonly cnp.float64_t [::1,:] converged_kalman_gain
    # cdef readonly cnp.float64_t converged_determinant

    # ### Temporary arrays
    # These matrices are used to temporarily hold selected observation vectors,
    # design matrices, and observation covariance matrices in the case of
    # missing data.  
    # `forecast_error_fac` is a forecast error covariance matrix **factorization** $(p \times p)$.
    # Depending on the method for handling the inverse of the forecast error covariance matrix, it may be:
    # - a Cholesky factorization if `cholesky_solve` is used
    # - an inverse calculated via Cholesky factorization if `cholesky_inverse` is used
    # - an LU factorization if `lu_solve` is used
    # - an inverse calculated via LU factorization if `lu_inverse` is used
    # cdef readonly cnp.float64_t [::1,:] forecast_error_fac
    # `forecast_error_ipiv` holds pivot indices if an LU decomposition is used
    # cdef readonly int [:] forecast_error_ipiv
    # `forecast_error_work` is a work array for matrix inversion if an LU
    # decomposition is used
    # cdef readonly cnp.float64_t [::1,:] forecast_error_work
    # These hold the memory allocations of the anonymous temporary arrays
    # cdef readonly cnp.float64_t [::1,:] tmp0, tmp00
    # These hold the memory allocations of the named temporary arrays  
    # (these are all time-varying in the last dimension)
    # cdef readonly cnp.float64_t [::1,:] tmp2
    # cdef readonly cnp.float64_t [::1,:,:] tmp1, tmp3

    # Holds the determinant across calculations (this is done because after
    # convergence, it doesn't need to be re-calculated anymore)
    # cdef readonly cnp.float64_t determinant

    # ### Pointers to current-iteration arrays
    # cdef cnp.float64_t * _obs
    # cdef cnp.float64_t * _design
    # cdef cnp.float64_t * _obs_intercept
    # cdef cnp.float64_t * _obs_cov
    # cdef cnp.float64_t * _transition
    # cdef cnp.float64_t * _state_intercept
    # cdef cnp.float64_t * _selection
    # cdef cnp.float64_t * _state_cov
    # cdef cnp.float64_t * _selected_state_cov
    # cdef cnp.float64_t * _initial_state
    # cdef cnp.float64_t * _initial_state_cov

    # cdef cnp.float64_t * _input_state
    # cdef cnp.float64_t * _input_state_cov

    # cdef cnp.float64_t * _forecast
    # cdef cnp.float64_t * _forecast_error
    # cdef cnp.float64_t * _forecast_error_cov
    # cdef cnp.float64_t * _filtered_state
    # cdef cnp.float64_t * _filtered_state_cov
    # cdef cnp.float64_t * _predicted_state
    # cdef cnp.float64_t * _predicted_state_cov

    # cdef cnp.float64_t * _kalman_gain
    # cdef cnp.float64_t * _loglikelihood

    # cdef cnp.float64_t * _converged_forecast_error_cov
    # cdef cnp.float64_t * _converged_filtered_state_cov
    # cdef cnp.float64_t * _converged_predicted_state_cov
    # cdef cnp.float64_t * _converged_kalman_gain

    # cdef cnp.float64_t * _forecast_error_fac
    # cdef int * _forecast_error_ipiv
    # cdef cnp.float64_t * _forecast_error_work

    # cdef cnp.float64_t * _tmp0
    # cdef cnp.float64_t * _tmp00
    # cdef cnp.float64_t * _tmp1
    # cdef cnp.float64_t * _tmp2
    # cdef cnp.float64_t * _tmp3

    # ### Pointers to current-iteration Kalman filtering functions
    # cdef int (*forecasting)(
    #     dKalmanFilter, dStatespace
    # )
    # cdef cnp.float64_t (*inversion)(
    #     dKalmanFilter, dStatespace, cnp.float64_t
    # ) except *
    # cdef int (*updating)(
    #     dKalmanFilter, dStatespace
    # )
    # cdef cnp.float64_t (*calculate_loglikelihood)(
    #     dKalmanFilter, dStatespace, cnp.float64_t
    # )
    # cdef int (*prediction)(
    #     dKalmanFilter, dStatespace
    # )

    # ### Define some constants
    # cdef readonly int k_endog, k_states, k_posdef, k_endog2, k_states2, k_endogstates
    # cdef readonly ldwork
    
    def __init__(self,
                 dStatespace model,
                 int filter_method=FILTER_CONVENTIONAL,
                 int inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY,
                 int stability_method=STABILITY_FORCE_SYMMETRY,
                 int conserve_memory=MEMORY_STORE_ALL,
                 int filter_timing=TIMING_INIT_PREDICTED,
                 cnp.float64_t tolerance=1e-19,
                 int loglikelihood_burn=0):

        # Save the model
        self.model = model

        # Initialize filter parameters
        self.tolerance = tolerance
        self.inversion_method = inversion_method
        self.stability_method = stability_method
        self.conserve_memory = conserve_memory
        self.filter_timing = filter_timing
        self.loglikelihood_burn = loglikelihood_burn

        # Initialize the constant values
        self.time_invariant = self.model.time_invariant

        # TODO replace with optimal work array size
        self.ldwork = self.model.k_endog

        # Set the filter method
        self.set_dimensions()
        self.set_filter_method(filter_method, True)

        # Initialize time and convergence status
        self.t = 0
        self.converged = 0
        self.period_converged = 0

    def __reduce__(self):
        args = (self.model, self.filter_method, self.inversion_method,
                self.stability_method,  self.conserve_memory, self.filter_timing,
                self.tolerance, self.loglikelihood_burn)
        state = {'t': self.t,
                 'converged' : self.converged ,
                 'converged_determinant' : self.converged_determinant,
                 'determinant' : self.determinant,
                 'period_converged' : self.period_converged,
                 'converged_filtered_state_cov': np.array(self.converged_filtered_state_cov, copy=True, order='F'),
                 'converged_forecast_error_cov': np.array(self.converged_forecast_error_cov, copy=True, order='F'),
                 'converged_kalman_gain': np.array(self.converged_kalman_gain, copy=True, order='F'),
                 'converged_predicted_state_cov': np.array(self.converged_predicted_state_cov, copy=True, order='F'),
                 'filtered_state': np.array(self.filtered_state, copy=True, order='F'),
                 'filtered_state_cov': np.array(self.filtered_state_cov, copy=True, order='F'),
                 'forecast': np.array(self.forecast, copy=True, order='F'),
                 'forecast_error': np.array(self.forecast_error, copy=True, order='F'),
                 'forecast_error_cov': np.array(self.forecast_error_cov, copy=True, order='F'),
                 'forecast_error_fac': np.array(self.forecast_error_fac, copy=True, order='F'),
                 'forecast_error_ipiv': np.array(self.forecast_error_ipiv, copy=True, order='F'),
                 'forecast_error_work': np.array(self.forecast_error_work, copy=True, order='F'),
                 'kalman_gain': np.array(self.kalman_gain, copy=True, order='F'),
                 'loglikelihood': np.array(self.loglikelihood, copy=True, order='F'),
                 'predicted_state': np.array(self.predicted_state, copy=True, order='F'),
                 'predicted_state_cov': np.array(self.predicted_state_cov, copy=True, order='F'),
                 'standardized_forecast_error': np.array(self.standardized_forecast_error, copy=True, order='F'),
                 'tmp0': np.array(self.tmp0, copy=True, order='F'),
                 'tmp00': np.array(self.tmp00, copy=True, order='F'),
                 'tmp1': np.array(self.tmp1, copy=True, order='F'),
                 'tmp2': np.array(self.tmp2, copy=True, order='F'),
                 'tmp3': np.array(self.tmp3, copy=True, order='F'),
                 'tmp4': np.array(self.tmp4, copy=True, order='F')
                 }

        return (self.__class__, args, state)

    def __setstate__(self, state):
        self.t = state['t']
        self.converged  = state['converged']
        self.converged_determinant = state['converged_determinant']
        self.determinant = state['determinant']
        self.period_converged = state['period_converged']
        self.converged_filtered_state_cov = state['converged_filtered_state_cov']
        self.converged_forecast_error_cov = state['converged_forecast_error_cov']
        self.converged_kalman_gain = state['converged_kalman_gain']
        self.converged_predicted_state_cov = state['converged_predicted_state_cov']
        self.filtered_state = state['filtered_state']
        self.filtered_state_cov = state['filtered_state_cov']
        self.forecast = state['forecast']
        self.forecast_error = state['forecast_error']
        self.forecast_error_cov = state['forecast_error_cov']
        self.forecast_error_fac = state['forecast_error_fac']
        self.forecast_error_ipiv = state['forecast_error_ipiv']
        self.forecast_error_work = state['forecast_error_work']
        self.kalman_gain = state['kalman_gain']
        self.loglikelihood = state['loglikelihood']
        self.predicted_state = state['predicted_state']
        self.predicted_state_cov = state['predicted_state_cov']
        self.standardized_forecast_error = state['standardized_forecast_error']
        self.tmp0 = state['tmp0']
        self.tmp00 = state['tmp00']
        self.tmp1 = state['tmp1']
        self.tmp2 = state['tmp2']
        self.tmp3 = state['tmp3']
        self.tmp4 = state['tmp4']
        self._reinitialize_pointers()

    cdef void _reinitialize_pointers(self) except *:
        self._converged_forecast_error_cov = &self.converged_forecast_error_cov[0,0]
        self._converged_filtered_state_cov = &self.converged_filtered_state_cov[0,0]
        self._converged_predicted_state_cov = &self.converged_predicted_state_cov[0,0]
        self._converged_kalman_gain = &self.converged_kalman_gain[0,0]
        self._forecast_error_fac = &self.forecast_error_fac[0,0]
        self._forecast_error_work = &self.forecast_error_work[0,0]
        self._forecast_error_ipiv = &self.forecast_error_ipiv[0]
        self._tmp0 = &self.tmp0[0, 0]
        self._tmp00 = &self.tmp00[0, 0]

    cdef allocate_arrays(self):
        # Local variables
        cdef:
            cnp.npy_intp dim1[1]
            cnp.npy_intp dim2[2]
            cnp.npy_intp dim3[3]
        cdef int storage
        # #### Allocate arrays for calculations

        # Arrays for Kalman filter output

        # Forecast
        if self.conserve_memory & MEMORY_NO_FORECAST:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.forecast = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self.forecast_error = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        dim3[0] = self.k_endog; dim3[1] = self.k_endog; dim3[2] = storage;
        self.forecast_error_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT64, FORTRAN)
        # Standardized forecast errors
        if self.conserve_memory & MEMORY_NO_STD_FORECAST > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.standardized_forecast_error = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)

        # Filtered
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_states; dim2[1] = storage;
        self.filtered_state = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = storage;
        self.filtered_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT64, FORTRAN)

        # Predicted
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            storage = 2
        else:
            storage = self.model.nobs
        dim2[0] = self.k_states; dim2[1] = storage+1;
        self.predicted_state = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = storage+1;
        self.predicted_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT64, FORTRAN)

        # Kalman Gain
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim3[0] = self.k_states; dim3[1] = self.k_endog; dim3[2] = storage;
        self.kalman_gain = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT64, FORTRAN)

        # Likelihood
        if self.conserve_memory & MEMORY_NO_LIKELIHOOD > 0:
            storage = 1
        else:
            storage = self.model.nobs
        dim1[0] = storage
        self.loglikelihood = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT64, FORTRAN)

        # Converged matrices
        dim2[0] = self.k_endog; dim2[1] = self.k_endog;
        self.converged_forecast_error_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self._converged_forecast_error_cov = &self.converged_forecast_error_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.converged_filtered_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self._converged_filtered_state_cov = &self.converged_filtered_state_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.converged_predicted_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self._converged_predicted_state_cov = &self.converged_predicted_state_cov[0,0]
        dim2[0] = self.k_states; dim2[1] = self.k_endog;
        self.converged_kalman_gain = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self._converged_kalman_gain = &self.converged_kalman_gain[0,0]

        # #### Arrays for temporary calculations
        # *Note*: in math notation below, a $\\#$ will represent a generic
        # temporary array, and a $\\#_i$ will represent a named temporary array.

        # Arrays related to matrix factorizations / inverses
        dim2[0] = self.k_endog; dim2[1] = self.k_endog;
        self.forecast_error_fac = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self._forecast_error_fac = &self.forecast_error_fac[0,0]
        dim2[0] = self.ldwork; dim2[1] = self.ldwork;
        self.forecast_error_work = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self._forecast_error_work = &self.forecast_error_work[0,0]
        dim1[0] = self.k_endog;
        self.forecast_error_ipiv = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_INT, FORTRAN)
        self._forecast_error_ipiv = &self.forecast_error_ipiv[0]

        # Holds arrays of dimension $(m \times m)$ and $(m \times r)$
        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.tmp0 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self._tmp0 = &self.tmp0[0, 0]

        dim2[0] = self.k_states; dim2[1] = self.k_states;
        self.tmp00 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self._tmp00 = &self.tmp00[0, 0]

        # Optionally we may not want to store temporary arrays required  
        # for smoothing
        if self.conserve_memory & MEMORY_NO_SMOOTHING > 0:
            storage = 1
        else:
            storage = self.model.nobs

        # Holds arrays of dimension $(m \times p \times T)$  
        # $\\#_1 = P_t Z_t'$
        dim3[0] = self.k_states; dim3[1] = self.k_endog; dim3[2] = storage;
        self.tmp1 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT64, FORTRAN)

        # Holds arrays of dimension $(p \times T)$  
        # $\\#_2 = F_t^{-1} v_t$
        dim2[0] = self.k_endog; dim2[1] = storage;
        self.tmp2 = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)

        # Holds arrays of dimension $(p \times m \times T)$  
        # $\\#_3 = F_t^{-1} Z_t$
        dim3[0] = self.k_endog; dim3[1] = self.k_states; dim3[2] = storage;
        self.tmp3 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT64, FORTRAN)

        # Holds arrays of dimension $(p \times p \times T)$  
        # $\\#_4 = F_t^{-1} H_t$
        dim3[0] = self.k_endog; dim3[1] = self.k_endog; dim3[2] = storage;
        self.tmp4 = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT64, FORTRAN)

    @cython.cfunc
    @cython.returns(cython.void)
    def set_dimensions(self):
        """
        Set dimensions for the Kalman filter

        These are used *only* to define the shapes of the Kalman filter output
        and temporary arrays in memory. They will not change between iterations
        of the filter.

        They only differ from the dStatespace versions in the case
        that the FILTER_COLLAPSED flag is set, in which case model.k_endog
        and kfilter.k_endog will be different
        (since kfilter.k_endog = model.k_states).

        Across *iterations* of the Kalman filter, both model.k_* and
        kfilter.k_* are fixed, although model._k_* may be different from either
        when there is missing data in a given period's observations.

        The actual dimension of the *data* being considered at a given
        iteration is always given by model._k_* variables, which take into
        account both FILTER_COLLAPSED and missing data.

        But, the dimension *in memory* of the Kalman filter arrays will always
        be given by kfilter.k_*.

        The following relations will always hold:

        kfilter.k_endog = model.k_states if self.filter_method & FILTER_COLLAPSED else model.k_endog
        kfilter.k_endog = model._k_endog + model._nmissing
        """
        self.k_endog = self.model.k_states if self.filter_method & FILTER_COLLAPSED else self.model.k_endog
        self.k_states = self.model.k_states
        self.k_posdef = self.model.k_posdef
        self.k_endog2 = self.k_endog**2
        self.k_states2 = self.k_states**2
        self.k_posdef2 = self.k_posdef**2
        self.k_endogstates = self.k_endog * self.k_states
        self.k_statesposdef = self.k_states * self.k_posdef

    @cython.ccall
    def set_filter_method(self, int filter_method, int force_reset=True):
        """
        set_filter_method(self, filter_method, force_reset=True)

        Change the filter method.
        """
        if not filter_method == self.filter_method or force_reset:
            # Check for invalid filter methods
            if filter_method & FILTER_COLLAPSED and self.k_endog <= self.k_states:
                raise RuntimeError('Cannot collapse observation vector if the'
                                   ' state dimension is equal to or larger than the'
                                   ' dimension of the observation vector.')

            # Change the smoother output flag
            self.filter_method = filter_method

            # Reset dimensions
            self.set_dimensions()

            # Reset matrices
            self.allocate_arrays()

            # Seek to the beginning
            self.seek(0, True)

    @cython.ccall
    def seek(self, unsigned int t, int reset_convergence=True):
        """
        seek(self, t, reset_convergence = True)

        Change the time-state of the filter

        Is usually called to reset the filter to the beginning.
        """
        if not t == 0 and t >= self.model.nobs:
            raise IndexError("Observation index out of range")
        self.t = t

        if reset_convergence:
            self.converged = 0
            self.period_converged = 0

    def __iter__(self):
        return self

    def __call__(self, int filter_method=-1):
        """
        Iterate the filter across the entire set of observations.
        """
        cdef int i

        # Reset the filter method if necessary
        if not filter_method == -1:
            self.set_filter_method(filter_method)

        # Reset the filter
        self.seek(0, True)

        # Perform forward filtering iterations
        for i in range(self.model.nobs):
            next(self)

    def __next__(self):
        """
        Perform an iteration of the Kalman filter
        """
        cdef int filtered_t = self.t
        cdef int predicted_t = self.t
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1

        # Get time subscript, and stop the iterator if at the end
        if not self.t < self.model.nobs:
            raise StopIteration

        # Clear values
        if self.t == 0 or not (self.conserve_memory & MEMORY_NO_LIKELIHOOD):
            self.loglikelihood[self.t] = 0

        # Initialize pointers to current-iteration objects
        self.initialize_statespace_object_pointers()
        self.initialize_filter_object_pointers()

        # Initialize pointers to appropriate Kalman filtering functions
        self.initialize_function_pointers()

        # Convert base arrays into "selected" arrays  
        # - State covariance matrix? $Q_t \to R_t Q_t R_t`$
        # - Missing values: $y_t \to W_t y_t$, $Z_t \to W_t Z_t$, $H_t \to W_t H_t$
        # self.select_state_cov()
        # self.select_missing()
        # self.transform()

        # Post-convergence: copy previous iteration arrays
        self.post_convergence()

        # Prediction step (alternate timing)
        if self.filter_timing == TIMING_INIT_FILTERED:
            # We need to shift back to the previous filtered_* arrays, or to
            # the initial_* arrays if we're at time t==0
            if self.t == 0:
                self._filtered_state = self.model._initial_state
                self._filtered_state_cov = self.model._initial_state_cov
            else:
                self._filtered_state = &self.filtered_state[0, filtered_t-1]
                self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t-1]

            # Perform the prediction step
            self.prediction(self, self.model)
            # self._prediction()

            # Aids to numerical stability
            self.numerical_stability()

            # Now shift back to the current filtered_* arrays (so they can be
            # set in the updating step)
            self._filtered_state = &self.filtered_state[0, filtered_t]
            self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t]

        # Form forecasts
        self.forecasting(self, self.model)
        # self._forecasting()

        # Perform `forecast_error_cov` inversion (or decomposition)
        self.determinant = self.inversion(self, self.model, self.determinant)
        # self.determinant = self._inversion()

        # Updating step
        self.updating(self, self.model)
        # self._updating()

        # Retrieve the loglikelihood
        if not self.conserve_memory & MEMORY_NO_LIKELIHOOD or self.t >= self.loglikelihood_burn:
            self._loglikelihood[0] = (
                self._loglikelihood[0] +
                self.calculate_loglikelihood(self, self.model, self.determinant) +
                # self._calculate_loglikelihood() +
                self.model.collapse_loglikelihood
            )

        # Prediction step (default timing)
        if self.filter_timing == TIMING_INIT_PREDICTED:
            self.prediction(self, self.model)
            # self._prediction()

            # Aids to numerical stability
            self.numerical_stability()

        # Last prediction step (alternate timing)
        if self.filter_timing == TIMING_INIT_FILTERED and self.t == self.model.nobs-1:
            self._predicted_state = &self.predicted_state[0, predicted_t+1]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t+1]
            self.prediction(self, self.model)

        # Check for convergence
        self.check_convergence()

        # If conserving memory, migrate storage: t->t-1, t+1->t
        self.migrate_storage()

        # Advance the time
        self.t += 1

    @cython.cfunc
    @cython.returns(cython.void)
    def _forecasting(self):
        dforecast_univariate(self, self.model)

    @cython.cfunc
    @cython.returns(cnp.float64_t)
    def _inversion(self):
        dinverse_noop_univariate(self, self.model, self.determinant)

    @cython.cfunc
    @cython.returns(cython.void)
    def _updating(self):
        dupdating_univariate(self, self.model)

    @cython.cfunc
    @cython.returns(cnp.float64_t)
    def _calculate_loglikelihood(self):
        return dloglikelihood_univariate(self, self.model, self.determinant)

    @cython.cfunc
    @cython.returns(cython.void)
    def _prediction(self):
        dprediction_univariate(self, self.model)

    cdef void initialize_statespace_object_pointers(self) except *:
        cdef:
            int transform_diagonalize = 0
            int transform_generalized_collapse = 0

        # Determine which transformations need to be made
        transform_generalized_collapse = self.filter_method & FILTER_COLLAPSED
        transform_diagonalize = self.filter_method & FILTER_UNIVARIATE

        # Initialize object-level pointers to statespace arrays
        #self.model.initialize_object_pointers(self.t)
        self.model.seek(self.t, transform_diagonalize, transform_generalized_collapse)

        # Handle missing data
        if self.model._nmissing > 0 or (self.model.has_missing and self.filter_method & FILTER_UNIVARIATE):
            # TODO there is likely a way to allow convergence and the univariate filter, but it
            # doesn't work "out-of-the-box" right now
            self.converged = 0

    @cython.cfunc
    @cython.returns(cython.void)
    def initialize_filter_object_pointers(self):
        cdef:
            int t = self.t
            int inc = 1
        # Indices for arrays that may or may not be stored completely
        cdef:
            int forecast_t = t
            int filtered_t = t
            int predicted_t = t
            int gain_t = t
            int smoothing_t = t
            int loglikelihood_t = t
            int std_forecast_t = t
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            forecast_t = 1
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            gain_t = 0
        if self.conserve_memory & MEMORY_NO_SMOOTHING > 0:
            smoothing_t = 0
        if self.conserve_memory & MEMORY_NO_LIKELIHOOD > 0:
            loglikelihood_t = 0
        if self.conserve_memory & MEMORY_NO_STD_FORECAST > 0:
            std_forecast_t = 0

        # Initialize object-level pointers to input arrays
        self._input_state = &self.predicted_state[0, predicted_t]
        self._input_state_cov = &self.predicted_state_cov[0, 0, predicted_t]

        # Copy initialization arrays to input arrays if we're starting the
        # filter
        if t == 0 and self.filter_timing == TIMING_INIT_PREDICTED:
            # `predicted_state[:,0]` $= a_1 =$ `initial_state`  
            # `predicted_state_cov[:,:,0]` $= P_1 =$ `initial_state_cov`  
            # Under the default timing assumption (TIMING_INIT_PREDICTED), the
            # recursion takes $a_t, P_t$ as input, and as a last step computes
            # $a_{t+1}, P_{t+1}$, which can be input for the next recursion.
            # This means that the filter ends by computing $a_{T+1}, P_{T+1}$,
            # so that the predicted_* arrays have time-dimension T+1, rather than
            # T like all the other arrays.
            # Note that $a_{T+1}, P_{T+1}$ should not be in use anywhere.
            # TODO phase out any use of these, and eventually stop computing it
            # This means that the zeroth entry in the time-dimension can hold the
            # input array (even though it is no different than what is held in the
            # initial_state_* arrays).
            blas.dcopy(
                &self.model._k_states, self.model._initial_state, &inc,
                self._input_state, &inc)
            blas.dcopy(
                &self.model._k_states2, self.model._initial_state_cov, &inc,
                self._input_state_cov, &inc)

        # Initialize object-level pointers to output arrays
        self._forecast = &self.forecast[0, forecast_t]
        self._forecast_error = &self.forecast_error[0, forecast_t]
        self._forecast_error_cov = &self.forecast_error_cov[0, 0, forecast_t]
        self._standardized_forecast_error = &self.standardized_forecast_error[0, std_forecast_t]

        self._filtered_state = &self.filtered_state[0, filtered_t]
        self._filtered_state_cov = &self.filtered_state_cov[0, 0, filtered_t]
        
        if self.filter_timing == TIMING_INIT_PREDICTED:
            self._predicted_state = &self.predicted_state[0, predicted_t + 1]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t + 1]
        else:
            self._predicted_state = &self.predicted_state[0, predicted_t]
            self._predicted_state_cov = &self.predicted_state_cov[0, 0, predicted_t]

        self._kalman_gain = &self.kalman_gain[0, 0, gain_t]

        self._loglikelihood = &self.loglikelihood[loglikelihood_t]

        # Initialize object-level pointers to named temporary arrays
        self._tmp1 = &self.tmp1[0, 0, smoothing_t]
        self._tmp2 = &self.tmp2[0, smoothing_t]
        self._tmp3 = &self.tmp3[0, 0, smoothing_t]
        self._tmp4 = &self.tmp4[0, 0, smoothing_t]

    cdef void initialize_function_pointers(self) except *:
        # Filtering method
        if self.filter_method & FILTER_UNIVARIATE:
            self.forecasting = dforecast_univariate
            self.updating = dupdating_univariate
            self.inversion = dinverse_noop_univariate
            self.calculate_loglikelihood = dloglikelihood_univariate
            self.prediction = dprediction_univariate

        # Conventional method
        elif self.filter_method & FILTER_CONVENTIONAL:
            self.forecasting = dforecast_conventional
            self.updating = dupdating_conventional
            self.calculate_loglikelihood = dloglikelihood_conventional
            self.prediction = dprediction_conventional

            # Inversion method
            if self.inversion_method & INVERT_UNIVARIATE and self.k_endog == 1:
                self.inversion = dinverse_univariate
            elif self.inversion_method & SOLVE_CHOLESKY:
                self.inversion = dsolve_cholesky
            elif self.inversion_method & SOLVE_LU:
                self.inversion = dsolve_lu
            elif self.inversion_method & INVERT_CHOLESKY:
                self.inversion = dinverse_cholesky
            elif self.inversion_method & INVERT_LU:
                self.inversion = dinverse_lu
            else:
                raise NotImplementedError("Invalid inversion method")
        else:
            raise NotImplementedError("Invalid filtering method")

        # Handle completely missing data, can always just use conventional 
        # methods
        if self.model._nmissing == self.model.k_endog:
            # Change the forecasting step to set the forecast at the intercept
            # $d_t$, so that the forecast error is $v_t = y_t - d_t$.
            self.forecasting = dforecast_missing_conventional

            # Change the updating step to just copy $a_{t|t} = a_t$ and
            # $P_{t|t} = P_t$
            self.updating = dupdating_missing_conventional

            # Change the inversion step to inverse to nans.
            self.inversion = dinverse_missing_conventional

            # Change the loglikelihood calculation to give zero.
            self.calculate_loglikelihood = dloglikelihood_missing_conventional

            # The prediction step is the same as the conventional Kalman
            # filter

    @cython.cfunc
    @cython.returns(cython.void)
    def post_convergence(self):
        cdef:
            int inc = 1

        if self.converged:
            # $F_t$
            blas.dcopy(
                &self.k_endog2, self._converged_forecast_error_cov, &inc,
                self._forecast_error_cov, &inc)
            # $P_{t|t}$
            blas.dcopy(
                &self.k_states2, self._converged_filtered_state_cov, &inc,
                self._filtered_state_cov, &inc)
            # $P_t$
            blas.dcopy(
                &self.k_states2, self._converged_predicted_state_cov, &inc,
                self._predicted_state_cov, &inc)
            # $K_t$
            blas.dcopy(
                &self.k_endogstates, self._converged_kalman_gain, &inc,
                self._kalman_gain, &inc)
            # $|F_t|$
            self.determinant = self.converged_determinant

    @cython.cfunc
    @cython.returns(cython.void)
    def numerical_stability(self):
        cdef int i, j
        cdef int predicted_t = self.t
        cdef cnp.float64_t value

        if self.conserve_memory & MEMORY_NO_PREDICTED:
            predicted_t = 1

        if self.filter_timing == TIMING_INIT_PREDICTED:
            predicted_t += 1

        if self.stability_method & STABILITY_FORCE_SYMMETRY:
            # Enforce symmetry of predicted covariance matrix  
            # $P_{t+1} = 0.5 * (P_{t+1} + P_{t+1}')$  
            # See Grewal (2001), Section 6.3.1.1
            for i in range(self.k_states):
                for j in range(i, self.k_states):
                    value = 0.5 * (
                        self.predicted_state_cov[i, j, predicted_t] +
                        self.predicted_state_cov[j, i, predicted_t]
                    )
                    self.predicted_state_cov[i, j, predicted_t] = value
                    self.predicted_state_cov[j, i, predicted_t] = value

    @cython.cfunc
    @cython.returns(cython.void)
    def check_convergence(self):
        cdef:
            int inc = 1, missing_flag = 0
            cnp.float64_t alpha = 1.0
            cnp.float64_t beta = 0.0
            cnp.float64_t gamma = -1.0
        # Indices for arrays that may or may not be stored completely
        cdef:
            int forecast_t = self.t
            int filtered_t = self.t
            int predicted_t = self.t
            int gain_t = self.t
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            forecast_t = 1
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            filtered_t = 1
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            predicted_t = 1
        if self.conserve_memory & MEMORY_NO_GAIN > 0:
            gain_t = 0

        # Figure out if there is a missing value
        if self.model.nmissing[self.t] > 0 or (not self.t == 0 and self.model.nmissing[self.t - 1] > 0):
            missing_flag = 1

        if self.time_invariant and not self.converged and not missing_flag:
            # #### Check for steady-state convergence
            # 
            # `tmp0` array used here, dimension $(m \times m)$  
            # `tmp00` array used here, dimension $(1 \times 1)$  
            if self.filter_timing == TIMING_INIT_PREDICTED:
                blas.dcopy(
                    &self.k_states2, self._input_state_cov, &inc,
                    self._tmp0, &inc)
                blas.daxpy(&self.k_states2, &gamma, self._predicted_state_cov, &inc, self._tmp0, &inc)
            elif self.t > 0:
                blas.dcopy(
                    &self.k_states2, &self.predicted_state_cov[0,0,predicted_t], &inc,
                    self._tmp0, &inc)
                blas.daxpy(&self.k_states2, &gamma, &self.predicted_state_cov[0,0,predicted_t-1], &inc, self._tmp0, &inc)
            else:
                return


            if blas.ddot(&self.k_states2, self._tmp0, &inc, self._tmp0, &inc) < self.tolerance:
                self.converged = 1
                self.period_converged = self.t


            # If we just converged, copy the current iteration matrices to the
            # converged storage
            if self.converged == 1:
                # $F_t$
                blas.dcopy(
                    &self.k_endog2, &self.forecast_error_cov[0, 0, forecast_t], &inc,
                    self._converged_forecast_error_cov, &inc)
                # $P_{t|t}$
                blas.dcopy(
                    &self.k_states2, &self.filtered_state_cov[0, 0, filtered_t], &inc,
                    self._converged_filtered_state_cov, &inc)
                # $P_t$
                blas.dcopy(
                    &self.k_states2, &self.predicted_state_cov[0, 0, predicted_t], &inc,
                    self._converged_predicted_state_cov, &inc)
                # $|F_t|$
                self.converged_determinant = self.determinant
                # $K_t$
                blas.dcopy(
                    &self.k_endogstates, &self.kalman_gain[0, 0, gain_t], &inc,
                    self._converged_kalman_gain, &inc)

    @cython.cfunc
    @cython.returns(cython.void)
    def migrate_storage(self):
        cdef:
            int inc = 1

        # Forecast: 1 -> 0
        if self.conserve_memory & MEMORY_NO_FORECAST > 0:
            blas.dcopy(&self.k_endog, &self.forecast[0, 1], &inc, &self.forecast[0, 0], &inc)
            blas.dcopy(&self.k_endog, &self.forecast_error[0, 1], &inc, &self.forecast_error[0, 0], &inc)
            blas.dcopy(&self.k_endog2, &self.forecast_error_cov[0, 0, 1], &inc, &self.forecast_error_cov[0, 0, 0], &inc)

        # Filtered: 1 -> 0
        if self.conserve_memory & MEMORY_NO_FILTERED > 0:
            blas.dcopy(&self.k_states, &self.filtered_state[0, 1], &inc, &self.filtered_state[0, 0], &inc)
            blas.dcopy(&self.k_states2, &self.filtered_state_cov[0, 0, 1], &inc, &self.filtered_state_cov[0, 0, 0], &inc)

        # Predicted: 1 -> 0
        if self.conserve_memory & MEMORY_NO_PREDICTED > 0:
            blas.dcopy(&self.k_states, &self.predicted_state[0, 1], &inc, &self.predicted_state[0, 0], &inc)
            blas.dcopy(&self.k_states2, &self.predicted_state_cov[0, 0, 1], &inc, &self.predicted_state_cov[0, 0, 0], &inc)

            # Predicted: 2 -> 1
            if self.filter_timing == TIMING_INIT_PREDICTED:
                blas.dcopy(&self.k_states, &self.predicted_state[0, 2], &inc, &self.predicted_state[0, 1], &inc)
                blas.dcopy(&self.k_states2, &self.predicted_state_cov[0, 0, 2], &inc, &self.predicted_state_cov[0, 0, 1], &inc)
