#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models

Author: Chad Fulton  
License: Simplified-BSD
"""
import warnings

cimport cython

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.math cimport log as dlog, abs as dabs, exp as dexp

cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack

cimport sm2.tsa.statespace._tools as tools

from sm2.src.math cimport zabs, zlog, NPY_PI

cdef int FORTRAN = 1

## State Space Representation
cdef class dStatespace(object):
    """
    dStatespace(obs, design, obs_intercept, obs_cov, transition, state_intercept, selection, state_cov)

    *See Durbin and Koopman (2012), Chapter 4 for all notation*
    """

    # ### State space representation
    # 
    # $$
    # \begin{align}
    # y_t & = Z_t \alpha_t + d_t + \varepsilon_t \hspace{3em} & \varepsilon_t & \sim N(0, H_t) \\\\
    # \alpha_{t+1} & = T_t \alpha_t + c_t + R_t \eta_t & \eta_t & \sim N(0, Q_t) \\\\
    # & & \alpha_1 & \sim N(a_1, P_1)
    # \end{align}
    # $$
    # 
    # $y_t$ is $p \times 1$  
    # $\varepsilon_t$ is $p \times 1$  
    # $\alpha_t$ is $m \times 1$  
    # $\eta_t$ is $r \times 1$  
    # $t = 1, \dots, T$

    # `nobs` $\equiv T$ is the length of the time-series  
    # `k_endog` $\equiv p$ is dimension of observation space  
    # `k_states` $\equiv m$ is the dimension of the state space  
    # `k_posdef` $\equiv r$ is the dimension of the state shocks  
    # *Old notation: T, n, k, g*
    # cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # `obs` $\equiv y_t$ is the **observation vector** $(p \times T)$  
    # `design` $\equiv Z_t$ is the **design vector** $(p \times m \times T)$  
    # `obs_intercept` $\equiv d_t$ is the **observation intercept** $(p \times T)$  
    # `obs_cov` $\equiv H_t$ is the **observation covariance matrix** $(p \times p \times T)$  
    # `transition` $\equiv T_t$ is the **transition matrix** $(m \times m \times T)$  
    # `state_intercept` $\equiv c_t$ is the **state intercept** $(m \times T)$  
    # `selection` $\equiv R_t$ is the **selection matrix** $(m \times r \times T)$  
    # `state_cov` $\equiv Q_t$ is the **state covariance matrix** $(r \times r \times T)$  
    # `selected_state_cov` $\equiv R Q_t R'$ is the **selected state covariance matrix** $(m \times m \times T)$  
    # `initial_state` $\equiv a_1$ is the **initial state mean** $(m \times 1)$  
    # `initial_state_cov` $\equiv P_1$ is the **initial state covariance matrix** $(m \times m)$
    #
    # With the exception of `obs`, these are *optionally* time-varying. If they are instead time-invariant,
    # then the dimension of length $T$ is instead of length $1$.
    #
    # *Note*: the initial vectors' notation 1-indexed as in Durbin and Koopman,
    # but in the recursions below it will be 0-indexed in the Python arrays.
    # 
    # *Old notation: y, -, mu, beta_tt_init, P_tt_init*
    # cdef readonly cnp.float64_t [::1,:] obs, obs_intercept, state_intercept
    # cdef readonly cnp.float64_t [:] initial_state
    # cdef readonly cnp.float64_t [::1,:] initial_state_cov
    # *Old notation: H, R, F, G, Q*, G Q* G'*
    # cdef readonly cnp.float64_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    # `missing` is a $(p \times T)$ boolean matrix where a row is a $(p \times 1)$ vector
    # in which the $i$th position is $1$ if $y_{i,t}$ is to be considered a missing value.  
    # *Note:* This is created as the output of np.isnan(obs).
    # cdef readonly int [::1,:] missing
    # `nmissing` is an `T \times 0` integer vector holding the number of *missing* observations
    # $p - p_t$
    # cdef readonly int [:] nmissing

    # Flag for a time-invariant model, which requires that *all* of the
    # possibly time-varying arrays are time-invariant.
    # cdef readonly int time_invariant

    # Flag for initialization.
    # cdef readonly int initialized

    # Flags for performance improvements
    # TODO need to add this to the UI in representation
    # cdef public int diagonal_obs_cov
    # cdef public int subset_design
    # cdef public int companion_transition

    # Temporary arrays
    # cdef cnp.float64_t [::1,:] tmp

    # Temporary selection arrays
    # cdef readonly cnp.float64_t [:] selected_obs
    # The following are contiguous memory segments which are then used to
    # store the data in the above matrices.
    # cdef readonly cnp.float64_t [:] selected_design
    # cdef readonly cnp.float64_t [:] selected_obs_cov

    # Temporary transformation arrays
    # cdef readonly cnp.float64_t [::1,:] transform_cholesky
    # cdef readonly cnp.float64_t [::1,:] transform_obs_cov
    # cdef readonly cnp.float64_t [::1,:] transform_design
    # cdef readonly cnp.float64_t transform_determinant

    # cdef readonly cnp.float64_t [:] collapse_obs
    # cdef readonly cnp.float64_t [:] collapse_obs_tmp
    # cdef readonly cnp.float64_t [::1,:] collapse_design
    # cdef readonly cnp.float64_t [::1,:] collapse_obs_cov
    # cdef readonly cnp.float64_t [::1,:] collapse_cholesky
    # cdef readonly cnp.float64_t collapse_loglikelihood

    # Pointers  
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

    # Current location dimensions
    # cdef int _k_endog, _k_states, _k_posdef, _k_endog2, _k_states2, _k_posdef2, _k_endogstates, _k_statesposdef
    # cdef int _nmissing

    # ### Initialize state space model
    # *Note*: The initial state and state covariance matrix must be provided.
    def __init__(self,
                 cnp.float64_t [::1, :]   obs,
                 cnp.float64_t [::1, :, :] design,
                 cnp.float64_t [::1, :]   obs_intercept,
                 cnp.float64_t [::1, :, :] obs_cov,
                 cnp.float64_t [::1, :, :] transition,
                 cnp.float64_t [::1, :]   state_intercept,
                 cnp.float64_t [::1, :, :] selection,
                 cnp.float64_t [::1, :, :] state_cov,
                 diagonal_obs_cov=-1):

        # Local variables
        cdef:
            int t, i, j
            cnp.npy_intp dim1[1]
            cnp.npy_intp dim2[2]
            cnp.npy_intp dim3[3]

        # #### State space representation variables  
        # **Note**: these arrays share data with the versions defined in
        # Python and passed to this constructor, so if they are updated in
        # Python they will also be updated here.
        self.obs = obs
        self.design = design
        self.obs_intercept = obs_intercept
        self.obs_cov = obs_cov
        self.transition = transition
        self.state_intercept = state_intercept
        self.selection = selection
        self.state_cov = state_cov

        # Dimensions
        self.k_endog = obs.shape[0]
        self.k_states = selection.shape[0]
        self.k_posdef = selection.shape[1]
        self.nobs = obs.shape[1]

        # #### Validate matrix dimensions
        #
        # Make sure that the given state-space matrices have consistent sizes
        tools.validate_matrix_shape('design', &self.design.shape[0],
                                    self.k_endog, self.k_states, self.nobs)
        tools.validate_vector_shape('observation intercept', &self.obs_intercept.shape[0],
                                    self.k_endog, self.nobs)
        tools.validate_matrix_shape('observation covariance matrix', &self.obs_cov.shape[0],
                                    self.k_endog, self.k_endog, self.nobs)
        tools.validate_matrix_shape('transition', &self.transition.shape[0],
                                    self.k_states, self.k_states, self.nobs)
        tools.validate_vector_shape('state intercept', &self.state_intercept.shape[0],
                                    self.k_states, self.nobs)
        tools.validate_matrix_shape('state covariance matrix', &self.state_cov.shape[0],
                                    self.k_posdef, self.k_posdef, self.nobs)

        # Check for a time-invariant model
        self.time_invariant = (
            self.design.shape[2] == 1           and
            self.obs_intercept.shape[1] == 1    and
            self.obs_cov.shape[2] == 1          and
            self.transition.shape[2] == 1       and
            self.state_intercept.shape[1] == 1  and
            self.selection.shape[2] == 1        and
            self.state_cov.shape[2] == 1
        )

        # Set the flag for initialization to be false
        self.initialized = False
        self.initial_state = None
        self.initial_state_cov = None

        # Unless it is specified, check for a diagonal covariance matrix
        if diagonal_obs_cov == -1:
            diagonal_obs_cov = True
            for t in range(self.obs_cov.shape[2]):
                for i in range(self.k_endog):
                    for j in range(self.k_endog):
                        if i == j:
                            continue
                        if not (dabs(self.obs_cov[i, j, t]) < 1e-9):
                            diagonal_obs_cov = False
                            break
        self.diagonal_obs_cov = diagonal_obs_cov

        # Allocate selected state covariance matrix
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = 1
        # (we only allocate memory for time-varying array if necessary)
        if self.state_cov.shape[2] > 1 or self.selection.shape[2] > 1:
            dim3[2] = self.nobs
        self.selected_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT64, FORTRAN)

        # Handle missing data
        self.missing = np.array(np.isnan(obs), dtype=np.int32, order="F")
        self.nmissing = np.array(np.sum(self.missing, axis=0), dtype=np.int32)
        self.has_missing = np.sum(self.nmissing) > 0

        # Create the temporary array
        # Holds arrays of dimension $(m \times m)$
        dim2[0] = self.k_states; dim2[1] = max(self.k_states, self.k_posdef)
        self.tmp = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)

        # Arrays for missing data
        dim1[0] = self.k_endog;
        self.selected_obs = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT64, FORTRAN)
        dim1[0] = self.k_endog;
        self.selected_obs_intercept = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT64, FORTRAN)
        dim1[0] = self.k_endog * self.k_states;
        self.selected_design = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT64, FORTRAN)
        dim1[0] = self.k_endog**2;
        self.selected_obs_cov = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT64, FORTRAN)

        # Arrays for transformations
        dim2[0] = self.k_endog; dim2[1] = self.k_endog
        self.transform_cholesky = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self.transform_obs_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        dim2[0] = self.k_endog; dim2[1] = self.k_states
        self.transform_design = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)

        dim1[0] = self.k_states
        self.collapse_obs = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT64, FORTRAN)
        self.collapse_obs_tmp = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT64, FORTRAN)
        dim2[0] = self.k_states; dim2[1] = self.k_states
        self.collapse_design = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self.collapse_obs_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)
        self.collapse_cholesky = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)

        # Initialize location
        self.t = 0

        # Initialize dimensions
        self.set_dimensions(self.k_endog, self.k_states, self.k_posdef)

    def __reduce__(self):
        init = (np.array(self.obs, copy=True, order='F'), np.array(self.design, copy=True, order='F'),
                np.array(self.obs_intercept, copy=True, order='F'), np.array(self.obs_cov, copy=True, order='F'),
                np.array(self.transition, copy=True, order='F'), np.array(self.state_intercept, copy=True, order='F'),
                np.array(self.selection, copy=True, order='F'), np.array(self.state_cov, copy=True, order='F'),
                self.diagonal_obs_cov)
        state = {'initialized': self.initialized,
                 'initial_state': None,
                 'initial_state_cov': None,
                 'missing': np.array(self.missing, copy=True, order='F'),
                 'nmissing': np.array(self.nmissing, copy=True, order='F'),
                 'has_missing': self.has_missing,
                 'tmp': np.array(self.tmp, copy=True, order='F'),
                 'selected_state_cov': np.array(self.selected_state_cov, copy=True, order='F'),
                 'selected_obs': np.array(self.selected_obs, copy=True, order='F'),
                 'selected_obs_intercept': np.array(self.selected_obs_intercept, copy=True, order='F'),
                 'selected_design': np.array(self.selected_design, copy=True, order='F'),
                 'selected_obs_cov': np.array(self.selected_obs_cov, copy=True, order='F'),
                 'transform_cholesky': np.array(self.transform_cholesky, copy=True, order='F'),
                 'transform_obs_cov': np.array(self.transform_obs_cov, copy=True, order='F'),
                 'transform_design': np.array(self.transform_design, copy=True, order='F'),
                 'collapse_obs': np.array(self.collapse_obs, copy=True, order='F'),
                 'collapse_obs_tmp': np.array(self.collapse_obs_tmp, copy=True, order='F'),
                 'collapse_design': np.array(self.collapse_design, copy=True, order='F'),
                 'collapse_obs_cov': np.array(self.collapse_obs_cov, copy=True, order='F'),
                 'collapse_cholesky': np.array(self.collapse_cholesky, copy=True, order='F'),
                 't': self.t,
                 'collapse_loglikelihood': self.collapse_loglikelihood,
                 'companion_transition': self.companion_transition,
                 'transform_determinant': self.transform_determinant,
                 }
        if self.initialized:
            state['initial_state'] = np.array(self.initial_state, copy=True, order='F')
            state['initial_state_cov'] = np.array(self.initial_state_cov, copy=True, order='F')
        return (self.__class__, init, state)

    def __setstate__(self, state):
        self.initial_state = state['initial_state']
        self.initial_state_cov = state['initial_state_cov']
        self.initialized = state['initialized']
        self.selected_state_cov = state['selected_state_cov']
        self.missing = state['missing']
        self.nmissing =state['nmissing']
        self.has_missing = state['has_missing']
        self.tmp = state['tmp']
        self.selected_obs  = state['selected_obs']
        self.selected_obs_intercept  = state['selected_obs_intercept']
        self.selected_design  = state['selected_design']
        self.selected_obs_cov  =state['selected_obs_cov']
        self.transform_cholesky  = state['transform_cholesky']
        self.transform_obs_cov  = state['transform_obs_cov']
        self.transform_design = state['transform_design']
        self.collapse_obs = state['collapse_obs']
        self.collapse_obs_tmp = state['collapse_obs_tmp']
        self.collapse_design = state['collapse_design']
        self.collapse_obs_cov = state['collapse_obs_cov']
        self.collapse_cholesky = state['collapse_cholesky']
        self.t = state['t']
        self.collapse_loglikelihood = state['collapse_loglikelihood']
        self.companion_transition = state['companion_transition']
        self.transform_determinant = state['transform_determinant']

    # ## Initialize: known values
    #
    # Initialize the filter with specific values, assumed to be known with
    # certainty or else as filled with parameters from a maximum likelihood
    # estimation run.
    def initialize_known(self, cnp.float64_t [:] initial_state, cnp.float64_t [::1, :] initial_state_cov):
        """
        initialize_known(initial_state, initial_state_cov)
        """
        tools.validate_vector_shape('inital state', &initial_state.shape[0], self.k_states, None)
        tools.validate_matrix_shape('initial state covariance', &initial_state_cov.shape[0], self.k_states, self.k_states, None)

        self.initial_state = initial_state
        self.initial_state_cov = initial_state_cov

        self.initialized = True

    # ## Initialize: approximate diffuse priors
    #
    # Durbin and Koopman note that this initialization should only be coupled
    # with the standard Kalman filter for "approximate exploratory work" and
    # can lead to "large rounding errors" (p. 125).
    # 
    # *Note:* see Durbin and Koopman section 5.6.1
    def initialize_approximate_diffuse(self, cnp.float64_t variance=1e2):
        """
        initialize_approximate_diffuse(variance=1e2)
        """
        cdef cnp.npy_intp dim[1]
        dim[0] = self.k_states
        self.initial_state = cnp.PyArray_ZEROS(1, dim, cnp.NPY_FLOAT64, FORTRAN)
        self.initial_state_cov = np.eye(self.k_states, dtype=float).T * variance

        self.initialized = True

    # ## Initialize: stationary process
    # *Note:* see Durbin and Koopman section 5.6.2
    def initialize_stationary(self, complex_step=False):
        """
        initialize_stationary()
        """
        cdef cnp.npy_intp dim1[1]
        cdef cnp.npy_intp dim2[2]
        cdef int i, info, inc = 1
        cdef int k_states2 = self.k_states**2
        cdef cnp.float64_t asum, tol = 1e-9
        cdef cnp.float64_t scalar
        cdef int [::1,:] ipiv

        # Create selected state covariance matrix
        dselect_cov(self.k_states, self.k_posdef,
                                   &self.tmp[0, 0],
                                   &self.selection[0, 0, 0],
                                   &self.state_cov[0, 0, 0],
                                   &self.selected_state_cov[0, 0, 0])

        # Initial state mean
        asum = blas.dasum(&self.k_states, &self.state_intercept[0, 0], &inc)

        dim1[0] = self.k_states
        self.initial_state = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT64, FORTRAN)
        if asum > tol:
            dim2[0] = self.k_states
            dim2[1] = self.k_states
            ipiv = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_INT32, FORTRAN)

            # I - T
            blas.dcopy(
                &k_states2, &self.transition[0, 0, 0], &inc,
                &self.tmp[0,0], &inc)
            scalar = -1.0
            blas.dscal(&k_states2, &scalar, &self.tmp[0, 0], &inc)
            for i in range(self.k_states):
                self.tmp[i, i] = self.tmp[i, i] + 1

            # c
            blas.dcopy(
                &self.k_states, &self.state_intercept[0, 0], &inc,
                &self.initial_state[0], &inc)

            # Solve (I - T) x = c
            lapack.dgetrf(
                &self.k_states, &self.k_states, &self.tmp[0, 0], &self.k_states,
                &ipiv[0, 0], &info)
            lapack.dgetrs(
                'N', &self.k_states, &inc, &self.tmp[0, 0], &self.k_states,
                &ipiv[0, 0], &self.initial_state[0], &self.k_states, &info)

        dim2[0] = self.k_states; dim2[1] = self.k_states
        self.initial_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT64, FORTRAN)

        # Create a copy of the transition matrix (to avoid overwriting it)
        blas.dcopy(
            &k_states2, &self.transition[0, 0, 0], &inc,
            &self.tmp[0, 0], &inc)

        # Copy the selected state covariance to the initial state covariance
        # (it will be overwritten with the appropriate matrix)
        blas.dcopy(
            &k_states2, &self.selected_state_cov[0, 0, 0], &inc,
            &self.initial_state_cov[0, 0], &inc)

        # Solve the discrete Lyapunov equation to the get initial state
        # covariance matrix
        tools._dsolve_discrete_lyapunov(
            &self.tmp[0, 0],
            &self.initial_state_cov[0, 0],
            self.k_states, complex_step)

        self.initialized = True

    def __iter__(self):
        return self

    def __next__(self):
        """
        Advance to the next location
        """
        if self.t >= self.nobs:
            raise StopIteration
        else:
            self.seek(self.t + 1, 0, 0)

    cpdef seek(self, unsigned int t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse):
        cdef int previous_t = self.t

        # Set the global time indicator, if valid
        if t >= self.nobs:
            raise IndexError("Observation index out of range")
        self.t = t

        # Indices for possibly time-varying arrays
        cdef:
            int k_endog
            int design_t = 0
            int obs_intercept_t = 0
            int obs_cov_t = 0
            int transition_t = 0
            int state_intercept_t = 0
            int selection_t = 0
            int state_cov_t = 0

        # Get indices for possibly time-varying arrays
        if not self.time_invariant:
            if self.design.shape[2] > 1:
                design_t = t
            if self.obs_intercept.shape[1] > 1:
                obs_intercept_t = t
            if self.obs_cov.shape[2] > 1:
                obs_cov_t = t
            if self.transition.shape[2] > 1:
                transition_t = t
            if self.state_intercept.shape[1] > 1:
                state_intercept_t = t
            if self.selection.shape[2] > 1:
                selection_t = t
            if self.state_cov.shape[2] > 1:
                state_cov_t = t

        # Initialize object-level pointers to statespace arrays
        self._obs = &self.obs[0, t]
        self._design = &self.design[0, 0, design_t]
        self._obs_intercept = &self.obs_intercept[0, obs_intercept_t]
        self._obs_cov = &self.obs_cov[0, 0, obs_cov_t]
        self._transition = &self.transition[0, 0, transition_t]
        self._state_intercept = &self.state_intercept[0, state_intercept_t]
        self._selection = &self.selection[0, 0, selection_t]
        self._state_cov = &self.state_cov[0, 0, state_cov_t]

        # Initialize object-level pointers to initialization
        if not self.initialized:
            raise RuntimeError("Statespace model not initialized.")
        self._initial_state = &self.initial_state[0]
        self._initial_state_cov = &self.initial_state_cov[0,0]

        # Create the selected state covariance matrix
        self.select_state_cov(t)

        # Handle missing data
        # Note: this modifies object pointers and _* dimensions
        k_endog = self.select_missing(t)

        # Set dimensions
        self.set_dimensions(k_endog, self.k_states, self.k_posdef)

        # Handle transformations
        self.transform(t, previous_t, transform_diagonalize, transform_generalized_collapse)

    @cython.cfunc
    @cython.returns(cython.void)
    def set_dimensions(self, unsigned int k_endog, unsigned int k_states, unsigned int k_posdef):
        self._k_endog = k_endog
        self._k_states = k_states
        self._k_posdef = k_posdef
        self._k_endog2 = k_endog**2
        self._k_states2 = k_states**2
        self._k_posdef2 = k_posdef**2
        self._k_endogstates = k_endog * k_states
        self._k_statesposdef = k_states * k_posdef

    @cython.cfunc
    @cython.returns(cython.void)
    def select_state_cov(self, unsigned int t):
        cdef int selected_state_cov_t = 0

        # ### Get selected state covariance matrix
        if t == 0 or self.selected_state_cov.shape[2] > 1:
            selected_state_cov_t = t
            self._selected_state_cov = &self.selected_state_cov[0, 0, selected_state_cov_t]

            dselect_cov(
                self.k_states, self.k_posdef,
                &self.tmp[0, 0],
                self._selection,
                self._state_cov,
                self._selected_state_cov)
        else:
            self._selected_state_cov = &self.selected_state_cov[0, 0, 0]

    @cython.cfunc
    @cython.returns(int)
    def select_missing(self, unsigned int t):
        # Note: this assumes that object pointers are already initialized
        # Note: this assumes that transform_... will be done *later*
        cdef int k_endog = self.k_endog

        # Set the current iteration nmissing
        self._nmissing = self.nmissing[t]

        # ### Perform missing selections
        # In Durbin and Koopman (2012), these are represented as matrix
        # multiplications, i.e. $Z_t^* = W_t Z_t$ where $W_t$ is a row
        # selection matrix (it contains a subset of rows of the identity
        # matrix).
        #
        # It's more efficient, though, to just copy over the data directly,
        # which is what is done here. Note that the `selected_*` arrays are
        # defined as single-dimensional, so the assignment indexes below are
        # set such that the arrays can be interpreted by the BLAS and LAPACK
        # functions as two-dimensional, column-major arrays.
        #
        # In the case that all data is missing (e.g. this is what happens in
        # forecasting), we actually set don't change the dimension, but we set
        # the design matrix to the zeros array.
        if self._nmissing == self.k_endog:
            self._select_missing_entire_obs(t)
        elif self._nmissing > 0:
            self._select_missing_partial_obs(t)
            k_endog = self.k_endog - self._nmissing

        # Return the number of non-missing endogenous variables
        return k_endog

    @cython.cfunc
    @cython.returns(cython.void)
    def _select_missing_entire_obs(self, unsigned int t):
        cdef:
            int i, j

        # Design matrix is set to zeros
        for i in range(self.k_states):
            for j in range(self.k_endog):
                self.selected_design[j + i * self.k_endog] = 0.0
        self._design = &self.selected_design[0]

    @cython.cfunc
    @cython.returns(cython.void)
    def _select_missing_partial_obs(self, unsigned int t):
        cdef:
            int i, j, k, l
            int inc = 1
            int design_t = 0
            int obs_cov_t = 0
            int k_endog = self.k_endog - self._nmissing

        k = 0
        for i in range(self.k_endog):
            if not self.missing[i, t]:
                self.selected_obs[k] = self._obs[i]
                self.selected_obs_intercept[k] = self._obs_intercept[i]

                # i is rows, k is rows
                blas.dcopy(
                    &self.k_states, &self._design[i], &self.k_endog,
                    &self.selected_design[k], &k_endog)

                # i, k is columns, j, l is rows
                l = 0
                for j in range(self.k_endog):
                    if not self.missing[j, t]:
                        self.selected_obs_cov[l + k * k_endog] = self._obs_cov[j + i * self.k_endog]
                        l += 1
                k += 1
        self._obs = &self.selected_obs[0]
        self._obs_intercept = &self.selected_obs_intercept[0]
        self._design = &self.selected_design[0]
        self._obs_cov = &self.selected_obs_cov[0]

    cdef void transform(self, unsigned int t, unsigned int previous_t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse) except *:
        # Reset the collapsed loglikelihood
        self.collapse_loglikelihood = 0

        if transform_generalized_collapse and not self._k_endog <= self._k_states:
            k_endog = self.transform_generalized_collapse(t, previous_t)
            # Reset dimensions
            self.set_dimensions(k_endog, self._k_states, self._k_posdef)
        elif transform_diagonalize and not self.diagonal_obs_cov:
            self.transform_diagonalize(t, previous_t)

    cdef void transform_diagonalize(self, unsigned int t, unsigned int previous_t) except *:
        # Note: this assumes that initialize_object_pointers has *already* been done
        # Note: this assumes that select_missing has *already* been done
        # TODO need unit tests, especially for the missing case
        # TODO need to also transform observation intercept
        cdef:
            int i, j, inc=1
            int obs_cov_t, design_t
            int info
            int reset_missing
            cnp.float64_t * _transform_obs_cov = &self.transform_obs_cov[0, 0]
            cnp.float64_t * _transform_cholesky = &self.transform_cholesky[0, 0]

        # Compute the cholesky decomposition of *self._obs_cov
        if self.diagonal_obs_cov:
            return

        # Handle missing data
        if self.nmissing[t] == self.k_endog:
            return

        reset_missing = 0
        for i in range(self.k_endog):
            reset_missing = reset_missing + (not self.missing[i, t] == self.missing[i, previous_t])

        # Perform the LDL decomposition, if necessary
        if t == 0 or self.obs_cov.shape[2] > 1 or reset_missing:
            # Make sure we don't have an observation intercept
            if not np.sum(self.obs_intercept) == 0 or self.obs_intercept.shape[2] > 1:
                raise RuntimeError('The univariate method with non-diagonal '
                                   'observation covariance matrix does not '
                                   'currently support an observation intercept.')

            # LDL decomposition
            blas.dcopy(
                &self._k_endog2, self._obs_cov, &inc,
                _transform_cholesky, &inc)
            info = tools._dldl(_transform_cholesky, self._k_endog)

            # Check for errors
            if info > 0:
                warnings.warn("Positive semi-definite observation covariance matrix encountered at period %d" % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in LDL factorization of observation covariance matrix encountered at period %d' % t)

            # Currently both L and D are stored in transform_cholesky
            for i in range(self._k_endog): # i is rows
                for j in range(self._k_endog): # j is columns
                    # Diagonal elements come from the diagonal
                    if i == j:
                        _transform_obs_cov[i + i * self._k_endog] = _transform_cholesky[i + i * self._k_endog]
                    # Other elements are zero
                    else:
                        _transform_obs_cov[i + j * self._k_endog] = 0

                    # Zero out the upper triangle of the cholesky
                    if j > i:
                        _transform_cholesky[i + j * self._k_endog] = 0

                # Convert from L to C simply by setting the diagonal elements to ones
                _transform_cholesky[i + i * self._k_endog] = 1
        
        # Solve for y_t^*
        # (unless this is a completely missing observation)
        if not self._nmissing == self.k_endog:
            # If we have some missing elements, selected_obs is already populated
            if self._nmissing == 0:
                blas.dcopy(
                    &self._k_endog, &self.obs[0, t], &inc,
                    &self.selected_obs[0], &inc)
            lapack.dtrtrs(
                "L", "N", "U", &self._k_endog, &inc,
                _transform_cholesky, &self.k_endog,
                &self.selected_obs[0], &self._k_endog, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Singular factorization of observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in factorization of observation covariance matrix encountered at period %d' % t)

            # Setup the pointer
            self._obs = &self.selected_obs[0]

        # Solve for Z_t^*, if necessary
        if t == 0 or self.design.shape[2] > 1 or reset_missing:
            blas.dcopy(
                &self._k_endogstates, self._design, &inc,
                &self.transform_design[0, 0], &inc)
            lapack.dtrtrs(
                "L", "N", "U", &self._k_endog, &self._k_states,
                _transform_cholesky, &self.k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Singular factorization of observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in factorization of observation covariance matrix encountered at period %d' % t)

        # Setup final pointers            
        self._design = &self.transform_design[0, 0]
        self._obs_cov = &self.transform_obs_cov[0, 0]

    cdef int transform_generalized_collapse(self, unsigned int t, unsigned int previous_t) except *:
        # Note: this assumes that initialize_object_pointers has *already* been done
        # Note: this assumes that select_missing has *already* been done
        # TODO need unit tests, especially for the missing case
        cdef:
            int i, j, inc=1
            int obs_cov_t, design_t
            int info
            int reset_missing
            cnp.float64_t alpha = 1.0
            cnp.float64_t beta = 0.0
            cnp.float64_t gamma = -1.0
            int k_states = self._k_states
            int k_states2 = self._k_states2
            int k_endogstates = self._k_endogstates

        # $y_t^* = \bar A^* y_t = C_t Z_t' H_t^{-1} y_t$  
        # $Z_t^* = C_t^{-1}$  
        # $H_t^* = I_m$  

        # Make sure we have enough observations to perform collapse
        if self.k_endog < self.k_states:
            raise RuntimeError('Cannot collapse observation vector it the'
                               ' state dimension is larger than the dimension'
                               ' of the observation vector.')

        # Adjust for a VAR transition (i.e. design = [#, 0], where the zeros
        # correspond to all states except the first k_posdef states)
        if self.subset_design:
            k_states = self._k_posdef
            k_states2 = self._k_posdef2
            k_endogstates = self._k_endog * self._k_posdef

        # Handle missing data
        if self.nmissing[t] == self.k_endog:
            return self.k_states
        reset_missing = 0
        for i in range(self.k_endog):
            reset_missing = reset_missing + (not self.missing[i, t] == self.missing[i, previous_t])

        # Initialize the transformation
        if self.collapse_obs_cov[0, 0] == 0:
            # Set H_t^* to identity
            for i in range(k_states):
                self.collapse_obs_cov[i, i] = 1

            # Make sure we don't have an observation intercept
            if not np.sum(self.obs_intercept) == 0 or self.obs_intercept.shape[2] > 1:
                raise RuntimeError('The observation collapse transformation'
                                   ' does not currently support an observation'
                                   ' intercept.')

        # Perform the Cholesky decomposition of H_t, if necessary
        if t == 0 or self.obs_cov.shape[2] > 1 or reset_missing:
            # Cholesky decomposition: $H = L L'$  
            blas.dcopy(
                &self._k_endog2, self._obs_cov, &inc,
                &self.transform_cholesky[0, 0], &inc)
            # Use LDA=self.k_endog so that we can use the memoryview slicing below
            lapack.dpotrf(
                "L", &self._k_endog, &self.transform_cholesky[0, 0], &self.k_endog, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Non-positive-definite observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in observation covariance matrix encountered at period %d' % t)

            # Calculate the determinant (just the squared product of the
            # diagonals, in the Cholesky decomposition case)
            self.transform_determinant = 1.0
            for i in range(self._k_endog):
                if not self.transform_cholesky[i, i] == 0:
                    self.transform_determinant = self.transform_determinant * self.transform_cholesky[i, i]
            self.transform_determinant = self.transform_determinant**2

        # Get $Z_t \equiv C^{-1}$, if necessary  
        if t == 0 or self.obs_cov.shape[2] > 1 or self.design.shape[2] > 1 or reset_missing:
            # Calculate $H_t^{-1} Z_t \equiv (Z_t' H_t^{-1})'$ via Cholesky solver
            blas.dcopy(
                &self._k_endogstates, self._design, &inc,
                &self.transform_design[0, 0], &inc)
            lapack.dpotrs(
                "L", &self._k_endog, &k_states,
                &self.transform_cholesky[0, 0], &self.k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &info)

            # Check for errors
            if not info == 0:
                raise np.linalg.LinAlgError('Invalid value in calculation of H_t^{-1}Z matrix encountered at period %d' % t)
        
            # Calculate $(H_t^{-1} Z_t)' Z_t$  
            # $(m \times m) = (m \times p) (p \times p) (p \times m)$
            blas.dgemm(
                "T", "N", &k_states, &k_states, &self._k_endog,
                &alpha, self._design, &self._k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &beta, &self.collapse_cholesky[0, 0], &self._k_states)

            # Calculate $(Z_t' H_t^{-1} Z_t)^{-1}$ via Cholesky inversion  
            lapack.dpotrf("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)
            lapack.dpotri("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)

            # Calculate $C_t$ (the upper triangular cholesky decomposition of $(Z_t' H_t^{-1} Z_t)^{-1}$)  
            lapack.dpotrf("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Non-positive-definite ZHZ matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in ZHZ matrix encountered at period %d' % t)

            # Calculate $C_t'^{-1} \equiv Z_t$  
            # Do so by solving the system: $C_t' x = I$  
            # (Recall that collapse_obs_cov is an identity matrix)
            blas.dcopy(
                &self._k_states2, &self.collapse_obs_cov[0, 0], &inc,
                &self.collapse_design[0, 0], &inc)
            lapack.dtrtrs(
                "U", "T", "N", &k_states, &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_design[0, 0], &self._k_states,
                &info)

        # Calculate $\bar y_t^* = \bar A_t^* y_t = C_t Z_t' H_t^{-1} y_t$  
        # (unless this is a completely missing observation)
        self.collapse_loglikelihood = 0
        if not self._nmissing == self.k_endog:
            # If we have some missing elements, selected_obs is already populated
            if self._nmissing == 0:
                blas.dcopy(
                    &self.k_endog, &self.obs[0, t], &inc,
                    &self.selected_obs[0], &inc)
            # $\\# = Z_t' H_t^{-1} y_t$
            blas.dgemv(
                "T", &self._k_endog, &k_states,
                &alpha, &self.transform_design[0, 0], &self._k_endog,
                &self.selected_obs[0], &inc,
                &beta, &self.collapse_obs[0], &inc)
            # $y_t^* = C_t \\#$  
            blas.dtrmv(
                "U", "N", "N", &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_obs[0], &inc)

            # Get residuals for loglikelihood calculation
            # Note: Durbin and Koopman (2012) appears to have an error in the
            # formula here. They have $e_t = y_t - Z_t \bar y_t^*$, whereas it
            # should be: $e_t = y_t - Z_t C_t' \bar y_t^*$
            # See Jungbacker and Koopman (2014), section 2.5 where $e_t$ is
            # defined. In this case, $Z_t^dagger = Z_t C_t$ where
            # $C_t C_t' = (Z_t' \Sigma_\varepsilon^{-1} Z_t)^{-1}$.
            # 

            # $ \\# = C_t' y_t^*$
            blas.dcopy(
                &k_states, &self.collapse_obs[0], &inc,
                &self.collapse_obs_tmp[0], &inc)
            blas.dtrmv(
                "U", "T", "N", &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_obs_tmp[0], &inc)

            # $e_t = - Z_t C_t' y_t^* + y_t$
            blas.dgemv(
                "N", &self._k_endog, &k_states,
                &gamma, self._design, &self._k_endog,
                &self.collapse_obs_tmp[0], &inc,
                &alpha, &self.selected_obs[0], &inc)

            # Calculate e_t' H_t^{-1} e_t via Cholesky solver  
            # $H_t^{-1} = (L L')^{-1} = L^{-1}' L^{-1}$  
            # So we want $e_t' L^{-1}' L^{-1} e_t = (L^{-1} e_t)' L^{-1} e_t$  
            # We have $L$ in `transform_cholesky`, so we want to do a linear  
            # solve of $L x = e_t$  where L is lower triangular
            lapack.dtrtrs(
                "L", "N", "N", &self._k_endog, &inc,
                &self.transform_cholesky[0, 0], &self.k_endog,
                &self.selected_obs[0], &self._k_endog,
                &info)

            # Calculate loglikelihood contribution of this observation

            # $e_t' H_t^{-1} e_t = (L^{-1} e_t)' L^{-1} e_t = \sum_i e_{i,t}**2$  
            self.collapse_loglikelihood = 0
            for i in range(self._k_endog):
                self.collapse_loglikelihood = self.collapse_loglikelihood + self.selected_obs[i]**2
            
            # (p-m) log( 2*pi) + log( |H_t| )
            self.collapse_loglikelihood = (
                self.collapse_loglikelihood +
                (self._k_endog - k_states)*dlog(2*NPY_PI) + 
                dlog(self.transform_determinant)
            )

            # -0.5 * ...
            self.collapse_loglikelihood = -0.5 * self.collapse_loglikelihood

        # Set pointers
        self._obs = &self.collapse_obs[0]
        self._design = &self.collapse_design[0, 0]
        self._obs_cov = &self.collapse_obs_cov[0, 0]

        # TODO can I replace this with k_states? I think I should be able to
        return self._k_states


# ### Selected covariance matrice
cdef int dselect_cov(int k, int k_posdef,
                         cnp.float64_t* tmp,
                         cnp.float64_t* selection,
                         cnp.float64_t* cov,
                         cnp.float64_t* selected_cov):
    cdef:
        cnp.float64_t alpha = 1.0
        cnp.float64_t beta = 0.0

    # Only need to do something if there is a covariance matrix
    # (i.e k_posdof == 0)
    if k_posdef > 0:

        # #### Calculate selected state covariance matrix  
        # $Q_t^* = R_t Q_t R_t'$
        # 
        # Combine a selection matrix and a covariance matrix to get
        # a simplified (but possibly singular) "selected" covariance
        # matrix (see e.g. Durbin and Koopman p. 43)

        # `tmp0` array used here, dimension $(m \times r)$  

        # TODO this does not require two ?gemm calls, since we know that it
        # is just selection rows and columns of the Q matrix

        # $\\#_0 = 1.0 * R_t Q_t$  
        # $(m \times r) = (m \times r) (r \times r)$
        blas.dgemm(
            "N", "N", &k, &k_posdef, &k_posdef,
            &alpha, selection, &k,
            cov, &k_posdef,
            &beta, tmp, &k)
        # $Q_t^* = 1.0 * \\#_0 R_t'$  
        # $(m \times m) = (m \times r) (m \times r)'$
        blas.dgemm(
            "N", "T", &k, &k, &k_posdef,
            &alpha, tmp, &k,
            selection, &k,
            &beta, selected_cov, &k)

## State Space Representation
cdef class zStatespace(object):
    """
    zStatespace(obs, design, obs_intercept, obs_cov, transition, state_intercept, selection, state_cov)

    *See Durbin and Koopman (2012), Chapter 4 for all notation*
    """

    # ### State space representation
    # 
    # $$
    # \begin{align}
    # y_t & = Z_t \alpha_t + d_t + \varepsilon_t \hspace{3em} & \varepsilon_t & \sim N(0, H_t) \\\\
    # \alpha_{t+1} & = T_t \alpha_t + c_t + R_t \eta_t & \eta_t & \sim N(0, Q_t) \\\\
    # & & \alpha_1 & \sim N(a_1, P_1)
    # \end{align}
    # $$
    # 
    # $y_t$ is $p \times 1$  
    # $\varepsilon_t$ is $p \times 1$  
    # $\alpha_t$ is $m \times 1$  
    # $\eta_t$ is $r \times 1$  
    # $t = 1, \dots, T$

    # `nobs` $\equiv T$ is the length of the time-series  
    # `k_endog` $\equiv p$ is dimension of observation space  
    # `k_states` $\equiv m$ is the dimension of the state space  
    # `k_posdef` $\equiv r$ is the dimension of the state shocks  
    # *Old notation: T, n, k, g*
    # cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # `obs` $\equiv y_t$ is the **observation vector** $(p \times T)$  
    # `design` $\equiv Z_t$ is the **design vector** $(p \times m \times T)$  
    # `obs_intercept` $\equiv d_t$ is the **observation intercept** $(p \times T)$  
    # `obs_cov` $\equiv H_t$ is the **observation covariance matrix** $(p \times p \times T)$  
    # `transition` $\equiv T_t$ is the **transition matrix** $(m \times m \times T)$  
    # `state_intercept` $\equiv c_t$ is the **state intercept** $(m \times T)$  
    # `selection` $\equiv R_t$ is the **selection matrix** $(m \times r \times T)$  
    # `state_cov` $\equiv Q_t$ is the **state covariance matrix** $(r \times r \times T)$  
    # `selected_state_cov` $\equiv R Q_t R'$ is the **selected state covariance matrix** $(m \times m \times T)$  
    # `initial_state` $\equiv a_1$ is the **initial state mean** $(m \times 1)$  
    # `initial_state_cov` $\equiv P_1$ is the **initial state covariance matrix** $(m \times m)$
    #
    # With the exception of `obs`, these are *optionally* time-varying. If they are instead time-invariant,
    # then the dimension of length $T$ is instead of length $1$.
    #
    # *Note*: the initial vectors' notation 1-indexed as in Durbin and Koopman,
    # but in the recursions below it will be 0-indexed in the Python arrays.
    # 
    # *Old notation: y, -, mu, beta_tt_init, P_tt_init*
    # cdef readonly cnp.complex128_t [::1,:] obs, obs_intercept, state_intercept
    # cdef readonly cnp.complex128_t [:] initial_state
    # cdef readonly cnp.complex128_t [::1,:] initial_state_cov
    # *Old notation: H, R, F, G, Q*, G Q* G'*
    # cdef readonly cnp.complex128_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    # `missing` is a $(p \times T)$ boolean matrix where a row is a $(p \times 1)$ vector
    # in which the $i$th position is $1$ if $y_{i,t}$ is to be considered a missing value.  
    # *Note:* This is created as the output of np.isnan(obs).
    # cdef readonly int [::1,:] missing
    # `nmissing` is an `T \times 0` integer vector holding the number of *missing* observations
    # $p - p_t$
    # cdef readonly int [:] nmissing

    # Flag for a time-invariant model, which requires that *all* of the
    # possibly time-varying arrays are time-invariant.
    # cdef readonly int time_invariant

    # Flag for initialization.
    # cdef readonly int initialized

    # Flags for performance improvements
    # TODO need to add this to the UI in representation
    # cdef public int diagonal_obs_cov
    # cdef public int subset_design
    # cdef public int companion_transition

    # Temporary arrays
    # cdef cnp.complex128_t [::1,:] tmp

    # Temporary selection arrays
    # cdef readonly cnp.complex128_t [:] selected_obs
    # The following are contiguous memory segments which are then used to
    # store the data in the above matrices.
    # cdef readonly cnp.complex128_t [:] selected_design
    # cdef readonly cnp.complex128_t [:] selected_obs_cov

    # Temporary transformation arrays
    # cdef readonly cnp.complex128_t [::1,:] transform_cholesky
    # cdef readonly cnp.complex128_t [::1,:] transform_obs_cov
    # cdef readonly cnp.complex128_t [::1,:] transform_design
    # cdef readonly cnp.complex128_t transform_determinant

    # cdef readonly cnp.complex128_t [:] collapse_obs
    # cdef readonly cnp.complex128_t [:] collapse_obs_tmp
    # cdef readonly cnp.complex128_t [::1,:] collapse_design
    # cdef readonly cnp.complex128_t [::1,:] collapse_obs_cov
    # cdef readonly cnp.complex128_t [::1,:] collapse_cholesky
    # cdef readonly cnp.complex128_t collapse_loglikelihood

    # Pointers  
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

    # Current location dimensions
    # cdef int _k_endog, _k_states, _k_posdef, _k_endog2, _k_states2, _k_posdef2, _k_endogstates, _k_statesposdef
    # cdef int _nmissing

    # ### Initialize state space model
    # *Note*: The initial state and state covariance matrix must be provided.
    def __init__(self,
                 cnp.complex128_t [::1, :]   obs,
                 cnp.complex128_t [::1, :, :] design,
                 cnp.complex128_t [::1, :]   obs_intercept,
                 cnp.complex128_t [::1, :, :] obs_cov,
                 cnp.complex128_t [::1, :, :] transition,
                 cnp.complex128_t [::1, :]   state_intercept,
                 cnp.complex128_t [::1, :, :] selection,
                 cnp.complex128_t [::1, :, :] state_cov,
                 diagonal_obs_cov=-1):

        # Local variables
        cdef:
            int t, i, j
            cnp.npy_intp dim1[1]
            cnp.npy_intp dim2[2]
            cnp.npy_intp dim3[3]

        # #### State space representation variables  
        # **Note**: these arrays share data with the versions defined in
        # Python and passed to this constructor, so if they are updated in
        # Python they will also be updated here.
        self.obs = obs
        self.design = design
        self.obs_intercept = obs_intercept
        self.obs_cov = obs_cov
        self.transition = transition
        self.state_intercept = state_intercept
        self.selection = selection
        self.state_cov = state_cov

        # Dimensions
        self.k_endog = obs.shape[0]
        self.k_states = selection.shape[0]
        self.k_posdef = selection.shape[1]
        self.nobs = obs.shape[1]

        # #### Validate matrix dimensions
        #
        # Make sure that the given state-space matrices have consistent sizes
        tools.validate_matrix_shape('design', &self.design.shape[0],
                                    self.k_endog, self.k_states, self.nobs)
        tools.validate_vector_shape('observation intercept', &self.obs_intercept.shape[0],
                                    self.k_endog, self.nobs)
        tools.validate_matrix_shape('observation covariance matrix', &self.obs_cov.shape[0],
                                    self.k_endog, self.k_endog, self.nobs)
        tools.validate_matrix_shape('transition', &self.transition.shape[0],
                                    self.k_states, self.k_states, self.nobs)
        tools.validate_vector_shape('state intercept', &self.state_intercept.shape[0],
                                    self.k_states, self.nobs)
        tools.validate_matrix_shape('state covariance matrix', &self.state_cov.shape[0],
                                    self.k_posdef, self.k_posdef, self.nobs)

        # Check for a time-invariant model
        self.time_invariant = (
            self.design.shape[2] == 1           and
            self.obs_intercept.shape[1] == 1    and
            self.obs_cov.shape[2] == 1          and
            self.transition.shape[2] == 1       and
            self.state_intercept.shape[1] == 1  and
            self.selection.shape[2] == 1        and
            self.state_cov.shape[2] == 1
        )

        # Set the flag for initialization to be false
        self.initialized = False
        self.initial_state = None
        self.initial_state_cov = None

        # Unless it is specified, check for a diagonal covariance matrix
        if diagonal_obs_cov == -1:
            diagonal_obs_cov = True
            for t in range(self.obs_cov.shape[2]):
                for i in range(self.k_endog):
                    for j in range(self.k_endog):
                        if i == j:
                            continue
                        if not (zabs(self.obs_cov[i, j, t]) < 1e-9):
                            diagonal_obs_cov = False
                            break
        self.diagonal_obs_cov = diagonal_obs_cov

        # Allocate selected state covariance matrix
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = 1
        # (we only allocate memory for time-varying array if necessary)
        if self.state_cov.shape[2] > 1 or self.selection.shape[2] > 1:
            dim3[2] = self.nobs
        self.selected_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX128, FORTRAN)

        # Handle missing data
        self.missing = np.array(np.isnan(obs), dtype=np.int32, order="F")
        self.nmissing = np.array(np.sum(self.missing, axis=0), dtype=np.int32)
        self.has_missing = np.sum(self.nmissing) > 0

        # Create the temporary array
        # Holds arrays of dimension $(m \times m)$
        dim2[0] = self.k_states; dim2[1] = max(self.k_states, self.k_posdef)
        self.tmp = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)

        # Arrays for missing data
        dim1[0] = self.k_endog;
        self.selected_obs = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX128, FORTRAN)
        dim1[0] = self.k_endog;
        self.selected_obs_intercept = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX128, FORTRAN)
        dim1[0] = self.k_endog * self.k_states;
        self.selected_design = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX128, FORTRAN)
        dim1[0] = self.k_endog**2;
        self.selected_obs_cov = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX128, FORTRAN)

        # Arrays for transformations
        dim2[0] = self.k_endog; dim2[1] = self.k_endog
        self.transform_cholesky = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self.transform_obs_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        dim2[0] = self.k_endog; dim2[1] = self.k_states
        self.transform_design = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)

        dim1[0] = self.k_states
        self.collapse_obs = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX128, FORTRAN)
        self.collapse_obs_tmp = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX128, FORTRAN)
        dim2[0] = self.k_states; dim2[1] = self.k_states
        self.collapse_design = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self.collapse_obs_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)
        self.collapse_cholesky = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)

        # Initialize location
        self.t = 0

        # Initialize dimensions
        self.set_dimensions(self.k_endog, self.k_states, self.k_posdef)

    def __reduce__(self):
        init = (np.array(self.obs, copy=True, order='F'), np.array(self.design, copy=True, order='F'),
                np.array(self.obs_intercept, copy=True, order='F'), np.array(self.obs_cov, copy=True, order='F'),
                np.array(self.transition, copy=True, order='F'), np.array(self.state_intercept, copy=True, order='F'),
                np.array(self.selection, copy=True, order='F'), np.array(self.state_cov, copy=True, order='F'),
                self.diagonal_obs_cov)
        state = {'initialized': self.initialized,
                 'initial_state': None,
                 'initial_state_cov': None,
                 'missing': np.array(self.missing, copy=True, order='F'),
                 'nmissing': np.array(self.nmissing, copy=True, order='F'),
                 'has_missing': self.has_missing,
                 'tmp': np.array(self.tmp, copy=True, order='F'),
                 'selected_state_cov': np.array(self.selected_state_cov, copy=True, order='F'),
                 'selected_obs': np.array(self.selected_obs, copy=True, order='F'),
                 'selected_obs_intercept': np.array(self.selected_obs_intercept, copy=True, order='F'),
                 'selected_design': np.array(self.selected_design, copy=True, order='F'),
                 'selected_obs_cov': np.array(self.selected_obs_cov, copy=True, order='F'),
                 'transform_cholesky': np.array(self.transform_cholesky, copy=True, order='F'),
                 'transform_obs_cov': np.array(self.transform_obs_cov, copy=True, order='F'),
                 'transform_design': np.array(self.transform_design, copy=True, order='F'),
                 'collapse_obs': np.array(self.collapse_obs, copy=True, order='F'),
                 'collapse_obs_tmp': np.array(self.collapse_obs_tmp, copy=True, order='F'),
                 'collapse_design': np.array(self.collapse_design, copy=True, order='F'),
                 'collapse_obs_cov': np.array(self.collapse_obs_cov, copy=True, order='F'),
                 'collapse_cholesky': np.array(self.collapse_cholesky, copy=True, order='F'),
                 't': self.t,
                 'collapse_loglikelihood': self.collapse_loglikelihood,
                 'companion_transition': self.companion_transition,
                 'transform_determinant': self.transform_determinant,
                 }
        if self.initialized:
            state['initial_state'] = np.array(self.initial_state, copy=True, order='F')
            state['initial_state_cov'] = np.array(self.initial_state_cov, copy=True, order='F')
        return (self.__class__, init, state)

    def __setstate__(self, state):
        self.initial_state = state['initial_state']
        self.initial_state_cov = state['initial_state_cov']
        self.initialized = state['initialized']
        self.selected_state_cov = state['selected_state_cov']
        self.missing = state['missing']
        self.nmissing =state['nmissing']
        self.has_missing = state['has_missing']
        self.tmp = state['tmp']
        self.selected_obs  = state['selected_obs']
        self.selected_obs_intercept  = state['selected_obs_intercept']
        self.selected_design  = state['selected_design']
        self.selected_obs_cov  =state['selected_obs_cov']
        self.transform_cholesky  = state['transform_cholesky']
        self.transform_obs_cov  = state['transform_obs_cov']
        self.transform_design = state['transform_design']
        self.collapse_obs = state['collapse_obs']
        self.collapse_obs_tmp = state['collapse_obs_tmp']
        self.collapse_design = state['collapse_design']
        self.collapse_obs_cov = state['collapse_obs_cov']
        self.collapse_cholesky = state['collapse_cholesky']
        self.t = state['t']
        self.collapse_loglikelihood = state['collapse_loglikelihood']
        self.companion_transition = state['companion_transition']
        self.transform_determinant = state['transform_determinant']

    # ## Initialize: known values
    #
    # Initialize the filter with specific values, assumed to be known with
    # certainty or else as filled with parameters from a maximum likelihood
    # estimation run.
    def initialize_known(self, cnp.complex128_t [:] initial_state, cnp.complex128_t [::1, :] initial_state_cov):
        """
        initialize_known(initial_state, initial_state_cov)
        """
        tools.validate_vector_shape('inital state', &initial_state.shape[0], self.k_states, None)
        tools.validate_matrix_shape('initial state covariance', &initial_state_cov.shape[0], self.k_states, self.k_states, None)

        self.initial_state = initial_state
        self.initial_state_cov = initial_state_cov

        self.initialized = True

    # ## Initialize: approximate diffuse priors
    #
    # Durbin and Koopman note that this initialization should only be coupled
    # with the standard Kalman filter for "approximate exploratory work" and
    # can lead to "large rounding errors" (p. 125).
    # 
    # *Note:* see Durbin and Koopman section 5.6.1
    def initialize_approximate_diffuse(self, cnp.complex128_t variance=1e2):
        """
        initialize_approximate_diffuse(variance=1e2)
        """
        cdef cnp.npy_intp dim[1]
        dim[0] = self.k_states
        self.initial_state = cnp.PyArray_ZEROS(1, dim, cnp.NPY_COMPLEX128, FORTRAN)
        self.initial_state_cov = np.eye(self.k_states, dtype=complex).T * variance

        self.initialized = True

    # ## Initialize: stationary process
    # *Note:* see Durbin and Koopman section 5.6.2
    def initialize_stationary(self, complex_step=False):
        """
        initialize_stationary()
        """
        cdef cnp.npy_intp dim1[1]
        cdef cnp.npy_intp dim2[2]
        cdef int i, info, inc = 1
        cdef int k_states2 = self.k_states**2
        cdef cnp.float64_t asum, tol = 1e-9
        cdef cnp.complex128_t scalar
        cdef int [::1,:] ipiv

        # Create selected state covariance matrix
        zselect_cov(self.k_states, self.k_posdef,
                                   &self.tmp[0, 0],
                                   &self.selection[0, 0, 0],
                                   &self.state_cov[0, 0, 0],
                                   &self.selected_state_cov[0, 0, 0])

        # Initial state mean
        asum = blas.dzasum(&self.k_states, &self.state_intercept[0, 0], &inc)

        dim1[0] = self.k_states
        self.initial_state = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX128, FORTRAN)
        if asum > tol:
            dim2[0] = self.k_states
            dim2[1] = self.k_states
            ipiv = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_INT32, FORTRAN)

            # I - T
            blas.zcopy(
                &k_states2, &self.transition[0, 0, 0], &inc,
                &self.tmp[0,0], &inc)
            scalar = -1.0
            blas.zscal(&k_states2, &scalar, &self.tmp[0, 0], &inc)
            for i in range(self.k_states):
                self.tmp[i, i] = self.tmp[i, i] + 1

            # c
            blas.zcopy(
                &self.k_states, &self.state_intercept[0, 0], &inc,
                &self.initial_state[0], &inc)

            # Solve (I - T) x = c
            lapack.zgetrf(
                &self.k_states, &self.k_states, &self.tmp[0, 0], &self.k_states,
                &ipiv[0, 0], &info)
            lapack.zgetrs(
                'N', &self.k_states, &inc, &self.tmp[0, 0], &self.k_states,
                &ipiv[0, 0], &self.initial_state[0], &self.k_states, &info)

        dim2[0] = self.k_states; dim2[1] = self.k_states
        self.initial_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX128, FORTRAN)

        # Create a copy of the transition matrix (to avoid overwriting it)
        blas.zcopy(
            &k_states2, &self.transition[0, 0, 0], &inc,
            &self.tmp[0, 0], &inc)

        # Copy the selected state covariance to the initial state covariance
        # (it will be overwritten with the appropriate matrix)
        blas.zcopy(
            &k_states2, &self.selected_state_cov[0, 0, 0], &inc,
            &self.initial_state_cov[0, 0], &inc)

        # Solve the discrete Lyapunov equation to the get initial state
        # covariance matrix
        tools._zsolve_discrete_lyapunov(
            &self.tmp[0, 0],
            &self.initial_state_cov[0, 0],
            self.k_states, complex_step)

        self.initialized = True

    def __iter__(self):
        return self

    def __next__(self):
        """
        Advance to the next location
        """
        if self.t >= self.nobs:
            raise StopIteration
        else:
            self.seek(self.t + 1, 0, 0)

    cpdef seek(self, unsigned int t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse):
        cdef int previous_t = self.t

        # Set the global time indicator, if valid
        if t >= self.nobs:
            raise IndexError("Observation index out of range")
        self.t = t

        # Indices for possibly time-varying arrays
        cdef:
            int k_endog
            int design_t = 0
            int obs_intercept_t = 0
            int obs_cov_t = 0
            int transition_t = 0
            int state_intercept_t = 0
            int selection_t = 0
            int state_cov_t = 0

        # Get indices for possibly time-varying arrays
        if not self.time_invariant:
            if self.design.shape[2] > 1:
                design_t = t
            if self.obs_intercept.shape[1] > 1:
                obs_intercept_t = t
            if self.obs_cov.shape[2] > 1:
                obs_cov_t = t
            if self.transition.shape[2] > 1:
                transition_t = t
            if self.state_intercept.shape[1] > 1:
                state_intercept_t = t
            if self.selection.shape[2] > 1:
                selection_t = t
            if self.state_cov.shape[2] > 1:
                state_cov_t = t

        # Initialize object-level pointers to statespace arrays
        self._obs = &self.obs[0, t]
        self._design = &self.design[0, 0, design_t]
        self._obs_intercept = &self.obs_intercept[0, obs_intercept_t]
        self._obs_cov = &self.obs_cov[0, 0, obs_cov_t]
        self._transition = &self.transition[0, 0, transition_t]
        self._state_intercept = &self.state_intercept[0, state_intercept_t]
        self._selection = &self.selection[0, 0, selection_t]
        self._state_cov = &self.state_cov[0, 0, state_cov_t]

        # Initialize object-level pointers to initialization
        if not self.initialized:
            raise RuntimeError("Statespace model not initialized.")
        self._initial_state = &self.initial_state[0]
        self._initial_state_cov = &self.initial_state_cov[0,0]

        # Create the selected state covariance matrix
        self.select_state_cov(t)

        # Handle missing data
        # Note: this modifies object pointers and _* dimensions
        k_endog = self.select_missing(t)

        # Set dimensions
        self.set_dimensions(k_endog, self.k_states, self.k_posdef)

        # Handle transformations
        self.transform(t, previous_t, transform_diagonalize, transform_generalized_collapse)

    @cython.cfunc
    @cython.returns(cython.void)
    def set_dimensions(self, unsigned int k_endog, unsigned int k_states, unsigned int k_posdef):
        self._k_endog = k_endog
        self._k_states = k_states
        self._k_posdef = k_posdef
        self._k_endog2 = k_endog**2
        self._k_states2 = k_states**2
        self._k_posdef2 = k_posdef**2
        self._k_endogstates = k_endog * k_states
        self._k_statesposdef = k_states * k_posdef

    @cython.cfunc
    @cython.returns(cython.void)
    def select_state_cov(self, unsigned int t):
        cdef int selected_state_cov_t = 0

        # ### Get selected state covariance matrix
        if t == 0 or self.selected_state_cov.shape[2] > 1:
            selected_state_cov_t = t
            self._selected_state_cov = &self.selected_state_cov[0, 0, selected_state_cov_t]

            zselect_cov(
                self.k_states, self.k_posdef,
                &self.tmp[0, 0],
                self._selection,
                self._state_cov,
                self._selected_state_cov)
        else:
            self._selected_state_cov = &self.selected_state_cov[0, 0, 0]

    @cython.cfunc
    @cython.returns(int)
    def select_missing(self, unsigned int t):
        # Note: this assumes that object pointers are already initialized
        # Note: this assumes that transform_... will be done *later*
        cdef int k_endog = self.k_endog

        # Set the current iteration nmissing
        self._nmissing = self.nmissing[t]

        # ### Perform missing selections
        # In Durbin and Koopman (2012), these are represented as matrix
        # multiplications, i.e. $Z_t^* = W_t Z_t$ where $W_t$ is a row
        # selection matrix (it contains a subset of rows of the identity
        # matrix).
        #
        # It's more efficient, though, to just copy over the data directly,
        # which is what is done here. Note that the `selected_*` arrays are
        # defined as single-dimensional, so the assignment indexes below are
        # set such that the arrays can be interpreted by the BLAS and LAPACK
        # functions as two-dimensional, column-major arrays.
        #
        # In the case that all data is missing (e.g. this is what happens in
        # forecasting), we actually set don't change the dimension, but we set
        # the design matrix to the zeros array.
        if self._nmissing == self.k_endog:
            self._select_missing_entire_obs(t)
        elif self._nmissing > 0:
            self._select_missing_partial_obs(t)
            k_endog = self.k_endog - self._nmissing

        # Return the number of non-missing endogenous variables
        return k_endog

    @cython.cfunc
    @cython.returns(cython.void)
    def _select_missing_entire_obs(self, unsigned int t):
        cdef:
            int i, j

        # Design matrix is set to zeros
        for i in range(self.k_states):
            for j in range(self.k_endog):
                self.selected_design[j + i * self.k_endog] = 0.0
        self._design = &self.selected_design[0]

    @cython.cfunc
    @cython.returns(cython.void)
    def _select_missing_partial_obs(self, unsigned int t):
        cdef:
            int i, j, k, l
            int inc = 1
            int design_t = 0
            int obs_cov_t = 0
            int k_endog = self.k_endog - self._nmissing

        k = 0
        for i in range(self.k_endog):
            if not self.missing[i, t]:
                self.selected_obs[k] = self._obs[i]
                self.selected_obs_intercept[k] = self._obs_intercept[i]

                # i is rows, k is rows
                blas.zcopy(
                    &self.k_states, &self._design[i], &self.k_endog,
                    &self.selected_design[k], &k_endog)

                # i, k is columns, j, l is rows
                l = 0
                for j in range(self.k_endog):
                    if not self.missing[j, t]:
                        self.selected_obs_cov[l + k * k_endog] = self._obs_cov[j + i * self.k_endog]
                        l += 1
                k += 1
        self._obs = &self.selected_obs[0]
        self._obs_intercept = &self.selected_obs_intercept[0]
        self._design = &self.selected_design[0]
        self._obs_cov = &self.selected_obs_cov[0]

    cdef void transform(self, unsigned int t, unsigned int previous_t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse) except *:
        # Reset the collapsed loglikelihood
        self.collapse_loglikelihood = 0

        if transform_generalized_collapse and not self._k_endog <= self._k_states:
            k_endog = self.transform_generalized_collapse(t, previous_t)
            # Reset dimensions
            self.set_dimensions(k_endog, self._k_states, self._k_posdef)
        elif transform_diagonalize and not self.diagonal_obs_cov:
            self.transform_diagonalize(t, previous_t)

    cdef void transform_diagonalize(self, unsigned int t, unsigned int previous_t) except *:
        # Note: this assumes that initialize_object_pointers has *already* been done
        # Note: this assumes that select_missing has *already* been done
        # TODO need unit tests, especially for the missing case
        # TODO need to also transform observation intercept
        cdef:
            int i, j, inc=1
            int obs_cov_t, design_t
            int info
            int reset_missing
            cnp.complex128_t * _transform_obs_cov = &self.transform_obs_cov[0, 0]
            cnp.complex128_t * _transform_cholesky = &self.transform_cholesky[0, 0]

        # Compute the cholesky decomposition of *self._obs_cov
        if self.diagonal_obs_cov:
            return

        # Handle missing data
        if self.nmissing[t] == self.k_endog:
            return

        reset_missing = 0
        for i in range(self.k_endog):
            reset_missing = reset_missing + (not self.missing[i, t] == self.missing[i, previous_t])

        # Perform the LDL decomposition, if necessary
        if t == 0 or self.obs_cov.shape[2] > 1 or reset_missing:
            # Make sure we don't have an observation intercept
            if not np.sum(self.obs_intercept) == 0 or self.obs_intercept.shape[2] > 1:
                raise RuntimeError('The univariate method with non-diagonal '
                                   'observation covariance matrix does not '
                                   'currently support an observation intercept.')

            # LDL decomposition
            blas.zcopy(
                &self._k_endog2, self._obs_cov, &inc,
                _transform_cholesky, &inc)
            info = tools._zldl(_transform_cholesky, self._k_endog)

            # Check for errors
            if info > 0:
                warnings.warn("Positive semi-definite observation covariance matrix encountered at period %d" % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in LDL factorization of observation covariance matrix encountered at period %d' % t)

            # Currently both L and D are stored in transform_cholesky
            for i in range(self._k_endog): # i is rows
                for j in range(self._k_endog): # j is columns
                    # Diagonal elements come from the diagonal
                    if i == j:
                        _transform_obs_cov[i + i * self._k_endog] = _transform_cholesky[i + i * self._k_endog]
                    # Other elements are zero
                    else:
                        _transform_obs_cov[i + j * self._k_endog] = 0

                    # Zero out the upper triangle of the cholesky
                    if j > i:
                        _transform_cholesky[i + j * self._k_endog] = 0

                # Convert from L to C simply by setting the diagonal elements to ones
                _transform_cholesky[i + i * self._k_endog] = 1
        
        # Solve for y_t^*
        # (unless this is a completely missing observation)
        if not self._nmissing == self.k_endog:
            # If we have some missing elements, selected_obs is already populated
            if self._nmissing == 0:
                blas.zcopy(
                    &self._k_endog, &self.obs[0, t], &inc,
                    &self.selected_obs[0], &inc)
            lapack.ztrtrs(
                "L", "N", "U", &self._k_endog, &inc,
                _transform_cholesky, &self.k_endog,
                &self.selected_obs[0], &self._k_endog, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Singular factorization of observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in factorization of observation covariance matrix encountered at period %d' % t)

            # Setup the pointer
            self._obs = &self.selected_obs[0]

        # Solve for Z_t^*, if necessary
        if t == 0 or self.design.shape[2] > 1 or reset_missing:
            blas.zcopy(
                &self._k_endogstates, self._design, &inc,
                &self.transform_design[0, 0], &inc)
            lapack.ztrtrs(
                "L", "N", "U", &self._k_endog, &self._k_states,
                _transform_cholesky, &self.k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Singular factorization of observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in factorization of observation covariance matrix encountered at period %d' % t)

        # Setup final pointers            
        self._design = &self.transform_design[0, 0]
        self._obs_cov = &self.transform_obs_cov[0, 0]

    cdef int transform_generalized_collapse(self, unsigned int t, unsigned int previous_t) except *:
        # Note: this assumes that initialize_object_pointers has *already* been done
        # Note: this assumes that select_missing has *already* been done
        # TODO need unit tests, especially for the missing case
        cdef:
            int i, j, inc=1
            int obs_cov_t, design_t
            int info
            int reset_missing
            cnp.complex128_t alpha = 1.0
            cnp.complex128_t beta = 0.0
            cnp.complex128_t gamma = -1.0
            int k_states = self._k_states
            int k_states2 = self._k_states2
            int k_endogstates = self._k_endogstates

        # $y_t^* = \bar A^* y_t = C_t Z_t' H_t^{-1} y_t$  
        # $Z_t^* = C_t^{-1}$  
        # $H_t^* = I_m$  

        # Make sure we have enough observations to perform collapse
        if self.k_endog < self.k_states:
            raise RuntimeError('Cannot collapse observation vector it the'
                               ' state dimension is larger than the dimension'
                               ' of the observation vector.')

        # Adjust for a VAR transition (i.e. design = [#, 0], where the zeros
        # correspond to all states except the first k_posdef states)
        if self.subset_design:
            k_states = self._k_posdef
            k_states2 = self._k_posdef2
            k_endogstates = self._k_endog * self._k_posdef

        # Handle missing data
        if self.nmissing[t] == self.k_endog:
            return self.k_states
        reset_missing = 0
        for i in range(self.k_endog):
            reset_missing = reset_missing + (not self.missing[i, t] == self.missing[i, previous_t])

        # Initialize the transformation
        if self.collapse_obs_cov[0, 0] == 0:
            # Set H_t^* to identity
            for i in range(k_states):
                self.collapse_obs_cov[i, i] = 1

            # Make sure we don't have an observation intercept
            if not np.sum(self.obs_intercept) == 0 or self.obs_intercept.shape[2] > 1:
                raise RuntimeError('The observation collapse transformation'
                                   ' does not currently support an observation'
                                   ' intercept.')

        # Perform the Cholesky decomposition of H_t, if necessary
        if t == 0 or self.obs_cov.shape[2] > 1 or reset_missing:
            # Cholesky decomposition: $H = L L'$  
            blas.zcopy(
                &self._k_endog2, self._obs_cov, &inc,
                &self.transform_cholesky[0, 0], &inc)
            # Use LDA=self.k_endog so that we can use the memoryview slicing below
            lapack.zpotrf(
                "L", &self._k_endog, &self.transform_cholesky[0, 0], &self.k_endog, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Non-positive-definite observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in observation covariance matrix encountered at period %d' % t)

            # Calculate the determinant (just the squared product of the
            # diagonals, in the Cholesky decomposition case)
            self.transform_determinant = 1.0
            for i in range(self._k_endog):
                if not self.transform_cholesky[i, i] == 0:
                    self.transform_determinant = self.transform_determinant * self.transform_cholesky[i, i]
            self.transform_determinant = self.transform_determinant**2

        # Get $Z_t \equiv C^{-1}$, if necessary  
        if t == 0 or self.obs_cov.shape[2] > 1 or self.design.shape[2] > 1 or reset_missing:
            # Calculate $H_t^{-1} Z_t \equiv (Z_t' H_t^{-1})'$ via Cholesky solver
            blas.zcopy(
                &self._k_endogstates, self._design, &inc,
                &self.transform_design[0, 0], &inc)
            lapack.zpotrs(
                "L", &self._k_endog, &k_states,
                &self.transform_cholesky[0, 0], &self.k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &info)

            # Check for errors
            if not info == 0:
                raise np.linalg.LinAlgError('Invalid value in calculation of H_t^{-1}Z matrix encountered at period %d' % t)
        
            # Calculate $(H_t^{-1} Z_t)' Z_t$  
            # $(m \times m) = (m \times p) (p \times p) (p \times m)$
            blas.zgemm(
                "T", "N", &k_states, &k_states, &self._k_endog,
                &alpha, self._design, &self._k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &beta, &self.collapse_cholesky[0, 0], &self._k_states)

            # Calculate $(Z_t' H_t^{-1} Z_t)^{-1}$ via Cholesky inversion  
            lapack.zpotrf("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)
            lapack.zpotri("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)

            # Calculate $C_t$ (the upper triangular cholesky decomposition of $(Z_t' H_t^{-1} Z_t)^{-1}$)  
            lapack.zpotrf("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Non-positive-definite ZHZ matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in ZHZ matrix encountered at period %d' % t)

            # Calculate $C_t'^{-1} \equiv Z_t$  
            # Do so by solving the system: $C_t' x = I$  
            # (Recall that collapse_obs_cov is an identity matrix)
            blas.zcopy(
                &self._k_states2, &self.collapse_obs_cov[0, 0], &inc,
                &self.collapse_design[0, 0], &inc)
            lapack.ztrtrs(
                "U", "T", "N", &k_states, &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_design[0, 0], &self._k_states,
                &info)

        # Calculate $\bar y_t^* = \bar A_t^* y_t = C_t Z_t' H_t^{-1} y_t$  
        # (unless this is a completely missing observation)
        self.collapse_loglikelihood = 0
        if not self._nmissing == self.k_endog:
            # If we have some missing elements, selected_obs is already populated
            if self._nmissing == 0:
                blas.zcopy(
                    &self.k_endog, &self.obs[0, t], &inc,
                    &self.selected_obs[0], &inc)
            # $\\# = Z_t' H_t^{-1} y_t$
            blas.zgemv(
                "T", &self._k_endog, &k_states,
                &alpha, &self.transform_design[0, 0], &self._k_endog,
                &self.selected_obs[0], &inc,
                &beta, &self.collapse_obs[0], &inc)
            # $y_t^* = C_t \\#$  
            blas.ztrmv(
                "U", "N", "N", &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_obs[0], &inc)

            # Get residuals for loglikelihood calculation
            # Note: Durbin and Koopman (2012) appears to have an error in the
            # formula here. They have $e_t = y_t - Z_t \bar y_t^*$, whereas it
            # should be: $e_t = y_t - Z_t C_t' \bar y_t^*$
            # See Jungbacker and Koopman (2014), section 2.5 where $e_t$ is
            # defined. In this case, $Z_t^dagger = Z_t C_t$ where
            # $C_t C_t' = (Z_t' \Sigma_\varepsilon^{-1} Z_t)^{-1}$.
            # 

            # $ \\# = C_t' y_t^*$
            blas.zcopy(
                &k_states, &self.collapse_obs[0], &inc,
                &self.collapse_obs_tmp[0], &inc)
            blas.ztrmv(
                "U", "T", "N", &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_obs_tmp[0], &inc)

            # $e_t = - Z_t C_t' y_t^* + y_t$
            blas.zgemv(
                "N", &self._k_endog, &k_states,
                &gamma, self._design, &self._k_endog,
                &self.collapse_obs_tmp[0], &inc,
                &alpha, &self.selected_obs[0], &inc)

            # Calculate e_t' H_t^{-1} e_t via Cholesky solver  
            # $H_t^{-1} = (L L')^{-1} = L^{-1}' L^{-1}$  
            # So we want $e_t' L^{-1}' L^{-1} e_t = (L^{-1} e_t)' L^{-1} e_t$  
            # We have $L$ in `transform_cholesky`, so we want to do a linear  
            # solve of $L x = e_t$  where L is lower triangular
            lapack.ztrtrs(
                "L", "N", "N", &self._k_endog, &inc,
                &self.transform_cholesky[0, 0], &self.k_endog,
                &self.selected_obs[0], &self._k_endog,
                &info)

            # Calculate loglikelihood contribution of this observation

            # $e_t' H_t^{-1} e_t = (L^{-1} e_t)' L^{-1} e_t = \sum_i e_{i,t}**2$  
            self.collapse_loglikelihood = 0
            for i in range(self._k_endog):
                self.collapse_loglikelihood = self.collapse_loglikelihood + self.selected_obs[i]**2
            
            # (p-m) log( 2*pi) + log( |H_t| )
            self.collapse_loglikelihood = (
                self.collapse_loglikelihood +
                (self._k_endog - k_states)*zlog(2*NPY_PI) + 
                zlog(self.transform_determinant)
            )

            # -0.5 * ...
            self.collapse_loglikelihood = -0.5 * self.collapse_loglikelihood

        # Set pointers
        self._obs = &self.collapse_obs[0]
        self._design = &self.collapse_design[0, 0]
        self._obs_cov = &self.collapse_obs_cov[0, 0]

        # TODO can I replace this with k_states? I think I should be able to
        return self._k_states


# ### Selected covariance matrice
cdef int zselect_cov(int k, int k_posdef,
                         cnp.complex128_t* tmp,
                         cnp.complex128_t* selection,
                         cnp.complex128_t* cov,
                         cnp.complex128_t* selected_cov):
    cdef:
        cnp.complex128_t alpha = 1.0
        cnp.complex128_t beta = 0.0

    # Only need to do something if there is a covariance matrix
    # (i.e k_posdof == 0)
    if k_posdef > 0:

        # #### Calculate selected state covariance matrix  
        # $Q_t^* = R_t Q_t R_t'$
        # 
        # Combine a selection matrix and a covariance matrix to get
        # a simplified (but possibly singular) "selected" covariance
        # matrix (see e.g. Durbin and Koopman p. 43)

        # `tmp0` array used here, dimension $(m \times r)$  

        # TODO this does not require two ?gemm calls, since we know that it
        # is just selection rows and columns of the Q matrix

        # $\\#_0 = 1.0 * R_t Q_t$  
        # $(m \times r) = (m \times r) (r \times r)$
        blas.zgemm(
            "N", "N", &k, &k_posdef, &k_posdef,
            &alpha, selection, &k,
            cov, &k_posdef,
            &beta, tmp, &k)
        # $Q_t^* = 1.0 * \\#_0 R_t'$  
        # $(m \times m) = (m \times r) (m \times r)'$
        blas.zgemm(
            "N", "T", &k, &k, &k_posdef,
            &alpha, tmp, &k,
            selection, &k,
            &beta, selected_cov, &k)

## State Space Representation
cdef class sStatespace(object):
    """
    sStatespace(obs, design, obs_intercept, obs_cov, transition, state_intercept, selection, state_cov)

    *See Durbin and Koopman (2012), Chapter 4 for all notation*
    """

    # ### State space representation
    # 
    # $$
    # \begin{align}
    # y_t & = Z_t \alpha_t + d_t + \varepsilon_t \hspace{3em} & \varepsilon_t & \sim N(0, H_t) \\\\
    # \alpha_{t+1} & = T_t \alpha_t + c_t + R_t \eta_t & \eta_t & \sim N(0, Q_t) \\\\
    # & & \alpha_1 & \sim N(a_1, P_1)
    # \end{align}
    # $$
    # 
    # $y_t$ is $p \times 1$  
    # $\varepsilon_t$ is $p \times 1$  
    # $\alpha_t$ is $m \times 1$  
    # $\eta_t$ is $r \times 1$  
    # $t = 1, \dots, T$

    # `nobs` $\equiv T$ is the length of the time-series  
    # `k_endog` $\equiv p$ is dimension of observation space  
    # `k_states` $\equiv m$ is the dimension of the state space  
    # `k_posdef` $\equiv r$ is the dimension of the state shocks  
    # *Old notation: T, n, k, g*
    # cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # `obs` $\equiv y_t$ is the **observation vector** $(p \times T)$  
    # `design` $\equiv Z_t$ is the **design vector** $(p \times m \times T)$  
    # `obs_intercept` $\equiv d_t$ is the **observation intercept** $(p \times T)$  
    # `obs_cov` $\equiv H_t$ is the **observation covariance matrix** $(p \times p \times T)$  
    # `transition` $\equiv T_t$ is the **transition matrix** $(m \times m \times T)$  
    # `state_intercept` $\equiv c_t$ is the **state intercept** $(m \times T)$  
    # `selection` $\equiv R_t$ is the **selection matrix** $(m \times r \times T)$  
    # `state_cov` $\equiv Q_t$ is the **state covariance matrix** $(r \times r \times T)$  
    # `selected_state_cov` $\equiv R Q_t R'$ is the **selected state covariance matrix** $(m \times m \times T)$  
    # `initial_state` $\equiv a_1$ is the **initial state mean** $(m \times 1)$  
    # `initial_state_cov` $\equiv P_1$ is the **initial state covariance matrix** $(m \times m)$
    #
    # With the exception of `obs`, these are *optionally* time-varying. If they are instead time-invariant,
    # then the dimension of length $T$ is instead of length $1$.
    #
    # *Note*: the initial vectors' notation 1-indexed as in Durbin and Koopman,
    # but in the recursions below it will be 0-indexed in the Python arrays.
    # 
    # *Old notation: y, -, mu, beta_tt_init, P_tt_init*
    # cdef readonly cnp.float32_t [::1,:] obs, obs_intercept, state_intercept
    # cdef readonly cnp.float32_t [:] initial_state
    # cdef readonly cnp.float32_t [::1,:] initial_state_cov
    # *Old notation: H, R, F, G, Q*, G Q* G'*
    # cdef readonly cnp.float32_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    # `missing` is a $(p \times T)$ boolean matrix where a row is a $(p \times 1)$ vector
    # in which the $i$th position is $1$ if $y_{i,t}$ is to be considered a missing value.  
    # *Note:* This is created as the output of np.isnan(obs).
    # cdef readonly int [::1,:] missing
    # `nmissing` is an `T \times 0` integer vector holding the number of *missing* observations
    # $p - p_t$
    # cdef readonly int [:] nmissing

    # Flag for a time-invariant model, which requires that *all* of the
    # possibly time-varying arrays are time-invariant.
    # cdef readonly int time_invariant

    # Flag for initialization.
    # cdef readonly int initialized

    # Flags for performance improvements
    # TODO need to add this to the UI in representation
    # cdef public int diagonal_obs_cov
    # cdef public int subset_design
    # cdef public int companion_transition

    # Temporary arrays
    # cdef cnp.float32_t [::1,:] tmp

    # Temporary selection arrays
    # cdef readonly cnp.float32_t [:] selected_obs
    # The following are contiguous memory segments which are then used to
    # store the data in the above matrices.
    # cdef readonly cnp.float32_t [:] selected_design
    # cdef readonly cnp.float32_t [:] selected_obs_cov

    # Temporary transformation arrays
    # cdef readonly cnp.float32_t [::1,:] transform_cholesky
    # cdef readonly cnp.float32_t [::1,:] transform_obs_cov
    # cdef readonly cnp.float32_t [::1,:] transform_design
    # cdef readonly cnp.float32_t transform_determinant

    # cdef readonly cnp.float32_t [:] collapse_obs
    # cdef readonly cnp.float32_t [:] collapse_obs_tmp
    # cdef readonly cnp.float32_t [::1,:] collapse_design
    # cdef readonly cnp.float32_t [::1,:] collapse_obs_cov
    # cdef readonly cnp.float32_t [::1,:] collapse_cholesky
    # cdef readonly cnp.float32_t collapse_loglikelihood

    # Pointers  
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

    # Current location dimensions
    # cdef int _k_endog, _k_states, _k_posdef, _k_endog2, _k_states2, _k_posdef2, _k_endogstates, _k_statesposdef
    # cdef int _nmissing

    # ### Initialize state space model
    # *Note*: The initial state and state covariance matrix must be provided.
    def __init__(self,
                 cnp.float32_t [::1, :]   obs,
                 cnp.float32_t [::1, :, :] design,
                 cnp.float32_t [::1, :]   obs_intercept,
                 cnp.float32_t [::1, :, :] obs_cov,
                 cnp.float32_t [::1, :, :] transition,
                 cnp.float32_t [::1, :]   state_intercept,
                 cnp.float32_t [::1, :, :] selection,
                 cnp.float32_t [::1, :, :] state_cov,
                 diagonal_obs_cov=-1):

        # Local variables
        cdef:
            int t, i, j
            cnp.npy_intp dim1[1]
            cnp.npy_intp dim2[2]
            cnp.npy_intp dim3[3]

        # #### State space representation variables  
        # **Note**: these arrays share data with the versions defined in
        # Python and passed to this constructor, so if they are updated in
        # Python they will also be updated here.
        self.obs = obs
        self.design = design
        self.obs_intercept = obs_intercept
        self.obs_cov = obs_cov
        self.transition = transition
        self.state_intercept = state_intercept
        self.selection = selection
        self.state_cov = state_cov

        # Dimensions
        self.k_endog = obs.shape[0]
        self.k_states = selection.shape[0]
        self.k_posdef = selection.shape[1]
        self.nobs = obs.shape[1]

        # #### Validate matrix dimensions
        #
        # Make sure that the given state-space matrices have consistent sizes
        tools.validate_matrix_shape('design', &self.design.shape[0],
                                    self.k_endog, self.k_states, self.nobs)
        tools.validate_vector_shape('observation intercept', &self.obs_intercept.shape[0],
                                    self.k_endog, self.nobs)
        tools.validate_matrix_shape('observation covariance matrix', &self.obs_cov.shape[0],
                                    self.k_endog, self.k_endog, self.nobs)
        tools.validate_matrix_shape('transition', &self.transition.shape[0],
                                    self.k_states, self.k_states, self.nobs)
        tools.validate_vector_shape('state intercept', &self.state_intercept.shape[0],
                                    self.k_states, self.nobs)
        tools.validate_matrix_shape('state covariance matrix', &self.state_cov.shape[0],
                                    self.k_posdef, self.k_posdef, self.nobs)

        # Check for a time-invariant model
        self.time_invariant = (
            self.design.shape[2] == 1           and
            self.obs_intercept.shape[1] == 1    and
            self.obs_cov.shape[2] == 1          and
            self.transition.shape[2] == 1       and
            self.state_intercept.shape[1] == 1  and
            self.selection.shape[2] == 1        and
            self.state_cov.shape[2] == 1
        )

        # Set the flag for initialization to be false
        self.initialized = False
        self.initial_state = None
        self.initial_state_cov = None

        # Unless it is specified, check for a diagonal covariance matrix
        if diagonal_obs_cov == -1:
            diagonal_obs_cov = True
            for t in range(self.obs_cov.shape[2]):
                for i in range(self.k_endog):
                    for j in range(self.k_endog):
                        if i == j:
                            continue
                        if not (dabs(self.obs_cov[i, j, t]) < 1e-9):
                            diagonal_obs_cov = False
                            break
        self.diagonal_obs_cov = diagonal_obs_cov

        # Allocate selected state covariance matrix
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = 1
        # (we only allocate memory for time-varying array if necessary)
        if self.state_cov.shape[2] > 1 or self.selection.shape[2] > 1:
            dim3[2] = self.nobs
        self.selected_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_FLOAT32, FORTRAN)

        # Handle missing data
        self.missing = np.array(np.isnan(obs), dtype=np.int32, order="F")
        self.nmissing = np.array(np.sum(self.missing, axis=0), dtype=np.int32)
        self.has_missing = np.sum(self.nmissing) > 0

        # Create the temporary array
        # Holds arrays of dimension $(m \times m)$
        dim2[0] = self.k_states; dim2[1] = max(self.k_states, self.k_posdef)
        self.tmp = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)

        # Arrays for missing data
        dim1[0] = self.k_endog;
        self.selected_obs = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT32, FORTRAN)
        dim1[0] = self.k_endog;
        self.selected_obs_intercept = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT32, FORTRAN)
        dim1[0] = self.k_endog * self.k_states;
        self.selected_design = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT32, FORTRAN)
        dim1[0] = self.k_endog**2;
        self.selected_obs_cov = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT32, FORTRAN)

        # Arrays for transformations
        dim2[0] = self.k_endog; dim2[1] = self.k_endog
        self.transform_cholesky = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self.transform_obs_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        dim2[0] = self.k_endog; dim2[1] = self.k_states
        self.transform_design = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)

        dim1[0] = self.k_states
        self.collapse_obs = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT32, FORTRAN)
        self.collapse_obs_tmp = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT32, FORTRAN)
        dim2[0] = self.k_states; dim2[1] = self.k_states
        self.collapse_design = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self.collapse_obs_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)
        self.collapse_cholesky = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)

        # Initialize location
        self.t = 0

        # Initialize dimensions
        self.set_dimensions(self.k_endog, self.k_states, self.k_posdef)

    def __reduce__(self):
        init = (np.array(self.obs, copy=True, order='F'), np.array(self.design, copy=True, order='F'),
                np.array(self.obs_intercept, copy=True, order='F'), np.array(self.obs_cov, copy=True, order='F'),
                np.array(self.transition, copy=True, order='F'), np.array(self.state_intercept, copy=True, order='F'),
                np.array(self.selection, copy=True, order='F'), np.array(self.state_cov, copy=True, order='F'),
                self.diagonal_obs_cov)
        state = {'initialized': self.initialized,
                 'initial_state': None,
                 'initial_state_cov': None,
                 'missing': np.array(self.missing, copy=True, order='F'),
                 'nmissing': np.array(self.nmissing, copy=True, order='F'),
                 'has_missing': self.has_missing,
                 'tmp': np.array(self.tmp, copy=True, order='F'),
                 'selected_state_cov': np.array(self.selected_state_cov, copy=True, order='F'),
                 'selected_obs': np.array(self.selected_obs, copy=True, order='F'),
                 'selected_obs_intercept': np.array(self.selected_obs_intercept, copy=True, order='F'),
                 'selected_design': np.array(self.selected_design, copy=True, order='F'),
                 'selected_obs_cov': np.array(self.selected_obs_cov, copy=True, order='F'),
                 'transform_cholesky': np.array(self.transform_cholesky, copy=True, order='F'),
                 'transform_obs_cov': np.array(self.transform_obs_cov, copy=True, order='F'),
                 'transform_design': np.array(self.transform_design, copy=True, order='F'),
                 'collapse_obs': np.array(self.collapse_obs, copy=True, order='F'),
                 'collapse_obs_tmp': np.array(self.collapse_obs_tmp, copy=True, order='F'),
                 'collapse_design': np.array(self.collapse_design, copy=True, order='F'),
                 'collapse_obs_cov': np.array(self.collapse_obs_cov, copy=True, order='F'),
                 'collapse_cholesky': np.array(self.collapse_cholesky, copy=True, order='F'),
                 't': self.t,
                 'collapse_loglikelihood': self.collapse_loglikelihood,
                 'companion_transition': self.companion_transition,
                 'transform_determinant': self.transform_determinant,
                 }
        if self.initialized:
            state['initial_state'] = np.array(self.initial_state, copy=True, order='F')
            state['initial_state_cov'] = np.array(self.initial_state_cov, copy=True, order='F')
        return (self.__class__, init, state)

    def __setstate__(self, state):
        self.initial_state = state['initial_state']
        self.initial_state_cov = state['initial_state_cov']
        self.initialized = state['initialized']
        self.selected_state_cov = state['selected_state_cov']
        self.missing = state['missing']
        self.nmissing =state['nmissing']
        self.has_missing = state['has_missing']
        self.tmp = state['tmp']
        self.selected_obs  = state['selected_obs']
        self.selected_obs_intercept  = state['selected_obs_intercept']
        self.selected_design  = state['selected_design']
        self.selected_obs_cov  =state['selected_obs_cov']
        self.transform_cholesky  = state['transform_cholesky']
        self.transform_obs_cov  = state['transform_obs_cov']
        self.transform_design = state['transform_design']
        self.collapse_obs = state['collapse_obs']
        self.collapse_obs_tmp = state['collapse_obs_tmp']
        self.collapse_design = state['collapse_design']
        self.collapse_obs_cov = state['collapse_obs_cov']
        self.collapse_cholesky = state['collapse_cholesky']
        self.t = state['t']
        self.collapse_loglikelihood = state['collapse_loglikelihood']
        self.companion_transition = state['companion_transition']
        self.transform_determinant = state['transform_determinant']

    # ## Initialize: known values
    #
    # Initialize the filter with specific values, assumed to be known with
    # certainty or else as filled with parameters from a maximum likelihood
    # estimation run.
    def initialize_known(self, cnp.float32_t [:] initial_state, cnp.float32_t [::1, :] initial_state_cov):
        """
        initialize_known(initial_state, initial_state_cov)
        """
        tools.validate_vector_shape('inital state', &initial_state.shape[0], self.k_states, None)
        tools.validate_matrix_shape('initial state covariance', &initial_state_cov.shape[0], self.k_states, self.k_states, None)

        self.initial_state = initial_state
        self.initial_state_cov = initial_state_cov

        self.initialized = True

    # ## Initialize: approximate diffuse priors
    #
    # Durbin and Koopman note that this initialization should only be coupled
    # with the standard Kalman filter for "approximate exploratory work" and
    # can lead to "large rounding errors" (p. 125).
    # 
    # *Note:* see Durbin and Koopman section 5.6.1
    def initialize_approximate_diffuse(self, cnp.float32_t variance=1e2):
        """
        initialize_approximate_diffuse(variance=1e2)
        """
        cdef cnp.npy_intp dim[1]
        dim[0] = self.k_states
        self.initial_state = cnp.PyArray_ZEROS(1, dim, cnp.NPY_FLOAT32, FORTRAN)
        self.initial_state_cov = np.eye(self.k_states, dtype=np.float32).T * variance

        self.initialized = True

    # ## Initialize: stationary process
    # *Note:* see Durbin and Koopman section 5.6.2
    def initialize_stationary(self, complex_step=False):
        """
        initialize_stationary()
        """
        cdef cnp.npy_intp dim1[1]
        cdef cnp.npy_intp dim2[2]
        cdef int i, info, inc = 1
        cdef int k_states2 = self.k_states**2
        cdef cnp.float64_t asum, tol = 1e-9
        cdef cnp.float32_t scalar
        cdef int [::1,:] ipiv

        # Create selected state covariance matrix
        sselect_cov(self.k_states, self.k_posdef,
                                   &self.tmp[0, 0],
                                   &self.selection[0, 0, 0],
                                   &self.state_cov[0, 0, 0],
                                   &self.selected_state_cov[0, 0, 0])

        # Initial state mean
        asum = blas.sasum(&self.k_states, &self.state_intercept[0, 0], &inc)

        dim1[0] = self.k_states
        self.initial_state = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_FLOAT32, FORTRAN)
        if asum > tol:
            dim2[0] = self.k_states
            dim2[1] = self.k_states
            ipiv = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_INT32, FORTRAN)

            # I - T
            blas.scopy(
                &k_states2, &self.transition[0, 0, 0], &inc,
                &self.tmp[0,0], &inc)
            scalar = -1.0
            blas.sscal(&k_states2, &scalar, &self.tmp[0, 0], &inc)
            for i in range(self.k_states):
                self.tmp[i, i] = self.tmp[i, i] + 1

            # c
            blas.scopy(
                &self.k_states, &self.state_intercept[0, 0], &inc,
                &self.initial_state[0], &inc)

            # Solve (I - T) x = c
            lapack.sgetrf(
                &self.k_states, &self.k_states, &self.tmp[0, 0], &self.k_states,
                &ipiv[0, 0], &info)
            lapack.sgetrs(
                'N', &self.k_states, &inc, &self.tmp[0, 0], &self.k_states,
                &ipiv[0, 0], &self.initial_state[0], &self.k_states, &info)

        dim2[0] = self.k_states; dim2[1] = self.k_states
        self.initial_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_FLOAT32, FORTRAN)

        # Create a copy of the transition matrix (to avoid overwriting it)
        blas.scopy(
            &k_states2, &self.transition[0, 0, 0], &inc,
            &self.tmp[0, 0], &inc)

        # Copy the selected state covariance to the initial state covariance
        # (it will be overwritten with the appropriate matrix)
        blas.scopy(
            &k_states2, &self.selected_state_cov[0, 0, 0], &inc,
            &self.initial_state_cov[0, 0], &inc)

        # Solve the discrete Lyapunov equation to the get initial state
        # covariance matrix
        tools._ssolve_discrete_lyapunov(
            &self.tmp[0, 0],
            &self.initial_state_cov[0, 0],
            self.k_states, complex_step)

        self.initialized = True

    def __iter__(self):
        return self

    def __next__(self):
        """
        Advance to the next location
        """
        if self.t >= self.nobs:
            raise StopIteration
        else:
            self.seek(self.t + 1, 0, 0)

    cpdef seek(self, unsigned int t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse):
        cdef int previous_t = self.t

        # Set the global time indicator, if valid
        if t >= self.nobs:
            raise IndexError("Observation index out of range")
        self.t = t

        # Indices for possibly time-varying arrays
        cdef:
            int k_endog
            int design_t = 0
            int obs_intercept_t = 0
            int obs_cov_t = 0
            int transition_t = 0
            int state_intercept_t = 0
            int selection_t = 0
            int state_cov_t = 0

        # Get indices for possibly time-varying arrays
        if not self.time_invariant:
            if self.design.shape[2] > 1:
                design_t = t
            if self.obs_intercept.shape[1] > 1:
                obs_intercept_t = t
            if self.obs_cov.shape[2] > 1:
                obs_cov_t = t
            if self.transition.shape[2] > 1:
                transition_t = t
            if self.state_intercept.shape[1] > 1:
                state_intercept_t = t
            if self.selection.shape[2] > 1:
                selection_t = t
            if self.state_cov.shape[2] > 1:
                state_cov_t = t

        # Initialize object-level pointers to statespace arrays
        self._obs = &self.obs[0, t]
        self._design = &self.design[0, 0, design_t]
        self._obs_intercept = &self.obs_intercept[0, obs_intercept_t]
        self._obs_cov = &self.obs_cov[0, 0, obs_cov_t]
        self._transition = &self.transition[0, 0, transition_t]
        self._state_intercept = &self.state_intercept[0, state_intercept_t]
        self._selection = &self.selection[0, 0, selection_t]
        self._state_cov = &self.state_cov[0, 0, state_cov_t]

        # Initialize object-level pointers to initialization
        if not self.initialized:
            raise RuntimeError("Statespace model not initialized.")
        self._initial_state = &self.initial_state[0]
        self._initial_state_cov = &self.initial_state_cov[0,0]

        # Create the selected state covariance matrix
        self.select_state_cov(t)

        # Handle missing data
        # Note: this modifies object pointers and _* dimensions
        k_endog = self.select_missing(t)

        # Set dimensions
        self.set_dimensions(k_endog, self.k_states, self.k_posdef)

        # Handle transformations
        self.transform(t, previous_t, transform_diagonalize, transform_generalized_collapse)

    @cython.cfunc
    @cython.returns(cython.void)
    def set_dimensions(self, unsigned int k_endog, unsigned int k_states, unsigned int k_posdef):
        self._k_endog = k_endog
        self._k_states = k_states
        self._k_posdef = k_posdef
        self._k_endog2 = k_endog**2
        self._k_states2 = k_states**2
        self._k_posdef2 = k_posdef**2
        self._k_endogstates = k_endog * k_states
        self._k_statesposdef = k_states * k_posdef

    @cython.cfunc
    @cython.returns(cython.void)
    def select_state_cov(self, unsigned int t):
        cdef int selected_state_cov_t = 0

        # ### Get selected state covariance matrix
        if t == 0 or self.selected_state_cov.shape[2] > 1:
            selected_state_cov_t = t
            self._selected_state_cov = &self.selected_state_cov[0, 0, selected_state_cov_t]

            sselect_cov(
                self.k_states, self.k_posdef,
                &self.tmp[0, 0],
                self._selection,
                self._state_cov,
                self._selected_state_cov)
        else:
            self._selected_state_cov = &self.selected_state_cov[0, 0, 0]

    @cython.cfunc
    @cython.returns(int)
    def select_missing(self, unsigned int t):
        # Note: this assumes that object pointers are already initialized
        # Note: this assumes that transform_... will be done *later*
        cdef int k_endog = self.k_endog

        # Set the current iteration nmissing
        self._nmissing = self.nmissing[t]

        # ### Perform missing selections
        # In Durbin and Koopman (2012), these are represented as matrix
        # multiplications, i.e. $Z_t^* = W_t Z_t$ where $W_t$ is a row
        # selection matrix (it contains a subset of rows of the identity
        # matrix).
        #
        # It's more efficient, though, to just copy over the data directly,
        # which is what is done here. Note that the `selected_*` arrays are
        # defined as single-dimensional, so the assignment indexes below are
        # set such that the arrays can be interpreted by the BLAS and LAPACK
        # functions as two-dimensional, column-major arrays.
        #
        # In the case that all data is missing (e.g. this is what happens in
        # forecasting), we actually set don't change the dimension, but we set
        # the design matrix to the zeros array.
        if self._nmissing == self.k_endog:
            self._select_missing_entire_obs(t)
        elif self._nmissing > 0:
            self._select_missing_partial_obs(t)
            k_endog = self.k_endog - self._nmissing

        # Return the number of non-missing endogenous variables
        return k_endog

    @cython.cfunc
    @cython.returns(cython.void)
    def _select_missing_entire_obs(self, unsigned int t):
        cdef:
            int i, j

        # Design matrix is set to zeros
        for i in range(self.k_states):
            for j in range(self.k_endog):
                self.selected_design[j + i * self.k_endog] = 0.0
        self._design = &self.selected_design[0]

    @cython.cfunc
    @cython.returns(cython.void)
    def _select_missing_partial_obs(self, unsigned int t):
        cdef:
            int i, j, k, l
            int inc = 1
            int design_t = 0
            int obs_cov_t = 0
            int k_endog = self.k_endog - self._nmissing

        k = 0
        for i in range(self.k_endog):
            if not self.missing[i, t]:
                self.selected_obs[k] = self._obs[i]
                self.selected_obs_intercept[k] = self._obs_intercept[i]

                # i is rows, k is rows
                blas.scopy(
                    &self.k_states, &self._design[i], &self.k_endog,
                    &self.selected_design[k], &k_endog)

                # i, k is columns, j, l is rows
                l = 0
                for j in range(self.k_endog):
                    if not self.missing[j, t]:
                        self.selected_obs_cov[l + k * k_endog] = self._obs_cov[j + i * self.k_endog]
                        l += 1
                k += 1
        self._obs = &self.selected_obs[0]
        self._obs_intercept = &self.selected_obs_intercept[0]
        self._design = &self.selected_design[0]
        self._obs_cov = &self.selected_obs_cov[0]

    cdef void transform(self, unsigned int t, unsigned int previous_t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse) except *:
        # Reset the collapsed loglikelihood
        self.collapse_loglikelihood = 0

        if transform_generalized_collapse and not self._k_endog <= self._k_states:
            k_endog = self.transform_generalized_collapse(t, previous_t)
            # Reset dimensions
            self.set_dimensions(k_endog, self._k_states, self._k_posdef)
        elif transform_diagonalize and not self.diagonal_obs_cov:
            self.transform_diagonalize(t, previous_t)

    cdef void transform_diagonalize(self, unsigned int t, unsigned int previous_t) except *:
        # Note: this assumes that initialize_object_pointers has *already* been done
        # Note: this assumes that select_missing has *already* been done
        # TODO need unit tests, especially for the missing case
        # TODO need to also transform observation intercept
        cdef:
            int i, j, inc=1
            int obs_cov_t, design_t
            int info
            int reset_missing
            cnp.float32_t * _transform_obs_cov = &self.transform_obs_cov[0, 0]
            cnp.float32_t * _transform_cholesky = &self.transform_cholesky[0, 0]

        # Compute the cholesky decomposition of *self._obs_cov
        if self.diagonal_obs_cov:
            return

        # Handle missing data
        if self.nmissing[t] == self.k_endog:
            return

        reset_missing = 0
        for i in range(self.k_endog):
            reset_missing = reset_missing + (not self.missing[i, t] == self.missing[i, previous_t])

        # Perform the LDL decomposition, if necessary
        if t == 0 or self.obs_cov.shape[2] > 1 or reset_missing:
            # Make sure we don't have an observation intercept
            if not np.sum(self.obs_intercept) == 0 or self.obs_intercept.shape[2] > 1:
                raise RuntimeError('The univariate method with non-diagonal '
                                   'observation covariance matrix does not '
                                   'currently support an observation intercept.')

            # LDL decomposition
            blas.scopy(
                &self._k_endog2, self._obs_cov, &inc,
                _transform_cholesky, &inc)
            info = tools._sldl(_transform_cholesky, self._k_endog)

            # Check for errors
            if info > 0:
                warnings.warn("Positive semi-definite observation covariance matrix encountered at period %d" % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in LDL factorization of observation covariance matrix encountered at period %d' % t)

            # Currently both L and D are stored in transform_cholesky
            for i in range(self._k_endog): # i is rows
                for j in range(self._k_endog): # j is columns
                    # Diagonal elements come from the diagonal
                    if i == j:
                        _transform_obs_cov[i + i * self._k_endog] = _transform_cholesky[i + i * self._k_endog]
                    # Other elements are zero
                    else:
                        _transform_obs_cov[i + j * self._k_endog] = 0

                    # Zero out the upper triangle of the cholesky
                    if j > i:
                        _transform_cholesky[i + j * self._k_endog] = 0

                # Convert from L to C simply by setting the diagonal elements to ones
                _transform_cholesky[i + i * self._k_endog] = 1
        
        # Solve for y_t^*
        # (unless this is a completely missing observation)
        if not self._nmissing == self.k_endog:
            # If we have some missing elements, selected_obs is already populated
            if self._nmissing == 0:
                blas.scopy(
                    &self._k_endog, &self.obs[0, t], &inc,
                    &self.selected_obs[0], &inc)
            lapack.strtrs(
                "L", "N", "U", &self._k_endog, &inc,
                _transform_cholesky, &self.k_endog,
                &self.selected_obs[0], &self._k_endog, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Singular factorization of observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in factorization of observation covariance matrix encountered at period %d' % t)

            # Setup the pointer
            self._obs = &self.selected_obs[0]

        # Solve for Z_t^*, if necessary
        if t == 0 or self.design.shape[2] > 1 or reset_missing:
            blas.scopy(
                &self._k_endogstates, self._design, &inc,
                &self.transform_design[0, 0], &inc)
            lapack.strtrs(
                "L", "N", "U", &self._k_endog, &self._k_states,
                _transform_cholesky, &self.k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Singular factorization of observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in factorization of observation covariance matrix encountered at period %d' % t)

        # Setup final pointers            
        self._design = &self.transform_design[0, 0]
        self._obs_cov = &self.transform_obs_cov[0, 0]

    cdef int transform_generalized_collapse(self, unsigned int t, unsigned int previous_t) except *:
        # Note: this assumes that initialize_object_pointers has *already* been done
        # Note: this assumes that select_missing has *already* been done
        # TODO need unit tests, especially for the missing case
        cdef:
            int i, j, inc=1
            int obs_cov_t, design_t
            int info
            int reset_missing
            cnp.float32_t alpha = 1.0
            cnp.float32_t beta = 0.0
            cnp.float32_t gamma = -1.0
            int k_states = self._k_states
            int k_states2 = self._k_states2
            int k_endogstates = self._k_endogstates

        # $y_t^* = \bar A^* y_t = C_t Z_t' H_t^{-1} y_t$  
        # $Z_t^* = C_t^{-1}$  
        # $H_t^* = I_m$  

        # Make sure we have enough observations to perform collapse
        if self.k_endog < self.k_states:
            raise RuntimeError('Cannot collapse observation vector it the'
                               ' state dimension is larger than the dimension'
                               ' of the observation vector.')

        # Adjust for a VAR transition (i.e. design = [#, 0], where the zeros
        # correspond to all states except the first k_posdef states)
        if self.subset_design:
            k_states = self._k_posdef
            k_states2 = self._k_posdef2
            k_endogstates = self._k_endog * self._k_posdef

        # Handle missing data
        if self.nmissing[t] == self.k_endog:
            return self.k_states
        reset_missing = 0
        for i in range(self.k_endog):
            reset_missing = reset_missing + (not self.missing[i, t] == self.missing[i, previous_t])

        # Initialize the transformation
        if self.collapse_obs_cov[0, 0] == 0:
            # Set H_t^* to identity
            for i in range(k_states):
                self.collapse_obs_cov[i, i] = 1

            # Make sure we don't have an observation intercept
            if not np.sum(self.obs_intercept) == 0 or self.obs_intercept.shape[2] > 1:
                raise RuntimeError('The observation collapse transformation'
                                   ' does not currently support an observation'
                                   ' intercept.')

        # Perform the Cholesky decomposition of H_t, if necessary
        if t == 0 or self.obs_cov.shape[2] > 1 or reset_missing:
            # Cholesky decomposition: $H = L L'$  
            blas.scopy(
                &self._k_endog2, self._obs_cov, &inc,
                &self.transform_cholesky[0, 0], &inc)
            # Use LDA=self.k_endog so that we can use the memoryview slicing below
            lapack.spotrf(
                "L", &self._k_endog, &self.transform_cholesky[0, 0], &self.k_endog, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Non-positive-definite observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in observation covariance matrix encountered at period %d' % t)

            # Calculate the determinant (just the squared product of the
            # diagonals, in the Cholesky decomposition case)
            self.transform_determinant = 1.0
            for i in range(self._k_endog):
                if not self.transform_cholesky[i, i] == 0:
                    self.transform_determinant = self.transform_determinant * self.transform_cholesky[i, i]
            self.transform_determinant = self.transform_determinant**2

        # Get $Z_t \equiv C^{-1}$, if necessary  
        if t == 0 or self.obs_cov.shape[2] > 1 or self.design.shape[2] > 1 or reset_missing:
            # Calculate $H_t^{-1} Z_t \equiv (Z_t' H_t^{-1})'$ via Cholesky solver
            blas.scopy(
                &self._k_endogstates, self._design, &inc,
                &self.transform_design[0, 0], &inc)
            lapack.spotrs(
                "L", &self._k_endog, &k_states,
                &self.transform_cholesky[0, 0], &self.k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &info)

            # Check for errors
            if not info == 0:
                raise np.linalg.LinAlgError('Invalid value in calculation of H_t^{-1}Z matrix encountered at period %d' % t)
        
            # Calculate $(H_t^{-1} Z_t)' Z_t$  
            # $(m \times m) = (m \times p) (p \times p) (p \times m)$
            blas.sgemm(
                "T", "N", &k_states, &k_states, &self._k_endog,
                &alpha, self._design, &self._k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &beta, &self.collapse_cholesky[0, 0], &self._k_states)

            # Calculate $(Z_t' H_t^{-1} Z_t)^{-1}$ via Cholesky inversion  
            lapack.spotrf("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)
            lapack.spotri("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)

            # Calculate $C_t$ (the upper triangular cholesky decomposition of $(Z_t' H_t^{-1} Z_t)^{-1}$)  
            lapack.spotrf("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Non-positive-definite ZHZ matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in ZHZ matrix encountered at period %d' % t)

            # Calculate $C_t'^{-1} \equiv Z_t$  
            # Do so by solving the system: $C_t' x = I$  
            # (Recall that collapse_obs_cov is an identity matrix)
            blas.scopy(
                &self._k_states2, &self.collapse_obs_cov[0, 0], &inc,
                &self.collapse_design[0, 0], &inc)
            lapack.strtrs(
                "U", "T", "N", &k_states, &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_design[0, 0], &self._k_states,
                &info)

        # Calculate $\bar y_t^* = \bar A_t^* y_t = C_t Z_t' H_t^{-1} y_t$  
        # (unless this is a completely missing observation)
        self.collapse_loglikelihood = 0
        if not self._nmissing == self.k_endog:
            # If we have some missing elements, selected_obs is already populated
            if self._nmissing == 0:
                blas.scopy(
                    &self.k_endog, &self.obs[0, t], &inc,
                    &self.selected_obs[0], &inc)
            # $\\# = Z_t' H_t^{-1} y_t$
            blas.sgemv(
                "T", &self._k_endog, &k_states,
                &alpha, &self.transform_design[0, 0], &self._k_endog,
                &self.selected_obs[0], &inc,
                &beta, &self.collapse_obs[0], &inc)
            # $y_t^* = C_t \\#$  
            blas.strmv(
                "U", "N", "N", &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_obs[0], &inc)

            # Get residuals for loglikelihood calculation
            # Note: Durbin and Koopman (2012) appears to have an error in the
            # formula here. They have $e_t = y_t - Z_t \bar y_t^*$, whereas it
            # should be: $e_t = y_t - Z_t C_t' \bar y_t^*$
            # See Jungbacker and Koopman (2014), section 2.5 where $e_t$ is
            # defined. In this case, $Z_t^dagger = Z_t C_t$ where
            # $C_t C_t' = (Z_t' \Sigma_\varepsilon^{-1} Z_t)^{-1}$.
            # 

            # $ \\# = C_t' y_t^*$
            blas.scopy(
                &k_states, &self.collapse_obs[0], &inc,
                &self.collapse_obs_tmp[0], &inc)
            blas.strmv(
                "U", "T", "N", &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_obs_tmp[0], &inc)

            # $e_t = - Z_t C_t' y_t^* + y_t$
            blas.sgemv(
                "N", &self._k_endog, &k_states,
                &gamma, self._design, &self._k_endog,
                &self.collapse_obs_tmp[0], &inc,
                &alpha, &self.selected_obs[0], &inc)

            # Calculate e_t' H_t^{-1} e_t via Cholesky solver  
            # $H_t^{-1} = (L L')^{-1} = L^{-1}' L^{-1}$  
            # So we want $e_t' L^{-1}' L^{-1} e_t = (L^{-1} e_t)' L^{-1} e_t$  
            # We have $L$ in `transform_cholesky`, so we want to do a linear  
            # solve of $L x = e_t$  where L is lower triangular
            lapack.strtrs(
                "L", "N", "N", &self._k_endog, &inc,
                &self.transform_cholesky[0, 0], &self.k_endog,
                &self.selected_obs[0], &self._k_endog,
                &info)

            # Calculate loglikelihood contribution of this observation

            # $e_t' H_t^{-1} e_t = (L^{-1} e_t)' L^{-1} e_t = \sum_i e_{i,t}**2$  
            self.collapse_loglikelihood = 0
            for i in range(self._k_endog):
                self.collapse_loglikelihood = self.collapse_loglikelihood + self.selected_obs[i]**2
            
            # (p-m) log( 2*pi) + log( |H_t| )
            self.collapse_loglikelihood = (
                self.collapse_loglikelihood +
                (self._k_endog - k_states)*dlog(2*NPY_PI) + 
                dlog(self.transform_determinant)
            )

            # -0.5 * ...
            self.collapse_loglikelihood = -0.5 * self.collapse_loglikelihood

        # Set pointers
        self._obs = &self.collapse_obs[0]
        self._design = &self.collapse_design[0, 0]
        self._obs_cov = &self.collapse_obs_cov[0, 0]

        # TODO can I replace this with k_states? I think I should be able to
        return self._k_states


# ### Selected covariance matrice
cdef int sselect_cov(int k, int k_posdef,
                         cnp.float32_t* tmp,
                         cnp.float32_t* selection,
                         cnp.float32_t* cov,
                         cnp.float32_t* selected_cov):
    cdef:
        cnp.float32_t alpha = 1.0
        cnp.float32_t beta = 0.0

    # Only need to do something if there is a covariance matrix
    # (i.e k_posdof == 0)
    if k_posdef > 0:

        # #### Calculate selected state covariance matrix  
        # $Q_t^* = R_t Q_t R_t'$
        # 
        # Combine a selection matrix and a covariance matrix to get
        # a simplified (but possibly singular) "selected" covariance
        # matrix (see e.g. Durbin and Koopman p. 43)

        # `tmp0` array used here, dimension $(m \times r)$  

        # TODO this does not require two ?gemm calls, since we know that it
        # is just selection rows and columns of the Q matrix

        # $\\#_0 = 1.0 * R_t Q_t$  
        # $(m \times r) = (m \times r) (r \times r)$
        blas.sgemm(
            "N", "N", &k, &k_posdef, &k_posdef,
            &alpha, selection, &k,
            cov, &k_posdef,
            &beta, tmp, &k)
        # $Q_t^* = 1.0 * \\#_0 R_t'$  
        # $(m \times m) = (m \times r) (m \times r)'$
        blas.sgemm(
            "N", "T", &k, &k, &k_posdef,
            &alpha, tmp, &k,
            selection, &k,
            &beta, selected_cov, &k)

## State Space Representation
cdef class cStatespace(object):
    """
    cStatespace(obs, design, obs_intercept, obs_cov, transition, state_intercept, selection, state_cov)

    *See Durbin and Koopman (2012), Chapter 4 for all notation*
    """

    # ### State space representation
    # 
    # $$
    # \begin{align}
    # y_t & = Z_t \alpha_t + d_t + \varepsilon_t \hspace{3em} & \varepsilon_t & \sim N(0, H_t) \\\\
    # \alpha_{t+1} & = T_t \alpha_t + c_t + R_t \eta_t & \eta_t & \sim N(0, Q_t) \\\\
    # & & \alpha_1 & \sim N(a_1, P_1)
    # \end{align}
    # $$
    # 
    # $y_t$ is $p \times 1$  
    # $\varepsilon_t$ is $p \times 1$  
    # $\alpha_t$ is $m \times 1$  
    # $\eta_t$ is $r \times 1$  
    # $t = 1, \dots, T$

    # `nobs` $\equiv T$ is the length of the time-series  
    # `k_endog` $\equiv p$ is dimension of observation space  
    # `k_states` $\equiv m$ is the dimension of the state space  
    # `k_posdef` $\equiv r$ is the dimension of the state shocks  
    # *Old notation: T, n, k, g*
    # cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # `obs` $\equiv y_t$ is the **observation vector** $(p \times T)$  
    # `design` $\equiv Z_t$ is the **design vector** $(p \times m \times T)$  
    # `obs_intercept` $\equiv d_t$ is the **observation intercept** $(p \times T)$  
    # `obs_cov` $\equiv H_t$ is the **observation covariance matrix** $(p \times p \times T)$  
    # `transition` $\equiv T_t$ is the **transition matrix** $(m \times m \times T)$  
    # `state_intercept` $\equiv c_t$ is the **state intercept** $(m \times T)$  
    # `selection` $\equiv R_t$ is the **selection matrix** $(m \times r \times T)$  
    # `state_cov` $\equiv Q_t$ is the **state covariance matrix** $(r \times r \times T)$  
    # `selected_state_cov` $\equiv R Q_t R'$ is the **selected state covariance matrix** $(m \times m \times T)$  
    # `initial_state` $\equiv a_1$ is the **initial state mean** $(m \times 1)$  
    # `initial_state_cov` $\equiv P_1$ is the **initial state covariance matrix** $(m \times m)$
    #
    # With the exception of `obs`, these are *optionally* time-varying. If they are instead time-invariant,
    # then the dimension of length $T$ is instead of length $1$.
    #
    # *Note*: the initial vectors' notation 1-indexed as in Durbin and Koopman,
    # but in the recursions below it will be 0-indexed in the Python arrays.
    # 
    # *Old notation: y, -, mu, beta_tt_init, P_tt_init*
    # cdef readonly cnp.complex64_t [::1,:] obs, obs_intercept, state_intercept
    # cdef readonly cnp.complex64_t [:] initial_state
    # cdef readonly cnp.complex64_t [::1,:] initial_state_cov
    # *Old notation: H, R, F, G, Q*, G Q* G'*
    # cdef readonly cnp.complex64_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    # `missing` is a $(p \times T)$ boolean matrix where a row is a $(p \times 1)$ vector
    # in which the $i$th position is $1$ if $y_{i,t}$ is to be considered a missing value.  
    # *Note:* This is created as the output of np.isnan(obs).
    # cdef readonly int [::1,:] missing
    # `nmissing` is an `T \times 0` integer vector holding the number of *missing* observations
    # $p - p_t$
    # cdef readonly int [:] nmissing

    # Flag for a time-invariant model, which requires that *all* of the
    # possibly time-varying arrays are time-invariant.
    # cdef readonly int time_invariant

    # Flag for initialization.
    # cdef readonly int initialized

    # Flags for performance improvements
    # TODO need to add this to the UI in representation
    # cdef public int diagonal_obs_cov
    # cdef public int subset_design
    # cdef public int companion_transition

    # Temporary arrays
    # cdef cnp.complex64_t [::1,:] tmp

    # Temporary selection arrays
    # cdef readonly cnp.complex64_t [:] selected_obs
    # The following are contiguous memory segments which are then used to
    # store the data in the above matrices.
    # cdef readonly cnp.complex64_t [:] selected_design
    # cdef readonly cnp.complex64_t [:] selected_obs_cov

    # Temporary transformation arrays
    # cdef readonly cnp.complex64_t [::1,:] transform_cholesky
    # cdef readonly cnp.complex64_t [::1,:] transform_obs_cov
    # cdef readonly cnp.complex64_t [::1,:] transform_design
    # cdef readonly cnp.complex64_t transform_determinant

    # cdef readonly cnp.complex64_t [:] collapse_obs
    # cdef readonly cnp.complex64_t [:] collapse_obs_tmp
    # cdef readonly cnp.complex64_t [::1,:] collapse_design
    # cdef readonly cnp.complex64_t [::1,:] collapse_obs_cov
    # cdef readonly cnp.complex64_t [::1,:] collapse_cholesky
    # cdef readonly cnp.complex64_t collapse_loglikelihood

    # Pointers  
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

    # Current location dimensions
    # cdef int _k_endog, _k_states, _k_posdef, _k_endog2, _k_states2, _k_posdef2, _k_endogstates, _k_statesposdef
    # cdef int _nmissing

    # ### Initialize state space model
    # *Note*: The initial state and state covariance matrix must be provided.
    def __init__(self,
                 cnp.complex64_t [::1, :]   obs,
                 cnp.complex64_t [::1, :, :] design,
                 cnp.complex64_t [::1, :]   obs_intercept,
                 cnp.complex64_t [::1, :, :] obs_cov,
                 cnp.complex64_t [::1, :, :] transition,
                 cnp.complex64_t [::1, :]   state_intercept,
                 cnp.complex64_t [::1, :, :] selection,
                 cnp.complex64_t [::1, :, :] state_cov,
                 diagonal_obs_cov=-1):

        # Local variables
        cdef:
            int t, i, j
            cnp.npy_intp dim1[1]
            cnp.npy_intp dim2[2]
            cnp.npy_intp dim3[3]

        # #### State space representation variables  
        # **Note**: these arrays share data with the versions defined in
        # Python and passed to this constructor, so if they are updated in
        # Python they will also be updated here.
        self.obs = obs
        self.design = design
        self.obs_intercept = obs_intercept
        self.obs_cov = obs_cov
        self.transition = transition
        self.state_intercept = state_intercept
        self.selection = selection
        self.state_cov = state_cov

        # Dimensions
        self.k_endog = obs.shape[0]
        self.k_states = selection.shape[0]
        self.k_posdef = selection.shape[1]
        self.nobs = obs.shape[1]

        # #### Validate matrix dimensions
        #
        # Make sure that the given state-space matrices have consistent sizes
        tools.validate_matrix_shape('design', &self.design.shape[0],
                                    self.k_endog, self.k_states, self.nobs)
        tools.validate_vector_shape('observation intercept', &self.obs_intercept.shape[0],
                                    self.k_endog, self.nobs)
        tools.validate_matrix_shape('observation covariance matrix', &self.obs_cov.shape[0],
                                    self.k_endog, self.k_endog, self.nobs)
        tools.validate_matrix_shape('transition', &self.transition.shape[0],
                                    self.k_states, self.k_states, self.nobs)
        tools.validate_vector_shape('state intercept', &self.state_intercept.shape[0],
                                    self.k_states, self.nobs)
        tools.validate_matrix_shape('state covariance matrix', &self.state_cov.shape[0],
                                    self.k_posdef, self.k_posdef, self.nobs)

        # Check for a time-invariant model
        self.time_invariant = (
            self.design.shape[2] == 1           and
            self.obs_intercept.shape[1] == 1    and
            self.obs_cov.shape[2] == 1          and
            self.transition.shape[2] == 1       and
            self.state_intercept.shape[1] == 1  and
            self.selection.shape[2] == 1        and
            self.state_cov.shape[2] == 1
        )

        # Set the flag for initialization to be false
        self.initialized = False
        self.initial_state = None
        self.initial_state_cov = None

        # Unless it is specified, check for a diagonal covariance matrix
        if diagonal_obs_cov == -1:
            diagonal_obs_cov = True
            for t in range(self.obs_cov.shape[2]):
                for i in range(self.k_endog):
                    for j in range(self.k_endog):
                        if i == j:
                            continue
                        if not (zabs(self.obs_cov[i, j, t]) < 1e-9):
                            diagonal_obs_cov = False
                            break
        self.diagonal_obs_cov = diagonal_obs_cov

        # Allocate selected state covariance matrix
        dim3[0] = self.k_states; dim3[1] = self.k_states; dim3[2] = 1
        # (we only allocate memory for time-varying array if necessary)
        if self.state_cov.shape[2] > 1 or self.selection.shape[2] > 1:
            dim3[2] = self.nobs
        self.selected_state_cov = cnp.PyArray_ZEROS(3, dim3, cnp.NPY_COMPLEX64, FORTRAN)

        # Handle missing data
        self.missing = np.array(np.isnan(obs), dtype=np.int32, order="F")
        self.nmissing = np.array(np.sum(self.missing, axis=0), dtype=np.int32)
        self.has_missing = np.sum(self.nmissing) > 0

        # Create the temporary array
        # Holds arrays of dimension $(m \times m)$
        dim2[0] = self.k_states; dim2[1] = max(self.k_states, self.k_posdef)
        self.tmp = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)

        # Arrays for missing data
        dim1[0] = self.k_endog;
        self.selected_obs = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX64, FORTRAN)
        dim1[0] = self.k_endog;
        self.selected_obs_intercept = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX64, FORTRAN)
        dim1[0] = self.k_endog * self.k_states;
        self.selected_design = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX64, FORTRAN)
        dim1[0] = self.k_endog**2;
        self.selected_obs_cov = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX64, FORTRAN)

        # Arrays for transformations
        dim2[0] = self.k_endog; dim2[1] = self.k_endog
        self.transform_cholesky = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self.transform_obs_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        dim2[0] = self.k_endog; dim2[1] = self.k_states
        self.transform_design = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)

        dim1[0] = self.k_states
        self.collapse_obs = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX64, FORTRAN)
        self.collapse_obs_tmp = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX64, FORTRAN)
        dim2[0] = self.k_states; dim2[1] = self.k_states
        self.collapse_design = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self.collapse_obs_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)
        self.collapse_cholesky = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)

        # Initialize location
        self.t = 0

        # Initialize dimensions
        self.set_dimensions(self.k_endog, self.k_states, self.k_posdef)

    def __reduce__(self):
        init = (np.array(self.obs, copy=True, order='F'), np.array(self.design, copy=True, order='F'),
                np.array(self.obs_intercept, copy=True, order='F'), np.array(self.obs_cov, copy=True, order='F'),
                np.array(self.transition, copy=True, order='F'), np.array(self.state_intercept, copy=True, order='F'),
                np.array(self.selection, copy=True, order='F'), np.array(self.state_cov, copy=True, order='F'),
                self.diagonal_obs_cov)
        state = {'initialized': self.initialized,
                 'initial_state': None,
                 'initial_state_cov': None,
                 'missing': np.array(self.missing, copy=True, order='F'),
                 'nmissing': np.array(self.nmissing, copy=True, order='F'),
                 'has_missing': self.has_missing,
                 'tmp': np.array(self.tmp, copy=True, order='F'),
                 'selected_state_cov': np.array(self.selected_state_cov, copy=True, order='F'),
                 'selected_obs': np.array(self.selected_obs, copy=True, order='F'),
                 'selected_obs_intercept': np.array(self.selected_obs_intercept, copy=True, order='F'),
                 'selected_design': np.array(self.selected_design, copy=True, order='F'),
                 'selected_obs_cov': np.array(self.selected_obs_cov, copy=True, order='F'),
                 'transform_cholesky': np.array(self.transform_cholesky, copy=True, order='F'),
                 'transform_obs_cov': np.array(self.transform_obs_cov, copy=True, order='F'),
                 'transform_design': np.array(self.transform_design, copy=True, order='F'),
                 'collapse_obs': np.array(self.collapse_obs, copy=True, order='F'),
                 'collapse_obs_tmp': np.array(self.collapse_obs_tmp, copy=True, order='F'),
                 'collapse_design': np.array(self.collapse_design, copy=True, order='F'),
                 'collapse_obs_cov': np.array(self.collapse_obs_cov, copy=True, order='F'),
                 'collapse_cholesky': np.array(self.collapse_cholesky, copy=True, order='F'),
                 't': self.t,
                 'collapse_loglikelihood': self.collapse_loglikelihood,
                 'companion_transition': self.companion_transition,
                 'transform_determinant': self.transform_determinant,
                 }
        if self.initialized:
            state['initial_state'] = np.array(self.initial_state, copy=True, order='F')
            state['initial_state_cov'] = np.array(self.initial_state_cov, copy=True, order='F')
        return (self.__class__, init, state)

    def __setstate__(self, state):
        self.initial_state = state['initial_state']
        self.initial_state_cov = state['initial_state_cov']
        self.initialized = state['initialized']
        self.selected_state_cov = state['selected_state_cov']
        self.missing = state['missing']
        self.nmissing =state['nmissing']
        self.has_missing = state['has_missing']
        self.tmp = state['tmp']
        self.selected_obs  = state['selected_obs']
        self.selected_obs_intercept  = state['selected_obs_intercept']
        self.selected_design  = state['selected_design']
        self.selected_obs_cov  =state['selected_obs_cov']
        self.transform_cholesky  = state['transform_cholesky']
        self.transform_obs_cov  = state['transform_obs_cov']
        self.transform_design = state['transform_design']
        self.collapse_obs = state['collapse_obs']
        self.collapse_obs_tmp = state['collapse_obs_tmp']
        self.collapse_design = state['collapse_design']
        self.collapse_obs_cov = state['collapse_obs_cov']
        self.collapse_cholesky = state['collapse_cholesky']
        self.t = state['t']
        self.collapse_loglikelihood = state['collapse_loglikelihood']
        self.companion_transition = state['companion_transition']
        self.transform_determinant = state['transform_determinant']

    # ## Initialize: known values
    #
    # Initialize the filter with specific values, assumed to be known with
    # certainty or else as filled with parameters from a maximum likelihood
    # estimation run.
    def initialize_known(self, cnp.complex64_t [:] initial_state, cnp.complex64_t [::1, :] initial_state_cov):
        """
        initialize_known(initial_state, initial_state_cov)
        """
        tools.validate_vector_shape('inital state', &initial_state.shape[0], self.k_states, None)
        tools.validate_matrix_shape('initial state covariance', &initial_state_cov.shape[0], self.k_states, self.k_states, None)

        self.initial_state = initial_state
        self.initial_state_cov = initial_state_cov

        self.initialized = True

    # ## Initialize: approximate diffuse priors
    #
    # Durbin and Koopman note that this initialization should only be coupled
    # with the standard Kalman filter for "approximate exploratory work" and
    # can lead to "large rounding errors" (p. 125).
    # 
    # *Note:* see Durbin and Koopman section 5.6.1
    def initialize_approximate_diffuse(self, cnp.complex64_t variance=1e2):
        """
        initialize_approximate_diffuse(variance=1e2)
        """
        cdef cnp.npy_intp dim[1]
        dim[0] = self.k_states
        self.initial_state = cnp.PyArray_ZEROS(1, dim, cnp.NPY_COMPLEX64, FORTRAN)
        self.initial_state_cov = np.eye(self.k_states, dtype=np.complex64).T * variance

        self.initialized = True

    # ## Initialize: stationary process
    # *Note:* see Durbin and Koopman section 5.6.2
    def initialize_stationary(self, complex_step=False):
        """
        initialize_stationary()
        """
        cdef cnp.npy_intp dim1[1]
        cdef cnp.npy_intp dim2[2]
        cdef int i, info, inc = 1
        cdef int k_states2 = self.k_states**2
        cdef cnp.float64_t asum, tol = 1e-9
        cdef cnp.complex64_t scalar
        cdef int [::1,:] ipiv

        # Create selected state covariance matrix
        cselect_cov(self.k_states, self.k_posdef,
                                   &self.tmp[0, 0],
                                   &self.selection[0, 0, 0],
                                   &self.state_cov[0, 0, 0],
                                   &self.selected_state_cov[0, 0, 0])

        # Initial state mean
        asum = blas.scasum(&self.k_states, &self.state_intercept[0, 0], &inc)

        dim1[0] = self.k_states
        self.initial_state = cnp.PyArray_ZEROS(1, dim1, cnp.NPY_COMPLEX64, FORTRAN)
        if asum > tol:
            dim2[0] = self.k_states
            dim2[1] = self.k_states
            ipiv = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_INT32, FORTRAN)

            # I - T
            blas.ccopy(
                &k_states2, &self.transition[0, 0, 0], &inc,
                &self.tmp[0,0], &inc)
            scalar = -1.0
            blas.cscal(&k_states2, &scalar, &self.tmp[0, 0], &inc)
            for i in range(self.k_states):
                self.tmp[i, i] = self.tmp[i, i] + 1

            # c
            blas.ccopy(
                &self.k_states, &self.state_intercept[0, 0], &inc,
                &self.initial_state[0], &inc)

            # Solve (I - T) x = c
            lapack.cgetrf(
                &self.k_states, &self.k_states, &self.tmp[0, 0], &self.k_states,
                &ipiv[0, 0], &info)
            lapack.cgetrs(
                'N', &self.k_states, &inc, &self.tmp[0, 0], &self.k_states,
                &ipiv[0, 0], &self.initial_state[0], &self.k_states, &info)

        dim2[0] = self.k_states; dim2[1] = self.k_states
        self.initial_state_cov = cnp.PyArray_ZEROS(2, dim2, cnp.NPY_COMPLEX64, FORTRAN)

        # Create a copy of the transition matrix (to avoid overwriting it)
        blas.ccopy(
            &k_states2, &self.transition[0, 0, 0], &inc,
            &self.tmp[0, 0], &inc)

        # Copy the selected state covariance to the initial state covariance
        # (it will be overwritten with the appropriate matrix)
        blas.ccopy(
            &k_states2, &self.selected_state_cov[0, 0, 0], &inc,
            &self.initial_state_cov[0, 0], &inc)

        # Solve the discrete Lyapunov equation to the get initial state
        # covariance matrix
        tools._csolve_discrete_lyapunov(
            &self.tmp[0, 0],
            &self.initial_state_cov[0, 0],
            self.k_states, complex_step)

        self.initialized = True

    def __iter__(self):
        return self

    def __next__(self):
        """
        Advance to the next location
        """
        if self.t >= self.nobs:
            raise StopIteration
        else:
            self.seek(self.t + 1, 0, 0)

    cpdef seek(self, unsigned int t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse):
        cdef int previous_t = self.t

        # Set the global time indicator, if valid
        if t >= self.nobs:
            raise IndexError("Observation index out of range")
        self.t = t

        # Indices for possibly time-varying arrays
        cdef:
            int k_endog
            int design_t = 0
            int obs_intercept_t = 0
            int obs_cov_t = 0
            int transition_t = 0
            int state_intercept_t = 0
            int selection_t = 0
            int state_cov_t = 0

        # Get indices for possibly time-varying arrays
        if not self.time_invariant:
            if self.design.shape[2] > 1:
                design_t = t
            if self.obs_intercept.shape[1] > 1:
                obs_intercept_t = t
            if self.obs_cov.shape[2] > 1:
                obs_cov_t = t
            if self.transition.shape[2] > 1:
                transition_t = t
            if self.state_intercept.shape[1] > 1:
                state_intercept_t = t
            if self.selection.shape[2] > 1:
                selection_t = t
            if self.state_cov.shape[2] > 1:
                state_cov_t = t

        # Initialize object-level pointers to statespace arrays
        self._obs = &self.obs[0, t]
        self._design = &self.design[0, 0, design_t]
        self._obs_intercept = &self.obs_intercept[0, obs_intercept_t]
        self._obs_cov = &self.obs_cov[0, 0, obs_cov_t]
        self._transition = &self.transition[0, 0, transition_t]
        self._state_intercept = &self.state_intercept[0, state_intercept_t]
        self._selection = &self.selection[0, 0, selection_t]
        self._state_cov = &self.state_cov[0, 0, state_cov_t]

        # Initialize object-level pointers to initialization
        if not self.initialized:
            raise RuntimeError("Statespace model not initialized.")
        self._initial_state = &self.initial_state[0]
        self._initial_state_cov = &self.initial_state_cov[0,0]

        # Create the selected state covariance matrix
        self.select_state_cov(t)

        # Handle missing data
        # Note: this modifies object pointers and _* dimensions
        k_endog = self.select_missing(t)

        # Set dimensions
        self.set_dimensions(k_endog, self.k_states, self.k_posdef)

        # Handle transformations
        self.transform(t, previous_t, transform_diagonalize, transform_generalized_collapse)

    @cython.cfunc
    @cython.returns(cython.void)
    def set_dimensions(self, unsigned int k_endog, unsigned int k_states, unsigned int k_posdef):
        self._k_endog = k_endog
        self._k_states = k_states
        self._k_posdef = k_posdef
        self._k_endog2 = k_endog**2
        self._k_states2 = k_states**2
        self._k_posdef2 = k_posdef**2
        self._k_endogstates = k_endog * k_states
        self._k_statesposdef = k_states * k_posdef

    @cython.cfunc
    @cython.returns(cython.void)
    def select_state_cov(self, unsigned int t):
        cdef int selected_state_cov_t = 0

        # ### Get selected state covariance matrix
        if t == 0 or self.selected_state_cov.shape[2] > 1:
            selected_state_cov_t = t
            self._selected_state_cov = &self.selected_state_cov[0, 0, selected_state_cov_t]

            cselect_cov(
                self.k_states, self.k_posdef,
                &self.tmp[0, 0],
                self._selection,
                self._state_cov,
                self._selected_state_cov)
        else:
            self._selected_state_cov = &self.selected_state_cov[0, 0, 0]

    @cython.cfunc
    @cython.returns(int)
    def select_missing(self, unsigned int t):
        # Note: this assumes that object pointers are already initialized
        # Note: this assumes that transform_... will be done *later*
        cdef int k_endog = self.k_endog

        # Set the current iteration nmissing
        self._nmissing = self.nmissing[t]

        # ### Perform missing selections
        # In Durbin and Koopman (2012), these are represented as matrix
        # multiplications, i.e. $Z_t^* = W_t Z_t$ where $W_t$ is a row
        # selection matrix (it contains a subset of rows of the identity
        # matrix).
        #
        # It's more efficient, though, to just copy over the data directly,
        # which is what is done here. Note that the `selected_*` arrays are
        # defined as single-dimensional, so the assignment indexes below are
        # set such that the arrays can be interpreted by the BLAS and LAPACK
        # functions as two-dimensional, column-major arrays.
        #
        # In the case that all data is missing (e.g. this is what happens in
        # forecasting), we actually set don't change the dimension, but we set
        # the design matrix to the zeros array.
        if self._nmissing == self.k_endog:
            self._select_missing_entire_obs(t)
        elif self._nmissing > 0:
            self._select_missing_partial_obs(t)
            k_endog = self.k_endog - self._nmissing

        # Return the number of non-missing endogenous variables
        return k_endog

    @cython.cfunc
    @cython.returns(cython.void)
    def _select_missing_entire_obs(self, unsigned int t):
        cdef:
            int i, j

        # Design matrix is set to zeros
        for i in range(self.k_states):
            for j in range(self.k_endog):
                self.selected_design[j + i * self.k_endog] = 0.0
        self._design = &self.selected_design[0]

    @cython.cfunc
    @cython.returns(cython.void)
    def _select_missing_partial_obs(self, unsigned int t):
        cdef:
            int i, j, k, l
            int inc = 1
            int design_t = 0
            int obs_cov_t = 0
            int k_endog = self.k_endog - self._nmissing

        k = 0
        for i in range(self.k_endog):
            if not self.missing[i, t]:
                self.selected_obs[k] = self._obs[i]
                self.selected_obs_intercept[k] = self._obs_intercept[i]

                # i is rows, k is rows
                blas.ccopy(
                    &self.k_states, &self._design[i], &self.k_endog,
                    &self.selected_design[k], &k_endog)

                # i, k is columns, j, l is rows
                l = 0
                for j in range(self.k_endog):
                    if not self.missing[j, t]:
                        self.selected_obs_cov[l + k * k_endog] = self._obs_cov[j + i * self.k_endog]
                        l += 1
                k += 1
        self._obs = &self.selected_obs[0]
        self._obs_intercept = &self.selected_obs_intercept[0]
        self._design = &self.selected_design[0]
        self._obs_cov = &self.selected_obs_cov[0]

    cdef void transform(self, unsigned int t, unsigned int previous_t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse) except *:
        # Reset the collapsed loglikelihood
        self.collapse_loglikelihood = 0

        if transform_generalized_collapse and not self._k_endog <= self._k_states:
            k_endog = self.transform_generalized_collapse(t, previous_t)
            # Reset dimensions
            self.set_dimensions(k_endog, self._k_states, self._k_posdef)
        elif transform_diagonalize and not self.diagonal_obs_cov:
            self.transform_diagonalize(t, previous_t)

    cdef void transform_diagonalize(self, unsigned int t, unsigned int previous_t) except *:
        # Note: this assumes that initialize_object_pointers has *already* been done
        # Note: this assumes that select_missing has *already* been done
        # TODO need unit tests, especially for the missing case
        # TODO need to also transform observation intercept
        cdef:
            int i, j, inc=1
            int obs_cov_t, design_t
            int info
            int reset_missing
            cnp.complex64_t * _transform_obs_cov = &self.transform_obs_cov[0, 0]
            cnp.complex64_t * _transform_cholesky = &self.transform_cholesky[0, 0]

        # Compute the cholesky decomposition of *self._obs_cov
        if self.diagonal_obs_cov:
            return

        # Handle missing data
        if self.nmissing[t] == self.k_endog:
            return

        reset_missing = 0
        for i in range(self.k_endog):
            reset_missing = reset_missing + (not self.missing[i, t] == self.missing[i, previous_t])

        # Perform the LDL decomposition, if necessary
        if t == 0 or self.obs_cov.shape[2] > 1 or reset_missing:
            # Make sure we don't have an observation intercept
            if not np.sum(self.obs_intercept) == 0 or self.obs_intercept.shape[2] > 1:
                raise RuntimeError('The univariate method with non-diagonal '
                                   'observation covariance matrix does not '
                                   'currently support an observation intercept.')

            # LDL decomposition
            blas.ccopy(
                &self._k_endog2, self._obs_cov, &inc,
                _transform_cholesky, &inc)
            info = tools._cldl(_transform_cholesky, self._k_endog)

            # Check for errors
            if info > 0:
                warnings.warn("Positive semi-definite observation covariance matrix encountered at period %d" % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in LDL factorization of observation covariance matrix encountered at period %d' % t)

            # Currently both L and D are stored in transform_cholesky
            for i in range(self._k_endog): # i is rows
                for j in range(self._k_endog): # j is columns
                    # Diagonal elements come from the diagonal
                    if i == j:
                        _transform_obs_cov[i + i * self._k_endog] = _transform_cholesky[i + i * self._k_endog]
                    # Other elements are zero
                    else:
                        _transform_obs_cov[i + j * self._k_endog] = 0

                    # Zero out the upper triangle of the cholesky
                    if j > i:
                        _transform_cholesky[i + j * self._k_endog] = 0

                # Convert from L to C simply by setting the diagonal elements to ones
                _transform_cholesky[i + i * self._k_endog] = 1
        
        # Solve for y_t^*
        # (unless this is a completely missing observation)
        if not self._nmissing == self.k_endog:
            # If we have some missing elements, selected_obs is already populated
            if self._nmissing == 0:
                blas.ccopy(
                    &self._k_endog, &self.obs[0, t], &inc,
                    &self.selected_obs[0], &inc)
            lapack.ctrtrs(
                "L", "N", "U", &self._k_endog, &inc,
                _transform_cholesky, &self.k_endog,
                &self.selected_obs[0], &self._k_endog, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Singular factorization of observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in factorization of observation covariance matrix encountered at period %d' % t)

            # Setup the pointer
            self._obs = &self.selected_obs[0]

        # Solve for Z_t^*, if necessary
        if t == 0 or self.design.shape[2] > 1 or reset_missing:
            blas.ccopy(
                &self._k_endogstates, self._design, &inc,
                &self.transform_design[0, 0], &inc)
            lapack.ctrtrs(
                "L", "N", "U", &self._k_endog, &self._k_states,
                _transform_cholesky, &self.k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Singular factorization of observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in factorization of observation covariance matrix encountered at period %d' % t)

        # Setup final pointers            
        self._design = &self.transform_design[0, 0]
        self._obs_cov = &self.transform_obs_cov[0, 0]

    cdef int transform_generalized_collapse(self, unsigned int t, unsigned int previous_t) except *:
        # Note: this assumes that initialize_object_pointers has *already* been done
        # Note: this assumes that select_missing has *already* been done
        # TODO need unit tests, especially for the missing case
        cdef:
            int i, j, inc=1
            int obs_cov_t, design_t
            int info
            int reset_missing
            cnp.complex64_t alpha = 1.0
            cnp.complex64_t beta = 0.0
            cnp.complex64_t gamma = -1.0
            int k_states = self._k_states
            int k_states2 = self._k_states2
            int k_endogstates = self._k_endogstates

        # $y_t^* = \bar A^* y_t = C_t Z_t' H_t^{-1} y_t$  
        # $Z_t^* = C_t^{-1}$  
        # $H_t^* = I_m$  

        # Make sure we have enough observations to perform collapse
        if self.k_endog < self.k_states:
            raise RuntimeError('Cannot collapse observation vector it the'
                               ' state dimension is larger than the dimension'
                               ' of the observation vector.')

        # Adjust for a VAR transition (i.e. design = [#, 0], where the zeros
        # correspond to all states except the first k_posdef states)
        if self.subset_design:
            k_states = self._k_posdef
            k_states2 = self._k_posdef2
            k_endogstates = self._k_endog * self._k_posdef

        # Handle missing data
        if self.nmissing[t] == self.k_endog:
            return self.k_states
        reset_missing = 0
        for i in range(self.k_endog):
            reset_missing = reset_missing + (not self.missing[i, t] == self.missing[i, previous_t])

        # Initialize the transformation
        if self.collapse_obs_cov[0, 0] == 0:
            # Set H_t^* to identity
            for i in range(k_states):
                self.collapse_obs_cov[i, i] = 1

            # Make sure we don't have an observation intercept
            if not np.sum(self.obs_intercept) == 0 or self.obs_intercept.shape[2] > 1:
                raise RuntimeError('The observation collapse transformation'
                                   ' does not currently support an observation'
                                   ' intercept.')

        # Perform the Cholesky decomposition of H_t, if necessary
        if t == 0 or self.obs_cov.shape[2] > 1 or reset_missing:
            # Cholesky decomposition: $H = L L'$  
            blas.ccopy(
                &self._k_endog2, self._obs_cov, &inc,
                &self.transform_cholesky[0, 0], &inc)
            # Use LDA=self.k_endog so that we can use the memoryview slicing below
            lapack.cpotrf(
                "L", &self._k_endog, &self.transform_cholesky[0, 0], &self.k_endog, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Non-positive-definite observation covariance matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in observation covariance matrix encountered at period %d' % t)

            # Calculate the determinant (just the squared product of the
            # diagonals, in the Cholesky decomposition case)
            self.transform_determinant = 1.0
            for i in range(self._k_endog):
                if not self.transform_cholesky[i, i] == 0:
                    self.transform_determinant = self.transform_determinant * self.transform_cholesky[i, i]
            self.transform_determinant = self.transform_determinant**2

        # Get $Z_t \equiv C^{-1}$, if necessary  
        if t == 0 or self.obs_cov.shape[2] > 1 or self.design.shape[2] > 1 or reset_missing:
            # Calculate $H_t^{-1} Z_t \equiv (Z_t' H_t^{-1})'$ via Cholesky solver
            blas.ccopy(
                &self._k_endogstates, self._design, &inc,
                &self.transform_design[0, 0], &inc)
            lapack.cpotrs(
                "L", &self._k_endog, &k_states,
                &self.transform_cholesky[0, 0], &self.k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &info)

            # Check for errors
            if not info == 0:
                raise np.linalg.LinAlgError('Invalid value in calculation of H_t^{-1}Z matrix encountered at period %d' % t)
        
            # Calculate $(H_t^{-1} Z_t)' Z_t$  
            # $(m \times m) = (m \times p) (p \times p) (p \times m)$
            blas.cgemm(
                "T", "N", &k_states, &k_states, &self._k_endog,
                &alpha, self._design, &self._k_endog,
                &self.transform_design[0, 0], &self._k_endog,
                &beta, &self.collapse_cholesky[0, 0], &self._k_states)

            # Calculate $(Z_t' H_t^{-1} Z_t)^{-1}$ via Cholesky inversion  
            lapack.cpotrf("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)
            lapack.cpotri("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)

            # Calculate $C_t$ (the upper triangular cholesky decomposition of $(Z_t' H_t^{-1} Z_t)^{-1}$)  
            lapack.cpotrf("U", &k_states, &self.collapse_cholesky[0, 0], &self.k_states, &info)

            # Check for errors
            if info > 0:
                raise np.linalg.LinAlgError('Non-positive-definite ZHZ matrix encountered at period %d' % t)
            elif info < 0:
                raise np.linalg.LinAlgError('Invalid value in ZHZ matrix encountered at period %d' % t)

            # Calculate $C_t'^{-1} \equiv Z_t$  
            # Do so by solving the system: $C_t' x = I$  
            # (Recall that collapse_obs_cov is an identity matrix)
            blas.ccopy(
                &self._k_states2, &self.collapse_obs_cov[0, 0], &inc,
                &self.collapse_design[0, 0], &inc)
            lapack.ctrtrs(
                "U", "T", "N", &k_states, &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_design[0, 0], &self._k_states,
                &info)

        # Calculate $\bar y_t^* = \bar A_t^* y_t = C_t Z_t' H_t^{-1} y_t$  
        # (unless this is a completely missing observation)
        self.collapse_loglikelihood = 0
        if not self._nmissing == self.k_endog:
            # If we have some missing elements, selected_obs is already populated
            if self._nmissing == 0:
                blas.ccopy(
                    &self.k_endog, &self.obs[0, t], &inc,
                    &self.selected_obs[0], &inc)
            # $\\# = Z_t' H_t^{-1} y_t$
            blas.cgemv(
                "T", &self._k_endog, &k_states,
                &alpha, &self.transform_design[0, 0], &self._k_endog,
                &self.selected_obs[0], &inc,
                &beta, &self.collapse_obs[0], &inc)
            # $y_t^* = C_t \\#$  
            blas.ctrmv(
                "U", "N", "N", &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_obs[0], &inc)

            # Get residuals for loglikelihood calculation
            # Note: Durbin and Koopman (2012) appears to have an error in the
            # formula here. They have $e_t = y_t - Z_t \bar y_t^*$, whereas it
            # should be: $e_t = y_t - Z_t C_t' \bar y_t^*$
            # See Jungbacker and Koopman (2014), section 2.5 where $e_t$ is
            # defined. In this case, $Z_t^dagger = Z_t C_t$ where
            # $C_t C_t' = (Z_t' \Sigma_\varepsilon^{-1} Z_t)^{-1}$.
            # 

            # $ \\# = C_t' y_t^*$
            blas.ccopy(
                &k_states, &self.collapse_obs[0], &inc,
                &self.collapse_obs_tmp[0], &inc)
            blas.ctrmv(
                "U", "T", "N", &k_states,
                &self.collapse_cholesky[0, 0], &self._k_states,
                &self.collapse_obs_tmp[0], &inc)

            # $e_t = - Z_t C_t' y_t^* + y_t$
            blas.cgemv(
                "N", &self._k_endog, &k_states,
                &gamma, self._design, &self._k_endog,
                &self.collapse_obs_tmp[0], &inc,
                &alpha, &self.selected_obs[0], &inc)

            # Calculate e_t' H_t^{-1} e_t via Cholesky solver  
            # $H_t^{-1} = (L L')^{-1} = L^{-1}' L^{-1}$  
            # So we want $e_t' L^{-1}' L^{-1} e_t = (L^{-1} e_t)' L^{-1} e_t$  
            # We have $L$ in `transform_cholesky`, so we want to do a linear  
            # solve of $L x = e_t$  where L is lower triangular
            lapack.ctrtrs(
                "L", "N", "N", &self._k_endog, &inc,
                &self.transform_cholesky[0, 0], &self.k_endog,
                &self.selected_obs[0], &self._k_endog,
                &info)

            # Calculate loglikelihood contribution of this observation

            # $e_t' H_t^{-1} e_t = (L^{-1} e_t)' L^{-1} e_t = \sum_i e_{i,t}**2$  
            self.collapse_loglikelihood = 0
            for i in range(self._k_endog):
                self.collapse_loglikelihood = self.collapse_loglikelihood + self.selected_obs[i]**2
            
            # (p-m) log( 2*pi) + log( |H_t| )
            self.collapse_loglikelihood = (
                self.collapse_loglikelihood +
                (self._k_endog - k_states)*zlog(2*NPY_PI) + 
                zlog(self.transform_determinant)
            )

            # -0.5 * ...
            self.collapse_loglikelihood = -0.5 * self.collapse_loglikelihood

        # Set pointers
        self._obs = &self.collapse_obs[0]
        self._design = &self.collapse_design[0, 0]
        self._obs_cov = &self.collapse_obs_cov[0, 0]

        # TODO can I replace this with k_states? I think I should be able to
        return self._k_states


# ### Selected covariance matrice
cdef int cselect_cov(int k, int k_posdef,
                         cnp.complex64_t* tmp,
                         cnp.complex64_t* selection,
                         cnp.complex64_t* cov,
                         cnp.complex64_t* selected_cov):
    cdef:
        cnp.complex64_t alpha = 1.0
        cnp.complex64_t beta = 0.0

    # Only need to do something if there is a covariance matrix
    # (i.e k_posdof == 0)
    if k_posdef > 0:

        # #### Calculate selected state covariance matrix  
        # $Q_t^* = R_t Q_t R_t'$
        # 
        # Combine a selection matrix and a covariance matrix to get
        # a simplified (but possibly singular) "selected" covariance
        # matrix (see e.g. Durbin and Koopman p. 43)

        # `tmp0` array used here, dimension $(m \times r)$  

        # TODO this does not require two ?gemm calls, since we know that it
        # is just selection rows and columns of the Q matrix

        # $\\#_0 = 1.0 * R_t Q_t$  
        # $(m \times r) = (m \times r) (r \times r)$
        blas.cgemm(
            "N", "N", &k, &k_posdef, &k_posdef,
            &alpha, selection, &k,
            cov, &k_posdef,
            &beta, tmp, &k)
        # $Q_t^* = 1.0 * \\#_0 R_t'$  
        # $(m \times m) = (m \times r) (m \times r)'$
        blas.cgemm(
            "N", "T", &k, &k, &k_posdef,
            &alpha, tmp, &k,
            selection, &k,
            &beta, selected_cov, &k)
