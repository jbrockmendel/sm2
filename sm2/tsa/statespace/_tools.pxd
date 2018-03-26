# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=False
"""
State Space Models - Cython Tools declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

from cython cimport Py_ssize_t

cimport numpy as cnp

cdef validate_matrix_shape(str name, Py_ssize_t* shape, int nrows, int ncols, object nobs=*)

cdef validate_vector_shape(str name, Py_ssize_t* shape, int nrows, object nobs=*)

cdef int _ssolve_discrete_lyapunov(cnp.float32_t* a, cnp.float32_t* q, int n, int complex_step=*) except *
cdef int _dsolve_discrete_lyapunov(cnp.float64_t* a, cnp.float64_t* q, int n, int complex_step=*) except *
cdef int _csolve_discrete_lyapunov(cnp.complex64_t* a, cnp.complex64_t* q, int n, int complex_step=*) except *
cdef int _zsolve_discrete_lyapunov(cnp.complex128_t* a, cnp.complex128_t* q, int n, int complex_step=*) except *

cdef int _sldl(cnp.float32_t* A, int n) except *
cdef int _dldl(cnp.float64_t* A, int n) except *
cdef int _cldl(cnp.complex64_t* A, int n) except *
cdef int _zldl(cnp.complex128_t* A, int n) except *

cpdef int sldl(cnp.float32_t[::1, :] A) except *
cpdef int dldl(cnp.float64_t[::1, :] A) except *
cpdef int cldl(cnp.complex64_t[::1, :] A) except *
cpdef int zldl(cnp.complex128_t[::1, :] A) except *

cdef int _sreorder_missing_diagonal(cnp.float32_t* a, int* missing, int n)
cdef int _dreorder_missing_diagonal(cnp.float64_t* a, int* missing, int n)
cdef int _creorder_missing_diagonal(cnp.complex64_t* a, int* missing, int n)
cdef int _zreorder_missing_diagonal(cnp.complex128_t* a, int* missing, int n)

cdef int _sreorder_missing_submatrix(cnp.float32_t* a, int* missing, int n)
cdef int _dreorder_missing_submatrix(cnp.float64_t* a, int* missing, int n)
cdef int _creorder_missing_submatrix(cnp.complex64_t* a, int* missing, int n)
cdef int _zreorder_missing_submatrix(cnp.complex128_t* a, int* missing, int n)

cdef int _sreorder_missing_rows(cnp.float32_t* a, int* missing, int n, int m)
cdef int _dreorder_missing_rows(cnp.float64_t* a, int* missing, int n, int m)
cdef int _creorder_missing_rows(cnp.complex64_t* a, int* missing, int n, int m)
cdef int _zreorder_missing_rows(cnp.complex128_t* a, int* missing, int n, int m)

cdef int _sreorder_missing_cols(cnp.float32_t* a, int* missing, int n, int m)
cdef int _dreorder_missing_cols(cnp.float64_t* a, int* missing, int n, int m)
cdef int _creorder_missing_cols(cnp.complex64_t* a, int* missing, int n, int m)
cdef int _zreorder_missing_cols(cnp.complex128_t* a, int* missing, int n, int m)

cpdef int sreorder_missing_matrix(cnp.float32_t[::1, :, :] A, int[::1, :] missing, int reorder_rows, int reorder_cols, int diagonal) except *
cpdef int dreorder_missing_matrix(cnp.float64_t[::1, :, :] A, int[::1, :] missing, int reorder_rows, int reorder_cols, int diagonal) except *
cpdef int creorder_missing_matrix(cnp.complex64_t[::1, :, :] A, int[::1, :] missing, int reorder_rows, int reorder_cols, int diagonal) except *
cpdef int zreorder_missing_matrix(cnp.complex128_t[::1, :, :] A, int[::1, :] missing, int reorder_rows, int reorder_cols, int diagonal) except *

cpdef int sreorder_missing_vector(cnp.float32_t[::1, :] A, int[::1, :] missing) except *
cpdef int dreorder_missing_vector(cnp.float64_t[::1, :] A, int[::1, :] missing) except *
cpdef int creorder_missing_vector(cnp.complex64_t[::1, :] A, int[::1, :] missing) except *
cpdef int zreorder_missing_vector(cnp.complex128_t[::1, :] A, int[::1, :] missing) except *

cdef int _scopy_missing_diagonal(cnp.float32_t* a, cnp.float32_t* b, int* missing, int n)
cdef int _dcopy_missing_diagonal(cnp.float64_t* a, cnp.float64_t* b, int* missing, int n)
cdef int _ccopy_missing_diagonal(cnp.complex64_t* a, cnp.complex64_t* b, int* missing, int n)
cdef int _zcopy_missing_diagonal(cnp.complex128_t* a, cnp.complex128_t* b, int* missing, int n)

cdef int _scopy_missing_submatrix(cnp.float32_t* a, cnp.float32_t* b, int* missing, int n)
cdef int _dcopy_missing_submatrix(cnp.float64_t* a, cnp.float64_t* b, int* missing, int n)
cdef int _ccopy_missing_submatrix(cnp.complex64_t* a, cnp.complex64_t* b, int* missing, int n)
cdef int _zcopy_missing_submatrix(cnp.complex128_t* a, cnp.complex128_t* b, int* missing, int n)

cdef int _scopy_missing_rows(cnp.float32_t* a, cnp.float32_t* b, int* missing, int n, int m)
cdef int _dcopy_missing_rows(cnp.float64_t* a, cnp.float64_t* b, int* missing, int n, int m)
cdef int _ccopy_missing_rows(cnp.complex64_t* a, cnp.complex64_t* b, int* missing, int n, int m)
cdef int _zcopy_missing_rows(cnp.complex128_t* a, cnp.complex128_t* b, int* missing, int n, int m)

cdef int _scopy_missing_cols(cnp.float32_t* a, cnp.float32_t* b, int* missing, int n, int m)
cdef int _dcopy_missing_cols(cnp.float64_t* a, cnp.float64_t* b, int* missing, int n, int m)
cdef int _ccopy_missing_cols(cnp.complex64_t* a, cnp.complex64_t* b, int* missing, int n, int m)
cdef int _zcopy_missing_cols(cnp.complex128_t* a, cnp.complex128_t* b, int* missing, int n, int m)

cpdef int scopy_missing_matrix(cnp.float32_t[::1, :, :] A, cnp.float32_t[::1, :, :] B, int[::1, :] missing, int copy_rows, int copy_cols, int diagonal) except *
cpdef int dcopy_missing_matrix(cnp.float64_t[::1, :, :] A, cnp.float64_t[::1, :, :] B, int[::1, :] missing, int copy_rows, int copy_cols, int diagonal) except *
cpdef int ccopy_missing_matrix(cnp.complex64_t[::1, :, :] A, cnp.complex64_t[::1, :, :] B, int[::1, :] missing, int copy_rows, int copy_cols, int diagonal) except *
cpdef int zcopy_missing_matrix(cnp.complex128_t[::1, :, :] A, cnp.complex128_t[::1, :, :] B, int[::1, :] missing, int copy_rows, int copy_cols, int diagonal) except *

cpdef int scopy_missing_vector(cnp.float32_t[::1, :] A, cnp.float32_t[::1, :] B, int[::1, :] missing) except *
cpdef int dcopy_missing_vector(cnp.float64_t[::1, :] A, cnp.float64_t[::1, :] B, int[::1, :] missing) except *
cpdef int ccopy_missing_vector(cnp.complex64_t[::1, :] A, cnp.complex64_t[::1, :] B, int[::1, :] missing) except *
cpdef int zcopy_missing_vector(cnp.complex128_t[::1, :] A, cnp.complex128_t[::1, :] B, int[::1, :] missing) except *

cdef int _scopy_index_diagonal(cnp.float32_t* a, cnp.float32_t* b, int* index, int n)
cdef int _dcopy_index_diagonal(cnp.float64_t* a, cnp.float64_t* b, int* index, int n)
cdef int _ccopy_index_diagonal(cnp.complex64_t* a, cnp.complex64_t* b, int* index, int n)
cdef int _zcopy_index_diagonal(cnp.complex128_t* a, cnp.complex128_t* b, int* index, int n)

cdef int _scopy_index_submatrix(cnp.float32_t* a, cnp.float32_t* b, int* index, int n)
cdef int _dcopy_index_submatrix(cnp.float64_t* a, cnp.float64_t* b, int* index, int n)
cdef int _ccopy_index_submatrix(cnp.complex64_t* a, cnp.complex64_t* b, int* index, int n)
cdef int _zcopy_index_submatrix(cnp.complex128_t* a, cnp.complex128_t* b, int* index, int n)

cdef int _scopy_index_rows(cnp.float32_t* a, cnp.float32_t* b, int* index, int n, int m)
cdef int _dcopy_index_rows(cnp.float64_t* a, cnp.float64_t* b, int* index, int n, int m)
cdef int _ccopy_index_rows(cnp.complex64_t* a, cnp.complex64_t* b, int* index, int n, int m)
cdef int _zcopy_index_rows(cnp.complex128_t* a, cnp.complex128_t* b, int* index, int n, int m)

cdef int _scopy_index_cols(cnp.float32_t* a, cnp.float32_t* b, int* index, int n, int m)
cdef int _dcopy_index_cols(cnp.float64_t* a, cnp.float64_t* b, int* index, int n, int m)
cdef int _ccopy_index_cols(cnp.complex64_t* a, cnp.complex64_t* b, int* index, int n, int m)
cdef int _zcopy_index_cols(cnp.complex128_t* a, cnp.complex128_t* b, int* index, int n, int m)

cpdef int scopy_index_matrix(cnp.float32_t[::1, :, :] A, cnp.float32_t[::1, :, :] B, int[::1, :] index, int copy_rows, int copy_cols, int diagonal) except *
cpdef int dcopy_index_matrix(cnp.float64_t[::1, :, :] A, cnp.float64_t[::1, :, :] B, int[::1, :] index, int copy_rows, int copy_cols, int diagonal) except *
cpdef int ccopy_index_matrix(cnp.complex64_t[::1, :, :] A, cnp.complex64_t[::1, :, :] B, int[::1, :] index, int copy_rows, int copy_cols, int diagonal) except *
cpdef int zcopy_index_matrix(cnp.complex128_t[::1, :, :] A, cnp.complex128_t[::1, :, :] B, int[::1, :] index, int copy_rows, int copy_cols, int diagonal) except *

cpdef int scopy_index_vector(cnp.float32_t[::1, :] A, cnp.float32_t[::1, :] B, int[::1, :] index) except *
cpdef int dcopy_index_vector(cnp.float64_t[::1, :] A, cnp.float64_t[::1, :] B, int[::1, :] index) except *
cpdef int ccopy_index_vector(cnp.complex64_t[::1, :] A, cnp.complex64_t[::1, :] B, int[::1, :] index) except *
cpdef int zcopy_index_vector(cnp.complex128_t[::1, :] A, cnp.complex128_t[::1, :] B, int[::1, :] index) except *
