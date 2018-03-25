# Math Functions
# Real and complex log and abs functions
from libc.math cimport log as dlog, abs as dabs, exp as dexp
cimport numpy as cnp

cdef extern from "numpy/npy_math.h":
    cnp.float64_t NPY_PI
    # npy_cabs and npy_clog are needed to define zabs, zlog, are not cimported
    # _from_ here
    cnp.float64_t npy_cabs(cnp.npy_cdouble z)
    cnp.npy_cdouble npy_clog(cnp.npy_cdouble z)


cdef inline cnp.float64_t zabs(cnp.complex128_t z):
    return npy_cabs((<cnp.npy_cdouble *> &z)[0])


cdef inline cnp.complex128_t zlog(cnp.complex128_t z):
    cdef:
        cnp.npy_cdouble x
    x = npy_clog((<cnp.npy_cdouble*> &z)[0])
    return (<cnp.complex128_t *> &x)[0]
