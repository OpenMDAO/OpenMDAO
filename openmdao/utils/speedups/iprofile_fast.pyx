
from cpython.ref cimport PyObject
from cpython.string cimport PyString_AsString

from cpython.pystate cimport (
    Py_tracefunc,
    PyTrace_CALL, PyTrace_EXCEPTION, PyTrace_LINE, PyTrace_RETURN,
    PyTrace_C_CALL, PyTrace_C_EXCEPTION, PyTrace_C_RETURN
)

from libc.stdio cimport printf

cdef extern from "Python.h":

    ctypedef struct PyCodeObject:
        int       co_argcount
        int       co_nlocals
        int       co_stacksize
        int       co_flags
        PyObject *co_code
        PyObject *co_consts
        PyObject *co_names
        PyObject *co_varnames
        PyObject *co_freevars
        PyObject *co_cellvars
        PyObject *co_filename
        PyObject *co_name
        int       co_firstlineno
        PyObject *co_lnotab


cdef extern from "frameobject.h":

    ctypedef struct PyFrameObject:
        PyFrameObject *f_back
        PyCodeObject  *f_code
        PyObject *f_builtins
        PyObject *f_globals
        PyObject *f_locals
        PyObject *f_trace
        PyObject *f_exc_type
        PyObject *f_exc_value
        PyObject *f_exc_traceback
        int f_lasti
        int f_lineno
        int f_restricted
        int f_iblock
        int f_nlocals
        int f_ncells
        int f_nfreevars
        int f_stacksize



ctypedef extern int (*Py_tracefunc)(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg)

cdef extern void PyEval_SetProfile(Py_tracefunc, PyObject*)


cdef public int profiler_callback(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg):

    if what == PyTrace_CALL:
        fname = frame.f_code.co_name
        printf((<object>fname).encode('UTF-8'))

    elif what == PyTrace_RETURN:
        printf('return\n')


    return 0


cdef void c_activate_iprof(PyObject *obj):
    PyEval_SetProfile(<Py_tracefunc>profiler_callback, <PyObject*>obj)

def activate_iprof(obj):
    c_activate_iprof(<PyObject*>obj)

