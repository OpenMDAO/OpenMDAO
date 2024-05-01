"""Define utils for use in testing."""
import json
import functools
import builtins
import os
import re
import sys
from itertools import zip_longest
from contextlib import contextmanager

import numpy as np
from scipy.sparse import coo_matrix

try:
    from parameterized import parameterized
except ImportError:
    parameterized = None

from openmdao.utils.general_utils import env_truthy, env_none


def _new_setup(self):
    import os
    import tempfile

    from openmdao.utils.mpi import MPI, multi_proc_exception_check
    self.startdir = os.getcwd()
    if MPI is None:
        self.tempdir = tempfile.mkdtemp(prefix='testdir-')
    elif MPI.COMM_WORLD.rank == 0:
        self.tempdir = tempfile.mkdtemp(prefix='testdir-')
        MPI.COMM_WORLD.bcast(self.tempdir, root=0)
    else:
        self.tempdir = MPI.COMM_WORLD.bcast(None, root=0)

    os.chdir(self.tempdir)
    if hasattr(self, 'original_setUp'):
        if MPI is not None and MPI.COMM_WORLD.size > 1:
            with multi_proc_exception_check(MPI.COMM_WORLD):
                self.original_setUp()
        else:
            self.original_setUp()


def _new_teardown(self):
    import os
    import shutil

    from openmdao.utils.mpi import MPI, multi_proc_exception_check
    if hasattr(self, 'original_tearDown'):
        if MPI is not None and MPI.COMM_WORLD.size > 1:
            with multi_proc_exception_check(MPI.COMM_WORLD):
                self.original_tearDown()
        else:
            self.original_tearDown()

    os.chdir(self.startdir)

    if MPI is None:
        rank = 0
    else:
        # make sure everyone's out of that directory before rank 0 deletes it
        MPI.COMM_WORLD.barrier()
        rank = MPI.COMM_WORLD.rank

    if rank == 0:
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass


def use_tempdirs(cls):
    """
    Decorate each test in a unittest.TestCase so it runs in its own directory.

    TestCase methods setUp and tearDown are replaced with _new_setup and
    _new_teardown, above.  Method _new_setup creates a temporary directory
    in which to run the test, stores it in self.tempdir, and then calls
    the original setUp method.  Method _new_teardown first runs the original
    tearDown method, and then returns to the original starting directory
    and deletes the temporary directory.

    Attributes
    ----------
    cls : TestCase
        TestCase being decorated to use a tempdir for each test.

    Returns
    -------
    TestCase
        The decorated TestCase class.
    """
    if env_truthy('USE_TEMPDIRS') or env_none('USE_TEMPDIRS'):
        if getattr(cls, 'setUp', None):
            setattr(cls, 'original_setUp', getattr(cls, 'setUp'))
        setattr(cls, 'setUp', _new_setup)

        if getattr(cls, 'tearDown', None):
            setattr(cls, 'original_tearDown', getattr(cls, 'tearDown'))
        setattr(cls, 'tearDown', _new_teardown)

    return cls


def require_pyoptsparse(optimizer=None):
    """
    Decorate test to raise a skiptest if a required pyoptsparse optimizer cannot be imported.

    Parameters
    ----------
    optimizer : str
        Pyoptsparse optimizer string. Default is None, which just checks for pyoptsparse.

    Returns
    -------
    TestCase or TestCase.method
        The decorated TestCase class or method.
    """
    def decorator(obj):

        import unittest
        try:
            from pyoptsparse import OPT

        except Exception:
            msg = "pyoptsparse is not installed."

            if not isinstance(obj, type):
                @functools.wraps(obj)
                def skip_wrapper(*args, **kwargs):
                    raise unittest.SkipTest(msg)
                obj = skip_wrapper
            obj.__unittest_skip__ = True
            obj.__unittest_skip_why__ = msg
            return obj

        if optimizer:
            try:
                OPT(optimizer)
            except Exception:
                msg = "pyoptsparse is not providing %s" % optimizer

                if not isinstance(obj, type):
                    @functools.wraps(obj)
                    def skip_wrapper(*args, **kwargs):
                        raise unittest.SkipTest(msg)
                    obj = skip_wrapper
                obj.__unittest_skip__ = True
                obj.__unittest_skip_why__ = msg

        return obj
    return decorator


if parameterized:
    def parameterized_name(testcase_func, num, param):
        """
        Generate a name for a parameterized test from the parameters.

        Parameters
        ----------
        testcase_func : str
            the root test function name
        num : int
            parameter number
        param : any
            parameter value

        Returns
        -------
        TestCase or TestCase.method
            The decorated TestCase class or method.
        """
        return "%s_%s" % (
            testcase_func.__name__,
            parameterized.to_safe_name("_".join(str(x) for x in param.args)),
        )
else:
    parameterized_name = None


class set_env_vars(object):
    """
    Decorate a function to temporarily set some environment variables.

    Parameters
    ----------
    **envs : dict
        Keyword args corresponding to environment variables to set.

    Attributes
    ----------
    envs : dict
        Saved mapping of environment var name to value.
    """

    def __init__(self, **envs):
        """
        Initialize attributes.
        """
        self.envs = envs

    def __call__(self, fnc):
        """
        Apply the decorator.

        Parameters
        ----------
        fnc : function
            The function being wrapped.
        """
        @functools.wraps(fnc)
        def wrap(*args, **kwargs):
            saved = {}
            try:
                for k, v in self.envs.items():
                    saved[k] = os.environ.get(k)
                    os.environ[k] = v  # will raise exception if v is not a string

                return fnc(*args, **kwargs)
            finally:
                # put environment back as it was
                for k, v in saved.items():
                    if v is None:
                        del os.environ[k]
                    else:
                        os.environ[k] = v

        return wrap


@contextmanager
def set_env_vars_context(**kwargs):
    """
    Context to temporarily set some environment variables.

    Parameters
    ----------
    **kwargs : dict
        Keyword args corresponding to environment variables to set.

    Yields
    ------
    None
    """
    saved = {}
    try:
        for k, v in kwargs.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v  # will raise exception if v is not a string

        yield

    finally:
        # put environment back as it was
        for k, v in saved.items():
            if v is None:
                del os.environ[k]
            else:
                os.environ[k] = v


@set_env_vars(OPENMDAO_CHECK_ALL_PARTIALS="1")
def force_check_partials(prob, *args, **kwargs):
    r"""
    Force the checking of partials even for components with _no_check_partials set.

    Parameters
    ----------
    prob : Problem
        The Problem being checked.
    *args : list
        Positional args.
    **kwargs : dict
        Keyword args.

    Returns
    -------
    dict
        Total derivative comparison data.
    """
    return prob.check_partials(*args, **kwargs)


class _ModelViewerDataTreeEncoder(json.JSONEncoder):
    """Special JSON encoder for writing model viewer data."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MissingImports(object):
    """
    ContextManager that emulates missing python packages or modules.

    Each import is checked to see if it starts with a missing import.

    For instance:

    >>> with MissingImports('matplotlib'):
    >>>    from matplotlib.pyplot import plt

    will fail because 'matplotlib.pyplot'.startswith('matplotlib') is True.

    This implementation modifies builtins.__import__ which is allowed but highly
    discouraged according to the documentation, but implementing a MetaPathFinder
    seemed like overkill.  Use at your own risk.

    Parameters
    ----------
    missing_imports : str or Sequence of str
        A string or sequence of strings that denotes modules that should appear to be absent
        for testing purposes.

    Attributes
    ----------
    missing_imports : str or Sequence of str
        A string or sequence of strings that denotes modules that should appear to be absent
        for testing purposes.
    _cached_import : None or builtin
        A cached import to emulate the missing import
    """

    def __init__(self, missing_imports):
        """
        Initialize attributes.
        """
        if isinstance(missing_imports, str):
            self.missing_imports = set([missing_imports])
        else:
            self.missing_imports = set(missing_imports)
        self._cached_import = None

    def __enter__(self):
        """
        Set cached import.
        """
        self._cached_import = builtins.__import__
        builtins.__import__ = self._emulate_missing_import

    def _emulate_missing_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        for mi in self.missing_imports:
            if name.startswith(mi):
                raise ImportError(f'No module named {name} due to missing import {mi}.')
        return self._cached_import(name, globals, locals, fromlist, level)

    def __exit__(self, type, value, traceback):
        """
        Exit the runtime context related to this object.

        Parameters
        ----------
        type : Exception class
            The type of the exception.
        value : Exception instance
            The exception instance raised.
        traceback : regex pattern
            Traceback object.
        """
        builtins.__import__ = self._cached_import


# this recognizes ints and floats with or without scientific notation.
# it does NOT recognize hex or complex numbers
num_rgx = re.compile(r"[-]?([0-9]+\.?[0-9]*|[0-9]*\.?[0-9]+)([eE][-+]?[0-9]+)?")


def snum_iter(s):
    """
    Iterate through a string, yielding numeric strings as numbers along with non-numeric strings.

    Parameters
    ----------
    s : str
        The string to iterate through.

    Yields
    ------
    str
        The next number or non-number.
    bool
        True if the string is a number, False otherwise.
    """
    if not s:
        return

    end = 0
    for m in num_rgx.finditer(s):
        mstart = m.start()

        if end != mstart:  # need to output the non-num string prior to this match
            yield (s[end:mstart], False)

        yield (float(m.group()), True)

        end = m.end()

    if end < len(s):  # yield any non-num at end of string
        yield (s[end:], False)


def snum_equal(s1, s2, atol=1e-6, rtol=1e-6):
    """
    Compare two strings, and if they contain numbers, compare the numbers subject to tolerance.

    Also compare the non-number parts of the strings exactly.

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    atol : float, optional
        Absolute tolerance. The default is 1e-6.
    rtol : float, optional
        Relative tolerance. The default is 1e-6.

    Returns
    -------
    bool
        True if the strings are equal within the tolerance, False otherwise.
    """
    for (s1, isnum1), (s2, isnum2) in zip_longest(snum_iter(s1), snum_iter(s2),
                                                  fillvalue=("", False)):
        if isnum1 and isnum2:
            if rtol is None and atol is None:
                if s1 != s2:
                    return False
            else:
                if rtol is not None and rel_num_diff(s1, s2) > rtol:
                    return False

                if atol is not None and abs(s1 - s2) > atol:
                    return False

        elif s1 != s2:
            return False

    return True


def rel_num_diff(n1, n2):
    """
    Return the relative numerical difference between two numbers.

    Parameters
    ----------
    n1 : float
        First number to compare.
    n2 : float
        Second number to compare.

    Returns
    -------
    float
        Relative difference between the numbers.
    """
    if n1 == 0.:
        return 0. if n2 == 0. else 1.0
    else:
        return abs(n2 - n1) / abs(n1)
