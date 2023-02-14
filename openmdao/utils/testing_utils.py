"""Define utils for use in testing."""
import json
import functools
import builtins
import os
from contextlib import contextmanager
from openmdao.utils.general_utils import env_truthy, env_none

try:
    from parameterized import parameterized
except ImportError:
    parameterized = None

import numpy as np


def _new_setup(self):
    import os
    import tempfile

    from openmdao.utils.mpi import MPI
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
        self.original_setUp()


def _new_teardown(self):
    import os
    import shutil

    from openmdao.utils.mpi import MPI
    if hasattr(self, 'original_tearDown'):
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
