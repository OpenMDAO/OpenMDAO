"""Define utils for use in testing."""
import json
import functools
import os
import shutil
import re
from itertools import zip_longest
from contextlib import contextmanager
from pathlib import Path
import tempfile

import numpy as np

try:
    from parameterized import parameterized
except ImportError:
    parameterized = None

from openmdao.utils.general_utils import env_truthy, env_none
from openmdao.utils.mpi import MPI


def _cleanup_workdir(self):
    os.chdir(self.startdir)

    if self.old_workdir:
        os.environ['OPENMDAO_WORKDIR'] = self.old_workdir

    if MPI is None:
        rank = 0
    else:
        # make sure everyone's out of that directory before rank 0 deletes it
        MPI.COMM_WORLD.barrier()
        rank = MPI.COMM_WORLD.rank

    if rank == 0:
        if not os.environ.get('OPENMDAO_KEEPDIRS'):
            try:
                shutil.rmtree(self.tempdir)
            except OSError:
                pass


def get_tempdir():
    """
    Return the same tempdir across all MPI processes.

    Returns
    -------
    str
        The tempdir.
    """
    if MPI is None:
        return tempfile.mkdtemp()
    elif MPI.COMM_WORLD.rank == 0:
        tempdir = tempfile.mkdtemp()
        MPI.COMM_WORLD.bcast(tempdir, root=0)
        return tempdir
    else:
        return MPI.COMM_WORLD.bcast(None, root=0)


def _new_setup(self):
    import os

    from openmdao.utils.mpi import MPI, multi_proc_exception_check
    self.startdir = os.getcwd()
    self.old_workdir = os.environ.get('OPENMDAO_WORKDIR', '')

    self.tempdir = get_tempdir()

    os.chdir(self.tempdir)
    # on mac tempdir is a symlink which messes some things up, so
    # use resolve to get the real directory path
    os.environ['OPENMDAO_WORKDIR'] = str(Path(self.tempdir).resolve())
    try:
        if hasattr(self, 'original_setUp'):
            if MPI is not None and MPI.COMM_WORLD.size > 1:
                with multi_proc_exception_check(MPI.COMM_WORLD):
                    self.original_setUp()
            else:
                self.original_setUp()
    except Exception:
        _cleanup_workdir(self)
        raise


def _new_teardown(self):
    from openmdao.utils.mpi import MPI, multi_proc_exception_check

    try:
        if hasattr(self, 'original_tearDown'):
            if MPI is not None and MPI.COMM_WORLD.size > 1:
                with multi_proc_exception_check(MPI.COMM_WORLD):
                    self.original_tearDown()
            else:
                self.original_tearDown()
    finally:
        _cleanup_workdir(self)


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


def _fix_comp_check_data(data):
    """
    Modify the data dict to match the problem format if there is only one fd step size.

    Parameters
    ----------
    data : dict
        Dictionary containing derivative information keyed by subjac.
    """
    names = ['J_fd', 'tol violation', 'magnitude', 'directional_fd_fwd', 'directional_fd_rev',
             'vals_at_max_error', 'abs error', 'rel error']

    for name in names:
        if name in data:
            data[name] = data[name][0]
    if 'steps' in data:
        del data['steps']


def compare_prob_vs_comp_check_partials(probdata, compdata, comp):
    """
    Compare the check_partials output for a Problem and a Component.

    Parameters
    ----------
    probdata : dict
        Problem check_partials data.
    compdata : dict
        Component check_partials data.
    comp : Component
        The component being checked.

    Returns
    -------
    dict
        Comparison data.
    """
    from openmdao.utils.assert_utils import assert_near_equal

    try:
        probcompdata = probdata[comp.pathname]
    except KeyError:
        raise KeyError(f"Component '{comp.pathname}' not found in Problem check_partials data.")

    compdata = compdata[comp.pathname]

    # check that the keys are the same
    assert set(probcompdata.keys()) == set(compdata.keys()), \
        f"Subjac keys don't match for {comp.pathname}"

    # check that the values are the same
    for key, probval in probcompdata.items():
        compval = compdata[key]

        if 'steps' not in probval:
            # if there is only one FD step, 'steps' gets removed from problem data
            compval = compval.copy()
            _fix_comp_check_data(compval)

        # check that the keys are the same
        assert set(probval.keys()) == set(compval.keys()), \
            f"For subjac key '{key}', inner dict keys don't match: {sorted(probval.keys())} " \
            f"!= {sorted(compval.keys())}."

        for key2 in probval:
            probval2 = probval[key2]
            compval2 = compval[key2]

            if key2 in ('J_fd', 'J_fwd', 'denom_idx'):
                assert_near_equal(probval2, compval2, 1e-6, 1e-6)
            elif key2 in ('rows', 'cols'):
                if probval2 is None:
                    assert probval2 is compval2
                else:
                    assert np.all(probval2 == compval2)
            else:
                for i in range(3):
                    p = probval2[i]
                    c = compval2[i]
                    if p is None and c is None:
                        continue
                    assert_near_equal(p, c, 1e-6, 1e-6), \
                        f"For key '{key}/{key2}', {probval2} != {compval2}."


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
