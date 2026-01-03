"""A bunch of MPI utilities."""

from contextlib import contextmanager
from importlib.util import find_spec
import io
import os
import sys
import traceback
import unittest
import functools

from openmdao.core.analysis_error import AnalysisError
from openmdao.utils.notebook_utils import notebook


_under_mpi_cache = None
"""True if running under MPI with multiple processes, otherwise False."""


def under_mpi():
    """
    Check if running under MPI with multiple processes.

    Uses a fast-path check via environment variables for the common case
    of not running under MPI. Only imports mpi4py if evidence suggests
    we might be running under MPI.

    Returns
    -------
    bool
        True if running under MPI with multiple processes, otherwise False.
    """
    global _under_mpi_cache

    if _under_mpi_cache is not None:
        return _under_mpi_cache

    # Check for explicit override
    openmdao_use_mpi = os.environ.get('OPENMDAO_USE_MPI', '').lower()
    if openmdao_use_mpi in ('1', 'true', 'yes', 'always'):
        _under_mpi_cache = True
        return _under_mpi_cache
    elif openmdao_use_mpi in ('0', 'false', 'no', 'never'):
        _under_mpi_cache = False
        return _under_mpi_cache

    # Fast path: Check common MPI environment variables
    # Most MPI runs will set at least one of these
    mpi_size_vars = [
        'OMPI_COMM_WORLD_SIZE',  # OpenMPI
        'PMI_SIZE',               # MPICH, Intel MPI, MVAPICH, MS-MPI
        'MPI_LOCALNRANKS',        # Various implementations
        'SLURM_NTASKS',           # SLURM scheduler
        'MP_PROCS',               # IBM Spectrum MPI
        'FJMPI_PROCS',            # Fujitsu MPI
        'NECMPI_PROCS',           # NEC MPI
    ]

    for var in mpi_size_vars:
        value = os.environ.get(var)
        if value:
            try:
                size = int(value)
                _under_mpi_cache = size > 1
                return _under_mpi_cache
            except ValueError:
                pass

    # Check for MPI presence indicators (suggest MPI but don't give size)
    mpi_presence_vars = [
        'MPI_SHEPHERD',  # SGI MPT
        'MPI_ROOT',      # HP MPI
        'MPI_FLAGS',     # Platform MPI
    ]

    has_mpi_indicator = any(var in os.environ for var in mpi_presence_vars)

    # If no MPI environment variables at all, assume not under MPI
    # This is the fast path for the common case
    if not has_mpi_indicator:
        # Double-check: is mpi4py even installed?
        if find_spec("mpi4py") is None:
            _under_mpi_cache = False
            return _under_mpi_cache

        # mpi4py exists but no env vars - likely not under MPI
        # but could be a weird setup, so default to False
        _under_mpi_cache = False
        return _under_mpi_cache

    # We found MPI presence indicators but not size
    # Need to actually check via mpi4py (slow path)
    try:
        from mpi4py import MPI
        _under_mpi_cache = MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        _under_mpi_cache = False

    return _under_mpi_cache


def _redirect_streams(to_fd):
    """
    Redirect stdout/stderr to the given file descriptor.

    Based on: http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/.

    Parameters
    ----------
    to_fd : int
        File descriptor to redirect to.
    """
    # get fd for sys.stdout/err,
    #   if not a valid stream object (e.g. subprocess 'DevNull') use os.devnull
    # Flush and close sys.stdout/err - also closes the file descriptors (fd)
    try:
        original_stdout_fd = sys.stdout.fileno()
        sys.stdout.close()
    except (AttributeError, io.UnsupportedOperation):
        with open(os.devnull) as devnull:
            original_stdout_fd = devnull.fileno()

    try:
        original_stderr_fd = sys.stderr.fileno()
        sys.stderr.close()
    except (AttributeError, io.UnsupportedOperation):
        with open(os.devnull) as devnull:
            original_stderr_fd = devnull.fileno()

    # Make original_stdout_fd point to the same file as to_fd
    os.dup2(to_fd, original_stdout_fd)
    os.dup2(to_fd, original_stderr_fd)

    # Create a new sys.stdout that points to the redirected fd
    sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))
    sys.stderr = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))


def use_proc_files():
    """
    Cause stdout/err from each MPI process to be written to [rank].out.
    """
    if MPI is not None:
        working_dir = os.environ.get('PROC_FILES_DIR')
        if not working_dir:
            ofile = open("%d.out" % MPI.COMM_WORLD.rank, 'wb')
        else:
            if not os.path.isdir(working_dir):
                raise RuntimeError("directory '%s' does not exist." % working_dir)
            ofile = open(os.path.join(working_dir, "%d.out" % MPI.COMM_WORLD.rank), 'wb')
        _redirect_streams(ofile.fileno())


def check_mpi_env():
    """
    Determine if the environment variable governing MPI usage is set.

    Returns
    -------
    bool
        True if MPI is required, False if it's to be skipped, None if not set.
    """
    mpi_selection = os.environ.get('OPENMDAO_USE_MPI', None)

    # If OPENMDAO_USE_MPI is set to a postive value, the run will fail
    # immediately if the import fails
    if str(mpi_selection).lower() in ['always', '1', 'true', 'yes', 'y', 'on']:
        return True

    # If set to something else, no import is attempted.
    if mpi_selection is not None:
        return False

    # If unset, the import will be attempted but give no warning if it fails.
    return None


use_mpi = check_mpi_env()
if use_mpi is True:
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError("Importing MPI failed and OPENMDAO_USE_MPI is true.")
elif use_mpi is False:
    MPI = None
else:
    try:
        if under_mpi():
            from mpi4py import MPI
        else:
            MPI = None

    except ImportError:
        MPI = None


if MPI:
    def debug(*msg):  # pragma: no cover
        """
        Print debug message to stdout.

        Parameters
        ----------
        *msg : tuple of str
            Strings to be printed.
        """
        newmsg = ["%d: " % MPI.COMM_WORLD.rank] + list(msg)
        for m in newmsg:
            sys.stdout.write("%s " % m)
        sys.stdout.write('\n')
        sys.stdout.flush()
else:
    def debug(*msg):  # pragma: no cover
        """
        Print debug message to stdout.

        Parameters
        ----------
        *msg : tuple of str
            Strings to be printed.
        """
        for m in msg:
            sys.stdout.write("%s " % str(m))
        sys.stdout.write('\n')


class FakeComm(object):
    """
    Fake MPI communicator class used if mpi4py is not installed.

    Attributes
    ----------
    rank : int
        index of current proc; value is 0 because there is only 1 proc.
    size : int
        number of procs in the comm; value is 1 since MPI is not available.
    """

    def __init__(self):
        """
        Initialize attributes.
        """
        self.rank = 0
        self.size = 1


@contextmanager
def multi_proc_fail_check(comm):  # pragma no cover
    """
    Raise an AnalysisError on all procs if it is raised on one.

    Wrap this around code that you want to globally fail if it fails
    on any MPI process in comm.  If not running under MPI, don't
    handle any exceptions.

    Parameters
    ----------
    comm : MPI communicator or None
        Communicator from the ParallelGroup that owns the calling solver.

    Yields
    ------
    None
    """
    if MPI is None:
        yield
    else:
        try:
            yield
        except AnalysisError:
            msg = traceback.format_exc()
        else:
            msg = ''

        fails = comm.allgather(msg)

        for i, f in enumerate(fails):
            if f:
                raise AnalysisError("AnalysisError raised in rank %d: traceback follows\n%s"
                                    % (i, f))


@contextmanager
def multi_proc_exception_check(comm):  # pragma no cover
    """
    Raise an exception on all procs if it is raised on one.

    Exception raised will be the one from the lowest rank where an exception occurred.

    Wrap this around code that you want to globally fail if it fails
    on any MPI process in comm.  If not running under MPI, don't
    handle any exceptions.

    Parameters
    ----------
    comm : MPI communicator or None
        Communicator from the ParallelGroup that owns the calling solver.

    Yields
    ------
    None
    """
    if MPI is None or comm is None or comm.size == 1:
        yield
    else:
        try:
            yield
        except Exception:
            exc_type, exc, tb = sys.exc_info()
            fail = 1
        else:
            fail = 0

        failed = comm.allreduce(fail)
        if failed:
            if fail:
                info = (MPI.COMM_WORLD.rank, exc_type, ''.join(traceback.format_tb(tb)),
                        exc)
            else:
                info = None

            gathered = [tup for tup in comm.allgather(info) if tup is not None]
            ranks = sorted(set([tup[0] for tup in gathered]))
            _, exc_type, tbtext, exc = gathered[0]

            if comm.rank == 0:
                raise exc_type(f"Exception raised on ranks {ranks}: first traceback:\n{tbtext}")
            else:
                raise exc_type(f"Exception raised on ranks {ranks}: {exc}")


if MPI:
    def check_mpi_exceptions(fn):
        """
        Wrap a function in multi_proc_exception_check.

        This should be used only as a method decorator on an instance that
        has a 'comm' attribute that refers to an MPI communicator.

        Parameters
        ----------
        fn : function
            The function being checked for possible memory leaks.

        Returns
        -------
        function
            A wrapper for fn that reports possible memory leaks.
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with multi_proc_exception_check(args[0].comm):
                return fn(*args, **kwargs)
        return wrapper
else:
    # do nothing decorator
    def check_mpi_exceptions(fn):
        """
        Wrap a function in multi_proc_exception_check.

        This does nothing if not running under MPI.

        Parameters
        ----------
        fn : function
            The function being checked for possible memory leaks.

        Returns
        -------
        function
            A wrapper for fn that reports possible memory leaks.
        """
        return fn


if MPI:
    def mpirun_tests():  # pragma no cover
        """
        Run individual tests under MPI.

        This is used in the "if __name__ == '__main__'" block to run an
        individual test in the file under MPI.  Note that if the test
        file has not been run under mpirun, this reverts to running
        unittest.main().
        """
        mod = __import__('__main__')

        tests = [arg for arg in sys.argv[1:] if not arg.startswith('-')]

        if tests:
            for test in tests:
                parts = test.split('.', 1)
                if len(parts) == 2:
                    tcase_name, method_name = parts
                    testcase = getattr(mod, tcase_name)(methodName=method_name)
                    setup = getattr(testcase, 'setUp', None)
                    if setup is not None:
                        setup()
                    getattr(testcase, method_name)()
                    teardown = getattr(testcase, 'tearDown', None)
                    if teardown:
                        teardown()
                else:
                    funcname = parts[0]
                    getattr(mod, funcname)()
        else:
            unittest.main()
else:
    mpirun_tests = unittest.main


if os.environ.get('USE_PROC_FILES') or os.environ.get('PROC_FILES_DIR'):
    if not notebook:
        use_proc_files()
