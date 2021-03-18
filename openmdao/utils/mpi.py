"""A bunch of MPI utilities."""

from contextlib import contextmanager
import io
import os
import sys
import traceback
import unittest
import functools

from openmdao.core.analysis_error import AnalysisError


def _redirect_streams(to_fd):
    """
    Redirect stdout/stderr to the given file descriptor.

    Based on: http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/.

    Parameters
    ----------
    to_fd : int
        File descriptor to redirect to.
    """
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Flush and close sys.stdout/err - also closes the file descriptors (fd)
    sys.stdout.close()
    sys.stderr.close()

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


# Attempt to import mpi4py.
# If OPENMDAO_REQUIRE_MPI is set to a recognized positive value, attempt import
# and raise exception on failure. If set to anything else, no import is attempted.
if 'OPENMDAO_REQUIRE_MPI' in os.environ:
    if os.environ['OPENMDAO_REQUIRE_MPI'].lower() in ['always', '1', 'true', 'yes']:
        from mpi4py import MPI
    else:
        MPI = None
# If OPENMDAO_REQUIRE_MPI is unset, attempt to import mpi4py, but continue on failure
# with a notification.
else:
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
        sys.stdout.write("Unable to import mpi4py. Parallel processing unavailable.\n")
        sys.stdout.flush()
    else:
        # If the import succeeded, but it doesn't look like a parallel
        # run was intended, don't use MPI
        if MPI.COMM_WORLD.size == 1:
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
def multi_proc_fail_check(comm):
    """
    Raise an AnalysisError on all procs if it is raised on one.

    Wrap this around code that you want to globally fail if it fails
    on any MPI process in comm.  If not running under MPI, don't
    handle any exceptions.

    Parameters
    ----------
    comm : MPI communicator or None
        Communicator from the ParallelGroup that owns the calling solver.
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
def multi_proc_exception_check(comm):
    """
    Raise an exception on all procs if it is raised on one.

    Wrap this around code that you want to globally fail if it fails
    on any MPI process in comm.  If not running under MPI, don't
    handle any exceptions.

    Parameters
    ----------
    comm : MPI communicator or None
        Communicator from the ParallelGroup that owns the calling solver.
    """
    if MPI is None or comm is None or comm.size == 1:
        yield
    else:
        try:
            yield
        except Exception:
            exc = sys.exc_info()
            fail = 1
        else:
            fail = 0

        failed = comm.allreduce(fail)
        if failed:
            if fail:
                msg = f"{exc[1]}"
            else:
                msg = None
            allmsgs = comm.allgather(msg)
            if fail:
                msg = f"Exception raised on rank {comm.rank}: {exc[1]}"
                raise exc[0](msg).with_traceback(exc[2])
            else:
                for m in allmsgs:
                    if m is not None:
                        raise RuntimeError(f"Exception raised on other rank: {m}.")


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
    def mpirun_tests():
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
    use_proc_files()
