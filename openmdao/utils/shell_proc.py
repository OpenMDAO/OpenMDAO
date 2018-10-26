"""Some basic shell utilities, used for ExternalCodeComp mostly."""
import os
import signal
import subprocess
import sys
import time

PIPE = subprocess.PIPE
STDOUT = subprocess.STDOUT
DEV_NULL = 'nul:' if sys.platform == 'win32' else '/dev/null'


class CalledProcessError(subprocess.CalledProcessError):
    """
    :class:`subprocess.CalledProcessError` plus `errormsg` attribute.

    Attributes
    ----------
    errormsg : str
        Error message saved for string access.
    """

    def __init__(self, returncode, cmd, errormsg):
        """
        Initialize.

        Parameters
        ----------
        returncode : int
            Error code for this error.
        cmd : str or list
            If a string, then this is the command line to execute, and the
            :class:`subprocess.Popen` ``shell`` argument is set True.
            Otherwise, this is a list of arguments; the first is the command
            to execute.
        errormsg : str
            Error message for this error.
        """
        super(CalledProcessError, self).__init__(returncode, cmd)
        self.errormsg = errormsg

    def __str__(self):
        """
        Return string of error message.

        Returns
        -------
        str
            Error message.
        """
        return 'Command %r returned non-zero exit status %d: %s' % (self.cmd, self.returncode,
                                                                    self.errormsg)


class ShellProc(subprocess.Popen):
    """
    A slight modification to :class:`subprocess.Popen`.

    If `args` is a string, then the ``shell`` argument is set True.
    Updates a copy of ``os.environ`` with `env` and opens files for any
    stream which is a :class:`str`.

    Attributes
    ----------
    _stdin_arg : str, file, or int
        Save handle to make closing easier.
    _stdout_arg : str, file, or int
        Save handle to make closing easier.
    _stderr_arg : str, file, or int
        Save handle to make closing easier.
    _inp : str, file, or int
        Save handle to make closing easier.
    _out : str, file, or int
        Save handle to make closing easier.
    _err : str, file, or int
        Save handle to make closing easier.
    """

    def __init__(self, args, stdin=None, stdout=None, stderr=None, env=None,
                 universal_newlines=False):
        """
        Initialize.

        Parameters
        ----------
        args : str or list
            If a string, then this is the command line to execute and the
            :class:`subprocess.Popen` ``shell`` argument is set True.
            Otherwise, this is a list of arguments; the first is the command
            to execute.
        stdin : str, file, or int
            Specify handling of stdin stream. If a string, a file
            of that name is opened. Otherwise, see the :mod:`subprocess`
            documentation.
        stdout : str, file, or int
            Specify handling of stdout stream. If a string, a file
            of that name is opened. Otherwise, see the :mod:`subprocess`
            documentation.
        stderr : str, file, or int
            Specify handling of stderr stream. If a string, a file
            of that name is opened. Otherwise, see the :mod:`subprocess`
            documentation.
        env : dict
            Environment variables for the command.
        universal_newlines : bool
            Set to True to turn on universal newlines.
        """
        environ = os.environ.copy()
        if env:
            environ.update(env)

        self._stdin_arg = stdin
        self._stdout_arg = stdout
        self._stderr_arg = stderr

        if isinstance(stdin, str):
            self._inp = open(stdin, 'r')
        else:
            self._inp = stdin

        if isinstance(stdout, str):
            self._out = open(stdout, 'w')
        else:
            self._out = stdout

        if isinstance(stderr, str):
            self._err = open(stderr, 'w')
        else:
            self._err = stderr

        shell = isinstance(args, str)

        try:
            if sys.platform == 'win32':
                subprocess.Popen.__init__(self, args, stdin=self._inp,
                                          stdout=self._out, stderr=self._err,
                                          shell=shell, env=environ,
                                          universal_newlines=universal_newlines)
            else:
                subprocess.Popen.__init__(self, args, stdin=self._inp,
                                          stdout=self._out, stderr=self._err,
                                          shell=shell, env=environ,
                                          universal_newlines=universal_newlines,
                                          # setsid to put this and any children in
                                          # same process group so we can kill them
                                          # all if necessary
                                          preexec_fn=os.setsid)

        except Exception:
            self.close_files()
            raise

    def close_files(self):
        """
        Close files that were implicitly opened.
        """
        if isinstance(self._stdin_arg, str):
            self._inp.close()
        if isinstance(self._stdout_arg, str):
            self._out.close()
        if isinstance(self._stderr_arg, str):
            self._err.close()

    def terminate(self, timeout=None):
        """
        Stop child process.

        If `timeout` is specified, then :meth:`wait` will be called to wait for the process
        to terminate.

        Parameters
        ----------
        timeout : float (seconds)
            Maximum time to wait for the process to stop.
            A value of zero implies an infinite maximum wait.

        Returns
        -------
        int
            Return Code
        str
            Error Message
        """
        if sys.platform == 'win32':
            subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=self.pid))
        else:
            os.killpg(os.getpgid(self.pid), signal.SIGTERM)

        if timeout is not None:
            return self.wait(timeout=timeout)

    def wait(self, poll_delay=0., timeout=0.):
        """
        Poll for command completion or timeout.

        Closes any files implicitly opened.

        Parameters
        ----------
        poll_delay : float (seconds)
            Time to delay between polling for command completion.
            A value of zero uses an internal default.
        timeout : float (seconds)
            Maximum time to wait for command completion.
            A value of zero implies an infinite maximum wait.

        Returns
        -------
        int
            Return Code
        str
            Error Message
        """
        return_code = None
        try:
            if poll_delay <= 0:
                poll_delay = max(0.1, timeout / 100.)
                poll_delay = min(10., poll_delay)
            npolls = int(timeout / poll_delay) + 1

            time.sleep(poll_delay)
            return_code = self.poll()
            while return_code is None:
                npolls -= 1
                if (timeout > 0) and (npolls < 0):
                    self.terminate()
                    break
                time.sleep(poll_delay)
                return_code = self.poll()
        finally:
            self.close_files()

        # self.returncode set by self.poll().
        if return_code is not None:
            self.errormsg = self.error_message(return_code)
        else:
            self.errormsg = 'Timed out'
        return (return_code, self.errormsg)

    def error_message(self, return_code):
        """
        Return error message for `return_code`.

        The error messages are derived from the operating system definitions.
        Some programs don't necessarily return exit codes conforming to these
        definitions.

        Parameters
        ----------
        return_code : int
            Return code from :meth:`poll`.

        Returns
        -------
        str
            Error Message string.
        """
        error_msg = ''
        if return_code:
            if return_code > 0:
                try:
                    err_msg = os.strerror(return_code)
                except OverflowError:
                    err_msg = "Process exited with unknown return code {}".format(return_code)
            elif sys.platform != 'win32':
                sig = -return_code
                if sig < signal.NSIG:
                    for item in signal.__dict__.keys():
                        if item.startswith('SIG'):
                            if getattr(signal, item) == sig:
                                error_msg = ': %s' % item
                                break

        return error_msg


def call(args, stdin=None, stdout=None, stderr=None, env=None,
         poll_delay=0., timeout=0.):
    """
    Run command with arguments.

    Parameters
    ----------
    args : str or list
        If a string, then this is the command line to execute and the
        :class:`subprocess.Popen` ``shell`` argument is set True.
        Otherwise, this is a list of arguments; the first is the command
        to execute.
    stdin : str, file, or int
        Specify handling of stdin stream. If a string, a file
        of that name is opened. Otherwise, see the :mod:`subprocess`
        documentation.
    stdout : str, file, or int
        Specify handling of stdout stream. If a string, a file
        of that name is opened. Otherwise, see the :mod:`subprocess`
        documentation.
    stderr : str, file, or int
        Specify handling of stderr stream. If a string, a file
        of that name is opened. Otherwise, see the :mod:`subprocess`
        documentation.
    env : dict
        Environment variables for the command.
    poll_delay : float (seconds)
        Time to delay between polling for command completion.
        A value of zero uses an internal default.
    timeout : float (seconds)
        Maximum time to wait for command completion.
        A value of zero implies an infinite maximum wait.

    Returns
    -------
    int
        Return Code
    str
        Error Message
    """
    process = ShellProc(args, stdin, stdout, stderr, env)
    return process.wait(poll_delay, timeout)


def check_call(args, stdin=None, stdout=None, stderr=None, env=None,
               poll_delay=0., timeout=0.):
    """
    Run command with arguments.

    Raises :class:`CalledProcessError` if process returns an error code.

    Parameters
    ----------
    args : str or list
        If a string, then this is the command line to execute, and the
        :class:`subprocess.Popen` ``shell`` argument is set True.
        Otherwise, this is a list of arguments; the first is the command
        to execute.
    stdin : str, file, or int
        Specify handling of stdin stream. If a string, a file
        of that name is opened. Otherwise, see the :mod:`subprocess`
        documentation.
    stdout : str, file, or int
        Specify handling of stdout stream. If a string, a file
        of that name is opened. Otherwise, see the :mod:`subprocess`
        documentation.
    stderr : str, file, or int
        Specify handling of stderr stream. If a string, a file
        of that name is opened. Otherwise, see the :mod:`subprocess`
        documentation.
    env : dict
        Environment variables for the command.
    poll_delay : float (seconds)
        Time to delay between polling for command completion.
        A value of zero uses an internal default.
    timeout : float (seconds)
        Maximum time to wait for command completion.
        A value of zero implies an infinite maximum wait.
    """
    process = ShellProc(args, stdin, stdout, stderr, env)
    return_code, error_msg = process.wait(poll_delay, timeout)
    if return_code:
        raise CalledProcessError(return_code, args, error_msg)
