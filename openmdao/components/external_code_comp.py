"""Define the ExternalCodeComp class."""
from __future__ import print_function

import os
import sys

from six import iteritems, itervalues

import numpy.distutils
from numpy.distutils.exec_command import find_executable

from openmdao.core.analysis_error import AnalysisError
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.shell_proc import STDOUT, DEV_NULL, ShellProc
from openmdao.utils.general_utils import warn_deprecation


class ExternalCodeComp(ExplicitComponent):
    """
    Run an external code as a component.

    Default stdin is the 'null' device, default stdout is the console, and
    default stderr is ``error.out``.

    Attributes
    ----------
    stdin : str or file object
        Input stream external code reads from.
    stdout : str or file object
        Output stream external code writes to.
    stderr : str or file object
        Error stream external code writes to.
    DEV_NULL : File object
        NULL device.
    STDOUT : File object
        Special value that can be used as the stderr argument to Popen and indicates
        that standard error should go into the same handle as standard output.
    return_code : int
        Exit status of the child process.

    Options
    -------
    options['command'] :  list([])
        Command to be executed. Command must be a list of command line args.
    options['env_vars'] :  dict({})
        Environment variables required by the command
    options['external_input_files'] :  list([])
        (optional) list of input file names to check the existence of before solve_nonlinear
    options['external_output_files'] :  list([])
        (optional) list of input file names to check the existence of after solve_nonlinear
    options['poll_delay'] :  float(0.0)
        Delay between polling for command completion. A value of zero will use
        an internally computed default.
    options['timeout'] :  float(0.0)
        Maximum time in seconds to wait for command completion. A value of zero
        implies an infinite wait. If the timeout interval is exceeded, an
        AnalysisError will be raised.
    options['fail_hard'] :  bool(True)
        Behavior on error returned from code, either raise a 'hard' error (RuntimeError) if True
        or a 'soft' error (AnalysisError) if False.
    options['allowed_return_codes'] :  list or set of int [0]
        List of return codes that are considered successful.
    """

    def __init__(self):
        """
        Intialize the ExternalCodeComp component.
        """
        super(ExternalCodeComp, self).__init__()

        self.STDOUT = STDOUT
        self.DEV_NULL = DEV_NULL

        # Input options for this Component
        self.options.declare('command', [], desc='command to be executed')
        self.options.declare('env_vars', {},
                             desc='Environment variables required by the command')
        self.options.declare('poll_delay', 0.0, lower=0.0,
                             desc='Delay between polling for command completion. A value of zero '
                             'will use an internally computed default')
        self.options.declare('timeout', 0.0, lower=0.0,
                             desc='Maximum time to wait for command completion. A value of zero '
                             'implies an infinite wait')
        self.options.declare('external_input_files', [],
                             desc='(optional) list of input file names to check the existence '
                             'of before solve_nonlinear')
        self.options.declare('external_output_files', [],
                             desc='(optional) list of input file names to check the existence of '
                             'after solve_nonlinear')
        self.options.declare('fail_hard', True,
                             desc="If True, external code errors raise a 'hard' exception "
                             "(RuntimeError).  Otherwise raise a 'soft' exception "
                             "(AnalysisError).")
        self.options.declare('allowed_return_codes', [0],
                             desc="Set of return codes that are considered successful.")

        # Outputs of the run of the component or items that will not work with the OptionsDictionary
        self.return_code = 0  # Return code from the command
        self.stdin = self.DEV_NULL
        self.stdout = None
        self.stderr = "external_code_comp_error.out"

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        # check for the command
        cmd = [c for c in self.options['command'] if c.strip()]
        if not cmd:
            logger.error("The command cannot be empty")
        else:
            program_to_execute = self.options['command'][0]
            command_full_path = find_executable(program_to_execute)

            if not command_full_path:
                logger.error("The command to be executed, '%s', "
                             "cannot be found" % program_to_execute)

        # Check for missing input files. This just generates a warning during
        # setup, since these files may be generated later during execution.
        missing = self._check_for_files(self.options['external_input_files'])
        if missing:
            logger.warning("The following input files are missing at setup "
                           "time: %s" % missing)

    def compute(self, inputs, outputs):
        """
        Run this component.

        User should call this method from their overriden compute method.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        """
        self.return_code = -12345678

        if not self.options['command']:
            raise ValueError('Empty command list')

        if self.options['fail_hard']:
            err_class = RuntimeError
        else:
            err_class = AnalysisError

        return_code = None

        try:
            missing = self._check_for_files(self.options['external_input_files'])
            if missing:
                raise err_class("The following input files are missing: %s"
                                % sorted(missing))
            return_code, error_msg = self._execute_local()

            if return_code is None:
                raise AnalysisError('Timed out after %s sec.' %
                                    self.options['timeout'])

            elif return_code not in self.options['allowed_return_codes']:
                if isinstance(self.stderr, str):
                    if os.path.exists(self.stderr):
                        stderrfile = open(self.stderr, 'r')
                        error_desc = stderrfile.read()
                        stderrfile.close()
                        err_fragment = "\nError Output:\n%s" % error_desc
                    else:
                        err_fragment = "\n[stderr %r missing]" % self.stderr
                else:
                    err_fragment = error_msg

                raise err_class('return_code = %d%s' % (return_code,
                                                        err_fragment))

            missing = self._check_for_files(self.options['external_output_files'])
            if missing:
                raise err_class("The following output files are missing: %s"
                                % sorted(missing))

        finally:
            self.return_code = -999999 if return_code is None else return_code

    def _check_for_files(self, files):
        """
        Check that specified files exist.

        Parameters
        ----------
        files : iterable
            Contains files to check.

        Returns
        -------
        list
            List of files that do not exist.
        """
        return [path for path in files if not os.path.exists(path)]

    def _execute_local(self):
        """
        Run the command.

        Returns
        -------
        int
            Return Code
        str
            Error Message
        """
        # Check to make sure command exists
        if isinstance(self.options['command'], str):
            program_to_execute = self.options['command']
        else:
            program_to_execute = self.options['command'][0]

        # Suppress message from find_executable function, we'll handle it
        numpy.distutils.log.set_verbosity(-1)

        command_full_path = find_executable(program_to_execute)
        if not command_full_path:
            msg = "The command to be executed, '%s', cannot be found" % program_to_execute
            raise ValueError(msg)

        command_for_shell_proc = self.options['command']
        if sys.platform == 'win32':
            command_for_shell_proc = ['cmd.exe', '/c'] + command_for_shell_proc

        self._process = \
            ShellProc(command_for_shell_proc, self.stdin,
                      self.stdout, self.stderr, self.options['env_vars'])

        try:
            return_code, error_msg = \
                self._process.wait(self.options['poll_delay'], self.options['timeout'])
        finally:
            self._process.close_files()
            self._process = None

        return (return_code, error_msg)


class ExternalCode(ExternalCodeComp):
    """
    Deprecated.
    """

    def __init__(self, *args, **kwargs):
        """
        Capture Initialize to throw warning.

        Parameters
        ----------
        *args : list
            Deprecated arguments.
        **kwargs : dict
            Deprecated arguments.
        """
        warn_deprecation("'ExternalCode' component has been deprecated. Use"
                         "'ExternalCodeComp' instead.")
        super(ExternalCode, self).__init__(*args, **kwargs)
