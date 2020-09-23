"""Define the ExternalCodeComp and ExternalCodeImplicitComp classes."""
import os
import sys
import re

import numpy.distutils
from numpy.distutils.exec_command import find_executable

from openmdao.core.analysis_error import AnalysisError
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.shell_proc import STDOUT, DEV_NULL, ShellProc


class ExternalCodeDelegate(object):
    """
    Handles all the methods related to running a code externally.

    Attributes
    ----------
    _comp : ExternalCodeComp or ExternalCodeImplicitComp object
        The external code object this delegate is associated with.
    """

    def __init__(self, comp):
        """
        Initialize.

        Parameters
        ----------
        comp : ExternalCodeComp or ExternalCodeImplicitComp object
            The external code object this delegate is associated with.
        """
        self._comp = comp

    def declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        comp = self._comp

        comp.options.declare('command', [], types=(list, str),
                             desc="Command to be executed. If command is a list, shell=True, "
                                  "otherwise shell=False")
        comp.options.declare('env_vars', {}, desc='Environment variables required by the command.')
        comp.options.declare('poll_delay', 0.0, lower=0.0,
                             desc='Delay between polling for command completion. '
                                  'A value of zero will use an internally computed default.')
        comp.options.declare('timeout', 0.0, lower=0.0,
                             desc='Maximum time to wait for command completion. '
                                  'A value of zero implies an infinite wait.')
        comp.options.declare('external_input_files', [],
                             desc='List of input files that must exist before execution, '
                                  'otherwise an Exception is raised.')
        comp.options.declare('external_output_files', [],
                             desc='List of output files that must exist after execution, '
                                  'otherwise an Exception is raised.')
        comp.options.declare('fail_hard', types=bool, default=True,
                             desc="If True, external code errors raise a 'hard' exception "
                                  "(RuntimeError), otherwise errors raise a 'soft' exception "
                                  "(AnalysisError).")
        comp.options.declare('allowed_return_codes', [0],
                             desc="List of return codes that are considered successful.")

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        # check for the command
        comp = self._comp

        cmd = [c for c in comp.options['command'] if c.strip()]
        if not cmd:
            logger.error("The command cannot be empty")
        else:
            program_to_execute = comp.options['command'][0]
            if sys.platform == 'win32':
                if not find_executable(program_to_execute):
                    missing = self._check_for_files([program_to_execute])
                    if missing:
                        logger.error("The command to be executed, '%s', "
                                     "cannot be found" % program_to_execute)
            else:
                if not find_executable(program_to_execute):
                    logger.error("The command to be executed, '%s', "
                                 "cannot be found" % program_to_execute)

        # Check for missing input files. This just generates a warning during
        # setup, since these files may be generated later during execution.
        missing = self._check_for_files(comp.options['external_input_files'])
        if missing:
            logger.warning("The following input files are missing at setup "
                           "time: %s" % missing)

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

    def run_component(self, command=None):
        """
        Run this component.

        User should call this method from their overriden compute method.

        Parameters
        ----------
        command : List
            Optional command. Otherwise use the command in self.options['command'].
        """
        comp = self._comp

        if not command:
            command = comp.options['command']

        comp.return_code = -12345678

        if not command:
            raise ValueError('Empty command list')

        if comp.options['fail_hard']:
            err_class = RuntimeError
        else:
            err_class = AnalysisError

        return_code = None

        try:
            missing = self._check_for_files(comp.options['external_input_files'])
            if missing:
                raise err_class("The following input files are missing: %s"
                                % sorted(missing))
            return_code, error_msg = self._execute_local(command)

            if return_code is None:
                raise AnalysisError('Timed out after %s sec.' %
                                    comp.options['timeout'])

            elif return_code not in comp.options['allowed_return_codes']:
                if isinstance(comp.stderr, str):
                    if os.path.exists(comp.stderr):
                        with open(comp.stderr, 'r') as stderrfile:
                            error_desc = stderrfile.read()
                        err_fragment = "\nError Output:\n%s" % error_desc
                    else:
                        err_fragment = "\n[stderr %r missing]" % comp.stderr
                else:
                    err_fragment = error_msg

                raise err_class('return_code = %d%s' % (return_code,
                                                        err_fragment))

            missing = self._check_for_files(comp.options['external_output_files'])
            if missing:
                raise err_class("The following output files are missing: %s"
                                % sorted(missing))

        finally:
            comp.return_code = -999999 if return_code is None else return_code

    def _execute_local(self, command):
        """
        Run the command.

        Parameters
        ----------
        command : List
            List containing OS command string.

        Returns
        -------
        int
            Return Code
        str
            Error Message
        """
        # Check to make sure command exists
        comp = self._comp

        if isinstance(command, str):
            program_to_execute = re.findall(r"^([\w\-]+)", command)[0]
        else:
            program_to_execute = command[0]

        # Suppress message from find_executable function, we'll handle it
        numpy.distutils.log.set_verbosity(-1)

        if sys.platform == 'win32':
            if not find_executable(program_to_execute):
                missing = self._check_for_files([program_to_execute])
                if missing:
                    raise ValueError("The command to be executed, '%s', "
                                     "cannot be found" % program_to_execute)
            if isinstance(command, list):
                command_for_shell_proc = ['cmd.exe', '/c'] + command
            else:
                command_for_shell_proc = 'cmd.exe /c ' + str(command)

        else:
            if not find_executable(program_to_execute):
                raise ValueError("The command to be executed, '%s', "
                                 "cannot be found" % program_to_execute)
            command_for_shell_proc = command

        comp._process = \
            ShellProc(command_for_shell_proc, comp.stdin,
                      comp.stdout, comp.stderr, comp.options['env_vars'])

        try:
            return_code, error_msg = \
                comp._process.wait(comp.options['poll_delay'], comp.options['timeout'])
        finally:
            comp._process.close_files()
            comp._process = None

        return (return_code, error_msg)


class ExternalCodeComp(ExplicitComponent):
    """
    Run an external code as a component.

    Default stdin is the 'null' device, default stdout is the console, and
    default stderr is ``external_code_comp_error.out``.

    Attributes
    ----------
    stdin : str or file object
        Input stream external code reads from.
    stdout : str or file object
        Output stream external code writes to.
    stderr : str or file object
        Error stream external code writes to.
    _external_code_runner: ExternalCodeDelegate object
        The delegate object that handles all the running of the external code for this object.
    return_code : int
        Exit status of the child process.
    """

    def __init__(self, **kwargs):
        """
        Intialize the ExternalCodeComp component.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        self._external_code_runner = ExternalCodeDelegate(self)
        super().__init__(**kwargs)

        self.stdin = DEV_NULL
        self.stdout = None
        self.stderr = "external_code_comp_error.out"

        self.return_code = 0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        Options are declared here because this class is intended to be subclassed by
        the end user. The `initialize` method is left available for user-defined options.
        """
        super()._declare_options()
        self._external_code_runner.declare_options()

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        # check for the command
        self._external_code_runner.check_config(logger)

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
        self._external_code_runner.run_component()


class ExternalCodeImplicitComp(ImplicitComponent):
    """
    Run an external code as a component.

    Default stdin is the 'null' device, default stdout is the console, and
    default stderr is ``external_code_comp_error.out``.

    Attributes
    ----------
    stdin : str or file object
        Input stream external code reads from.
    stdout : str or file object
        Output stream external code writes to.
    stderr : str or file object
        Error stream external code writes to.
    _external_code_runner: ExternalCodeDelegate object
        The delegate object that handles all the running of the external code for this object.
    return_code : int
        Exit status of the child process.
    """

    def __init__(self, **kwargs):
        """
        Intialize the ExternalCodeComp component.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        self._external_code_runner = ExternalCodeDelegate(self)
        super().__init__(**kwargs)

        self.stdin = DEV_NULL
        self.stdout = None
        self.stderr = "external_code_comp_error.out"

        self.return_code = 0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        Options are declared here because this class is intended to be subclassed by
        the end user. The `initialize` method is left available for user-defined options.
        """
        super()._declare_options()
        self._external_code_runner.declare_options()

        # ImplicitComponent has two separate commands to run.
        self.options.declare('command_apply', [],
                             desc='command to be executed for apply_nonlinear')
        self.options.declare('command_solve', [],
                             desc='command to be executed for solve_nonlinear')
        self.options.undeclare('command')

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        self._external_code_runner.check_config(logger)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute residuals given inputs and outputs.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        command = self.options['command_apply']
        if command:
            self._external_code_runner.run_component(command=command)

    def solve_nonlinear(self, inputs, outputs):
        """
        Compute outputs given inputs. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        command = self.options['command_solve']
        if command:
            self._external_code_runner.run_component(command=command)
