"""Define the Problem class and a FakeComm class for non-MPI users."""

import __main__
import shutil

import sys
import pprint
import os
from copy import deepcopy
import weakref
import pathlib
import textwrap
import traceback
import time
import atexit

from itertools import chain

from io import TextIOBase, StringIO

import numpy as np

from openmdao.core.constants import _SetupStatus
from openmdao.core.component import Component
from openmdao.core.driver import Driver, record_iteration
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.system import System, _iter_derivs
from openmdao.core.group import Group
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.core.constants import _DEFAULT_COLORING_DIR, _DEFAULT_OUT_STREAM, \
    _UNDEFINED
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.solvers.solver import SolverInfo
from openmdao.vectors.default_vector import DefaultVector
from openmdao.error_checking.check_config import _default_checks, _all_checks, \
    _all_non_redundant_checks
from openmdao.recorders.recording_iteration_stack import _RecIteration
from openmdao.recorders.recording_manager import RecordingManager, record_viewer_data, \
    record_model_options
from openmdao.utils.deriv_display import _print_deriv_table, _deriv_display, _deriv_display_compact
from openmdao.utils.mpi import MPI, FakeComm, multi_proc_exception_check, check_mpi_env
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.units import simplify_unit
from openmdao.utils.logger_utils import get_logger, TestLogger
from openmdao.utils.hooks import _setup_hooks, _reset_all_hooks
from openmdao.utils.record_util import create_local_meta
from openmdao.utils.array_utils import scatter_dist_to_local
from openmdao.utils.class_util import overrides_method
from openmdao.utils.reports_system import get_reports_to_activate, activate_reports, \
    clear_reports, _load_report_plugins
from openmdao.utils.general_utils import pad_name, _find_dict_meta, env_truthy, add_border, \
    match_includes_excludes, ProblemMetaclass, is_undefined
from openmdao.utils.om_warnings import issue_warning, DerivativesWarning, warn_deprecation, \
    OMInvalidCheckDerivativesOptionsWarning
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.file_utils import _get_outputs_dir, text2html, _get_work_dir
from openmdao.utils.testing_utils import _fix_comp_check_data
from openmdao.utils.name_maps import DISTRIBUTED

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


CITATION = """@article{openmdao_2019,
    Author={Justin S. Gray and John T. Hwang and Joaquim R. R. A.
            Martins and Kenneth T. Moore and Bret A. Naylor},
    Title="{OpenMDAO: An Open-Source Framework for Multidisciplinary
            Design, Analysis, and Optimization}",
    Journal="{Structural and Multidisciplinary Optimization}",
    Year={2019},
    Publisher={Springer},
    pdf={http://openmdao.org/pubs/openmdao_overview_2019.pdf},
    note= {In Press}
    }"""


# Used for naming Problems when no explicit name is given
# Also handles sub problems
_problem_names = []


def _clear_problem_names():
    global _problem_names
    _problem_names = []
    _reset_all_hooks()


def _get_top_script():
    """
    Return the absolute pathname of the top level script.

    Returns
    -------
    Path or None
        The absolute path, or None if it can't be resolved.
    """
    try:
        script_name = os.environ.get('OPENMDAO_SCRIPT_NAME')
        if script_name is not None:
            return pathlib.Path(script_name).resolve()
        else:
            return pathlib.Path(__main__.__file__).resolve()
    except Exception:
        # this will error out in some cases, e.g. inside of a jupyter notebook, so just
        # return None in that case.
        pass


def _default_prob_name():
    """
    Return the default problem name.

    Returns
    -------
    str
        The default problem name.
    """
    def_prob_name = os.environ.get('OPENMDAO_DEFAULT_PROBLEM', '')
    if def_prob_name:
        return def_prob_name

    name = _get_top_script()
    if name is None or env_truthy('TESTFLO_RUNNING'):
        return 'problem'

    return name.stem


class Problem(object, metaclass=ProblemMetaclass):
    """
    Top-level container for the systems and drivers.

    Parameters
    ----------
    model : <System> or None
        The top-level <System>. If not specified, an empty <Group> will be created.
    driver : <Driver> or None
        The driver for the problem. If not specified, a simple "Run Once" driver will be used.
    comm : MPI.Comm or <FakeComm> or None
        The MPI communicator for this Problem. If not specified, comm will be MPI.COMM_WORLD if
        MPI is active, else it will be None.
    name : str
        Problem name. Can be used to specify a Problem instance when multiple Problems
        exist.
    reports : str, bool, None, _UNDEFINED
        If _UNDEFINED, the OPENMDAO_REPORTS variable is used. Defaults to _UNDEFINED.
        If given, reports overrides OPENMDAO_REPORTS. If boolean, enable/disable all reports.
        Since none is acceptable in the environment variable, a value of reports=None
        is equivalent to reports=False. Otherwise, reports may be a sequence of
        strings giving the names of the reports to run.
    **options : named args
        All remaining named args are converted to options.

    Attributes
    ----------
    model : <System>
        Pointer to the top-level <System> object (root node in the tree).
    comm : MPI.Comm or <FakeComm>
        The global communicator.
    _driver : <Driver>
        Slot for the driver. The default driver is `Driver`, which just runs
        the model once.
    _mode : 'fwd' or 'rev'
        Derivatives calculation mode, 'fwd' for forward, and 'rev' for
        reverse (adjoint).
    _orig_mode : 'fwd', 'rev', or 'auto'
        Derivatives calculation mode assigned by the user.  If set to 'auto', _mode will be
        automatically assigned to 'fwd' or 'rev' based on relative sizes of design variables vs.
        responses.
    cite : str
        Listing of relevant citations that should be referenced when
        publishing work that uses this class.
    options : <OptionsDictionary>
        Dictionary with general options for the problem.
    model_options : dict
        A  dictionary of options to be passed to subsystems in the problem's model during
        the setup process.
        This dictionary is keyed by a path pattern string, and the associated value for each path
        pattern is a dictionary of {option_name: option_val}. Those subsystems within the
        hierarchy which match the path pattern and that have an option of the given name, will
        have the value of that option overridden by value given in the dictionary.
    recording_options : <OptionsDictionary>
        Dictionary with problem recording options.
    _rec_mgr : <RecordingManager>
        Object that manages all recorders added to this problem.
    _reports : list of str
        Names of reports to activate for this Problem.
    _check : bool
        If True, call check_config at the end of final_setup.
    _filtered_vars_to_record : dict
        Dictionary of lists of design vars, constraints, etc. to record.
    _logger : object or None
        Object for logging config checks if _check is True.
    _name : str
        Problem name. If no name given, a default name of the form 'problemN', where N is an
        integer, will be given to the problem so it can be referenced in command line tools
        that have an optional problem name argument
    _metadata : dict
        Problem level metadata.
    _run_counter : int
        The number of times run_driver or run_model has been called.
    _computing_coloring : bool
        When True, we are computing coloring.
    """

    def __init__(self, model=None, driver=None, comm=None, name=None, reports=_UNDEFINED,
                 **options):
        """
        Initialize attributes.
        """
        global _problem_names

        # this function doesn't do anything after the first call
        _load_report_plugins()

        self._driver = None
        self._reports = get_reports_to_activate(reports)

        self.cite = CITATION
        self._computing_coloring = False

        if comm is None:
            use_mpi = check_mpi_env()
            if use_mpi is False:
                comm = FakeComm()
            else:
                try:
                    from mpi4py import MPI
                    comm = MPI.COMM_WORLD
                except ImportError:
                    comm = FakeComm()

        self.comm = comm

        self._set_name(name)

        if model is None:
            self.model = Group()
        elif isinstance(model, Group):
            from openmdao.core.parallel_group import ParallelGroup
            if isinstance(model, ParallelGroup):
                raise TypeError(f"{self.msginfo}: The value provided for 'model' "
                                "cannot be a ParallelGroup.")
            self.model = model
        else:
            raise TypeError(self.msginfo +
                            ": The value provided for 'model' is not a Group.")

        if driver is None:
            driver = Driver()
        elif not isinstance(driver, Driver):
            raise TypeError(self.msginfo +
                            ": The value provided for 'driver' is not a valid Driver.")

        self._update_reports(driver)

        # can't use driver property here without causing a lint error, so just do it manually
        self._driver = driver

        self._metadata = {'setup_status': _SetupStatus.PRE_SETUP}
        self._run_counter = -1
        self._rec_mgr = RecordingManager()

        # General options
        self.options = OptionsDictionary(parent_name=type(self).__name__)
        default_workdir = options['work_dir'] if 'work_dir' in options else _get_work_dir()
        self.options.declare('work_dir', default=default_workdir,
                             desc='Working directory for the problem.')
        self.options.declare('coloring_dir', types=str,
                             default=os.path.join(default_workdir, 'coloring_files'),
                             desc='Directory containing coloring files (if any) for this Problem.')
        self.options.declare('group_by_pre_opt_post', types=bool,
                             default=True,
                             desc="If True, group subsystems of the top level model into "
                             "pre-optimization, optimization, and post-optimization, and only "
                             "iterate over the optimization subsystems during optimization.  This "
                             "applies only when the top level nonlinear solver is of type"
                             "NonlinearRunOnce.")
        self.options.declare('allow_post_setup_reorder', types=bool,
                             default=True,
                             desc="If True, the execution order of direct subsystems of any group "
                             "that sets its 'auto_order' option to True will be automatically "
                             "ordered according to data dependencies. If this option is False, the "
                             "'auto_order' option will be ignored and a warning will be issued for "
                             "each group that has set it to True. Note that subsystems of a Group "
                             "that form a cycle will never be reordered, regardless of the value of"
                             " the 'auto_order' option.")
        self.options.update(options)

        # Options passed to models
        self.model_options = {}

        # Case recording options
        self.recording_options = OptionsDictionary(parent_name=type(self).__name__)

        self.recording_options.declare('record_desvars', types=bool, default=True,
                                       desc='Set to True to record design variables at the '
                                            'problem level')
        self.recording_options.declare('record_objectives', types=bool, default=True,
                                       desc='Set to True to record objectives at the problem level')
        self.recording_options.declare('record_constraints', types=bool, default=True,
                                       desc='Set to True to record constraints at the '
                                            'problem level')
        self.recording_options.declare('record_responses', types=bool, default=False,
                                       desc='Set True to record constraints and objectives at the '
                                            'problem level.')
        self.recording_options.declare('record_inputs', types=bool, default=False,
                                       desc='Set True to record inputs at the '
                                            'problem level.')
        self.recording_options.declare('record_outputs', types=bool, default=True,
                                       desc='Set True to record outputs at the '
                                            'problem level.')
        self.recording_options.declare('record_residuals', types=bool, default=False,
                                       desc='Set True to record residuals at the '
                                            'problem level.')
        self.recording_options.declare('record_derivatives', types=bool, default=False,
                                       desc='Set to True to record derivatives for the problem '
                                            'level')
        self.recording_options.declare('record_abs_error', types=bool, default=True,
                                       desc='Set to True to record absolute error of '
                                            'model nonlinear solver')
        self.recording_options.declare('record_rel_error', types=bool, default=True,
                                       desc='Set to True to record relative error of model \
                                       nonlinear solver')
        self.recording_options.declare('includes', types=list, default=['*'],
                                       desc='Patterns for variables to include in recording. \
                                       Uses fnmatch wildcards')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                            '(processed post-includes). Uses fnmatch wildcards')

        # register hooks for any reports
        activate_reports(self._reports, self)

        # So Problem and driver can have hooks attached to their methods
        _setup_hooks(self)

        # call cleanup at system exit, if requested
        if 'cleanup' in os.environ.get('OPENMDAO_ATEXIT', '').split(','):
            atexit.register(self.cleanup)

    def _set_name(self, name):
        if not MPI or self.comm.rank == 0:
            # Set the Problem name so that it can be referenced from command line tools (e.g. check)
            # that accept a Problem argument, and to name the corresponding outputs subdirectory.
            if name:  # if name hasn't been used yet, use it. Otherwise, error
                if name in _problem_names:
                    issue_warning(f"The problem name '{name}' already exists")
                self._name = name
            else:  # No name given: look for a name, of the form, 'problemN', that hasn't been used
                problem_counter = len(_problem_names) + 1 if _problem_names else ''
                base = _default_prob_name()
                _name = f"{base}{problem_counter}"
                if _name in _problem_names:  # need to make it unique so append string of form '.N'
                    i = 1
                    while True:
                        _name = f"{base}{problem_counter}.{i}"
                        if _name not in _problem_names:
                            break
                        i += 1
                self._name = _name
            if self.comm.size > 1:
                self._name = self.comm.bcast(self._name, root=0)
        else:
            self._name = self.comm.bcast(None, root=0)

        _problem_names.append(self._name)

    def _has_active_report(self, name):
        """
        Return True if named report is active for this Problem.

        Parameters
        ----------
        name : str
            Name of the report.

        Returns
        -------
        bool
            True if the named report is active for this Problem.
        """
        return name in self._reports

    @property
    def driver(self):
        """
        Get the Driver for this Problem.
        """
        return self._driver

    def _update_reports(self, driver):
        if self._driver is not None:
            # remove any reports on previous driver
            clear_reports(self._driver)
        driver._set_problem(self)
        activate_reports(self._reports, driver)
        _setup_hooks(driver)

    @driver.setter
    def driver(self, driver):
        """
        Set this Problem's Driver.

        Parameters
        ----------
        driver : <Driver>
            Driver to be set to our _driver attribute.
        """
        self._update_reports(driver)
        self._driver = driver

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        if self._name is None:
            return type(self).__name__
        return f'{type(self).__name__} {self._name}'

    @property
    def _mode(self):
        """
        Return the derivative mode.

        Returns
        -------
        str
            Derivative mode, 'fwd' or 'rev'.
        """
        if self._metadata is None:
            return None
        return self._metadata['mode']

    def _get_inst_id(self):
        return self._name

    def is_local(self, name):
        """
        Return True if the named variable or system is local to the current process.

        Parameters
        ----------
        name : str
            Name of a variable or system.

        Returns
        -------
        bool
            True if the named system or variable is local to this process.
        """
        if self._metadata['setup_status'] < _SetupStatus.POST_SETUP:
            raise RuntimeError(f"{self.msginfo}: is_local('{name}') was called before setup() "
                               "completed.")

        abs_name = self.model._resolver.any2abs(name)
        if abs_name is None:  # no variable found
            sub = self.model._get_subsystem(name)
            return sub is not None and sub._is_local

        return self.model._resolver.is_local(abs_name)

    @property
    def _recording_iter(self):
        return self._metadata['recording_iter']

    def __getitem__(self, name):
        """
        Get an output/input variable.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.

        Returns
        -------
        float or ndarray or any python object
            the requested output/input variable.
        """
        return self.get_val(name, get_remote=None)

    def get_val(self, name, units=None, indices=None, get_remote=False, copy=False):
        """
        Get an output/input variable.

        Function is used if you want to specify display units.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        units : str, optional
            Units to convert to before return.
        indices : int or list of ints or tuple of ints or int ndarray or Iterable or None, optional
            Indices or slice to return.
        get_remote : bool or None
            If True, retrieve the value even if it is on a remote process.  Note that if the
            variable is remote on ANY process, this function must be called on EVERY process
            in the Problem's MPI communicator.
            If False, only retrieve the value if it is on the current process, or only the part
            of the value that's on the current process for a distributed variable.
            If None and the variable is remote or distributed, a RuntimeError will be raised.
        copy : bool, optional
            If True, return a copy of the value.  If False, return a reference to the value.

        Returns
        -------
        object
            The value of the requested output/input variable.
        """
        if self._metadata['setup_status'] <= _SetupStatus.POST_SETUP2:
            abs_names = self.model._resolver.absnames(name)
            val = self.model._get_cached_val(name, abs_names, get_remote=get_remote)
            if not is_undefined(val):
                if indices is not None:
                    val = val[indices]
                if units is not None:
                    val = self.model.convert2units(name, val, simplify_unit(units))
        else:
            val = self.model.get_val(name, units=units, indices=indices, get_remote=get_remote,
                                     from_src=True)

        if is_undefined(val):
            if get_remote:
                raise KeyError(f'{self.msginfo}: Variable name "{name}" not found.')
            else:
                raise RuntimeError(f"{self.model.msginfo}: Variable '{name}' is not local to "
                                   f"rank {self.comm.rank}. You can retrieve values from "
                                   "other processes using `get_val(<name>, get_remote=True)`.")

        if copy:
            return deepcopy(val)
        else:
            return val

    def __setitem__(self, name, value):
        """
        Set an output/input variable.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        value : float or ndarray or any python object
            value to set this variable to.
        """
        self.set_val(name, value)

    def set_val(self, name, val=None, units=None, indices=None):
        """
        Set an output/input variable.

        Function is used if you want to set a value using a different unit.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        val : object
            Value to set this variable to.
        units : str, optional
            Units that value is defined in.
        indices : int or list of ints or tuple of ints or int ndarray or Iterable or None, optional
            Indices or slice to set to specified value.
        """
        if self._metadata is None:
            raise RuntimeError(f"{self.msginfo}: '{name}' Cannot call set_val before setup.")

        self.model.set_val(name, val, units=units, indices=indices)

    def _set_initial_conditions(self):
        """
        Set all initial conditions that have been saved in cache after setup.
        """
        for value, set_units, pathname, name in self.model._initial_condition_cache.values():
            if pathname:
                system = self.model._get_subsystem(pathname)
                if system is None or not system._is_local:
                    pass
                else:
                    system.set_val(name, value, units=set_units)
            else:
                self.model.set_val(name, value, units=set_units)

        # Clean up cache
        self.model._initial_condition_cache = {}

    def _check_collected_errors(self):
        """
        If any collected errors are found, raise an exception containing all of them.
        """
        if self._metadata['saved_errors'] is None:
            return

        unique_errors = self._get_unique_saved_errors()

        # set the errors to None so that all future calls will immediately raise an exception.
        self._metadata['saved_errors'] = None

        if unique_errors:
            final_msg = [f"\nCollected errors for problem '{self._name}':"]
            for _, msg, exc_type, tback in unique_errors:
                final_msg.append(f"   {msg}")

                # if there's only one error, include its traceback if it exists.
                if len(unique_errors) == 1:
                    if isinstance(tback, str):
                        final_msg.append('Traceback (most recent call last):')
                        final_msg.append(tback)
                    else:
                        raise exc_type('\n'.join(final_msg)).with_traceback(tback)

            raise RuntimeError('\n'.join(final_msg))

    def run_model(self, case_prefix=None, reset_iter_counts=True):
        """
        Run the model by calling the root system's solve_nonlinear.

        Parameters
        ----------
        case_prefix : str or None
            Prefix to prepend to coordinates when recording.  None means keep the preexisting
            prefix.

        reset_iter_counts : bool
            If True and model has been run previously, reset all iteration counters.
        """
        if not self.model._have_output_solver_options_been_applied():
            raise RuntimeError(self.msginfo +
                               ": Before calling `run_model`, the `setup` method must be called "
                               "if set_output_solver_options has been called.")

        if self._metadata['setup_status'] < _SetupStatus.POST_SETUP:
            if self.model._order_set:
                raise RuntimeError(f"{self.msginfo}: Cannot call set_order without calling setup "
                                   "after")
            else:
                raise RuntimeError(self.msginfo +
                                   ": The `setup` method must be called before `run_model`.")

        old_prefix = self._recording_iter.prefix

        if case_prefix is not None:
            if not isinstance(case_prefix, str):
                raise TypeError(self.msginfo + ": The 'case_prefix' argument should be a string.")
            self._recording_iter.prefix = case_prefix

        try:
            if self.model.iter_count > 0 and reset_iter_counts:
                self.driver.iter_count = 0
                self.model._reset_iter_counts()

            self.final_setup()

            self._run_counter += 1
            record_model_options(self, self._run_counter)

            self.model._clear_iprint()
            self.model.run_solve_nonlinear()
        finally:
            self._recording_iter.prefix = old_prefix

    def run_driver(self, case_prefix=None, reset_iter_counts=True):
        """
        Run the driver on the model.

        Parameters
        ----------
        case_prefix : str or None
            Prefix to prepend to coordinates when recording.  None means keep the preexisting
            prefix.
        reset_iter_counts : bool
            If True and model has been run previously, reset all iteration counters.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        model = self.model
        driver = self.driver

        if self._metadata['setup_status'] < _SetupStatus.POST_SETUP:
            raise RuntimeError(self.msginfo +
                               ": The `setup` method must be called before `run_driver`.")

        if not model._have_output_solver_options_been_applied():
            raise RuntimeError(self.msginfo +
                               ": Before calling `run_driver`, the `setup` method must be called "
                               "if set_output_solver_options has been called.")

        if 'singular_jac_behavior' in driver.options:
            self._metadata['singular_jac_behavior'] = driver.options['singular_jac_behavior']

        old_prefix = self._recording_iter.prefix

        if case_prefix is not None:
            if not isinstance(case_prefix, str):
                raise TypeError(self.msginfo + ": The 'case_prefix' argument should be a string.")
            self._recording_iter.prefix = case_prefix

        try:
            if model.iter_count > 0 and reset_iter_counts:
                driver.iter_count = 0
                model._reset_iter_counts()

            self.final_setup()

            # for optimizing drivers, check that constraints are affected by design vars
            if driver.supports['optimization'] and self._metadata['use_derivatives']:
                driver.check_relevance()

            self._run_counter += 1
            record_model_options(self, self._run_counter)

            model._clear_iprint()

            return driver._run()
        finally:
            self._recording_iter.prefix = old_prefix

    def find_feasible(self, case_prefix=None, reset_iter_counts=True,
                      driver_scaling=True, exclude_desvars=None,
                      method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08,
                      x_scale=1., loss='linear', loss_tol=1.0E-8, f_scale=1.0,
                      max_nfev=None, tr_solver=None, tr_options=None, iprint=1):
        """
        Attempt to find design variable values which minimize the constraint violation.

        If the problem is feasible, this method should find the solution for which the
        violation of each constraint is zero.

        This approach uses a least-squares minimization of the constraint violation.  If
        the problem has a feasible solution, this should find the feasible solution
        closest to the current design variable values.

        Arguments method, ftol, xtol, gtol, x_scale, loss, f_scale, diff_step,
        tr_solver, tr_options, and verbose are passed to `scipy.optimize.least_squares`, see
        the documentation of that function for more information:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        Parameters
        ----------
        case_prefix : str or None
            Prefix to prepend to coordinates when recording.  None means keep the preexisting
            prefix.
        reset_iter_counts : bool
            If True and model has been run previously, reset all iteration counters.
        driver_scaling : bool
            If True, consider the constraint violation in driver-scaled units. Otherwise, it
            will be computed in the model's units.
        exclude_desvars : str or Sequence[str] or None
            If given, a pattern of one or more design variables to be excluded from
            the least-squares search.  The allows for finding a feasible (or least infeasible)
            solution when holding one or more design variables to their current values.
        method : {'trf', 'dogbox', or 'lm'}
            The method used by scipy.optimize.least_squares. One or 'trf', 'dogbox', or 'lm'.
        ftol : float or None
            The change in the cost function from one iteration to the next which triggers
            a termination of the minimization.
        xtol : float or None
            The change in the design variable vector norm from one iteration to the next
            which triggers a termination of the minimization.
        gtol : float or None
            The change in the gradient norm from one iteration to the next which triggers
            a termination of the minimization.
        x_scale : {float, array-like, or 'jac'}
            Additional scaling applied by the least-squares algorithm. Behavior is method-dependent.
            For additional details, see the scipy documentation.
        loss : {'linear', 'soft_l1', 'huber', 'cauchy', or 'arctan'}
            The loss aggregation method. Options of interest are:
            - 'linear' gives the standard "sum-of-squares".
            - 'soft_l1' gives a smooth approximation for the L1-norm of constraint violation.
            For other options, see the scipy documentation.
        loss_tol : float
            The tolerance on the loss value above which the algorithm is considered to have
            failed to find a feasible solution. This will result in the `DriverResult.success`
            attribute being False, and this method will return as _failed_.
        f_scale : float or None
            Value of margin between inlier and outlier residuals when loss is not 'linear'.
            For more information, see the scipy documentation.
        max_nfev : int or None
            The maximum allowable number of model evaluations.  If not provided scipy will
            determine it automatically based on the size of the design variable vector.
        tr_solver : {None, 'exact', or 'lsmr'}
            The solver used by trust region (trf) method.
            For more details, see the scipy documentation.
        tr_options : dict or None
            Additional options for the trust region (trf) method.
            For more details, see the scipy documentation.
        iprint : int
            Verbosity of the output. Use 2 for the full verbose least_squares output.
            Use 1 for a convergence summary, and 0 to suppress output.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        model = self.model
        driver = self.driver

        if self._metadata['setup_status'] < _SetupStatus.POST_SETUP:
            raise RuntimeError(self.msginfo +
                               ": The `setup` method must be called before `run_driver`.")

        if not model._have_output_solver_options_been_applied():
            raise RuntimeError(self.msginfo +
                               ": Before calling `run_driver`, the `setup` method must be called "
                               "if set_output_solver_options has been called.")

        if 'singular_jac_behavior' in driver.options:
            self._metadata['singular_jac_behavior'] = driver.options['singular_jac_behavior']

        old_prefix = self._recording_iter.prefix

        if case_prefix is not None:
            if not isinstance(case_prefix, str):
                raise TypeError(self.msginfo + ": The 'case_prefix' argument should be a string.")
            self._recording_iter.prefix = case_prefix

        try:
            if model.iter_count > 0 and reset_iter_counts:
                driver.iter_count = 0
                model._reset_iter_counts()

            self.final_setup()

            # for optimizing drivers, check that constraints are affected by design vars
            if driver.supports['optimization'] and self._metadata['use_derivatives']:
                driver.check_relevance()

            self._run_counter += 1
            record_model_options(self, self._run_counter)

            model._clear_iprint()

            return driver._find_feasible(driver_scaling=driver_scaling,
                                         exclude_desvars=exclude_desvars,
                                         method=method, ftol=ftol, xtol=xtol,
                                         gtol=gtol, x_scale=x_scale,
                                         loss=loss, loss_tol=loss_tol, f_scale=f_scale,
                                         max_nfev=max_nfev, tr_solver=tr_solver,
                                         tr_options=tr_options, iprint=iprint)
        finally:
            self._recording_iter.prefix = old_prefix

    def compute_jacvec_product(self, of, wrt, mode, seed, linearize=False):
        """
        Given a seed and 'of' and 'wrt' variables, compute the total jacobian vector product.

        Parameters
        ----------
        of : list of str
            Variables whose derivatives will be computed.
        wrt : list of str
            Derivatives will be computed with respect to these variables.
        mode : str
            Derivative direction ('fwd' or 'rev').
        seed : dict or list
            Either a dict keyed by 'wrt' varnames (fwd) or 'of' varnames (rev), containing
            dresidual (fwd) or doutput (rev) values, OR a list of dresidual or doutput
            values that matches the corresponding 'wrt' (fwd) or 'of' (rev) varname list.
        linearize : bool
            If False, assume the model is already in a linearized state, which it must be
            in order to produce correct results for the jvp/vjp. If True, this method
            will linearize the model before computing the jvp/vjp.

        Returns
        -------
        dict
            The total jacobian vector product, keyed by variable name.
        """
        if mode == 'fwd':
            if len(wrt) != len(seed):
                raise RuntimeError(self.msginfo +
                                   ": seed and 'wrt' list must be the same length in fwd mode.")
            lnames, rnames = of, wrt
            lkind, rkind = 'output', 'residual'
        else:  # rev
            if len(of) != len(seed):
                raise RuntimeError(self.msginfo +
                                   ": seed and 'of' list must be the same length in rev mode.")
            lnames, rnames = wrt, of
            lkind, rkind = 'residual', 'output'

        if linearize:
            self.model.run_linearize()

        resolver = self.model._resolver
        rvec = self.model._vectors[rkind]['linear']
        lvec = self.model._vectors[lkind]['linear']

        rvec.set_val(0.)

        # set seed values into dresids (fwd) or doutputs (rev)
        # seed may have keys that are inputs and must be converted into auto_ivcs
        try:
            seed[rnames[0]]
        except (IndexError, TypeError):
            # Seeds are given as a Sequence
            for i, name in enumerate(rnames):
                rvec[resolver.source(name)] = seed[i]
        else:
            # Seeds are given as a map
            for name in rnames:
                # rnames are abs_path
                rvec[resolver.source(name)] = seed[name]

        # We apply a -1 here because the derivative of the output is minus the derivative of
        # the residual in openmdao.
        data = rvec.asarray()
        data *= -1.

        self.model.run_solve_linear(mode)

        return {n: lvec[resolver.source(n)].copy() for n in lnames}

    def _setup_recording(self):
        """
        Set up case recording.
        """
        if self._rec_mgr.has_recorders():
            self._filtered_vars_to_record = self.driver._get_vars_to_record(self)
            self._rec_mgr.startup(self, self.comm)

    def add_recorder(self, recorder):
        """
        Add a recorder to the problem.

        Parameters
        ----------
        recorder : CaseRecorder
           A recorder instance.
        """
        self._rec_mgr.append(recorder)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        # shut down all recorders
        self._rec_mgr.shutdown()

        # clean up driver and model resources
        self.driver.cleanup()
        for system in self.model.system_iter(include_self=True, recurse=True):
            system.cleanup()

    def record(self, case_name):
        """
        Record the variables at the Problem level.

        Must be called after `final_setup` has been called. This can either
        happen automatically through `run_driver` or `run_model`, or it can be
        called manually.

        Parameters
        ----------
        case_name : str
            Name used to identify this Problem case.
        """
        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            raise RuntimeError(f"{self.msginfo}: Problem.record() cannot be called before "
                               "`Problem.run_model()`, `Problem.run_driver()`, or "
                               "`Problem.final_setup()`.")
        else:
            record_iteration(self, self, case_name)

    def _get_recorder_metadata(self, case_name):
        """
        Return metadata from the latest iteration for use in the recorder.

        Parameters
        ----------
        case_name : str
            Name of current case.

        Returns
        -------
        dict
            Metadata dictionary for the recorder.
        """
        return create_local_meta(case_name)

    def setup(self, check=None, logger=None, mode='auto', force_alloc_complex=False,
              distributed_vector_class=PETScVector, local_vector_class=DefaultVector,
              derivatives=True, parent=None):
        """
        Set up the model hierarchy.

        When `setup` is called, the model hierarchy is assembled, the processors are allocated
        (for MPI), and variables and connections are all assigned. This method traverses down
        the model hierarchy to call `setup` on each subsystem, and then traverses up the model
        hierarchy to call `configure` on each subsystem.

        Parameters
        ----------
        check : None, bool, list of str, or the strs ‘all’
            Determines what config checks, if any, are run after setup is complete.
            If None: no checks are run unless the 'checks' report is active, in which case the
            default reports will be run.
            If False, no checks are run
            If True, the default checks ('out_of_order', 'system', 'solvers', 'dup_inputs',
            'missing_recorders', 'unserializable_options', 'comp_has_no_outputs',
            'auto_ivc_warnings') are run
            If list of str, run those config checks
            If ‘all’, all the checks ('auto_ivc_warnings', 'comp_has_no_outputs', 'cycles',
            'dup_inputs', 'missing_recorders', 'all_unserializable_options', 'out_of_order',
            'promotions', 'solvers', 'system', 'unconnected_inputs') are run.
        logger : object
            Object for logging config checks if check is True.
        mode : str
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'auto', which will pick 'fwd' or 'rev' based on
            the direction resulting in the smallest number of linear solves required to
            compute derivatives.
        force_alloc_complex : bool
            If True, sufficient memory will be allocated to allow nonlinear vectors to store
            complex values while operating under complex step.
        distributed_vector_class : type
            Reference to the <Vector> class or factory function used to instantiate vectors
            and associated transfers involved in interprocess communication.
        local_vector_class : type
            Reference to the <Vector> class or factory function used to instantiate vectors
            and associated transfers involved in intraprocess communication.
        derivatives : bool
            If True, perform any memory allocations necessary for derivative computation.
        parent : Problem, System, Solver, or None
            The "parent" object of this problem instance in a tree of potentially nested problems.

        Returns
        -------
        <Problem>
            This enables the user to instantiate and setup in one line.
        """
        model = self.model
        comm = self.comm

        if not isinstance(self.model, Group):
            raise TypeError("The model for this Problem is of type "
                            f"'{self.model.__class__.__name__}'. "
                            "The model must be a Group or a sub-class of Group.")

        # A distributed vector type is required for MPI
        if comm.size > 1:
            if distributed_vector_class is PETScVector and PETScVector is None:
                raise ValueError(f"{self.msginfo}: Attempting to run in parallel under MPI but "
                                 "PETScVector could not be imported.")
            elif not distributed_vector_class.distributed:
                raise ValueError(f"{self.msginfo}: The `distributed_vector_class` argument must be "
                                 "a distributed vector class like `PETScVector` when running in "
                                 f"parallel under MPI but '{distributed_vector_class.__name__}' "
                                 "was specified which is not distributed.")

        if mode not in ['fwd', 'rev', 'auto']:
            msg = f"{self.msginfo}: Unsupported mode: '{mode}'. Use either 'fwd' or 'rev'."
            raise ValueError(msg)

        self._orig_mode = mode

        # this metadata will be shared by all Systems/Solvers in the system tree
        self._metadata.update({
            'name': self._name,  # the name of this Problem
            'pathname': None,  # the pathname of this Problem in the current tree of Problems
            'comm': comm,
            'work_dir': pathlib.Path(self.options['work_dir']),
            'coloring_dir': _DEFAULT_COLORING_DIR,  # directory for input coloring files
            'recording_iter': _RecIteration(comm.rank),  # manager of recorder iterations
            'local_vector_class': local_vector_class,
            'distributed_vector_class': distributed_vector_class,
            'solver_info': SolverInfo(),
            'use_derivatives': derivatives,
            'force_alloc_complex': force_alloc_complex,  # forces allocation of complex vectors
            'vars_to_gather': {},  # vars that are remote somewhere. does not include distrib vars
            'static_mode': False,  # used to determine where various 'static'
                                   # and 'dynamic' data structures are stored.
                                   # Dynamic ones are added during System
                                   # setup/configure. They are wiped out and re-created during
                                   # each Problem setup.  Static ones are added outside of
                                   # Problem setup and they are never wiped out or re-created.
            'config_info': None,  # used during config to determine if additional updates required
            'parallel_groups': [],  # list of pathnames of parallel groups in this model (all procs)
            'setup_status': _SetupStatus.PRE_SETUP,
            'model_ref': weakref.ref(model),  # ref to the model (needed to get out-of-scope
                                              # src data for inputs)
            'has_par_deriv_color': False,  # True if any dvs/responses have parallel deriv colors
            'mode': mode,  # mode (derivative direction) set by the user.  'auto' by default
            'orig_mode': mode,  # mode (derivative direction) set by the user.  'auto' by default
            'abs_in2prom_info': {},  # map of abs input name to list of length = sys tree height
                                     # down to var location, to allow quick resolution of local
                                     # src_shape/src_indices due to promotes.  For example,
                                     # for abs_in of a.b.c.d, dict entry would be
                                     # [None, None, None], corresponding to levels
                                     # a, a.b, and a.b.c, with one of the Nones replaced
                                     # by promotes info.  Dict entries are only created if
                                     # src_indices are applied to the variable somewhere.
            'reports_dir': None,  # directory where reports will be written
            'saved_errors': [],  # store setup errors here until after final_setup
            'checking': False,  # True if check_totals or check_partials is running
            'model_options': self.model_options,  # A dict of options passed to all systems in tree
            'allow_post_setup_reorder': self.options['allow_post_setup_reorder'],  # see option
            'singular_jac_behavior': 'warn',  # How to handle singular jac conditions
            'parallel_deriv_color': None,  # None unless derivatives involving a parallel deriv
                                           # colored dv/response are currently being computed.
            'seed_vars': None,  # set of names of seed variables. Seed variables are those that
                                # have their derivative value set to 1.0 at the beginning of the
                                # current derivative solve.
            'coloring_randgen': None,  # If total coloring is being computed, will contain a random
                                       # number generator, else None.
            'randomize_subjacs': True,  # If True, randomize subjacs before computing total sparsity
            'randomize_seeds': False,  # If True, randomize seed vectors when computing total
                                       # sparsity
            'group_by_pre_opt_post': self.options['group_by_pre_opt_post'],  # see option
            'relevance_cache': {},  # cache of relevance objects
            'rel_array_cache': {},  # cache of relevance arrays
            'ncompute_totals': 0,  # number of times compute_totals has been called
            'jax_group': None,  # not None if a Group is currently performing a jax operation
        })

        model_comm = self.driver._setup_comm(comm)

        if parent:
            if isinstance(parent, Problem):
                parent_prob_meta = parent._metadata
            elif isinstance(parent, System):
                parent_prob_meta = parent._problem_meta
            else:
                raise ValueError('Problem parent must be another Problem or System instance.')
        else:
            parent_prob_meta = None

        if parent_prob_meta and parent_prob_meta['pathname']:
            self._metadata['pathname'] = parent_prob_meta['pathname'] + f'/{self._name}'
        else:
            self._metadata['pathname'] = self._name

        # We don't want to delete the outputs directory because we may be using the coloring files
        # from a previous run.
        # Start setup by deleting any existing reports so that the files
        # that are in that directory are all from this run and not a previous run
        reports_dirpath = self.get_reports_dir()
        if not MPI or (self.comm is not None and self.comm.rank == 0):
            if os.path.isdir(reports_dirpath):
                try:
                    shutil.rmtree(reports_dirpath)
                except FileNotFoundError:
                    # Folder already removed by another proccess
                    pass
        self._metadata['reports_dir'] = self.get_reports_dir()

        try:
            model._setup(model_comm, self._metadata)
        finally:
            # whenever we're outside of model._setup, static mode should be True so that anything
            # added outside of _setup will persist.
            self._metadata['static_mode'] = True

        # Cache all args for final setup.
        self._check = check
        self._logger = logger

        self._metadata['setup_status'] = _SetupStatus.POST_SETUP

        return self

    def final_setup(self):
        """
        Perform final setup phase on problem in preparation for run.

        This is the second phase of setup, and is done automatically at the start of `run_driver`
        and `run_model`. At the beginning of final_setup, we have a model hierarchy with defined
        variables, solvers, case_recorders, and derivative settings. During this phase, the vectors
        are created and populated, the drivers and solvers are initialized, and the recorders are
        started, and the rest of the framework is prepared for execution.
        """
        driver = self.driver
        model = self.model

        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            first = True
            self._metadata['static_mode'] = False
            try:
                if self._metadata['setup_status'] < _SetupStatus.POST_SETUP2:
                    model._setup_part2()
                    self._check_collected_errors()

                responses = model.get_responses(recurse=True, use_prom_ivc=True)
                designvars = model.get_design_vars(recurse=True, use_prom_ivc=True)
                response_size, desvar_size = driver._update_voi_meta(model, responses,
                                                                     designvars)

                model._final_setup()
            finally:
                self._metadata['static_mode'] = True

            # update mode if it's been set to 'auto'
            if self._orig_mode == 'auto':
                mode = 'rev' if response_size < desvar_size else 'fwd'
            else:
                mode = self._orig_mode

            self._metadata['mode'] = mode
        else:
            first = False
            mode = self._metadata['mode']

            responses = model.get_responses(recurse=True, use_prom_ivc=True)
            designvars = model.get_design_vars(recurse=True, use_prom_ivc=True)
            response_size, desvar_size = driver._update_voi_meta(model, responses, designvars)

            # If set_solver_print is called after an initial run, in a multi-run scenario,
            #  this part of _final_setup still needs to happen so that change takes effect
            #  in subsequent runs
            model._setup_solver_print()

        driver._setup_driver(self)

        if first:
            if coloring_mod._use_total_sparsity:
                coloring = driver._coloring_info.coloring
                if coloring is not None:
                    # if we're using simultaneous total derivatives then our effective size is less
                    # than the full size
                    if coloring._fwd and coloring._rev:
                        pass  # we're doing both!
                    elif mode == 'fwd' and coloring._fwd:
                        desvar_size = coloring.total_solves()
                    elif mode == 'rev' and coloring._rev:
                        response_size = coloring.total_solves()

            if ((mode == 'fwd' and desvar_size > response_size) or
                    (mode == 'rev' and response_size > desvar_size)):
                issue_warning(f"Inefficient choice of derivative mode.  You chose '{mode}' for a "
                              f"problem with {desvar_size} design variables and {response_size} "
                              "response variables (objectives and nonlinear constraints).",
                              category=DerivativesWarning)

        if (not self._metadata['allow_post_setup_reorder'] and
                self._metadata['setup_status'] == _SetupStatus.PRE_SETUP and self.model._order_set):
            raise RuntimeError(f"{self.msginfo}: Cannot call set_order without calling setup after")

        # set up recording, including any new recorders since last setup
        # TODO: We should be smarter and only setup the recording when new recorders have
        # been added.
        if self._metadata['setup_status'] >= _SetupStatus.POST_SETUP:
            driver._setup_recording()
            self._setup_recording()
            record_viewer_data(self)

        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            self._metadata['setup_status'] = _SetupStatus.POST_FINAL_SETUP
            self._set_initial_conditions()

        if self.model.comm.size > 1:
            # this updates any source values that are attached to remote inputs
            self.model._resolve_remote_sets()

        if self._check and 'checks' not in self._reports:
            if self._check is True:
                checks = _default_checks
            else:
                checks = self._check
            if self.comm.rank == 0:
                logger = self._logger
            else:
                logger = TestLogger()
            self.check_config(logger, checks=checks)

    def set_setup_status(self, status, **setup_kwargs):
        """
        Set the setup status of the problem, running any setup steps that haven't been run yet.

        If the status is already at or beyond the requested status, this method does nothing,
        i.e., it won't reset the status to an earlier step.

        Parameters
        ----------
        status : _SetupStatus
            The status to set the problem to.
        **setup_kwargs : dict
            Keyword arguments to pass to the setup method if it hasn't already been called.
        """
        if self._metadata['setup_status'] >= status:
            return

        if status >= _SetupStatus.POST_SETUP:
            if self._metadata['setup_status'] < _SetupStatus.POST_SETUP:
                self.setup(**setup_kwargs)
                self._metadata['setup_status'] = _SetupStatus.POST_SETUP

        if status >= _SetupStatus.POST_SETUP2:
            if self._metadata['setup_status'] < _SetupStatus.POST_SETUP2:
                self.model._setup_part2()
                self._metadata['setup_status'] = _SetupStatus.POST_SETUP2

        if status >= _SetupStatus.POST_FINAL_SETUP:
            if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
                self.final_setup()
                self._metadata['setup_status'] = _SetupStatus.POST_FINAL_SETUP

    def check_partials(self, out_stream=_DEFAULT_OUT_STREAM, includes=None, excludes=None,
                       compact_print=False, abs_err_tol=0.0, rel_err_tol=1e-6,
                       method='fd', step=None, form='forward', step_calc='abs',
                       minimum_step=1e-12, force_dense=True, show_only_incorrect=False,
                       rich_print=True):
        """
        Check partial derivatives comprehensively for all components in your model.

        Parameters
        ----------
        out_stream : file-like object
            Where to send human readable output. By default it goes to stdout.
            Set to None to suppress.
        includes : None or list_like
            List of glob patterns for pathnames to include in the check. Default is None, which
            includes all components in the model.
        excludes : None or list_like
            List of glob patterns for pathnames to exclude from the check. Default is None, which
            excludes nothing.
        compact_print : bool
            Set to True to just print the essentials, one line per input-output pair.
        abs_err_tol : float
            Threshold value for absolute error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Default is 1.0E-6.
        rel_err_tol : float
            Threshold value for relative error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Note at times there may be a
            significant relative error due to a minor absolute error.  Default is 1.0E-6.
        method : str
            Method, 'fd' for finite difference or 'cs' for complex step. Default is 'fd'.
        step : None, float, or list/tuple of float
            Step size(s) for approximation. Default is None, which means 1e-6 for 'fd' and 1e-40 for
            'cs'.
        form : str
            Form for finite difference, can be 'forward', 'backward', or 'central'. Default
            'forward'.
        step_calc : str
            Step type for computing the size of the finite difference step. It can be 'abs' for
            absolute, 'rel_avg' for a size relative to the absolute value of the vector input, or
            'rel_element' for a size relative to each value in the vector input. In addition, it
            can be 'rel_legacy' for a size relative to the norm of the vector.  For backwards
            compatibilty, it can be 'rel', which is now equivalent to 'rel_avg'. Defaults to None,
            in which case the approximation method provides its default value.
        minimum_step : float
            Minimum step size allowed when using one of the relative step_calc options.
        force_dense : bool
            If True, analytic derivatives will be coerced into arrays. Default is True.
        show_only_incorrect : bool, optional
            Set to True if output should print only the subjacs found to be incorrect.
        rich_print : bool, optional
            If True, print using rich if available.

        Returns
        -------
        dict of dicts of dicts
            First key is the component name.
            Second key is the (output, input) tuple of strings.
            The third key is one of:
            'tol violation', 'magnitude', 'J_fd', 'J_fwd', 'J_rev', 'vals_at_max_error',
            and 'rank_inconsistent'.
            For 'tol violation' and 'vals_at_max_error' the value is a tuple containing values for
            forward - fd, adjoint - fd, forward - adjoint. For
            'magnitude' the value is a tuple indicating the maximum magnitude of values found in
            Jfwd, Jrev, and Jfd.
            The boolean 'rank_inconsistent' indicates if the derivative wrt a serial variable is
            inconsistent across MPI ranks.
        """
        model = self.model
        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP or model.iter_count == 0:
            self.run_model()

        if not model._use_derivatives:
            raise RuntimeError(self.msginfo +
                               ": Can't check partials.  Derivative support has been turned off.")

        includes = [includes] if isinstance(includes, str) else includes
        excludes = [excludes] if isinstance(excludes, str) else excludes

        self.set_solver_print(level=0)

        partials_data = {}

        # OPENMDAO_CHECK_ALL_PARTIALS overrides _no_check_partials (used for testing)
        force_check_partials = env_truthy('OPENMDAO_CHECK_ALL_PARTIALS')

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        worst = None
        incorrect_msg = "\n** The following partial Jacobians are incorrect **\n"

        for comp in model.system_iter(typ=Component, include_self=True):
            if comp._no_check_partials and not force_check_partials:
                continue

            # skip any Component with no outputs
            if len(comp._var_allprocs_abs2meta['output']) == 0:
                continue

            # skip any ExplicitComponent with no inputs (e.g. IndepVarComp)
            if (len(comp._var_allprocs_abs2meta['input']) == 0 and
                    isinstance(comp, ExplicitComponent)):
                continue

            if not match_includes_excludes(comp.pathname, includes, excludes):
                continue

            comp_stream = None if out_stream is None else StringIO()

            partials, wrst = comp.check_partials(out_stream=comp_stream,
                                                 compact_print=compact_print,
                                                 abs_err_tol=abs_err_tol, rel_err_tol=rel_err_tol,
                                                 method=method, step=step, form=form,
                                                 step_calc=step_calc,
                                                 minimum_step=minimum_step,
                                                 force_dense=force_dense,
                                                 show_only_incorrect=show_only_incorrect,
                                                 show_worst=False,
                                                 rich_print=rich_print)

            if out_stream is not None:
                comp_content = comp_stream.getvalue()
                if comp_content:
                    if show_only_incorrect:
                        if incorrect_msg:
                            print(incorrect_msg, file=out_stream, end='')
                            incorrect_msg = ''

                    print(comp_content, file=out_stream)

            partials_data.update(partials)

            if wrst is not None:
                if worst is None or wrst[0] > worst[0]:
                    worst = wrst + (type(comp).__name__, comp.pathname)

        if worst is not None:
            _, table_data, headers, col_meta, ctype, cpath = worst
            print(file=out_stream)
            print(add_border(f"Sub Jacobian with Largest Tolerance Violation: {ctype} '{cpath}'",
                             '#'), file=out_stream)
            _print_deriv_table([table_data], headers, out_stream, col_meta=col_meta)

        if step is None or isinstance(step, (float, int)):
            _fix_check_data(partials_data)

        return partials_data

    def check_totals(self, of=None, wrt=None, out_stream=_DEFAULT_OUT_STREAM, compact_print=False,
                     driver_scaling=False, abs_err_tol=0.0, rel_err_tol=1e-6, method='fd',
                     step=None, form=None, step_calc='abs', show_progress=False,
                     show_only_incorrect=False, directional=False, sort=True, rich_print=True):
        """
        Check total derivatives for the model vs. finite difference.

        Parameters
        ----------
        of : list of variable name str or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name str or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        out_stream : file-like object
            Where to send human readable output. By default it goes to stdout.
            Set to None to suppress.
        compact_print : bool
            Set to True to just print the essentials, one line per input-output pair.
        driver_scaling : bool
            When True, return derivatives that are scaled according to either the adder and scaler
            or the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is False, which is unscaled.
        abs_err_tol : float
            Threshold value for absolute error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Default is 1.0E-6.
        rel_err_tol : float
            Threshold value for relative error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Note at times there may be a
            significant relative error due to a minor absolute error.  Default is 1.0E-6.
        method : str
            Method, 'fd' for finite difference or 'cs' for complex step. Default is 'fd'.
        step : None, float, or list/tuple of float
            Step size for approximation. Default is None, which means 1e-6 for 'fd' and 1e-40 for
            'cs'.
        form : str
            Form for finite difference, can be 'forward', 'backward', or 'central'. Default
            None, which defaults to 'forward' for FD.
        step_calc : str
            Step type for computing the size of the finite difference step. It can be 'abs' for
            absolute, 'rel_avg' for a size relative to the absolute value of the vector input, or
            'rel_element' for a size relative to each value in the vector input. In addition, it
            can be 'rel_legacy' for a size relative to the norm of the vector.  For backwards
            compatibilty, it can be 'rel', which is now equivalent to 'rel_avg'. Defaults to None,
            in which case the approximation method provides its default value..
        show_progress : bool
            True to show progress of check_totals.
        show_only_incorrect : bool, optional
            Set to True if output should print only the subjacs found to be incorrect.
        directional : bool
            If True, compute a single directional derivative for each 'of' in rev mode or each
            'wrt' in fwd mode.
        sort : bool
            If True, sort the subjacobian keys alphabetically.
        rich_print : bool, optional
            If True, print using rich if available.

        Returns
        -------
        Dict of Dicts of Tuples of Floats
            First key:
                is the (output, input) tuple of strings;
            Second key:
                'tol violation', 'magnitude', 'J_fd', 'J_fwd', 'J_rev', 'vals_at_max_error',
                and 'rank_inconsistent'.

            For 'tol violation' and 'vals_at_max_error' the value is a tuple containing values for
            forward - fd, reverse - fd, forward - reverse. For 'magnitude' the value is a tuple
            indicating the maximum magnitude of values found in Jfwd, Jrev, and Jfd.
        """
        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        # Check to see if approximation options are the same as that used to compute totals
        # If yes, issue a warning
        if self.model._owns_approx_jac and method in self.model._approx_schemes:
            scheme = self.model._get_approx_scheme(method)

            # get approx options. Fill in with defaults, as needed
            approx_options = scheme.DEFAULT_OPTIONS.copy()
            approx_options.update(self.model._owns_approx_jac_meta)

            # get check options. Fill in with defaults, as needed
            check_options = scheme.DEFAULT_OPTIONS.copy()
            if step:
                check_options['step'] = step
            if method == 'fd':
                if form:
                    check_options['form'] = form
                if step_calc:
                    check_options['step_calc'] = step_calc

            # Compare the approx and check options
            all_same = True
            if approx_options['step'] != check_options['step']:
                all_same = False
            elif method == 'fd':
                if approx_options['form'] != check_options['form']:
                    all_same = False
                if approx_options['step_calc'] != check_options['step_calc']:
                    all_same = False

            if all_same:
                msg = "Checking totals using the same " \
                      "method and options as are used to compute the " \
                      "totals will not provide any relevant " \
                      "information on the " \
                      "accuracy.\n" \
                      "To correct this, change the options to do the " \
                      "check_totals or on the call to approx_totals " \
                      "for the model."

                issue_warning(msg, prefix=self.msginfo,
                              category=OMInvalidCheckDerivativesOptionsWarning)

        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            raise RuntimeError(self.msginfo + ": run_model must be called before total "
                               "derivatives can be checked.")

        model = self.model

        if method == 'cs' and not model._outputs._alloc_complex:
            msg = "\n" + self.msginfo + ": To enable complex step, specify "\
                  "'force_alloc_complex=True' when calling " + \
                  "setup on the problem, e.g. 'problem.setup(force_alloc_complex=True)'"
            raise RuntimeError(msg)

        if model.iter_count < 1:
            self.run_model()

        if wrt is None:
            if not self.driver._designvars:
                raise RuntimeError("Driver is not providing any design variables "
                                   "for compute_totals.")

        lcons = []
        if of is None:
            if not self.driver._responses:
                raise RuntimeError("Driver is not providing any response variables "
                                   "for compute_totals.")
            lcons = [n for n, meta in self.driver._cons.items() if meta['linear']]
            if lcons:
                # if driver has linear constraints, construct a full list of driver responses
                # in order to avoid using any driver coloring that won't include the linear
                # constraints. (The driver coloring would only be used if the supplied of and
                # wrt lists were None or identical to the driver's lists.)
                of = list(self.driver._responses)

        # Calculate Total Derivatives
        total_info = _TotalJacInfo(self, of, wrt, return_format='flat_dict',
                                   approx=model._owns_approx_jac,
                                   driver_scaling=driver_scaling, directional=directional)
        self._metadata['checking'] = True
        try:
            Jcalc = total_info.compute_totals()
        finally:
            self._metadata['checking'] = False
        Jcalc_name = f"J_{total_info.mode}"

        if step is None:
            if method == 'cs':
                steps = [ComplexStep.DEFAULT_OPTIONS['step']]
            else:
                steps = [FiniteDifference.DEFAULT_OPTIONS['step']]
        elif isinstance(step, (float, int)):
            steps = [step]
        else:
            steps = step

        approx = model._owns_approx_jac
        approx_of = model._owns_approx_of
        approx_wrt = model._owns_approx_wrt
        approx_jac_meta = model._owns_approx_jac_meta
        old_jac = model._jacobian
        old_subjacs = model._subjacs_info.copy()
        old_schemes = model._approx_schemes

        Jfds = []
        # prevent form from showing as None in check_totals output
        if form is None and method == 'fd':
            form = FiniteDifference.DEFAULT_OPTIONS['form']

        for step in steps:
            # Approximate FD
            fd_args = {
                'step': step,
                'form': form,
                'step_calc': step_calc,
                'method': method,
                'directional': directional,
            }

            model._approx_schemes = {}

            model.approx_totals(method=method, step=step, form=form,
                                step_calc=step_calc if method == 'fd' else None)
            fd_tot_info = _TotalJacInfo(self, of, wrt, return_format='flat_dict',
                                        approx=True, driver_scaling=driver_scaling,
                                        directional=directional)
            if directional:
                # for fd, use the same fwd mode seeds as the analytical derives used
                fd_tot_info.seeds = total_info.seeds
                Jcalc, Jcalc_slices = total_info._get_as_directional()

            if show_progress:
                Jfd = fd_tot_info.compute_totals(progress_out_stream=out_stream)
            else:
                Jfd = fd_tot_info.compute_totals()

            if directional:
                Jfd, Jfd_slices = fd_tot_info._get_as_directional(total_info.mode)
                Jfds.append((fd_tot_info.J, step))
            else:
                Jfds.append((Jfd, step))

        # reset the _owns_approx_jac flag after approximation is complete.
        if not approx:
            model._jacobian = old_jac
            model._owns_approx_jac = False
            model._owns_approx_of = approx_of
            model._owns_approx_wrt = approx_wrt
            model._owns_approx_jac_meta = approx_jac_meta
            model._subjacs_info = old_subjacs
            model._approx_schemes = old_schemes

        # Assemble and Return all metrics.
        data = {'': {}}
        resp = self.driver._responses
        do_steps = len(Jfds) > 1

        Jcalc_items = Jcalc.items()
        if sort:
            Jcalc_items = sorted(Jcalc_items, key=lambda x: x[0])

        for Jfd, step in Jfds:
            for key, val in Jcalc_items:
                if key not in data['']:
                    data[''][key] = {}
                meta = data[''][key]
                if 'J_fd' not in meta:
                    meta['J_fd'] = []
                    meta['steps'] = []
                meta['steps'].append(step)
                if directional:
                    if self._mode == 'fwd':
                        if 'directional_fd_fwd' not in meta:
                            meta['directional_fd_fwd'] = []
                        _, wrt = key
                        # check directional fwd against fd (one must have negative seed)
                        meta['J_fwd'] = total_info.J[:, Jcalc_slices['wrt'][wrt].start]
                        meta['J_fd'].append(Jfd[:, Jcalc_slices['wrt'][wrt].start])
                        meta['directional_fd_fwd'].append((meta['J_fd'][-1], meta['J_fwd']))
                    else:  # rev
                        if 'directional_fd_rev' not in meta:
                            meta['directional_fd_rev'] = []
                        of, _ = key
                        # check directional rev against fd (different seeds)
                        dhat = total_info.J[Jcalc_slices['of'][of].start, :]  # first row of 'of'
                        d = total_info.seeds['fwd']  # used as direction for fd
                        mhat = Jfd[Jfd_slices['of'][of], 0]
                        m = total_info.seeds['rev'][Jcalc_slices['of'][of]]

                        dhat_dot_d = dhat.dot(d)
                        mhat_dot_m = mhat.dot(m)

                        # Dot product test for adjoint validity.
                        meta['directional_fd_rev'].append((dhat_dot_d, mhat_dot_m))
                        meta['J_rev'] = dhat_dot_d
                        meta['J_fd'].append(mhat_dot_m)
                else:
                    meta[Jcalc_name] = val
                    meta['J_fd'].append(Jfd[key])

                # Display whether indices were declared when response was added.
                of = key[0]
                if of in resp and resp[of]['indices'] is not None:
                    data[''][key]['indices'] = resp[of]['indices'].indexed_src_size

        incon_keys = model._get_inconsistent_keys()

        # force iterator to run so that error info will be added to partials_data
        err_iter = list(_iter_derivs(data[''], show_only_incorrect, fd_args, True,
                                     set(), model.matrix_free, abs_err_tol, rel_err_tol,
                                     incon_keys, sort=sort))

        if out_stream is not None:
            if compact_print:
                _deriv_display_compact(model, err_iter, data[''], out_stream,
                                       totals=True, show_only_incorrect=show_only_incorrect,
                                       rich_print=rich_print)
            else:
                _deriv_display(model, err_iter, data[''], rel_err_tol, abs_err_tol,
                               out_stream, fd_args, totals=True, lcons=lcons,
                               show_only_incorrect=show_only_incorrect, rich_print=rich_print)

        if not do_steps:
            _fix_check_data(data)

        if incon_keys:
            # insert inconsistent key info into first (of, wrt) entry in data to avoid
            # having to change the format of the data dict.
            for key, meta in data[''].items():
                meta['inconsistent_keys'] = incon_keys
                break

        return data['']

    def compute_totals(self, of=None, wrt=None, return_format='flat_dict', debug_print=False,
                       driver_scaling=False, use_abs_names=False, get_remote=True,
                       coloring_info=None):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Parameters
        ----------
        of : list of variable name str or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name str or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : str
            Format to return the derivatives. Can be 'dict', 'flat_dict', or 'array'.
            Default is a 'flat_dict', which returns them in a dictionary whose keys are
            tuples of form (of, wrt).
        debug_print : bool
            Set to True to print out some debug information during linear solve.
        driver_scaling : bool
            When True, return derivatives that are scaled according to either the adder and scaler
            or the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is False, which is unscaled.
        use_abs_names : bool
            This is deprecated and has no effect.
        get_remote : bool
            If True, the default, the full distributed total jacobian will be retrieved.
        coloring_info : ColoringMeta, None, or False
            If False, do no coloring.  If None, use driver coloring info to compute the coloring.
            Otherwise use the given coloring info object to provide the coloring, if it exists.

        Returns
        -------
        object
            Derivatives in form requested by 'return_format'.
        """
        if use_abs_names:
            warn_deprecation("The use_abs_names argument to compute_totals is deprecated and has "
                             "no effect.")

        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            with multi_proc_exception_check(self.comm):
                self.final_setup()

        total_info = _TotalJacInfo(self, of, wrt, return_format, approx=self.model._owns_approx_jac,
                                   driver_scaling=driver_scaling, get_remote=get_remote,
                                   debug_print=debug_print, coloring_info=coloring_info)
        return total_info.compute_totals()

    def set_solver_print(self, level=2, depth=1e99, type_='all', debug_print=None):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int or None
            Iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
            A value of None will leave solving printing unchanged, which is useful
            when using this method to enable or disable debug printing only.
        depth : int
            How deep to recurse. For example, you can set this to 0 if you only want
            to print the top level linear and nonlinear solver messages. Default
            prints everything.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        debug_print : bool or None
            If None, leave solver debug printing unchanged, otherwise turn it on or off
            depending on whether debug_print is True or False. Note debug_print only
            affects nonlinear solvers.
        """
        self.model.set_solver_print(level=level, depth=depth, type_=type_, debug_print=debug_print)

    def list_problem_vars(self,
                          show_promoted_name=True,
                          print_arrays=False,
                          driver_scaling=True,
                          desvar_opts=[],
                          cons_opts=[],
                          objs_opts=[],
                          out_stream=_DEFAULT_OUT_STREAM
                          ):
        """
        Print all design variables and responses (objectives and constraints).

        Parameters
        ----------
        show_promoted_name : bool
            If True, then show the promoted names of the variables.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions.
            Default is False.
        driver_scaling : bool, optional
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.
        desvar_opts : list of str
            List of optional columns to be displayed in the desvars table.
            Allowed values are:
            ['lower', 'upper', 'ref', 'ref0', 'indices', 'adder', 'scaler', 'parallel_deriv_color',
            'cache_linear_solution', 'units', 'min', 'max'].
        cons_opts : list of str
            List of optional columns to be displayed in the cons table.
            Allowed values are:
            ['lower', 'upper', 'equals', 'ref', 'ref0', 'indices', 'adder', 'scaler',
            'linear', 'parallel_deriv_color', 'cache_linear_solution', 'units', 'min', 'max'].
        objs_opts : list of str
            List of optional columns to be displayed in the objs table.
            Allowed values are:
            ['ref', 'ref0', 'indices', 'adder', 'scaler', 'units',
            'parallel_deriv_color', 'cache_linear_solution'].
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        dict
            Name, size, val, and other requested parameters of design variables, constraints,
            and objectives.
        """
        warn_deprecation(msg='Method `list_problem_vars` has been renamed `list_driver_vars`.\n'
                         'Please update your code to use list_driver_vars to avoid this warning.')
        return self.list_driver_vars(show_promoted_name=show_promoted_name,
                                     print_arrays=print_arrays,
                                     driver_scaling=driver_scaling,
                                     desvar_opts=desvar_opts,
                                     cons_opts=cons_opts,
                                     objs_opts=objs_opts,
                                     out_stream=out_stream)

    def list_driver_vars(self,
                         show_promoted_name=True,
                         print_arrays=False,
                         driver_scaling=True,
                         desvar_opts=None,
                         cons_opts=None,
                         objs_opts=None,
                         out_stream=_DEFAULT_OUT_STREAM,
                         return_format='list'
                         ):
        """
        Print all design variables and responses (objectives and constraints).

        Parameters
        ----------
        show_promoted_name : bool
            If True, then show the promoted names of the variables.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions.
            Default is False.
        driver_scaling : bool, optional
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.
        desvar_opts : list of str
            List of optional columns to be displayed in the desvars table. All are displayed by
            defualt.
            Allowed values are:
            ['lower', 'upper', 'ref', 'ref0', 'indices', 'adder', 'scaler', 'parallel_deriv_color',
            'cache_linear_solution', 'units', 'min', 'max'].
        cons_opts : list of str
            List of optional columns to be displayed in the cons table. All are displayed by
            defualt.
            Allowed values are:
            ['lower', 'upper', 'equals', 'ref', 'ref0', 'indices', 'adder', 'scaler',
            'linear', 'parallel_deriv_color', 'cache_linear_solution', 'units', 'min', 'max'].
        objs_opts : list of str
            List of optional columns to be displayed in the objs table. All are displayed by
            defualt.
            Allowed values are:
            ['ref', 'ref0', 'indices', 'adder', 'scaler', 'units',
            'parallel_deriv_color', 'cache_linear_solution'].
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.
        return_format : str
            Indicates the desired format of the return value. Can have value of 'list' or 'dict'.
            If 'dict', the return value is a dictionary mapping {kind: {name: metadata}}.
            If 'list', the return value is a dictionary mapping {kind: [(name, metadata), ...]}.

        Returns
        -------
        dict
            Name, size, val, and other requested parameters of design variables, constraints,
            and objectives.
        """
        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            raise RuntimeError(f"{self.msginfo}: Problem.list_driver_vars() cannot be called "
                               "before `Problem.run_model()`, `Problem.run_driver()`, or "
                               "`Problem.final_setup()`.")

        default_col_names = ['name', 'val', 'size']

        # Design vars
        desvars = self.driver._designvars
        vals = self.driver.get_design_var_values(get_remote=True, driver_scaling=driver_scaling)
        if not driver_scaling:
            desvars = deepcopy(desvars)
            for meta in desvars.values():
                scaler = meta['scaler'] if meta.get('scaler') is not None else 1.
                adder = meta['adder'] if meta.get('adder') is not None else 0.
                if 'lower' in meta:
                    meta['lower'] = meta['lower'] / scaler - adder
                if 'upper' in meta:
                    meta['upper'] = meta['upper'] / scaler - adder
        header = "Design Variables"
        if desvar_opts is None:
            desvar_opts = ['lower', 'upper', 'ref', 'ref0', 'indices', 'adder', 'scaler',
                           'parallel_deriv_color', 'cache_linear_solution', 'units', 'min', 'max']
        def_desvar_opts = [opt for opt in ('indices',) if opt not in desvar_opts and
                           _find_dict_meta(desvars, opt)]
        desvar_opts = [opt for opt in desvar_opts if _find_dict_meta(desvars, opt)]
        col_names = default_col_names + def_desvar_opts + desvar_opts
        if out_stream:
            self._write_var_info_table(header, col_names, desvars, vals,
                                       show_promoted_name=show_promoted_name,
                                       print_arrays=print_arrays,
                                       col_spacing=2, out_stream=out_stream)

        des_vars = [[i, j] for i, j in desvars.items()]
        for d in des_vars:
            d[1] = {i: j for i, j in d[1].items() if i in col_names}
            d[1]['val'] = vals[d[0]]
        des_vars = [tuple(d) for d in des_vars]

        # Constraints
        cons = self.driver._cons
        vals = self.driver.get_constraint_values(driver_scaling=driver_scaling)
        if not driver_scaling:
            cons = deepcopy(cons)
            for meta in cons.values():
                scaler = meta['scaler'] if meta.get('scaler') is not None else 1.
                adder = meta['adder'] if meta.get('adder') is not None else 0.
                if 'lower' in meta:
                    meta['lower'] = meta['lower'] / scaler - adder
                if 'upper' in meta:
                    meta['upper'] = meta['upper'] / scaler - adder
        header = "Constraints"
        if cons_opts is None:
            cons_opts = ['lower', 'upper', 'equals', 'ref', 'ref0', 'indices', 'adder', 'scaler',
                         'linear', 'parallel_deriv_color', 'cache_linear_solution', 'units', 'min',
                         'max']
        # detect any cons that use aliases
        def_cons_opts = [opt for opt in ('indices', 'alias') if opt not in cons_opts and
                         _find_dict_meta(cons, opt)]
        cons_opts = [opt for opt in cons_opts if _find_dict_meta(cons, opt)]
        col_names = default_col_names + def_cons_opts + cons_opts
        if out_stream:
            self._write_var_info_table(header, col_names, cons, vals,
                                       show_promoted_name=show_promoted_name,
                                       print_arrays=print_arrays,
                                       col_spacing=2, out_stream=out_stream)

        cons_vars = [[i, j] for i, j in cons.items()]
        for c in cons_vars:
            c[1] = {i: j for i, j in c[1].items() if i in col_names}
            c[1]['val'] = vals[c[0]]
        cons_vars = [tuple(c) for c in cons_vars]

        objs = self.driver._objs
        vals = self.driver.get_objective_values(driver_scaling=driver_scaling)
        header = "Objectives"
        if objs_opts is None:
            objs_opts = ['ref', 'ref0', 'indices', 'adder', 'scaler', 'units',
                         'parallel_deriv_color', 'cache_linear_solution']
        def_obj_opts = [opt for opt in ('indices',) if opt not in objs_opts and
                        _find_dict_meta(objs, opt)]
        objs_opts = [opt for opt in objs_opts if _find_dict_meta(objs, opt)]
        col_names = default_col_names + def_obj_opts + objs_opts
        if out_stream:
            self._write_var_info_table(header, col_names, objs, vals,
                                       show_promoted_name=show_promoted_name,
                                       print_arrays=print_arrays,
                                       col_spacing=2, out_stream=out_stream)

        obj_vars = [[i, j] for i, j in objs.items()]
        for o in obj_vars:
            o[1] = {i: j for i, j in o[1].items() if i in col_names}
            o[1]['val'] = vals[o[0]]
        obj_vars = [tuple(o) for o in obj_vars]

        if return_format == 'dict':
            prob_vars = {'design_vars': {k: v for k, v in des_vars},
                         'constraints': {k: v for k, v in cons_vars},
                         'objectives': {k: v for k, v in obj_vars}}
        else:
            prob_vars = {'design_vars': des_vars,
                         'constraints': cons_vars,
                         'objectives': obj_vars}

        return prob_vars

    def _write_var_info_table(self, header, col_names, meta, vals, print_arrays=False,
                              show_promoted_name=True, col_spacing=1,
                              out_stream=_DEFAULT_OUT_STREAM):
        """
        Write a table of information for the problem variable in meta and vals.

        Parameters
        ----------
        header : str
            The header line for the table.
        col_names : list of str
            List of column labels.
        meta : dict
            Dictionary of metadata for each problem variable.
        vals : dict
            Dictionary of values for each problem variable.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions
            Default is False.
        show_promoted_name : bool
            If True, then show the promoted names of the variables.
        col_spacing : int
            Number of spaces between columns in the table.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.
        """
        if out_stream is None:
            return
        elif out_stream is _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout
        elif not isinstance(out_stream, TextIOBase):
            raise TypeError("Invalid output stream specified for 'out_stream'")

        resolver = self.model._resolver

        # Gets the current numpy print options for consistent decimal place
        #   printing between arrays and floats
        print_options = np.get_printoptions()
        np_precision = print_options['precision']

        # Get the values for all the elements in the tables
        rows = []
        for name, meta in meta.items():

            row = {}
            vname = meta['name'] if meta.get('alias') else name

            for col_name in col_names:
                if col_name == 'name':
                    if show_promoted_name:
                        if resolver.is_local(vname, 'input'):
                            row[col_name] = resolver.abs2prom(vname, 'input')
                        elif resolver.is_local(vname, 'output'):
                            row[col_name] = resolver.abs2prom(vname, 'output')
                        else:
                            # Promoted auto_ivc name. Keep it promoted
                            row[col_name] = vname
                    else:
                        row[col_name] = vname

                elif col_name == 'val':
                    row[col_name] = vals[name]
                elif col_name == 'min':
                    min_val = min(vals[name])
                    # Rounding to match float precision to numpy precision
                    row[col_name] = np.round(min_val, np_precision)
                elif col_name == 'max':
                    max_val = max(vals[name])
                    # Rounding to match float precision to numpy precision
                    row[col_name] = np.round(max_val, np_precision)
                else:
                    row[col_name] = meta[col_name]
            rows.append(row)

        col_space = ' ' * col_spacing
        print(add_border(header, '-'), file=out_stream)

        # loop through the rows finding the max widths
        max_width = {}
        for col_name in col_names:
            max_width[col_name] = len(col_name)
        for row in rows:
            for col_name in col_names:
                cell = row[col_name]
                if isinstance(cell, np.ndarray) and cell.size > 1:
                    norm = np.linalg.norm(cell)
                    out = f'|{np.round(norm, np_precision)}|'
                else:
                    out = str(cell)
                max_width[col_name] = max(len(out), max_width[col_name])

        # print col headers
        header_div = ''
        header_col_names = ''
        for col_name in col_names:
            header_div += '-' * max_width[col_name] + col_space
            header_col_names += pad_name(col_name, max_width[col_name], quotes=False) + col_space
        print(header_col_names, file=out_stream)
        print(header_div[:-1], file=out_stream)

        # print rows with var info
        for row in rows:
            have_array_values = []  # keep track of which values are arrays
            row_string = ''
            for col_name in col_names:
                cell = row[col_name]
                if isinstance(cell, np.ndarray) and cell.size > 1:
                    norm = np.linalg.norm(cell)
                    out = f'|{np.round(norm, np_precision)}|'
                    have_array_values.append(col_name)
                else:
                    out = str(cell)
                row_string += pad_name(out, max_width[col_name], quotes=False) + col_space
            print(row_string, file=out_stream)

            if print_arrays:
                spaces = (max_width['name'] + col_spacing) * ' '
                for col_name in have_array_values:
                    print(f"{spaces}{col_name}:", file=out_stream)
                    print(textwrap.indent(pprint.pformat(row[col_name]), spaces), file=out_stream)
                    print(file=out_stream)

        print(file=out_stream)

    def load_case(self, case):
        """
        Pull all input and output variables from a case into the model.

        Parameters
        ----------
        case : Case or dict
            A Case from a CaseReader, or a dictionary with key 'inputs' mapped to the
            output of problem.model.list_inputs and key 'outputs' mapped to the output
            of prob.model.list_outputs. Both list_inputs and list_outputs should be called
            with `prom_name=True` and `return_format='dict'`.
        """
        model = self.model

        # if model overrides load_case, then call the overloaded method
        if overrides_method('load_case', model, System):
            model.load_case(case)
            return

        # find all subsystems that override the load_case method
        system_overrides = {}
        for subsys in model.system_iter(include_self=False, recurse=True):
            if overrides_method('load_case', subsys, System):
                system_overrides[subsys.pathname] = subsys

        def set_later(var_name):
            # determine if variable should be set later via an overridden load_case method
            for pathname in system_overrides:
                if var_name.startswith(pathname + '.'):
                    return True
            return False

        case_is_dict = isinstance(case, dict)

        if case_is_dict:
            # case data comes from list_inputs/list_outputs, keyed on absolute pathname
            # we need it to be keyed on promoted name
            if 'inputs' in case:
                inputs = {meta['prom_name']: meta for meta in case['inputs'].values()}
            else:
                inputs = None
            if 'outputs' in case:
                outputs = {meta['prom_name']: meta for meta in case['outputs'].values()}
            else:
                outputs = None
        else:
            inputs = case.inputs
            outputs = case.outputs

        abs2idx = model._var_allprocs_abs2idx
        resolver = self.model._resolver

        if inputs:
            for abs_name in inputs:
                if set_later(abs_name):
                    continue

                if resolver.is_abs(abs_name, 'input'):
                    if case_is_dict:
                        val = inputs[abs_name]['val']
                    else:
                        val = case.inputs[abs_name]

                    if model.comm.size > 1 and resolver.flags(abs_name, 'input') & DISTRIBUTED:
                        sizes = model._var_sizes['input'][:, abs2idx[abs_name]]
                        model.set_val(abs_name, scatter_dist_to_local(val, model.comm, sizes))
                    else:
                        model.set_val(abs_name, val)
                else:
                    issue_warning(f"{model.msginfo}: Input variable, '{abs_name}', recorded "
                                  "in the case is not found in the model.")

        if outputs:
            for name in outputs:
                if set_later(name):
                    continue

                if resolver.is_prom(name):
                    if case_is_dict:
                        val = outputs[name]['val']
                    else:
                        val = outputs[name]

                    for abs_name in resolver.absnames(name):
                        if set_later(abs_name):
                            continue

                        if model.comm.size > 1 and resolver.flags(abs_name) & DISTRIBUTED:
                            sizes = model._var_sizes['output'][:, abs2idx[abs_name]]
                            model.set_val(abs_name, scatter_dist_to_local(val, model.comm, sizes))
                        else:
                            model.set_val(abs_name, val)

                else:
                    issue_warning(f"{model.msginfo}: Output variable, '{name}', recorded "
                                  "in the case is not found in the model.")

        # call the overridden load_case method on applicable subsystems (in top-down order)
        for sys_name in sorted(system_overrides.keys()):
            system_overrides[sys_name].load_case(case)

    def check_config(self, logger=None, checks=_default_checks, out_file='openmdao_checks.out'):
        """
        Perform optional error checks on a Problem.

        Parameters
        ----------
        logger : object
            Logging object.
        checks : list of str or None or the str 'all'
            Determines what config checks are run.
            If None, no checks are run
            If list of str, run those config checks
            If ‘all’, all the checks ('auto_ivc_warnings', 'comp_has_no_outputs', 'cycles',
            'dup_inputs', 'missing_recorders', 'out_of_order', 'promotions', 'solvers',
            'system', 'unconnected_inputs') are run.
        out_file : str or None
            If not None, output will be written to this file in addition to stdout.
        """
        if checks is None:
            return

        reports_dir_exists = os.path.isdir(self.get_reports_dir())
        check_file_path = None
        if logger is None:
            if out_file is not None:
                if reports_dir_exists:
                    check_file_path = str(self.get_reports_dir() / out_file)
                else:
                    check_file_path = out_file
            logger = get_logger('check_config', out_file=check_file_path, use_format=True)

        if checks == 'all':
            checks = sorted(_all_non_redundant_checks)

        if logger is None and checks:
            if out_file is None:
                check_file_path = None
            else:
                check_file_path = str(self.get_outputs_dir(mkdir=True) / out_file)
            logger = get_logger('check_config', out_file=check_file_path, use_format=True)

        for c in checks:
            if c not in _all_checks:
                print(f"WARNING: '{c}' is not a recognized check.  Available checks are: "
                      f"{sorted(_all_checks)}")
                continue
            logger.info(f'checking {c}...')
            beg = time.perf_counter()
            _all_checks[c](self, logger)
            end = time.perf_counter()
            logger.info(f"    {c} check complete ({(end - beg):.6f} sec).")

        if checks and check_file_path is not None and reports_dir_exists:
            # turn text file written to reports dir into an html file to be viewable from the
            # 'openmdao view_reports' command
            with open(check_file_path, 'r') as f:
                txt = f.read()

            path = self.get_reports_dir() / 'checks.html'
            with open(path, 'w') as f:
                f.write(text2html(txt))

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        if self._metadata is None or \
           self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            raise RuntimeError(f"{self.msginfo}: set_complex_step_mode cannot be called before "
                               "`Problem.run_model()`, `Problem.run_driver()`, or "
                               "`Problem.final_setup()`.")

        if active and not self.model._outputs._alloc_complex:
            raise RuntimeError(f"{self.msginfo}: To enable complex step, specify "
                               "'force_alloc_complex=True' when calling setup on the problem, "
                               "e.g. 'problem.setup(force_alloc_complex=True)'")

        self.model._set_complex_step_mode(active)

    def get_reports_dir(self, force=False):
        """
        Get the path to the directory where the report files should go.

        If it doesn't exist, it will be created.

        Parameters
        ----------
        force : bool
            If True, create the reports directory if it doesn't exist, even if this Problem does
            not have any active reports. This can happen when running testflo.

        Returns
        -------
        pathlib.Path
            The path to the directory where reports should be written.
        """
        return self.get_outputs_dir('reports', mkdir=force or len(self._reports) > 0)

    def get_outputs_dir(self, *subdirs, mkdir=False):
        """
        Get the path under which all output files of this problem are to be placed.

        Parameters
        ----------
        *subdirs : str
            Subdirectories nested under the relevant problem output directory.
            To create {prob_output_dir}/a/b one would pass `prob.get_outputs_dir('a', 'b')`.
        mkdir : bool
            If True, attempt to create this directory if it does not exist.

        Returns
        -------
        pathlib.Path
           The path of the outputs directory for the problem.
        """
        return _get_outputs_dir(self, *subdirs, mkdir=mkdir)

    def get_coloring_dir(self, mode, mkdir=False):
        """
        Get the path to the directory for the coloring files.

        Parameters
        ----------
        mode : str
            Must be one of 'input' or 'output'. A problem will always write its coloring files to
            its standard output directory in `{prob_name}_out/coloring_files`, but input coloring
            files to be loaded may be read from a different directory specifed by the problem's
            `coloring_dir` option.
        mkdir : bool
            If True, attempt to create this directory if it does not exist.

        Returns
        -------
        pathlib.Path
            The path to the directory where reports should be written.
        """
        if mode == 'input':
            return pathlib.Path(self.options['coloring_dir'])
        elif mode == 'output':
            return self.get_outputs_dir('coloring_files', mkdir=mkdir)
        else:
            raise ValueError(f"{self.msginfo}: get_coloring_dir requires mode"
                             "to be one of 'input' or 'output'.")

    def list_indep_vars(self, include_design_vars=True, options=None,
                        print_arrays=False, out_stream=_DEFAULT_OUT_STREAM):
        """
        Retrieve the independent variables in the Problem.

        Returns a dictionary mapping the promoted names of indep_vars which the user is
        expected to provide to the metadata for the associated independent variable.

        A output is designated as an independent variable if it is tagged with
        'openmdao:indep_var'. This includes IndepVarComp by default, and users are
        able to apply this tag to their own component outputs if they wish
        to provide components with IndepVarComp-like capability.

        Parameters
        ----------
        include_design_vars : bool
            If True, include design variables in the list of problem inputs.
            The user may provide values for these but ultimately they will
            be overwritten by the Driver.
            Default is False.
        options : list of str or None
            List of optional columns to be displayed in the independent variable table.
            Allowed values are:
            ['name', 'units', 'shape', 'size', 'desc', 'ref', 'ref0', 'res_ref',
            'distributed', 'lower', 'upper', 'tags', 'shape_by_conn', 'copy_shape', 'compute_shape',
            'global_size', 'global_shape', 'value'].
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        dict
            A dictionary mapping the promoted names of all independent variables
            in the model to their metadata.
        """
        model = self.model
        if model._outputs is None:
            raise RuntimeError("list_indep_vars requires that final_setup has been "
                               "run for the Problem.")

        design_vars = model.get_design_vars(recurse=True, use_prom_ivc=True, get_sizes=False)

        problem_indep_vars = []

        col_names = ['name', 'units', 'val']
        if options is not None:
            col_names.extend(options)

        abs2meta = model._var_allprocs_abs2meta['output']
        abs2disc = model._var_allprocs_discrete['output']
        abs2prom = model._resolver.abs2prom

        seen = set()
        for absname, meta in chain(abs2meta.items(), abs2disc.items()):
            if 'openmdao:indep_var' in meta['tags']:
                name = abs2prom(absname, 'output')
                if (include_design_vars or name not in design_vars) and name not in seen:
                    meta = {key: meta[key] for key in col_names if key in meta}
                    meta['val'] = self.get_val(name)
                    problem_indep_vars.append((name, meta))
                    seen.add(name)

        if out_stream is not None:
            header = f'Problem {self._name} Independent Variables'
            if problem_indep_vars:
                meta = {key: meta for key, meta in problem_indep_vars}
                vals = {key: self.get_val(key) for key in meta}
                self._write_var_info_table(header, col_names, meta, vals, print_arrays=print_arrays,
                                           show_promoted_name=True, col_spacing=1,
                                           out_stream=out_stream)
            else:
                if out_stream is _DEFAULT_OUT_STREAM:
                    out_stream = sys.stdout
                hr = '-' * len(header)
                print(f'{hr}\n{header}\n{hr}', file=out_stream)
                print('None found', file=out_stream)

        return problem_indep_vars

    def list_pre_post(self, outfile=None):
        """
        Display the pre and post optimization components.

        Parameters
        ----------
        outfile : file-like or str or None
            Where to send human readable output. Default is None, which sends output to stdout.
        """
        if self._metadata is None or self._metadata['setup_status'] < _SetupStatus.POST_SETUP:
            raise RuntimeError(f"{self.msginfo}: list_pre_post can't be called before setup.")

        if outfile is None:
            out = sys.stdout
        else:
            out = open(outfile, 'w')

        model = self.model
        if model._pre_components:
            print("\nPre-optimization components:", file=out)
            for name in sorted(model._pre_components):
                print(f"    {name}", file=out)
        else:
            print("\nPre-optimization components: []", file=out)

        if model._post_components:
            print("\nPost-optimization components:", file=out)
            for name in sorted(model._post_components):
                print(f"    {name}", file=out)
        else:
            print("\nPost-optimization components: []", file=out)

    def _any_rank_has_saved_errors(self):
        """
        Return True if any rank has saved errors.

        Returns
        -------
        bool
            True if any rank has errors.
        """
        if self._metadata is None:
            return False
        else:
            if MPI and self.comm is not None and self.comm.size > 1:
                if self._metadata['saved_errors'] is None:
                    nerrs = 0
                else:
                    nerrs = len(self._metadata['saved_errors'])
                return self.comm.allreduce(nerrs, op=MPI.SUM) > 0
            else:
                return bool(self._metadata['saved_errors'])

    def _get_unique_saved_errors(self):
        """
        Get a list of unique saved errors.

        Returns
        -------
        list
            List of unique saved errors.
        """
        unique_errors = []
        if self._metadata is not None:
            if self._any_rank_has_saved_errors():
                # traceback won't pickle, so convert to string
                if self.comm.size > 1:
                    saved = [(ident, msg, exc_type, ''.join(traceback.format_tb(tback)))
                             for ident, msg, exc_type, tback in self._metadata['saved_errors']]
                    all_errors = self.comm.allgather(saved)
                else:
                    all_errors = [self._metadata['saved_errors']]

                seen = set()
                for errors in all_errors:
                    for ident, msg, exc_type, tback in errors:
                        if (ident is None and msg not in seen) or ident not in seen:
                            unique_errors.append((ident, msg, exc_type, tback))
                            seen.add(ident)
                            seen.add(msg)

        return unique_errors

    def get_total_coloring(self, coloring_info=None, of=None, wrt=None, run_model=None):
        """
        Get the total coloring.

        If necessary, dynamically generate it.

        Parameters
        ----------
        coloring_info : dict
            Coloring metadata dict.
        of : list of str or None
            List of response names.
        wrt : list of str or None
            List of design variable names.
        run_model : bool or None
            If False, don't run model.  If None, use problem._run_counter to determine if model
            should be run.

        Returns
        -------
        Coloring or None
            Coloring object, possibly dynamically generated, or None.
        """
        if coloring_mod._use_total_sparsity:
            coloring = None
            # if no coloring_info is supplied, copy the coloring_info from the driver but
            # remove any existing coloring, and force dynamic coloring
            if coloring_info is None:
                coloring_info = self.driver._coloring_info.copy()
                coloring_info.coloring = None
                coloring_info.dynamic = True

            if coloring_info.do_compute_coloring():
                if coloring_info.dynamic:
                    do_run = run_model if run_model is not None else self._run_counter < 0
                    coloring = \
                        coloring_mod.dynamic_total_coloring(
                            self.driver, run_model=do_run,
                            fname=self.model.get_coloring_fname(mode='output'),
                            of=of, wrt=wrt)
            else:
                return coloring_info.coloring

            return coloring


def _fix_check_data(data):
    """
    Modify the data dict to match the old format if there is only one fd step size.

    Parameters
    ----------
    data : dict
        Dictionary containing derivative information keyed by system name.
    """
    for sdata in data.values():
        for dct in sdata.values():
            _fix_comp_check_data(dct)
