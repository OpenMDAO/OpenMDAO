"""Define the Problem class and a FakeComm class for non-MPI users."""

import __main__
import shutil

import sys
import pprint
import os
import weakref
import pathlib
import textwrap
import traceback

from collections import defaultdict, namedtuple
from itertools import product

from io import StringIO, TextIOBase

import numpy as np
import scipy.sparse as sparse

from openmdao.core.constants import _SetupStatus
from openmdao.core.component import Component
from openmdao.core.driver import Driver, record_iteration, SaveOptResult
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.system import System, _OptStatus
from openmdao.core.group import Group
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.core.constants import _DEFAULT_OUT_STREAM, _UNDEFINED
from openmdao.jacobians.dictionary_jacobian import _CheckingJacobian
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.solvers.solver import SolverInfo
from openmdao.vectors.default_vector import DefaultVector
from openmdao.error_checking.check_config import _default_checks, _all_checks, \
    _all_non_redundant_checks
from openmdao.recorders.recording_iteration_stack import _RecIteration
from openmdao.recorders.recording_manager import RecordingManager, record_viewer_data, \
    record_model_options
from openmdao.utils.mpi import MPI, FakeComm, multi_proc_exception_check, check_mpi_env
from openmdao.utils.name_maps import name2abs_names
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.units import simplify_unit
from openmdao.utils.name_maps import abs_key2rel_key
from openmdao.utils.logger_utils import get_logger, TestLogger
from openmdao.utils.hooks import _setup_hooks, _reset_all_hooks
from openmdao.utils.record_util import create_local_meta
from openmdao.utils.array_utils import scatter_dist_to_local
from openmdao.utils.class_util import overrides_method
from openmdao.utils.reports_system import get_reports_to_activate, activate_reports, \
    clear_reports, get_reports_dir, _load_report_plugins
from openmdao.utils.general_utils import pad_name, LocalRangeIterable, \
    _find_dict_meta, env_truthy, add_border, match_includes_excludes, inconsistent_across_procs
from openmdao.utils.om_warnings import issue_warning, DerivativesWarning, warn_deprecation, \
    OMInvalidCheckDerivativesOptionsWarning
import openmdao.utils.coloring as coloring_mod
from openmdao.visualization.tables.table_builder import generate_table

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.utils.name_maps import rel_key2abs_key, rel_name2abs_name


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

# Used to keep track of the current Problem tree if there are any subproblems
_prob_setup_stack = []


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


class Problem(object):
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
    _warned : bool
        Bool to check if `value` deprecation warning has occured yet
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
        self._warned = False
        self._computing_coloring = False

        # Set the Problem name so that it can be referenced from command line tools (e.g. check)
        # that accept a Problem argument, and to name the corresponding reports subdirectory.

        if name:  # if name hasn't been used yet, use it. Otherwise, error
            if name not in _problem_names:
                self._name = name
            else:
                raise ValueError(f"The problem name '{name}' already exists")
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
        _problem_names.append(self._name)

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

        self.comm = comm

        self._metadata = None
        self._run_counter = -1
        self._rec_mgr = RecordingManager()

        # General options
        self.options = OptionsDictionary(parent_name=type(self).__name__)
        self.options.declare('coloring_dir', types=str,
                             default=os.path.join(os.getcwd(), 'coloring_files'),
                             desc='Directory containing coloring files (if any) for this Problem.')
        self.options.declare('group_by_pre_opt_post', types=bool,
                             default=False,
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

        # Start a run by deleting any existing reports so that the files
        #   that are in that directory are all from this run and not a previous run
        reports_dirpath = pathlib.Path(get_reports_dir()).joinpath(f'{self._name}')
        if self.comm.rank == 0:
            if os.path.isdir(reports_dirpath):
                shutil.rmtree(reports_dirpath)

        # register hooks for any reports
        activate_reports(self._reports, self)

        # So Problem and driver can have hooks attached to their methods
        _setup_hooks(self)

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

    def _get_var_abs_name(self, name):
        if name in self.model._var_allprocs_abs2meta:
            return name
        elif name in self.model._var_allprocs_prom2abs_list['output']:
            return self.model._var_allprocs_prom2abs_list['output'][name][0]
        elif name in self.model._var_allprocs_prom2abs_list['input']:
            abs_names = self.model._var_allprocs_prom2abs_list['input'][name]
            if len(abs_names) == 1:
                return abs_names[0]
            else:
                raise KeyError(f"{self.msginfo}: Using promoted name `{name}' is ambiguous and "
                               f"matches unconnected inputs {sorted(abs_names)}. Use absolute name "
                               "to disambiguate.")

        raise KeyError(f'{self.msginfo}: Variable "{name}" not found.')

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
        if self._metadata is None:
            raise RuntimeError(f"{self.msginfo}: is_local('{name}') was called before setup() "
                               "completed.")

        try:
            abs_name = self._get_var_abs_name(name)
        except KeyError:
            sub = self.model._get_subsystem(name)
            return sub is not None and sub._is_local

        # variable exists, but may be remote
        return abs_name in self.model._var_abs2meta['input'] or \
            abs_name in self.model._var_abs2meta['output']

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

    def get_val(self, name, units=None, indices=None, get_remote=False):
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

        Returns
        -------
        object
            The value of the requested output/input variable.
        """
        if self._metadata['setup_status'] == _SetupStatus.POST_SETUP:
            abs_names = name2abs_names(self.model, name)
            if abs_names:
                val = self.model._get_cached_val(name, abs_names, get_remote=get_remote)
                if val is not _UNDEFINED:
                    if indices is not None:
                        val = val[indices]
                    if units is not None:
                        val = self.model.convert2units(name, val, simplify_unit(units))
            else:
                raise KeyError(f'{self.model.msginfo}: Variable "{name}" not found.')
        else:
            val = self.model.get_val(name, units=units, indices=indices, get_remote=get_remote,
                                     from_src=True)

        if val is _UNDEFINED:
            if get_remote:
                raise KeyError(f'{self.msginfo}: Variable name "{name}" not found.')
            else:
                raise RuntimeError(f"{self.model.msginfo}: Variable '{name}' is not local to "
                                   f"rank {self.comm.rank}. You can retrieve values from "
                                   "other processes using `get_val(<name>, get_remote=True)`.")

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
                if system is None:
                    self.model.set_val(pathname + '.' + name, value, units=set_units)
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

        if self._mode is None:
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

        if self._mode is None:
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

    def compute_jacvec_product(self, of, wrt, mode, seed):
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

        rvec = self.model._vectors[rkind]['linear']
        lvec = self.model._vectors[lkind]['linear']

        rvec.set_val(0.)

        conns = self.model._conn_global_abs_in2out

        # set seed values into dresids (fwd) or doutputs (rev)
        # seed may have keys that are inputs and must be converted into auto_ivcs
        try:
            seed[rnames[0]]
        except (IndexError, TypeError):
            for i, name in enumerate(rnames):
                if name in conns:
                    rvec[conns[name]] = seed[i]
                else:
                    rvec[name] = seed[i]
        else:
            for name in rnames:
                if name in conns:
                    rvec[conns[name]] = seed[name]
                else:
                    rvec[name] = seed[name]

        # We apply a -1 here because the derivative of the output is minus the derivative of
        # the residual in openmdao.
        data = rvec.asarray()
        data *= -1.

        self.model.run_solve_linear(mode)

        if mode == 'fwd':
            return {n: lvec[n].copy() for n in lnames}
        else:
            # may need to convert some lnames to auto_ivc names
            return {n: lvec[conns[n] if n in conns else n].copy() for n in lnames}

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

    def setup(self, check=False, logger=None, mode='auto', force_alloc_complex=False,
              distributed_vector_class=PETScVector, local_vector_class=DefaultVector,
              derivatives=True):
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
            If None or False, no checks are run
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

        model_comm = self.driver._setup_comm(comm)

        # this metadata will be shared by all Systems/Solvers in the system tree
        self._metadata = {
            'name': self._name,  # the name of this Problem
            'pathname': None,  # the pathname of this Problem in the current tree of Problems
            'comm': comm,
            'coloring_dir': self.options['coloring_dir'],  # directory for coloring files
            'recording_iter': _RecIteration(comm.rank),  # manager of recorder iterations
            'local_vector_class': local_vector_class,
            'distributed_vector_class': distributed_vector_class,
            'solver_info': SolverInfo(),
            'use_derivatives': derivatives,
            'force_alloc_complex': force_alloc_complex,  # forces allocation of complex vectors
            'vars_to_gather': {},  # vars that are remote somewhere. does not include distrib vars
            'prom2abs': {'input': {}, 'output': {}},  # includes ALL promotes including buried ones
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
            'reports_dir': self.get_reports_dir(),  # directory where reports will be written
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
            'group_by_pre_opt_post': self.options['group_by_pre_opt_post'],  # see option
            'relevance_cache': {},  # cache of relevance objects
            'rel_array_cache': {},  # cache of relevance arrays
            'ncompute_totals': 0,  # number of times compute_totals has been called
        }

        if _prob_setup_stack:
            self._metadata['pathname'] = _prob_setup_stack[-1]._metadata['pathname'] + '/' + \
                self._name
        else:
            self._metadata['pathname'] = self._name

        _prob_setup_stack.append(self)
        try:
            model._setup(model_comm, self._metadata)
        finally:
            _prob_setup_stack.pop()

            # whenever we're outside of model._setup, static mode should be True so that anything
            # added outside of _setup will persist.
            self._metadata['static_mode'] = True

        # Cache all args for final setup.
        self._check = check
        self._logger = logger

        self._metadata['setup_status'] = _SetupStatus.POST_SETUP

        self._check_collected_errors()

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

        response_size, desvar_size = driver._update_voi_meta(self.model)

        # update mode if it's been set to 'auto'
        if self._orig_mode == 'auto':
            mode = 'rev' if response_size < desvar_size else 'fwd'
        else:
            mode = self._orig_mode

        self._metadata['mode'] = mode

        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            self.model._final_setup()

        # If set_solver_print is called after an initial run, in a multi-run scenario,
        #  this part of _final_setup still needs to happen so that change takes effect
        #  in subsequent runs
        if self._metadata['setup_status'] >= _SetupStatus.POST_FINAL_SETUP:
            self.model._setup_solver_print()

        driver._setup_driver(self)

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
        if self._metadata['setup_status'] >= _SetupStatus.POST_SETUP:
            driver._setup_recording()
            self._setup_recording()
            record_viewer_data(self)

        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            self._metadata['setup_status'] = _SetupStatus.POST_FINAL_SETUP
            self._set_initial_conditions()

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

    def check_partials(self, out_stream=_DEFAULT_OUT_STREAM, includes=None, excludes=None,
                       compact_print=False, abs_err_tol=1e-6, rel_err_tol=1e-6,
                       method='fd', step=None, form='forward', step_calc='abs',
                       minimum_step=1e-12, force_dense=True, show_only_incorrect=False):
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

        Returns
        -------
        dict of dicts of dicts
            First key is the component name.
            Second key is the (output, input) tuple of strings.
            Third key is one of ['rel error', 'abs error', 'magnitude', 'J_fd', 'J_fwd', 'J_rev',
            'rank_inconsistent'].
            For 'rel error', 'abs error', and 'magnitude' the value is a tuple containing norms for
            forward - fd, adjoint - fd, forward - adjoint.
            For 'J_fd', 'J_fwd', 'J_rev' the value is a numpy array representing the computed
            Jacobian for the three different methods of computation.
            The boolean 'rank_inconsistent' indicates if the derivative wrt a serial variable is
            inconsistent across MPI ranks.
        """
        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            self.final_setup()

        model = self.model

        if not model._use_derivatives:
            raise RuntimeError(self.msginfo +
                               ": Can't check partials.  Derivative support has been turned off.")

        # TODO: Once we're tracking iteration counts, run the model if it has not been run before.

        includes = [includes] if isinstance(includes, str) else includes
        excludes = [excludes] if isinstance(excludes, str) else excludes

        comps = []

        # OPENMDAO_CHECK_ALL_PARTIALS overrides _no_check_partials (used for testing)
        force_check_partials = env_truthy('OPENMDAO_CHECK_ALL_PARTIALS')

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

            comps.append(comp)

        # Check to make sure the method and settings used for checking
        #   is different from the method used to calc the derivatives
        # Could do this later in this method but at that point some computations could have been
        #   made and it would just waste time before the user is told there is an error and the
        #   program errs out
        requested_method = method
        alloc_complex = model._outputs._alloc_complex
        abs2meta_in = model._var_allprocs_abs2meta['input']
        abs2meta_out = model._var_allprocs_abs2meta['output']

        for comp in comps:
            local_opts = comp._get_check_partial_options()

            for keypats, meta in comp._declared_partials_patterns.items():

                # Get the complete set of options, including defaults
                #    for the computing of the derivs for this component
                if 'method' not in meta:
                    meta_with_defaults = {}
                    meta_with_defaults['method'] = 'exact'
                elif meta['method'] == 'cs':
                    meta_with_defaults = ComplexStep.DEFAULT_OPTIONS.copy()
                else:
                    meta_with_defaults = FiniteDifference.DEFAULT_OPTIONS.copy()
                meta_with_defaults.update(meta)

                _, wrtpats = keypats
                # For each of the partials, check to see if the
                #   check partials options are different than the options used to compute
                #   the partials
                for _, wrtvars in comp._find_wrt_matches(wrtpats):
                    for var in wrtvars:
                        # we now have individual vars like 'x'
                        # get the options for checking partials
                        fd_options, _ = _get_fd_options(var, requested_method, local_opts, step,
                                                        form, step_calc, alloc_complex,
                                                        minimum_step)
                        # compare the compute options to the check options
                        if fd_options['method'] != meta_with_defaults['method']:
                            all_same = False
                        else:
                            all_same = True
                            if fd_options['method'] == 'fd':
                                option_names = ['form', 'step', 'step_calc', 'minimum_step',
                                                'directional']
                            else:
                                option_names = ['step', 'directional']
                            for name in option_names:
                                if fd_options[name] != meta_with_defaults[name]:
                                    all_same = False
                                    break
                        if all_same:
                            msg = f"Checking partials with respect " \
                                  f"to variable '{var}' in component " \
                                  f"'{comp.pathname}' using the same " \
                                  "method and options as are used to compute the " \
                                  "component's derivatives " \
                                  "will not provide any relevant information on the " \
                                  "accuracy.\n" \
                                  "To correct this, change the options to do the \n" \
                                  "check_partials using either:\n" \
                                  "     - arguments to Problem.check_partials. \n" \
                                  "     - arguments to Component.set_check_partial_options"

                            issue_warning(msg, prefix=self.msginfo,
                                          category=OMInvalidCheckDerivativesOptionsWarning)

        self.set_solver_print(level=0)

        # This is a defaultdict of (defaultdict of dicts).
        partials_data = defaultdict(lambda: defaultdict(dict))

        # Caching current point to restore after setups.
        input_cache = model._inputs.asarray(copy=True)
        output_cache = model._outputs.asarray(copy=True)

        # Keep track of derivative keys that are declared dependent so that we don't print them
        # unless they are in error.
        indep_key = {}

        # Directional derivative directions for matrix free comps.
        mfree_directions = {}

        # Analytic Jacobians
        print_reverse = False
        for mode in ('fwd', 'rev'):
            model._inputs.set_val(input_cache)
            model._outputs.set_val(output_cache)
            # Make sure we're in a valid state
            model.run_apply_nonlinear()

            jac_key = 'J_' + mode

            for comp in comps:

                # Only really need to linearize once.
                if mode == 'fwd':
                    comp.run_linearize(sub_do_ln=False)

                matrix_free = comp.matrix_free
                c_name = comp.pathname
                if mode == 'fwd':
                    indep_key[c_name] = set()

                with comp._unscaled_context():

                    of_list = comp._get_partials_ofs()
                    wrt_list = comp._get_partials_wrts()

                    # Matrix-free components need to calculate their Jacobian by matrix-vector
                    # product.
                    if matrix_free:
                        print_reverse = True
                        local_opts = comp._get_check_partial_options()

                        dstate = comp._doutputs
                        if mode == 'fwd':
                            dinputs = comp._dinputs
                            doutputs = comp._dresiduals
                            in_list = wrt_list
                            out_list = of_list
                        else:
                            dinputs = comp._dresiduals
                            doutputs = comp._dinputs
                            in_list = of_list
                            out_list = wrt_list

                        for inp in in_list:
                            inp_abs = rel_name2abs_name(comp, inp)
                            if mode == 'fwd':
                                directional = inp in local_opts and local_opts[inp]['directional']
                            else:
                                directional = c_name in mfree_directions

                            try:
                                flat_view = dinputs._abs_get_val(inp_abs)
                            except KeyError:
                                # Implicit state
                                flat_view = dstate._abs_get_val(inp_abs)

                            if directional:
                                n_in = 1
                                idxs = range(1)
                                if c_name not in mfree_directions:
                                    mfree_directions[c_name] = {}

                                if inp in mfree_directions[c_name]:
                                    perturb = mfree_directions[c_name][inp]
                                else:
                                    perturb = 2.0 * np.random.random(len(flat_view)) - 1.0
                                    mfree_directions[c_name][inp] = perturb

                            else:
                                n_in = len(flat_view)
                                idxs = LocalRangeIterable(comp, inp_abs, use_vec_offset=False)
                                perturb = 1.0

                            for idx in idxs:

                                dinputs.set_val(0.0)
                                dstate.set_val(0.0)

                                if directional:
                                    flat_view[:] = perturb
                                elif idx is not None:
                                    flat_view[idx] = perturb

                                # Matrix Vector Product
                                self._metadata['checking'] = True
                                try:
                                    comp.run_apply_linear(mode)
                                finally:
                                    self._metadata['checking'] = False

                                for out in out_list:
                                    out_abs = rel_name2abs_name(comp, out)

                                    try:
                                        derivs = doutputs._abs_get_val(out_abs)
                                    except KeyError:
                                        # Implicit state
                                        derivs = dstate._abs_get_val(out_abs)

                                    if mode == 'fwd':
                                        key = out, inp
                                        deriv = partials_data[c_name][key]

                                        # Allocate first time
                                        if jac_key not in deriv:
                                            shape = (len(derivs), n_in)
                                            deriv[jac_key] = np.zeros(shape)

                                        if idx is not None:
                                            deriv[jac_key][:, idx] = derivs

                                    else:  # rev
                                        key = inp, out
                                        deriv = partials_data[c_name][key]

                                        if directional:
                                            # Dot product test for adjoint validity.
                                            m = mfree_directions[c_name][out]
                                            d = mfree_directions[c_name][inp]
                                            mhat = derivs
                                            dhat = deriv['J_fwd'][:, idx]

                                            deriv['directional_fwd_rev'] = mhat.dot(m) - dhat.dot(d)
                                        else:
                                            meta = abs2meta_in[out_abs] if out_abs in abs2meta_in \
                                                else abs2meta_out[out_abs]
                                            if not meta['distributed']:  # serial input or state
                                                if inconsistent_across_procs(comp.comm, derivs,
                                                                             return_array=False):
                                                    deriv['rank_inconsistent'] = True

                                        # Allocate first time
                                        if jac_key not in deriv:
                                            shape = (n_in, len(derivs))
                                            deriv[jac_key] = np.zeros(shape)

                                        if idx is not None:
                                            deriv[jac_key][idx, :] = derivs

                    # These components already have a Jacobian with calculated derivatives.
                    else:

                        if mode == 'rev':
                            # Skip reverse mode because it is not different than forward.
                            continue

                        subjacs = comp._jacobian._subjacs_info

                        for rel_key in product(of_list, wrt_list):
                            abs_key = rel_key2abs_key(comp, rel_key)
                            of, wrt = abs_key

                            # No need to calculate partials; they are already stored
                            try:
                                deriv_value = subjacs[abs_key]['val']
                                rows = subjacs[abs_key]['rows']
                            except KeyError:
                                deriv_value = rows = None

                            # Testing for pairs that are not dependent so that we suppress printing
                            # them unless the fd is non zero. Note: subjacs_info is empty for
                            # undeclared partials, which is the default behavior now.
                            try:
                                if not subjacs[abs_key]['dependent']:
                                    indep_key[c_name].add(rel_key)
                            except KeyError:
                                indep_key[c_name].add(rel_key)

                            if wrt in comp._var_abs2meta['input']:
                                wrt_meta = comp._var_abs2meta['input'][wrt]
                            else:
                                wrt_meta = comp._var_abs2meta['output'][wrt]

                            if deriv_value is None:
                                # Missing derivatives are assumed 0.
                                in_size = wrt_meta['size']
                                out_size = comp._var_abs2meta['output'][of]['size']
                                deriv_value = np.zeros((out_size, in_size))

                            if force_dense:
                                if rows is not None:
                                    try:
                                        in_size = wrt_meta['size']
                                    except KeyError:
                                        in_size = wrt_meta['size']
                                    out_size = comp._var_abs2meta['output'][of]['size']
                                    tmp_value = np.zeros((out_size, in_size))
                                    # if a scalar value is provided (in declare_partials),
                                    # expand to the correct size array value for zipping
                                    if deriv_value.size == 1:
                                        deriv_value *= np.ones(rows.size)
                                    for i, j, val in zip(rows, subjacs[abs_key]['cols'],
                                                         deriv_value):
                                        tmp_value[i, j] += val
                                    deriv_value = tmp_value

                                elif sparse.issparse(deriv_value):
                                    deriv_value = deriv_value.todense()

                            partials_data[c_name][rel_key][jac_key] = deriv_value.copy()

        model._inputs.set_val(input_cache)
        model._outputs.set_val(output_cache)
        model.run_apply_nonlinear()

        # Finite Difference to calculate Jacobian
        if step is None or isinstance(step, (float, int)):
            steps = [step]
        else:
            steps = step

        do_steps = len(steps) > 1

        alloc_complex = model._outputs._alloc_complex
        all_fd_options = {}
        comps_could_not_cs = set()
        requested_method = method
        for comp in comps:

            c_name = comp.pathname
            all_fd_options[c_name] = {}

            of = comp._get_partials_ofs()
            wrt = comp._get_partials_wrts()

            actual_steps = defaultdict(list)

            for i, step in enumerate(steps):
                approximations = {'fd': FiniteDifference(),
                                  'cs': ComplexStep()}

                added_wrts = set()

                # Load up approximation objects with the requested settings.

                local_opts = comp._get_check_partial_options()
                for rel_key in product(of, wrt):
                    abs_key = rel_key2abs_key(comp, rel_key)
                    local_wrt = rel_key[1]

                    fd_options, could_not_cs = _get_fd_options(local_wrt, requested_method,
                                                               local_opts, step, form, step_calc,
                                                               alloc_complex, minimum_step)

                    actual_steps[rel_key].append(fd_options['step'])

                    if could_not_cs:
                        comps_could_not_cs.add(c_name)

                    # Determine if fd or cs.
                    method = requested_method

                    all_fd_options[c_name][local_wrt] = fd_options
                    if c_name in mfree_directions:
                        vector = mfree_directions[c_name].get(local_wrt)
                    else:
                        vector = None

                    # prevent adding multiple approxs with same wrt (and confusing users with
                    # warnings)
                    if abs_key[1] not in added_wrts:
                        approximations[fd_options['method']].add_approximation(abs_key, comp,
                                                                               fd_options,
                                                                               vector=vector)
                        added_wrts.add(abs_key[1])

                approx_jac = _CheckingJacobian(comp)
                for approximation in approximations.values():
                    # Perform the FD here.
                    approximation.compute_approximations(comp, jac=approx_jac)

                with multi_proc_exception_check(comp.comm):
                    if approx_jac._errors:
                        raise RuntimeError('\n'.join(approx_jac._errors))

                for abs_key, partial in approx_jac.items():
                    rel_key = abs_key2rel_key(comp, abs_key)
                    deriv = partials_data[c_name][rel_key]
                    _of, _wrt = rel_key

                    if 'J_fd' not in deriv:
                        deriv['J_fd'] = []
                        deriv['steps'] = []
                    deriv['J_fd'].append(partial)
                    deriv['steps'] = actual_steps[rel_key]

                    # If this is a directional derivative, convert the analytic to a directional
                    # one.
                    if _wrt in local_opts and local_opts[_wrt]['directional']:
                        if i == 0:  # only do this on the first iteration
                            deriv[f'J_fwd'] = np.atleast_2d(np.sum(deriv['J_fwd'], axis=1)).T

                        if comp.matrix_free:
                            if i == 0:  # only do this on the first iteration
                                deriv['J_rev'] = np.atleast_2d(np.sum(deriv['J_rev'], axis=0)).T

                            # Dot product test for adjoint validity.
                            m = mfree_directions[c_name][_of].flatten()
                            d = mfree_directions[c_name][_wrt].flatten()
                            mhat = partial.flatten()
                            dhat = deriv['J_rev'].flatten()

                            if 'directional_fd_rev' not in deriv:
                                deriv['directional_fd_rev'] = []
                            deriv['directional_fd_rev'].append(dhat.dot(d) - mhat.dot(m))

        # Conversion of defaultdict to dicts
        partials_data = {comp_name: dict(data) for comp_name, data in partials_data.items()}

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if len(comps_could_not_cs) > 0:
            msg = "The following components requested complex step, but force_alloc_complex " + \
                  "has not been set to True, so finite difference was used: "
            msg += str(list(comps_could_not_cs))
            msg += "\nTo enable complex step, specify 'force_alloc_complex=True' when calling " + \
                   "setup on the problem, e.g. 'problem.setup(force_alloc_complex=True)'"
            issue_warning(msg, category=DerivativesWarning)

        _assemble_derivative_data(partials_data, rel_err_tol, abs_err_tol, out_stream,
                                  compact_print, comps, all_fd_options, indep_key=indep_key,
                                  print_reverse=print_reverse,
                                  show_only_incorrect=show_only_incorrect)

        if not do_steps:
            _fix_check_data(partials_data)

        return partials_data

    def check_totals(self, of=None, wrt=None, out_stream=_DEFAULT_OUT_STREAM, compact_print=False,
                     driver_scaling=False, abs_err_tol=1e-6, rel_err_tol=1e-6, method='fd',
                     step=None, form=None, step_calc='abs', show_progress=False,
                     show_only_incorrect=False, directional=False, sort=False):
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

        Returns
        -------
        Dict of Dicts of Tuples of Floats

            First key:
                is the (output, input) tuple of strings;
            Second key:
                is one of ['rel error', 'abs error', 'magnitude', 'fdstep'];

            For 'rel error', 'abs error', 'magnitude' the value is: A tuple containing norms for
                forward - fd, adjoint - fd, forward - adjoint.
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

        # TODO: Once we're tracking iteration counts, run the model if it has not been run before.

        if wrt is None:
            if not self.driver._designvars:
                raise RuntimeError("Driver is not providing any design variables "
                                   "for compute_totals.")

        lcons = []
        if of is None:
            if not self.driver._responses:
                raise RuntimeError("Driver is not providing any response variables "
                                   "for compute_totals.")
            lcons = [n for n, meta in self.driver._cons.items()
                     if ('linear' in meta and meta['linear'])]
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
                        directional_fd_fwd = total_info.J[:, Jcalc_slices['wrt'][wrt].start] - \
                            Jfd[:, Jcalc_slices['wrt'][wrt].start]
                        meta['directional_fd_fwd'].append(directional_fd_fwd)
                        meta['J_fwd'] = total_info.J[:, Jcalc_slices['wrt'][wrt].start]
                        meta['J_fd'].append(Jfd[:, Jcalc_slices['wrt'][wrt].start])
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
                        meta['directional_fd_rev'].append(dhat_dot_d - mhat_dot_m)
                        meta['J_rev'] = dhat_dot_d
                        meta['J_fd'].append(mhat_dot_m)
                else:
                    meta[Jcalc_name] = val
                    meta['J_fd'].append(Jfd[key])

                # Display whether indices were declared when response was added.
                of = key[0]
                if of in resp and resp[of]['indices'] is not None:
                    data[''][key]['indices'] = resp[of]['indices'].indexed_src_size

        _assemble_derivative_data(data, rel_err_tol, abs_err_tol, out_stream, compact_print,
                                  [model], {'': fd_args}, totals=total_info, lcons=lcons,
                                  show_only_incorrect=show_only_incorrect, sort=sort)

        if not do_steps:
            _fix_check_data(data)

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

    def set_solver_print(self, level=2, depth=1e99, type_='all'):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            Iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        depth : int
            How deep to recurse. For example, you can set this to 0 if you only want
            to print the top level linear and nonlinear solver messages. Default
            prints everything.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        self.model.set_solver_print(level=level, depth=depth, type_=type_)

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
        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            raise RuntimeError(f"{self.msginfo}: Problem.list_driver_vars() cannot be called "
                               "before `Problem.run_model()`, `Problem.run_driver()`, or "
                               "`Problem.final_setup()`.")

        default_col_names = ['name', 'val', 'size']

        # Design vars
        desvars = self.driver._designvars
        vals = self.driver.get_design_var_values(get_remote=True, driver_scaling=driver_scaling)
        header = "Design Variables"
        def_desvar_opts = [opt for opt in ('indices',) if opt not in desvar_opts and
                           _find_dict_meta(desvars, opt)]
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
        header = "Constraints"
        # detect any cons that use aliases
        def_cons_opts = [opt for opt in ('indices', 'alias') if opt not in cons_opts and
                         _find_dict_meta(cons, opt)]
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
        def_obj_opts = [opt for opt in ('indices',) if opt not in objs_opts and
                        _find_dict_meta(objs, opt)]
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

        abs2prom = self.model._var_abs2prom

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
                        if vname in abs2prom['input']:
                            row[col_name] = abs2prom['input'][vname]
                        elif vname in abs2prom['output']:
                            row[col_name] = abs2prom['output'][vname]
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

        if isinstance(case, dict):
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
        prom2abs_in = model._var_allprocs_prom2abs_list['input']
        prom2abs_out = model._var_allprocs_prom2abs_list['output']
        abs2meta_in = model._var_allprocs_abs2meta['input']
        abs2meta_out = model._var_allprocs_abs2meta['output']
        abs2meta_disc_in = model._var_allprocs_discrete['input']
        abs2meta_disc_out = model._var_allprocs_discrete['output']

        if inputs:
            for name in inputs:
                if set_later(name):
                    continue

                if name in prom2abs_in:
                    for abs_name in prom2abs_in[name]:
                        if set_later(abs_name):
                            continue

                        if isinstance(case, dict):
                            val = inputs[name]['val']
                        else:
                            # need a unique abs_name to get value from a case
                            # if there is a matching abs_name in the case, use that value
                            # otherwise use the value of the first matching abs_name
                            case_abs_names = case._prom2abs['input'][name]
                            if abs_name in case_abs_names:
                                val = case.inputs[abs_name]
                            else:
                                val = case.inputs[case_abs_names[0]]
                        try:
                            varmeta = abs2meta_in[abs_name]
                        except KeyError:
                            # Var may be discrete
                            varmeta = abs2meta_disc_in[abs_name]
                        if varmeta['distributed'] and model.comm.size > 1:
                            sizes = model._var_sizes['input'][:, abs2idx[abs_name]]
                            model.set_val(abs_name, scatter_dist_to_local(val, model.comm, sizes))
                        else:
                            model.set_val(abs_name, val)
                else:
                    issue_warning(f"{model.msginfo}: Input variable, '{name}', recorded "
                                  "in the case is not found in the model.")

        if outputs:
            for name in outputs:
                if set_later(name):
                    continue

                # auto_ivc output may point to a promoted input name
                if name in prom2abs_out:
                    prom2abs = prom2abs_out
                    loc = 'output'
                else:
                    prom2abs = prom2abs_in
                    loc = 'input'

                if name in prom2abs:
                    if isinstance(case, dict):
                        val = outputs[name]['val']
                    else:
                        val = outputs[name]

                    for abs_name in prom2abs[name]:
                        if set_later(abs_name):
                            continue
                        

                        if loc == 'output':
                            try:
                                varmeta = abs2meta_out[abs_name]
                            except KeyError:
                                varmeta = abs2meta_disc_out[abs_name]
                        else:
                            try:
                                varmeta = abs2meta_in[abs_name]
                            except KeyError:
                                varmeta = abs2meta_disc_in[abs_name]
                        if 'distributed' in varmeta:
                            if varmeta['distributed'] and model.comm.size > 1:
                                sizes = model._var_sizes['output'][:, abs2idx[abs_name]]
                                model.set_val(abs_name, scatter_dist_to_local(val, model.comm, sizes))
                            else:
                                model.set_val(abs_name, val)
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

        if logger is None:
            logger = get_logger('check_config', out_file=out_file, use_format=True)

        if checks == 'all':
            checks = sorted(_all_non_redundant_checks)

        for c in checks:
            if c not in _all_checks:
                print(f"WARNING: '{c}' is not a recognized check.  Available checks are: "
                      f"{sorted(_all_checks)}")
                continue
            logger.info(f'checking {c}')
            _all_checks[c](self, logger)

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
        str
            The path to the directory where reports should be written.
        """
        reports_dirpath = pathlib.Path(get_reports_dir()).joinpath(f'{self._name}')

        if self.comm.rank == 0 and (force or self._reports):
            pathlib.Path(reports_dirpath).mkdir(parents=True, exist_ok=True)

        return reports_dirpath

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

        design_vars = model.get_design_vars(recurse=True,
                                            use_prom_ivc=True,
                                            get_sizes=False)

        problem_indep_vars = []
        indep_var_names = set()

        col_names = ['name', 'units', 'val']
        if options is not None:
            col_names.extend(options)

        abs2meta = model._var_allprocs_abs2meta['output']

        prom2src = {}
        for prom in self.model._var_allprocs_prom2abs_list['input']:
            src = model.get_source(prom)
            if 'openmdao:indep_var' in abs2meta[src]['tags']:
                prom2src[prom] = src

        for prom, src in prom2src.items():
            name = prom if src.startswith('_auto_ivc.') else src

            if (include_design_vars or name not in design_vars) \
                    and name not in indep_var_names:
                meta = abs2meta[src]
                meta = {key: meta[key] for key in col_names if key in meta}
                meta['val'] = self.get_val(prom)
                problem_indep_vars.append((name, meta))
                indep_var_names.add(name)

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
                print(f'None found', file=out_stream)

        return problem_indep_vars

    def iter_count_iter(self, include_driver=True, include_solvers=True, include_systems=False):
        """
        Yield iteration counts for driver, solvers and/or systems.

        Parameters
        ----------
        include_driver : bool
            If True, include the driver in the iteration counts.
        include_solvers : bool
            If True, include solvers in the iteration counts.
        include_systems : bool
            If True, include systems in the iteration counts.

        Yields
        ------
        str
            Name of the object.
        str
            Name of the counter.
        int
            Value of the counter.
        """
        if include_driver:
            yield ('Driver', 'iter_count', self.driver.iter_count)
        if include_solvers or include_systems:
            for s in self.model.system_iter(include_self=True, recurse=True):
                if include_systems:
                    for it in ('iter_count', 'iter_count_apply'):
                        val = getattr(s, it)
                        if val > 0:
                            yield (s.pathname, it, val)

                if include_solvers:
                    prefix = s.pathname + '.' if s.pathname else ''
                    if s.nonlinear_solver is not None and s.nonlinear_solver._iter_count > 0:
                        yield (prefix + 'nonlinear_solver', '_iter_count',
                               s.nonlinear_solver._iter_count)
                    if s.linear_solver is not None and s.linear_solver._iter_count > 0:
                        yield (prefix + 'linear_solver', '_iter_count',
                               s.linear_solver._iter_count)

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
                            fname=self.driver._get_total_coloring_fname(), of=of, wrt=wrt)
            else:
                return coloring_info.coloring

            return coloring


_ErrorTuple = namedtuple('ErrorTuple', ['forward', 'reverse', 'forward_reverse'])
_MagnitudeTuple = namedtuple('MagnitudeTuple', ['forward', 'reverse', 'fd'])


def _compute_deriv_errors(derivative_info, matrix_free, directional, totals):
    """
    Compute the errors between derivatives that were computed using different modes or methods.

    Error information in the derivative_info dict is updated by this function.

    Parameters
    ----------
    derivative_info : dict
        Metadata dict corresponding to a particular (of, wrt) pair.
    matrix_free : bool
        True if the current dirivatives are computed in a matrix free manner.
    directional : bool
        True if the current dirivtives are directional.
    totals : bool or _TotalJacInfo
        _TotalJacInfo if the current derivatives are total derivatives.

    Returns
    -------
    float
        The norm of the FD jacobian.
    """
    nan = float('nan')

    def safe_norm(arr):
        return 0. if arr is None or arr.size == 0 else np.linalg.norm(arr)

    Jforward = derivative_info.get('J_fwd')
    Jreverse = derivative_info.get('J_rev')
    forward = Jforward is not None
    reverse = Jreverse is not None

    rev_norm = fwd_norm = fwd_rev_error = None
    calc_norm = 0.
    if reverse:
        rev_norm = calc_norm = safe_norm(Jreverse)
    if forward:
        fwd_norm = calc_norm = safe_norm(Jforward)

    try:
        fdinfo = derivative_info['J_fd']
        steps = derivative_info['steps']
    except KeyError:
        # this can happen when a partial is not declared, which means it should be zero
        fdinfo = (np.zeros(1),)
        steps = (None,)

    if matrix_free:
        if directional:
            if forward and reverse:
                fwd_rev_error = safe_norm(derivative_info['directional_fwd_rev'])
            else:
                fwd_rev_error = None
        elif not totals:
            fwd_rev_error = safe_norm(Jforward - Jreverse)

    derivative_info['abs error'] = []
    derivative_info['rel error'] = []
    derivative_info['magnitude'] = []
    derivative_info['steps'] = []
    fdnorms = []

    for i, fd in enumerate(fdinfo):
        step = steps[i]
        fd_norm = safe_norm(fd)
        fdnorms.append(fd_norm)

        fwd_error = rev_error = None

        if reverse and not directional:
            rev_error = safe_norm(Jreverse - fd)

        if forward:
            fwd_error = safe_norm(Jforward - fd)

        if directional:
            if reverse:
                rev_error = safe_norm(derivative_info['directional_fd_rev'][i])
                if not totals:
                    rev_norm = None
            if forward and totals:
                fwd_error = safe_norm(derivative_info['directional_fd_fwd'][i])

        derivative_info['abs error'].append(_ErrorTuple(fwd_error, rev_error, fwd_rev_error))
        derivative_info['magnitude'].append(_MagnitudeTuple(fwd_norm, rev_norm, fd_norm))
        derivative_info['steps'].append(step)

        # If fd_norm is zero, let's use calc_norm as the divisor for the relative
        # error check. That way we don't accidentally squelch a legitimate problem.
        div_norm = fd_norm if fd_norm != 0. else calc_norm

        if div_norm == 0.:
            derivative_info['rel error'].append(_ErrorTuple(None if fwd_error is None else nan,
                                                            None if rev_error is None else nan,
                                                            None if fwd_rev_error is None else nan))
        else:
            if matrix_free and not totals:
                derivative_info['rel error'].append(_ErrorTuple(fwd_error / div_norm,
                                                                rev_error / div_norm,
                                                                fwd_rev_error / div_norm))
            else:
                derivative_info['rel error'].append(_ErrorTuple(
                    None if fwd_error is None else fwd_error / div_norm,
                    None if rev_error is None else rev_error / div_norm,
                    None if fwd_rev_error is None else fwd_rev_error / div_norm))

    return np.max(fdnorms)


def _errors_above_tol(deriv_info, abs_error_tol, rel_error_tol):
    """
    Return if either abs or rel tolerances are violated when comparing a group of derivatives.

    Parameters
    ----------
    deriv_info : dict
        Metadata dict corresponding to a particular (of, wrt) pair.
    abs_error_tol : float
        Absolute error tolerance.
    rel_error_tol : float
        Relative error tolerance.

    Returns
    -------
    bool
        True if absolute tolerance is violated.
    bool
        True if relative tolerance is violated.
    """
    abs_errs = deriv_info['abs error']
    rel_errs = deriv_info['rel error']

    above_abs = above_rel = False

    for abs_err in abs_errs:
        for error in abs_err:
            if error is not None and not np.isnan(error) and error >= abs_error_tol:
                above_abs = True
                break
        if above_abs:
            break

    for rel_err in rel_errs:
        for error in rel_err:
            if error is not None and not np.isnan(error) and error >= rel_error_tol:
                above_rel = True
                break
        if above_rel:
            break

    return above_abs, above_rel


def _iter_derivs(derivatives, sys_name, show_only_incorrect, global_options, totals,
                 matrix_free, abs_error_tol=1e-6, rel_error_tol=1e-6, incon_keys=()):
    """
    Iterate over all of the derivatives.

    If show_only_incorrect is True, only the derivatives with abs or rel errors outside of
    tolerance or derivatives wrt serial variables that are inconsistent across ranks will be
    returned.

    Parameters
    ----------
    derivatives : dict
        Dict of metadata for derivative groups, keyed on (of, wrt) pairs.
    sys_name : str
        Name of the current system.
    show_only_incorrect : bool
        If True, yield only derivatives with errors outside of tolerance.
    global_options : dict
        Dictionary containing the options for the approximation.
    totals : bool or _TotalJacInfo
        Set to _TotalJacInfo if we are doing check_totals to skip a bunch of stuff.
    matrix_free : bool
        True if the system computes matrix free derivatives.
    abs_error_tol : float
        Absolute error tolerance.
    rel_error_tol : float
        Relative error tolerance.
    incon_keys : set or tuple
        Keys where there are serial d_inputs variables that are inconsistent across processes.

    Yields
    ------
    tuple
        The (of, wrt) pair for the current derivatives being compared.
    float
        The FD norm.
    dict
        The FD options.
    bool
        True if the current derivatives are directional.
    bool
        True if the differences for the current derivatives are above the absolute error tolerance.
    bool
        True if the differences for the current derivatives are above the relative error tolerance.
    bool
        True if the current derivative was computed where some serial d_inputs variables were not
        consistent across processes.
    """
    # Sorted keys ensures deterministic ordering
    sorted_keys = sorted(derivatives)

    for key in sorted_keys:
        _, wrt = key

        inconsistent = False
        derivative_info = derivatives[key]

        if totals:
            fd_opts = global_options['']
        else:
            fd_opts = global_options[sys_name][wrt]
            if key in incon_keys:
                inconsistent = True

        directional = bool(fd_opts) and fd_opts.get('directional')

        fd_norm = _compute_deriv_errors(derivative_info, matrix_free, directional, totals)

        above_abs, above_rel = _errors_above_tol(derivative_info, abs_error_tol, rel_error_tol)

        if show_only_incorrect and not (above_abs or above_rel or inconsistent):
            continue

        yield key, fd_norm, fd_opts, directional, above_abs, above_rel, inconsistent


def _fix_check_data(data):
    """
    Modify the data dict to match the old format if there is only one fd step size.

    Parameters
    ----------
    data : dict
        Dictionary containing derivative information keyed by system name.
    """
    names = ['J_fd', 'abs error', 'rel error', 'magnitude', 'directional_fd_fwd',
             'directional_fd_rev']

    for sdata in data.values():
        for dct in sdata.values():
            for name in names:
                if name in dct:
                    dct[name] = dct[name][0]
            if 'steps' in dct:
                del dct['steps']


def _assemble_derivative_data(derivative_data, rel_error_tol, abs_error_tol, out_stream,
                              compact_print, system_list, global_options, totals=False,
                              indep_key=None, print_reverse=False,
                              show_only_incorrect=False, lcons=None, sort=False):
    """
    Compute the relative and absolute errors in the given derivatives and print to the out_stream.

    Parameters
    ----------
    derivative_data : dict
        Dictionary containing derivative information keyed by system name.
    rel_error_tol : float
        Relative error tolerance.
    abs_error_tol : float
        Absolute error tolerance.
    out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
    compact_print : bool
        If results should be printed verbosely or in a table.
    system_list : iterable
        The systems (in the proper order) that were checked.
    global_options : dict
        Dictionary containing the options for the approximation.
    totals : bool or _TotalJacInfo
        Set to _TotalJacInfo if we are doing check_totals to skip a bunch of stuff.
    indep_key : dict of sets, optional
        Keyed by component name, contains the of/wrt keys that are declared not dependent.
    print_reverse : bool, optional
        Set to True if compact_print results need to include columns for reverse mode.
    show_only_incorrect : bool, optional
        Set to True if output should print only the subjacs found to be incorrect.
    lcons : list or None
        For total derivatives only, list of outputs that are actually linear constraints.
    sort : bool
        If True, sort subjacobian keys alphabetically.
    """
    suppress_output = out_stream is None

    # Keep track of the worst subjac in terms of relative error for fwd and rev
    if not suppress_output and show_only_incorrect:
        if totals:
            out_stream.write('\n** Only writing information about incorrect total derivatives **'
                             '\n\n')
        else:
            out_stream.write('\n** Only writing information about components with '
                             'incorrect Jacobians **\n\n')

    worst_subjac = None
    incon_keys = ()

    for system in system_list:
        # Match header to appropriate type.
        if isinstance(system, Component):
            sys_type = 'Component'
        elif isinstance(system, Group):
            sys_type = 'Group'
        else:
            raise RuntimeError(f"Object type {type(system).__name__} is not a Component or Group.")

        sys_name = system.pathname
        sys_class_name = type(system).__name__
        matrix_free = system.matrix_free

        if sys_name not in derivative_data:
            issue_warning(f"No derivative data found for {sys_type} '{sys_name}'.",
                          category=DerivativesWarning)
            continue

        derivatives = derivative_data[sys_name]

        if totals:
            sys_name = 'Full Model'

        incon_keys = system._get_inconsistent_keys()

        num_bad_jacs = 0  # Keep track of number of bad derivative values for each component

        # Need to capture the output of a component's derivative
        # info so that it can be used if that component is the
        # worst subjac. That info is printed at the bottom of all the output
        out_buffer = StringIO()

        if not suppress_output:
            num_format = '{: 1.4e}'
            num_col_meta = {'format': num_format}

            if totals:
                title = f"Total Derivatives"
            else:
                title = f"{sys_type}: {sys_class_name} '{sys_name}'"

            print(f"{add_border(title, '-')}\n", file=out_buffer)

            table_data = []

        for key, fd_norm, fd_opts, directional, above_abs, above_rel, inconsistent in \
                _iter_derivs(derivatives, sys_name, show_only_incorrect,
                             global_options, totals, matrix_free,
                             abs_error_tol, rel_error_tol, incon_keys):

            # Skip printing the non-dependent keys if the derivatives are fine.
            if not compact_print:
                if indep_key and key in indep_key[sys_name] and fd_norm < abs_error_tol:
                    del derivatives[key]
                    continue

            of, wrt = key
            derivative_info = derivatives[key]

            if above_abs or above_rel or inconsistent:
                num_bad_jacs += 1

            if suppress_output:
                continue

            # Informative output for responses that were declared with an index.
            indices = derivative_info.get('indices')
            if indices is not None:
                of = f'{of} (index size: {indices})'

            abs_errs = derivative_info['abs error']
            rel_errs = derivative_info['rel error']
            magnitudes = derivative_info['magnitude']
            steps = derivative_info['steps']

            if len(steps) > 1:
                stepstrs = [f", step={step}" for step in steps]
            else:
                stepstrs = [""]

            if magnitudes[0].reverse is not None:
                Jrev = derivative_info['J_rev']

            # use forward even if both fwd and rev are defined
            if magnitudes[0].forward is not None:
                Jfor = derivative_info['J_fwd']

            if isinstance(wrt, str):
                wrt = f"'{wrt}'"
            if isinstance(of, str):
                of = f"'{of}'"

            if directional:
                wrt = f"(d){wrt}"

            if compact_print:
                err_desc = []
                if above_abs:
                    err_desc.append(' >ABS_TOL')
                if above_rel:
                    err_desc.append(' >REL_TOL')
                if inconsistent:
                    err_desc.append(' <RANK INCONSISTENT>')
                err_desc = ''.join(err_desc)

                for i in range(len(magnitudes)):
                    if magnitudes[0].reverse is not None:
                        calc_mag = magnitudes[i].reverse
                        calc_abs = abs_errs[i].reverse
                        calc_rel = rel_errs[i].reverse

                    # use forward even if both fwd and rev are defined
                    if magnitudes[0].forward is not None:
                        calc_mag = magnitudes[i].forward
                        calc_abs = abs_errs[i].forward
                        calc_rel = rel_errs[i].forward

                    if totals:
                        if len(steps) > 1:
                            table_data.append([of, wrt, steps[i], calc_mag, magnitudes[i].fd,
                                               calc_abs, calc_rel, err_desc])
                        else:
                            table_data.append([of, wrt, calc_mag, magnitudes[i].fd,
                                               calc_abs, calc_rel, err_desc])
                    else:
                        if print_reverse:
                            if len(steps) > 1:
                                table_data.append([of, wrt, steps[i], magnitudes[i].forward,
                                                   magnitudes[i].reverse, magnitudes[i].fd,
                                                   abs_errs[i].forward, abs_errs[i].reverse,
                                                   abs_errs[i].forward_reverse, rel_errs[i].forward,
                                                   rel_errs[i].reverse, rel_errs[i].forward_reverse,
                                                   err_desc])
                            else:
                                table_data.append([of, wrt, magnitudes[i].forward,
                                                   magnitudes[i].reverse, magnitudes[i].fd,
                                                   abs_errs[i].forward, abs_errs[i].reverse,
                                                   abs_errs[i].forward_reverse, rel_errs[i].forward,
                                                   rel_errs[i].reverse, rel_errs[i].forward_reverse,
                                                   err_desc])
                        else:
                            if len(steps) > 1:
                                table_data.append([of, wrt, steps[i], magnitudes[i].forward,
                                                   magnitudes[i].fd, abs_errs[i].forward,
                                                   rel_errs[i].forward, err_desc])
                            else:
                                table_data.append([of, wrt, magnitudes[i].forward, magnitudes[i].fd,
                                                   abs_errs[i].forward, rel_errs[i].forward,
                                                   err_desc])
                            assert abs_errs[i].forward_reverse is None
                            assert rel_errs[i].forward_reverse is None
                            assert abs_errs[i].reverse is None
                            assert rel_errs[i].reverse is None

                        # See if this component has the greater error in the derivative computation
                        # compared to the other components so far
                        for err in rel_errs[i][:2]:
                            if err is None or np.isnan(err):
                                continue

                            if worst_subjac is None or err > worst_subjac[2]:
                                worst_subjac = (sys_class_name, sys_name, err, table_data[-1])

            else:  # not compact print
                if fd_norm == 0.:
                    if magnitudes[0].forward is None:
                        divname = 'Jrev'
                    else:
                        divname = 'Jfor'
                else:
                    divname = 'Jfd'

                if out_stream:
                    # Magnitudes
                    out_buffer.write(f"  {sys_name}: {of} wrt {wrt}")
                    if not isinstance(of, tuple) and lcons and of.strip("'") in lcons:
                        out_buffer.write(" (Linear constraint)")

                    out_buffer.write('\n')
                    if magnitudes[0].forward is not None:
                        out_buffer.write(f'     Forward Magnitude: {magnitudes[0].forward:.6e}\n')

                    if magnitudes[0].reverse is not None:
                        out_buffer.write(f'     Reverse Magnitude: {magnitudes[0].reverse:.6e}\n')

                    fd_desc = f"{fd_opts['method']}:{fd_opts['form']}"
                    for i in range(len(magnitudes)):
                        out_buffer.write(f'          Fd Magnitude: '
                                         f'{magnitudes[i].fd:.6e} ({fd_desc}{stepstrs[i]})\n')
                    out_buffer.write('\n')

                    for i in range(len(magnitudes)):
                        # Absolute Errors
                        if directional:
                            if totals and abs_errs[i].forward is not None:
                                err = _format_error(abs_errs[i].forward, abs_error_tol)
                                out_buffer.write(f'    Absolute Error (Jfor - Jfd){stepstrs[i]} : '
                                                 f'{err}\n')

                            if abs_errs[i].reverse is not None:
                                err = _format_error(abs_errs[i].reverse, abs_error_tol)
                                out_buffer.write(f'    Absolute Error ([rev, fd] Dot Product Test)'
                                                 f'{stepstrs[i]} : {err}\n')
                        else:
                            if abs_errs[i].forward is not None:
                                err = _format_error(abs_errs[i].forward, abs_error_tol)
                                out_buffer.write(f'    Absolute Error (Jfor - Jfd){stepstrs[i]} : '
                                                 f'{err}\n')

                            if abs_errs[i].reverse is not None:
                                err = _format_error(abs_errs[i].reverse, abs_error_tol)
                                out_buffer.write(f'    Absolute Error (Jrev - Jfd){stepstrs[i]} : '
                                                 f'{err}\n')

                    if directional:
                        if abs_errs[0].forward_reverse is not None:
                            err = _format_error(abs_errs[0].forward_reverse, abs_error_tol)
                            out_buffer.write('    Absolute Error ([rev, for] Dot Product Test) : '
                                             f'{err}\n')
                    else:
                        if abs_errs[0].forward_reverse is not None:
                            err = _format_error(abs_errs[0].forward_reverse, abs_error_tol)
                            out_buffer.write(f'    Absolute Error (Jrev - Jfor) : {err}\n')

                out_buffer.write('\n')

                for i in range(len(magnitudes)):
                    # Relative Errors
                    if out_stream:
                        if directional:
                            if totals and rel_errs[i].forward is not None:
                                err = _format_error(rel_errs[i].forward, rel_error_tol)
                                out_buffer.write(f'    Relative Error (Jfor - Jfd) / {divname}'
                                                 f'{stepstrs[i]} : {err}\n')

                            if rel_errs[i].reverse is not None:
                                err = _format_error(rel_errs[i].reverse, rel_error_tol)
                                out_buffer.write(f'    Relative Error ([rev, fd] Dot Product Test) '
                                                 f'/ {divname}{stepstrs[i]} : {err}\n')
                        else:
                            if rel_errs[i].forward is not None:
                                err = _format_error(rel_errs[i].forward, rel_error_tol)
                                out_buffer.write(f'    Relative Error (Jfor - Jfd) / {divname}'
                                                 f'{stepstrs[i]} : {err}\n')

                            if rel_errs[i].reverse is not None:
                                err = _format_error(rel_errs[i].reverse, rel_error_tol)
                                out_buffer.write(f'    Relative Error (Jrev - Jfd) / {divname}'
                                                 f'{stepstrs[i]} : {err}\n')

                if out_stream:
                    if directional:
                        if rel_errs[0].forward_reverse is not None:
                            err = _format_error(rel_errs[0].forward_reverse, rel_error_tol)
                            out_buffer.write(f'    Relative Error ([rev, for] Dot Product Test) / '
                                             f'{divname} : {err}\n')
                    else:
                        if rel_errs[0].forward_reverse is not None:
                            err = _format_error(rel_errs[0].forward_reverse, rel_error_tol)
                            out_buffer.write(f'    Relative Error (Jrev - Jfor) / {divname} : '
                                             f'{err}\n')

                if inconsistent:
                    out_buffer.write('\n    * Inconsistent value across ranks *\n')

                if MPI and MPI.COMM_WORLD.size > 1:
                    out_buffer.write(f'\n    MPI Rank {MPI.COMM_WORLD.rank}\n')
                out_buffer.write('\n')

                with np.printoptions(linewidth=240):
                    # Raw Derivatives
                    if magnitudes[0].forward is not None:
                        if directional:
                            out_buffer.write('    Directional Derivative (Jfor)')
                        else:
                            out_buffer.write('    Raw Forward Derivative (Jfor)')
                        Jstr = textwrap.indent(str(Jfor), '    ')
                        out_buffer.write(f"\n{Jstr}\n\n")

                    fdtype = fd_opts['method'].upper()

                    if magnitudes[0].reverse is not None:
                        if directional:
                            if totals:
                                out_buffer.write('    Directional Derivative (Jrev) Dot Product')
                            else:
                                out_buffer.write('    Directional Derivative (Jrev)')
                        else:
                            out_buffer.write('    Raw Reverse Derivative (Jrev)')
                        Jstr = textwrap.indent(str(Jrev), '    ')
                        out_buffer.write(f"\n{Jstr}\n\n")

                    try:
                        fds = derivative_info['J_fd']
                    except KeyError:
                        fds = [0.]

                    for i in range(len(magnitudes)):
                        fd = fds[i]

                        if directional:
                            if totals and magnitudes[i].reverse is not None:
                                out_buffer.write(f'    Directional {fdtype} Derivative (Jfd) '
                                                 f'Dot Product{stepstrs[i]}\n    {fd}\n\n')
                            else:
                                out_buffer.write(f"    Directional {fdtype} Derivative (Jfd)"
                                                 f"{stepstrs[i]}\n    {fd}\n\n")
                        else:
                            Jstr = textwrap.indent(str(fd), '    ')
                            out_buffer.write(f"    Raw {fdtype} Derivative (Jfd){stepstrs[i]}"
                                             f"\n{Jstr}\n\n")

                out_buffer.write(' -' * 30 + '\n')

            # End of if compact print if/else
        # End of for of, wrt in sorted_keys

        if not suppress_output:
            if compact_print and table_data:
                headers = ["of '<variable>'", "wrt '<variable>'"]
                if len(steps) > 1:
                    headers.append('step')
                column_meta = [{}, {}]

                if print_reverse:
                    headers.extend(['fwd mag.', 'rev mag.', 'check mag.', 'a(fwd-chk)',
                                    'a(rev-chk)', 'a(fwd-rev)', 'r(fwd-chk)', 'r(rev-chk)',
                                    'r(fwd-rev)', 'error desc'])
                else:
                    headers.extend(['calc mag.', 'check mag.', 'a(cal-chk)', 'r(cal-chk)',
                                    'error desc'])

                column_meta.extend([num_col_meta.copy() for _ in range(len(headers) - 3)])
                column_meta.append({})
                print(generate_table(table_data, headers=headers, tablefmt='grid',
                                     column_meta=column_meta, missing_val='n/a'), file=out_buffer)

            if totals or not show_only_incorrect or num_bad_jacs > 0:
                out_stream.write(out_buffer.getvalue())

    # End of for system in system_list

    if not suppress_output:
        if compact_print and not totals and worst_subjac:
            class_name, name, _, worst_row = worst_subjac

            worst_header = f"Sub Jacobian with Largest Relative Error: {class_name} '{name}'"
            worst_table = generate_table([worst_row[:-1]], headers=headers[:-1],
                                         tablefmt='grid', column_meta=column_meta[:-1],
                                         missing_val='n/a')
            print(f"\n{add_border(worst_header, '#')}\n{worst_table}", file=out_stream)

    if incon_keys:
        # stick incon_keys into the first key's dict in order to avoid breaking existing code
        for key, dct in derivative_data['' if totals else sys_name].items():
            dct['inconsistent_keys'] = incon_keys
            break
        if not suppress_output:
            if totals:
                msgstart = "During computation of totals, the "
            else:
                msgstart = "The "
            ders = [f"{sof} wrt {swrt}" for sof, swrt in sorted(incon_keys)]
            print(f"\n{msgstart}following partial derivatives resulted in\n"
                  "inconsistent values across processes for certain serial inputs:\n"
                  f"{ders}.\nThis can happen if a component 'compute_jacvec_product' "
                  "or 'apply_linear'\nmethod does not properly reduce the value of a distributed "
                  "output when computing the\nderivative of that output with respect to a serial "
                  "input.\nOpenMDAO 3.25 changed the convention used "
                  "when transferring data between distributed and non-distributed \nvariables "
                  "within a matrix free component. See POEM 75 for details.")


def _format_cell(val):
    """
    Return string to represent deriv check value in compact display.

    Parameters
    ----------
    val : float or None
        The deriv check value.

    Returns
    -------
    str
        String which is the actual value or 'n/a' if val is None.
    """
    if val is None:
        return pad_name('n/a')

    if np.isnan(val):
        return pad_name('nan')
    return f'{val:.4e}'


def _format_error(error, tol):
    """
    Format the error, flagging if necessary.

    Parameters
    ----------
    error : float
        The absolute or relative error.
    tol : float
        Tolerance above which errors are flagged

    Returns
    -------
    str
        Formatted and possibly flagged error.
    """
    if np.isnan(error) or error < tol:
        return f'{error:.6e}'
    return f'{error:.6e} *'


def _get_fd_options(var, global_method, local_opts, global_step, global_form, global_step_calc,
                    alloc_complex, global_minimum_step):
    local_wrt = var

    # Determine if fd or cs.
    method = global_method
    if local_wrt in local_opts:
        local_method = local_opts[local_wrt]['method']
        if local_method:
            method = local_method

    # We can't use CS if we haven't allocated a complex vector, so we fall back on fd.
    if method == 'cs' and not alloc_complex:
        method = 'fd'
        could_not_cs = True
    else:
        could_not_cs = False

    fd_options = {'order': None,
                  'method': method}

    if method == 'cs':
        fd_options = ComplexStep.DEFAULT_OPTIONS.copy()
        fd_options['method'] = 'cs'

        fd_options['form'] = None
        fd_options['step_calc'] = None
        fd_options['minimum_step'] = None

    elif method == 'fd':
        fd_options = FiniteDifference.DEFAULT_OPTIONS.copy()
        fd_options['method'] = 'fd'

        fd_options['form'] = global_form
        fd_options['step_calc'] = global_step_calc
        fd_options['minimum_step'] = global_minimum_step

    if global_step and global_method == method:
        fd_options['step'] = global_step

    fd_options['directional'] = False

    # Precedence: component options > global options > defaults
    if local_wrt in local_opts:
        for name in ['form', 'step', 'step_calc', 'minimum_step', 'directional']:
            value = local_opts[local_wrt][name]
            if value is not None:
                fd_options[name] = value

    return fd_options, could_not_cs
