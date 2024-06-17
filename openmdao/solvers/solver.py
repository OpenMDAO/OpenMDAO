"""Define the base Solver, NonlinearSolver, and LinearSolver classes."""

import os
import pprint
import sys
import weakref

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.core.constants import _UNDEFINED
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.record_util import create_local_meta, check_path
from openmdao.utils.om_warnings import issue_warning, SolverWarning


class SolverInfo(object):
    """
    Communal object for storing some formatting for solver iprint.

    Attributes
    ----------
    prefix : str
        Prefix to prepend during this iprint.
    stack : list
        List of strings; strings are popped and appended as needed.
    """

    def __init__(self):
        """
        Initialize.
        """
        self.prefix = ""
        self.stack = []

    def clear(self):
        """
        Clear out the iprint stack, in case something is left over from a handled exception.
        """
        self.prefix = ""
        self.stack = []

    def pop(self):
        """
        Remove one level of solver depth in the printing.
        """
        last_string = self.stack.pop()
        nchar = len(last_string)
        self.prefix = self.prefix[:-nchar]

    def append_solver(self):
        """
        Add a new level for the main solver in a group.
        """
        new_str = '+  '
        self.prefix += new_str
        self.stack.append(new_str)

    def append_subsolver(self):
        """
        Add a new level for any sub-solver for your solver.
        """
        new_str = '|  '
        self.prefix += new_str
        self.stack.append(new_str)

    def append_precon(self):
        """
        Add a new level for any preconditioner to a linear solver.
        """
        new_str = '| precon:'
        self.prefix += new_str
        self.stack.append(new_str)

    def save_cache(self):
        """
        Save prefix and stack so that they can be restored later in event of an exception recovery.

        Returns
        -------
        tuple(str, list)
            Cache of current stack.
        """
        return (self.prefix, self.stack)

    def restore_cache(self, cache):
        """
        Restore previously saved iprint stack names.

        Parameters
        ----------
        cache : tuple(str, list)
            Cache of current stack.
        """
        self.prefix, self.stack = cache


class Solver(object):
    """
    Base solver class.

    This class is subclassed by NonlinearSolver and LinearSolver,
    which are in turn subclassed by actual solver implementations.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Solver options.

    Attributes
    ----------
    _system : <System>
        Pointer to the owning system.
    _depth : int
        How many subsolvers deep this solver is (0 means not a subsolver).
    _mode : str
        'fwd' or 'rev', applicable to linear solvers only.
    _iter_count : int
        Number of iterations for the current invocation of the solver.
    _rec_mgr : <RecordingManager>
        object that manages all recorders added to this solver
    cite : str
        Listing of relevant citations that should be referenced when
        publishing work that uses this class.
    options : <OptionsDictionary>
        Options dictionary.
    recording_options : <OptionsDictionary>
        Recording options dictionary.
    supports : <OptionsDictionary>
        Options dictionary describing what features are supported by this
        solver.
    _filtered_vars_to_record : Dict
        Dict of list of var names to record
    _norm0 : float
        Normalization factor
    _problem_meta : dict
        Problem level metadata.
    """

    # Object to store some formatting for iprint that is shared across all solvers.
    SOLVER = 'base_solver'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        self._system = None
        self._depth = 0
        self._mode = 'fwd'
        self._iter_count = 0
        self._problem_meta = None

        # Solver options
        self.options = OptionsDictionary(parent_name=self.msginfo)
        self.options.declare('maxiter', types=int, default=10,
                             desc='maximum number of iterations')
        self.options.declare('atol', default=1e-10,
                             desc='absolute error tolerance')
        self.options.declare('rtol', default=1e-10,
                             desc='relative error tolerance')
        self.options.declare('iprint', types=int, default=1,
                             desc='whether to print output')
        self.options.declare('err_on_non_converge', types=bool, default=False,
                             desc="When True, AnalysisError will be raised if we don't converge.")

        # Case recording options
        self.recording_options = OptionsDictionary(parent_name=self.msginfo)
        self.recording_options.declare('record_abs_error', types=bool, default=True,
                                       desc='Set to True to record absolute error at the \
                                       solver level')
        self.recording_options.declare('record_rel_error', types=bool, default=True,
                                       desc='Set to True to record relative error at the \
                                       solver level')
        self.recording_options.declare('record_inputs', types=bool, default=True,
                                       desc='Set to True to record inputs at the solver level')
        self.recording_options.declare('record_outputs', types=bool, default=True,
                                       desc='Set to True to record outputs at the solver level')
        self.recording_options.declare('record_solver_residuals', types=bool, default=False,
                                       desc='Set to True to record residuals at the solver level')
        self.recording_options.declare('includes', types=list, default=['*'],
                                       desc="Patterns for variables to include in recording. \
                                       Paths are relative to solver's Group. \
                                       Uses fnmatch wildcards")
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc="Patterns for vars to exclude in recording. \
                                       (processed post-includes) \
                                       Paths are relative to solver's Group. \
                                       Uses fnmatch wildcards"
                                       )
        # Case recording related
        self._filtered_vars_to_record = {}
        self._norm0 = 0.0

        # What the solver supports.
        self.supports = OptionsDictionary(parent_name=self.msginfo)
        self.supports.declare('gradients', types=bool, default=False)
        self.supports.declare('implicit_components', types=bool, default=False)
        self.supports.declare('linesearch', types=bool, default=False)

        self._declare_options()
        self.options.update(kwargs)

        self._rec_mgr = RecordingManager()

        self.cite = ""

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        # Default initialization
        if self._system is None:
            return type(self).__name__
        # Following Dead Weakref
        elif self._system() is None:
            return type(self).__name__
        return f"{type(self).__name__} in {self._system().msginfo}"

    def _inf_nan_failure(self):
        msg = (f"Solver '{self.SOLVER}' on system '{self._system().pathname}': "
               f"residuals contain 'inf' or 'NaN' after {self._iter_count} iterations.")
        self.report_failure(msg)

    def _convergence_failure(self):
        msg = (f"Solver '{self.SOLVER}' on system '{self._system().pathname}' failed to converge "
               f"in {self._iter_count} iterations.")
        self.report_failure(msg)

    def report_failure(self, msg):
        """
        Report a failure that has occurred.

        The msg may be printed or ignored depending on the iprint option, and may raise
        an AnalysisError depending on the err_on_non_converge option.

        Parameters
        ----------
        msg : str
            Message indicating the failure.
        """
        iprint = self.options['iprint']
        print_flag = self._system().comm.rank == 0 or os.environ.get('USE_PROC_FILES')

        if iprint > -1 and print_flag:
            print(self._solver_info.prefix + self.SOLVER + msg)

        # Raise AnalysisError if requested.
        if self.options['err_on_non_converge']:
            raise AnalysisError(msg)

    @property
    def _recording_iter(self):
        if self._problem_meta is None:
            raise RuntimeError(f"{self.msginfo}: Can't access recording_iter because "
                               "_setup_solvers has not been called.")
        return self._problem_meta['recording_iter']

    @property
    def _solver_info(self):
        if self._problem_meta is None:
            raise RuntimeError(f"{self.msginfo}: Can't access solver_info because _setup_solvers "
                               "has not been called.")
        return self._problem_meta['solver_info']

    def _assembled_jac_solver_iter(self):
        """
        Return an empty generator of lin solvers using assembled jacs.
        """
        for i in ():
            yield

    def add_recorder(self, recorder):
        """
        Add a recorder to the solver's RecordingManager.

        Parameters
        ----------
        recorder : <CaseRecorder>
           A recorder instance to be added to RecManager.
        """
        if MPI:
            raise RuntimeError(
                "Recording of Solvers when running parallel code is not supported yet")
        self._rec_mgr.append(recorder)

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Solver.
        """
        pass

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        # Default initialization
        if self._system is None:
            self._system = weakref.ref(system)
        # Following Dead Weakref
        elif self._system() is None:
            self._system = weakref.ref(system)
        # Assignment Mismatch
        elif self._system != weakref.ref(system):
            raise RuntimeError(f"{type(self).__name__} has already been assigned to "
                               f"{self._system().msginfo} and cannot also be assigned to "
                               f"{system.msginfo}.")

        self._depth = depth
        self._problem_meta = system._problem_meta

        if system.pathname:
            parent_name = self.msginfo
            self.options._parent_name = parent_name
            self.recording_options._parent_name = parent_name
            self.supports._parent_name = parent_name

        if isinstance(self, LinearSolver) and not system._use_derivatives:
            return

        self._rec_mgr.startup(self, self._problem_meta['comm'])

        myoutputs = myresiduals = myinputs = []
        incl = self.recording_options['includes']
        excl = self.recording_options['excludes']

        # doesn't matter if we're a linear or nonlinear solver.  The names for
        # inputs, outputs, and residuals are the same for both the 'linear' and 'nonlinear'
        # vectors.
        if system.pathname:
            incl = ['.'.join((system.pathname, i)) for i in incl]
            excl = ['.'.join((system.pathname, i)) for i in excl]

        if self.recording_options['record_solver_residuals']:
            myresiduals = [n for n in system._residuals._abs_iter() if check_path(n, incl, excl)]

        if self.recording_options['record_outputs']:
            myoutputs = [n for n in system._outputs._abs_iter() if check_path(n, incl, excl)]

        if self.recording_options['record_inputs']:
            myinputs = [n for n in system._inputs._abs_iter() if check_path(n, incl, excl)]

        self._filtered_vars_to_record = {
            'input': myinputs,
            'output': myoutputs,
            'residual': myresiduals
        }

    def _set_solver_print(self, level=2, type_='all'):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        self.options['iprint'] = level

    def _mpi_print(self, iteration, abs_res, rel_res):
        """
        Print residuals from an iteration if iprint == 2.

        Parameters
        ----------
        iteration : int
            iteration counter, 0-based.
        abs_res : float
            current absolute residual norm.
        rel_res : float
            current relative residual norm.
        """
        if (self.options['iprint'] == 2 and
                (self._system().comm.rank == 0 or os.environ.get('USE_PROC_FILES'))):

            prefix = self._solver_info.prefix
            solver_name = self.SOLVER

            if prefix.endswith('precon:'):
                solver_name = solver_name[3:]

            print(f"{prefix}{solver_name} {iteration} ; {abs_res:.9g} {rel_res:.9g}")

    def _mpi_print_header(self):
        """
        Print header text before solving.
        """
        if (self.options['iprint'] > 0 and
                (self._system().comm.rank == 0 or os.environ.get('USE_PROC_FILES'))):

            pathname = self._system().pathname
            if pathname:
                eqs = len(pathname) * "="
                prefix = self._solver_info.prefix
                print(f"{prefix}\n{prefix}{eqs}\n{prefix}{pathname}\n{prefix}{eqs}")

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        pass

    def _run_apply(self):
        """
        Run the appropriate apply method on the system.
        """
        pass

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        pass

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        bool
            Flag for indicating child linerization
        """
        return True

    def __str__(self):
        """
        Return a string representation of the solver.

        Returns
        -------
        str
            String representation of the solver.
        """
        return self.SOLVER

    def record_iteration(self, **kwargs):
        """
        Record an iteration of the current Solver.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments (used for abs and rel error).
        """
        if not self._rec_mgr._recorders:
            return

        metadata = create_local_meta(self.SOLVER)

        # Get the data
        data = {
            'abs': kwargs.get('abs') if self.recording_options['record_abs_error'] else None,
            'rel': kwargs.get('rel') if self.recording_options['record_rel_error'] else None,
            'input': {},
            'output': {},
            'residual': {}
        }

        system = self._system()
        vec_name = 'nonlinear' if isinstance(self, NonlinearSolver) else 'linear'
        filt = self._filtered_vars_to_record
        parallel = self._rec_mgr._check_parallel() if system.comm.size > 1 else False

        if self.recording_options['record_outputs']:
            data['output'] = system._retrieve_data_of_kind(filt, 'output', vec_name, parallel)

        if self.recording_options['record_inputs']:
            data['input'] = system._retrieve_data_of_kind(filt, 'input', vec_name, parallel)

        if self.recording_options['record_solver_residuals']:
            data['residual'] = system._retrieve_data_of_kind(filt, 'residual', vec_name, parallel)

        self._rec_mgr.record_iteration(self, data, metadata)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        # shut down all recorders
        self._rec_mgr.shutdown()

    def _set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Recurses to turn on or off complex stepping mode in all subsystems and their vectors.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        pass

    def _disallow_distrib_solve(self):
        """
        Raise an exception if our system or any subsystems are distributed or non-local.
        """
        s = self._system()
        if s.comm.size == 1:
            return

        from openmdao.core.group import Group
        if s._has_distrib_vars or (isinstance(s, Group) and s._contains_parallel_group):
            msg = "{} linear solver in {} cannot be used in or above a ParallelGroup or a " + \
                "distributed component."
            raise RuntimeError(msg.format(type(self).__name__, s.msginfo))

    def get_reports_dir(self):
        """
        Get the path to the directory where the report files should go.

        If it doesn't exist, it will be created.

        Returns
        -------
        str
            The path to the directory where reports should be written.
        """
        return self._system().get_reports_dir()

    def use_relevance(self):
        """
        Return True if relevance should be active.

        Returns
        -------
        bool
            True if relevance should be active.
        """
        return True


class NonlinearSolver(Solver):
    """
    Base class for nonlinear solvers.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _err_cache : dict
        Dictionary holding input and output vectors at start of iteration, if requested.
    _output_cache : ndarray or None
        Saved output values from last successful solve, if any.
    _prev_fail : bool
        If True, previous solve failed.
    _restarted : bool
        If True, solve was restarted from a sucessful point.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)
        self._err_cache = {}
        self._output_cache = None
        self._prev_fail = False
        self._restarted = False

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('debug_print', types=bool, default=False,
                             desc='If true, the values of input and output variables at '
                                  'the start of iteration are printed and written to a file '
                                  'after a failure to converge.')
        self.options.declare('stall_limit', default=0,
                             desc='Number of iterations after which, if the residual norms are '
                                  'identical within the stall_tol, then terminate as if max '
                                  'iterations were reached. Default is 0, which disables this '
                                  'feature.')
        self.options.declare('stall_tol', default=1e-12,
                             desc='When stall checking is enabled, the threshold below which the '
                                  'residual norm is considered unchanged.')
        self.options.declare('stall_tol_type', default='rel', values=('abs', 'rel'),
                             desc='Specifies whether the absolute or relative norm of the '
                                  'residual is used for stall detection.')
        self.options.declare('restart_from_successful', types=bool, default=False,
                             desc='If True, the states are cached after a successful solve and '
                                  'used to restart the solver in the case of a failed solve.')

    @property
    def linesearch(self):
        """
        Get the linesearch solver associated with this solver.

        Returns
        -------
        NonlinearSolver or None
            The linesearch associated with this solver, or None if it does not support one.
        """
        if not self.supports['linesearch']:
            return None
        else:
            return self._linesearch

    @linesearch.setter
    def linesearch(self, ls):
        if not self.supports['linesearch']:
            raise AttributeError(f'{self.msginfo}: This solver does not support a linesearch.')
        self._linesearch = ls

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)
        # The state caching only works if we throw an error on non-convergence, otherwise
        # the solver will disregard the caching option and issue a warning.
        if 'restart_from_successful' in self.options and self.options['restart_from_successful']:
            if not self.options['err_on_non_converge']:
                issue_warning(f"{self.msginfo}: Option 'restart_from_successful' does nothing "
                              "unless option 'err_on_non_converge' is set to True.",
                              category=SolverWarning)
                # reset to False so we won't waste memory allocating a cache array
                self.options['restart_from_successful'] = False

    def solve(self):
        """
        Run the solver.
        """
        try:
            self._solve()
        except Exception as err:
            if self.options['debug_print']:
                self._print_exc_debug_info()
            raise err

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        system = self._system()
        if self.options['debug_print']:
            self._err_cache['inputs'] = system._inputs._copy_views()
            self._err_cache['outputs'] = system._outputs._copy_views()

        if self.options['maxiter'] > 0:
            self._run_apply()
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _solve(self):
        """
        Run the iterative solver.
        """
        system = self._system()

        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        iprint = self.options['iprint']
        stall_limit = self.options['stall_limit']
        stall_tol = self.options['stall_tol']
        stall_tol_type = self.options['stall_tol_type']

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0

        self._mpi_print(self._iter_count, norm, norm / norm0)

        stalled = False
        stall_count = 0
        if stall_limit > 0:
            stall_norm = norm0

        force_one_iteration = system.under_complex_step

        while ((self._iter_count < maxiter and norm > atol and norm / norm0 > rtol and
                not stalled) or force_one_iteration):

            if system.under_complex_step:
                force_one_iteration = False

            with Recording(type(self).__name__, self._iter_count, self) as rec:
                ls = self.linesearch
                if stall_count == 3 and ls and not ls.options['print_bound_enforce']:

                    self.linesearch.options['print_bound_enforce'] = True

                    if self._system().pathname:
                        pathname = f"{self._system().pathname}."
                    else:
                        pathname = ""

                    msg = ("Your model has stalled three times and may be violating the bounds."
                           " In the future, turn on print_bound_enforce in your solver options "
                           f"here: \n{pathname}nonlinear_solver.linesearch.options"
                           "['print_bound_enforce']=True. \nThe bound(s) being violated now "
                           "are:\n")
                    issue_warning(msg, category=SolverWarning)

                    self._single_iteration()
                    self.linesearch.options['print_bound_enforce'] = False
                else:
                    self._single_iteration()

                self._iter_count += 1
                self._run_apply()
                norm = self._iter_get_norm()

                # Save the norm values in the context manager so they can also be recorded.
                rec.abs = norm
                if norm0 == 0:
                    norm0 = 1
                rec.rel = norm / norm0

                # Check if convergence is stalled.
                if stall_limit > 0:
                    norm_for_stall = rec.rel if stall_tol_type == 'rel' else rec.abs
                    norm_diff = np.abs(stall_norm - norm_for_stall)
                    if norm_diff <= stall_tol:
                        stall_count += 1
                        if stall_count >= stall_limit:
                            stalled = True
                    else:
                        stall_count = 0
                        stall_norm = norm_for_stall

            self._mpi_print(self._iter_count, norm, norm / norm0)

        # flag for the print statements. we only print on root if USE_PROC_FILES is not set to True
        print_flag = system.comm.rank == 0 or os.environ.get('USE_PROC_FILES')

        prefix = self._solver_info.prefix + self.SOLVER

        # Solver terminated early because a Nan in the norm doesn't satisfy the while-loop
        # conditionals.
        if np.isinf(norm) or np.isnan(norm):
            self._inf_nan_failure()

        # solver stalled.
        elif stalled:
            msg = (f"Solver '{self.SOLVER}' on system '{system.pathname}' stalled after "
                   f"{self._iter_count} iterations.")
            self.report_failure(msg)

        # Solver hit maxiter without meeting desired tolerances.
        elif norm > atol and norm / norm0 > rtol:
            self._convergence_failure()

        # Solver converged
        elif print_flag:
            if iprint == 1:
                print(prefix + ' Converged in {} iterations'.format(self._iter_count))
            elif iprint == 2:
                print(prefix + ' Converged')

    def _run_apply(self):
        """
        Run the apply_nonlinear method on the system.
        """
        self._recording_iter.push(('_run_apply', 0))
        try:
            self._system()._apply_nonlinear()
        finally:
            self._recording_iter.pop()

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        return self._system()._residuals.get_norm()

    def _disallow_discrete_outputs(self):
        """
        Raise an exception if any discrete outputs exist in our System.
        """
        if self._system()._var_allprocs_discrete['output']:
            raise RuntimeError("%s has a %s solver and contains discrete outputs %s." %
                               (self._system().msginfo, type(self).__name__,
                                sorted(self._system()._var_allprocs_discrete['output'])))

    def _print_exc_debug_info(self):
        coord = self._recording_iter.get_formatted_iteration_coordinate()

        out_strs = [f"\n# Inputs and outputs at start of iteration '{coord}':\n"]
        for vec_type, views in self._err_cache.items():
            out_strs.append('\n# nonlinear %s\n' % vec_type)
            out_strs.append(pprint.pformat(views))
            out_strs.append('\n')

        out_str = ''.join(out_strs)
        print(out_str)

        rank = MPI.COMM_WORLD.rank if MPI is not None else 0
        filename = f'solver_errors.{rank}.out'

        with open(filename, 'a') as f:
            f.write(out_str)
            print(f"Inputs and outputs at start of iteration have been saved to '{filename}'.")
            sys.stdout.flush()

    def _gs_iter(self):
        """
        Perform a Gauss-Seidel iteration over this Solver's subsystems.
        """
        system = self._system()
        for subsys in system._relevance.filter(system._all_subsystem_iter()):
            system._transfer('nonlinear', 'fwd', subsys.name)

            if subsys._is_local:
                try:
                    subsys._solve_nonlinear()
                except AnalysisError as err:
                    if 'reraise_child_analysiserror' not in self.options or \
                            self.options['reraise_child_analysiserror']:
                        raise err

    def _solve_with_cache_check(self):
        """
        Solve the nonlinear system, possibly after updating the output vector with cached values.

        Cached values, if any, are from the last successful nonlinear solve, and are only used
        if the 'restart_from_successful' option is True.
        """
        system = self._system()

        if (self.options['restart_from_successful'] and self.options['maxiter'] > 1 and
                not system.under_approx):
            try:
                # If we have a previous solver failure, we want to replace
                # the outputs using the cache.
                if self._prev_fail and self._output_cache is not None:
                    system._outputs.set_val(self._output_cache)
                    self._restarted = True
                else:
                    self._restarted = False

                self.solve()

                # If we make it here, the solver converged.
                self._prev_fail = False

                # Save the outputs upon a successful solve
                if self._output_cache is None:
                    self._output_cache = system._outputs.asarray(copy=True)
                else:
                    self._output_cache[:] = system._outputs.asarray()

            except Exception:
                # The solver failed so we need to set the flag to True
                self._prev_fail = True
                raise
        else:
            self.solve()


class LinearSolver(Solver):
    """
    Base class for linear solvers.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _assembled_jac : AssembledJacobian or None
        If not None, the AssembledJacobian instance used by this solver.
    _scope_in : set or None or _UNDEFINED
        Relevant input variables for the current matrix vector product.
    _scope_out : set or None or _UNDEFINED
        Relevant output variables for the current matrix vector product.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        self._assembled_jac = None
        self._scope_out = _UNDEFINED
        self._scope_in = _UNDEFINED

        super().__init__(**kwargs)

    def does_recursive_applies(self):
        """
        Return False.

        By default, assume linear solvers don't do recursive apply_linear calls.

        Returns
        -------
        bool
            True if solver makes recursive apply_linear calls on its subsystems.
        """
        return False

    def _set_matvec_scope(self, scope_out=_UNDEFINED, scope_in=_UNDEFINED):
        pass

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.options['assemble_jac']:
            yield self

    def add_recorder(self, recorder):
        """
        Add a recorder to the solver's RecordingManager.

        Parameters
        ----------
        recorder : <CaseRecorder>
           A recorder instance to be added to RecManager.
        """
        raise RuntimeError('Recording is not supported on Linear Solvers.')

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('assemble_jac', default=False, types=bool,
                             desc='Activates use of assembled jacobian by this solver.')

        self.supports.declare('assembled_jac', types=bool, default=True)

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)
        if self.options['assemble_jac'] and not self.supports['assembled_jac']:
            raise RuntimeError("Linear solver %s doesn't support assembled "
                               "jacobians." % self.msginfo)

    def solve(self, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.  Deprecated.
        """
        raise NotImplementedError("class %s does not implement solve()." % (type(self).__name__))

    def _solve(self):
        """
        Run the iterative solver.
        """
        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        iprint = self.options['iprint']

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0

        system = self._system()

        self._mpi_print(self._iter_count, norm, norm / norm0)

        while self._iter_count < maxiter and norm > atol and norm / norm0 > rtol:

            with Recording(type(self).__name__, self._iter_count, self) as rec:
                self._single_iteration()
                self._iter_count += 1
                self._run_apply()
                norm = self._iter_get_norm()

                # Save the norm values in the context manager so they can also be recorded.
                rec.abs = norm
                if norm0 == 0:
                    norm0 = 1
                rec.rel = norm / norm0

            self._mpi_print(self._iter_count, norm, norm / norm0)

        # flag for the print statements. we only print on root if USE_PROC_FILES is not set to True
        print_flag = system.comm.rank == 0 or os.environ.get('USE_PROC_FILES')

        prefix = self._solver_info.prefix + self.SOLVER

        # Solver terminated early because a Nan in the norm doesn't satisfy the while-loop
        # conditionals.
        if np.isinf(norm) or np.isnan(norm):
            self._inf_nan_failure()

        # Solver hit maxiter without meeting desired tolerances.
        elif (norm > atol and norm / norm0 > rtol):
            self._convergence_failure()

        # Solver converged
        elif print_flag:
            if iprint == 1:
                print(prefix + ' Converged in {} iterations'.format(self._iter_count))
            elif iprint == 2:
                print(prefix + ' Converged')

    def _run_apply(self):
        """
        Run the apply_linear method on the system.
        """
        self._recording_iter.push(('_run_apply', 0))

        system = self._system()
        scope_out, scope_in = system._get_matvec_scope()

        try:
            system._apply_linear(self._assembled_jac, self._mode, scope_out, scope_in)
        finally:
            self._recording_iter.pop()


class BlockLinearSolver(LinearSolver):
    """
    A base class for LinearBlockGS and LinearBlockJac.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _rhs_vec : ndarray
        Contains the values of the linear resids (fwd) or outputs (rev) saved at the beginning
        of the linear solve.
    """

    def __init__(self, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._rhs_vec = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        self.supports['assembled_jac'] = False

    def does_recursive_applies(self):
        """
        Return True.

        Block linear solvers make recursive apply_linear calls.

        Returns
        -------
        bool
            True if solver makes recursive apply_linear calls on its subsystems.
        """
        return True

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)
        self._rhs_vec = None

    def _create_rhs_vec(self):
        system = self._system()
        if self._mode == 'fwd':
            self._rhs_vec = system._dresiduals.asarray(True)
        else:
            self._rhs_vec = system._doutputs.asarray(True)

    def _update_rhs_vec(self):
        if self._rhs_vec is None:
            self._create_rhs_vec()

        if self._mode == 'fwd':
            self._rhs_vec[:] = self._system()._dresiduals.asarray()
        else:
            self._rhs_vec[:] = self._system()._doutputs.asarray()
        # print(self._system().pathname, "Updating RHS vec to", self._rhs_vec)  # DO NOT DELETE

    def _set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Recurses to turn on or off complex stepping mode in all subsystems and their vectors.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        if self._rhs_vec is None:
            self._create_rhs_vec()
        if active:
            self._rhs_vec = self._rhs_vec.astype(complex)
        else:
            self._rhs_vec = self._rhs_vec.real

    def _vars_union(self, slv_vars, sys_vars):
        """
        Return the union of the two 'set's of variables.

        The first 'set' comes from the this solver and the second from the set of variables
        from the current System that are relevent to the current matrix vector product. Note
        that this is called separately for input and output variables.

        Also handles cases where incoming variables are _UNDEFINED or None instead of sets.

        Parameters
        ----------
        slv_vars : set, None, or _UNDEFINED
            First variable set.
        sys_vars : set, None, or _UNDEFINED
            Second variable set.

        Returns
        -------
        set, None, or _UNDEFINED
            The combined variable 'set'.
        """
        if sys_vars is None or slv_vars is None:
            return None
        if slv_vars is _UNDEFINED:
            return sys_vars
        return sys_vars.union(slv_vars)

    def _run_apply(self, init=False):
        """
        Run the apply_linear method on the system.
        """
        system = self._system()
        self._recording_iter.push(('_run_apply', 0))
        try:
            scope_out, scope_in = system._get_matvec_scope()
            system._apply_linear(self._assembled_jac, self._mode,
                                 self._vars_union(self._scope_out, scope_out),
                                 self._vars_union(self._scope_in, scope_in))
        finally:
            self._recording_iter.pop()

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        self._update_rhs_vec()

        if self.options['maxiter'] > 1:
            self._run_apply()
            norm = self._iter_get_norm()
        else:
            return 1.0, 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Note: This has the side effect of modifying the residual vector in fwd mode
        and the output vector in rev mode.

        Returns
        -------
        float
            norm.
        """
        if self._mode == 'fwd':
            b_vec = self._system()._dresiduals
        else:  # rev
            b_vec = self._system()._doutputs

        b_vec -= self._rhs_vec  # compute Ax - rhs
        norm = b_vec.get_norm()
        b_vec += self._rhs_vec  # revert b_vec back to original value
        return norm

    def _set_matvec_scope(self, scope_out=_UNDEFINED, scope_in=_UNDEFINED):
        """
        Set the relevant variables for the current matrix vector product.

        Parameters
        ----------
        scope_out : set, None, or _UNDEFINED
            Outputs relevant to possible lower level calls to _apply_linear on Components.
        scope_in : set, None, or _UNDEFINED
            Inputs relevant to possible lower level calls to _apply_linear on Components.
        """
        self._scope_out = scope_out
        self._scope_in = scope_in

    def solve(self, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.  Deprecated.
        """
        self._mode = mode
        try:
            self._solve()
        finally:
            self._scope_out = self._scope_in = _UNDEFINED  # reset after solve is done
