"""Define the base Solver, NonlinearSolver, and LinearSolver classes."""

from collections import OrderedDict
import os
import pprint
import re
import sys
import weakref

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.record_util import create_local_meta, check_path
from openmdao.core.component import Component

_emptyset = set()


class SolverInfo(object):
    """
    Communal object for storing some formatting for solver iprint.

    Attributes
    ----------
    prefix : str
        Prefix to prepend during this iprint.
    stack : List
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
            cache of current stack
        """
        return (self.prefix, self.stack)

    def restore_cache(self, cache):
        """
        Restore previously saved iprint stack names.

        Parameters
        ----------
        cache : tuple(str, list)
            cache of current stack
        """
        self.prefix, self.stack = cache


class Solver(object):
    """
    Base solver class.

    This class is subclassed by NonlinearSolver and LinearSolver,
    which are in turn subclassed by actual solver implementations.

    Attributes
    ----------
    _system : <System>
        Pointer to the owning system.
    _depth : int
        How many subsolvers deep this solver is (0 means not a subsolver).
    _vec_names : [str, ...]
        List of right-hand-side (RHS) vector names.
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
    _filtered_vars_to_record: Dict
        Dict of list of var names to record
    _norm0: float
        Normalization factor
    _problem_meta : dict
        Problem level metadata.
    """

    # Object to store some formatting for iprint that is shared across all solvers.
    SOLVER = 'base_solver'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Solver options.
        """
        self._system = None
        self._depth = 0
        self._vec_names = None
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
        self.recording_options.declare('record_metadata', types=bool, desc='Deprecated. Recording '
                                       'of metadata will always be done',
                                       deprecation="The recording option, record_metadata, on "
                                       "Solver is "
                                       "deprecated. Recording of metadata will always be done",
                                       default=True)
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
        if self._system is None:
            return type(self).__name__
        return '{} in {}'.format(type(self).__name__, self._system().msginfo)

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
        self._system = weakref.ref(system)
        self._depth = depth
        self._problem_meta = system._problem_meta

        if system.pathname:
            parent_name = self.msginfo
            self.options._parent_name = parent_name
            self.recording_options._parent_name = parent_name
            self.supports._parent_name = parent_name

        if isinstance(self, LinearSolver) and not system._use_derivatives:
            return

        self._rec_mgr.startup(self)
        self._rec_mgr.record_metadata(self)

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
            myresiduals = [n for n in system._residuals._views if check_path(n, incl, excl)]

        if self.recording_options['record_outputs']:
            myoutputs = [n for n in system._outputs._views if check_path(n, incl, excl)]

        if self.recording_options['record_inputs']:
            myinputs = [n for n in system._inputs._views if check_path(n, incl, excl)]

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
        Print residuals from an iteration.

        Parameters
        ----------
        iteration : int
            iteration counter, 0-based.
        abs_res : float
            current absolute residual norm.
        rel_res : float
            current relative residual norm.
        """
        if (self.options['iprint'] == 2 and self._system().comm.rank == 0):

            prefix = self._solver_info.prefix
            solver_name = self.SOLVER

            if prefix.endswith('precon:'):
                solver_name = solver_name[3:]

            print_str = prefix + solver_name
            print_str += ' %d ; %.9g %.9g' % (iteration, abs_res, rel_res)
            print(print_str)

    def _mpi_print_header(self):
        """
        Print header text before solving.
        """
        pass

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

        self._mpi_print(self._iter_count, norm, norm / norm0)

        while self._iter_count < maxiter and norm > atol and norm / norm0 > rtol:
            with Recording(type(self).__name__, self._iter_count, self) as rec:
                self._single_iteration()
                self._iter_count += 1
                self._run_apply()
                norm = self._iter_get_norm()
                # With solvers, we want to record the norm AFTER the call, but the call needs to
                # be wrapped in the with for stack purposes, so we locally assign  norm & norm0
                # into the class.
                rec.abs = norm
                if norm0 == 0:
                    norm0 = 1
                rec.rel = norm / norm0

            self._mpi_print(self._iter_count, norm, norm / norm0)

        system = self._system()
        if system.comm.rank == 0 or os.environ.get('USE_PROC_FILES'):
            prefix = self._solver_info.prefix + self.SOLVER

            # Solver terminated early because a Nan in the norm doesn't satisfy the while-loop
            # conditionals.
            if np.isinf(norm) or np.isnan(norm):
                msg = "Solver '{}' on system '{}': residuals contain 'inf' or 'NaN' after {} " + \
                      "iterations."
                if iprint > -1:
                    print(prefix + msg.format(self.SOLVER, system.pathname,
                                              self._iter_count))

                # Raise AnalysisError if requested.
                if self.options['err_on_non_converge']:
                    raise AnalysisError(msg.format(self.SOLVER, system.pathname,
                                                   self._iter_count))

            # Solver hit maxiter without meeting desired tolerances.
            elif (norm > atol and norm / norm0 > rtol):
                msg = "Solver '{}' on system '{}' failed to converge in {} iterations."

                if iprint > -1:
                    print(prefix + msg.format(self.SOLVER, system.pathname,
                                              self._iter_count))

                # Raise AnalysisError if requested.
                if self.options['err_on_non_converge']:
                    raise AnalysisError(msg.format(self.SOLVER, system.pathname,
                                                   self._iter_count))

            # Solver converged
            elif iprint == 1:
                print(prefix + ' Converged in {} iterations'.format(self._iter_count))
            elif iprint == 2:
                print(prefix + ' Converged')

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
        boolean
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
        if (isinstance(s, Group) and (s._has_distrib_vars or s._contains_parallel_group)) or \
           (isinstance(s, Component) and s.options['distributed']):
            msg = "{} linear solver in {} cannot be used in or above a ParallelGroup or a " + \
                "distributed component."
            raise RuntimeError(msg.format(type(self).__name__, s.msginfo))


class NonlinearSolver(Solver):
    """
    Base class for nonlinear solvers.

    Attributes
    ----------
    _err_cache : dict
        Dictionary holding input and output vectors at start of iteration, if requested.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(NonlinearSolver, self).__init__(**kwargs)
        self._err_cache = OrderedDict()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('debug_print', types=bool, default=False,
                             desc='If true, the values of input and output variables at '
                                  'the start of iteration are printed and written to a file '
                                  'after a failure to converge.')

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

        out_strs = ["\n# Inputs and outputs at start of iteration '%s':\n" % coord]
        for vec_type, views in self._err_cache.items():
            out_strs.append('\n# nonlinear %s\n' % vec_type)
            out_strs.append(pprint.pformat(views))
            out_strs.append('\n')

        out_str = ''.join(out_strs)
        print(out_str)

        rank = MPI.COMM_WORLD.rank if MPI is not None else 0
        filename = 'solver_errors.%d.out' % rank

        with open(filename, 'a') as f:
            f.write(out_str)
            print("Inputs and outputs at start of iteration have been "
                  "saved to '%s'." % filename)
            sys.stdout.flush()

    def _gs_iter(self):
        """
        Perform a Gauss-Seidel iteration over this Solver's subsystems.
        """
        system = self._system()
        for isub, subsys in enumerate(system._subsystems_allprocs):
            system._transfer('nonlinear', 'fwd', isub)

            if subsys._is_local:
                try:
                    subsys._solve_nonlinear()
                except AnalysisError as err:
                    if 'reraise_child_analysiserror' not in self.options or \
                            self.options['reraise_child_analysiserror']:
                        raise err


class LinearSolver(Solver):
    """
    Base class for linear solvers.

    Attributes
    ----------
    _rel_systems : set of str
        Names of systems relevant to the current solve.
    _assembled_jac : AssembledJacobian or None
        If not None, the AssembledJacobian instance used by this solver.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        self._rel_systems = None
        self._assembled_jac = None
        super(LinearSolver, self).__init__(**kwargs)

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
        super(LinearSolver, self)._setup_solvers(system, depth)
        if self.options['assemble_jac'] and not self.supports['assembled_jac']:
            raise RuntimeError("Linear solver %s doesn't support assembled "
                               "jacobians." % self.msginfo)

    def solve(self, vec_names, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        """
        raise NotImplementedError("class %s does not implement solve()." % (type(self).__name__))

    def _run_apply(self):
        """
        Run the apply_linear method on the system.
        """
        self._recording_iter.push(('_run_apply', 0))

        system = self._system()
        scope_out, scope_in = system._get_scope()

        try:
            system._apply_linear(self._assembled_jac, self._vec_names, self._rel_systems,
                                 self._mode, scope_out, scope_in)
        finally:
            self._recording_iter.pop()


class BlockLinearSolver(LinearSolver):
    """
    A base class for LinearBlockGS and LinearBlockJac.
    """

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(BlockLinearSolver, self)._declare_options()
        self.supports['assembled_jac'] = False

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
        super(BlockLinearSolver, self)._setup_solvers(system, depth)
        if system._use_derivatives:
            self._create_rhs_vecs()

    def _create_rhs_vecs(self):
        self._rhs_vecs = rhs = {}
        system = self._system()
        for vec_name in system._lin_rel_vec_name_list:
            if self._mode == 'fwd':
                rhs[vec_name] = system._vectors['residual'][vec_name]._data.copy()
            else:
                rhs[vec_name] = system._vectors['output'][vec_name]._data.copy()

    def _update_rhs_vecs(self):
        system = self._system()
        for vec_name in system._lin_rel_vec_name_list:
            if self._mode == 'fwd':
                self._rhs_vecs[vec_name][:] = system._vectors['residual'][vec_name]._data
            else:
                self._rhs_vecs[vec_name][:] = system._vectors['output'][vec_name]._data

    def _set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Recurses to turn on or off complex stepping mode in all subsystems and their vectors.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        for vec_name in self._system()._lin_rel_vec_name_list:
            if active:
                self._rhs_vecs[vec_name] = self._rhs_vecs[vec_name].astype(np.complex)
            else:
                self._rhs_vecs[vec_name] = self._rhs_vecs[vec_name].real

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
        self._update_rhs_vecs()
        if self.options['maxiter'] > 1:
            self._run_apply()
            norm = self._iter_get_norm()
        else:
            norm = 1.0
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
        system = self._system()

        if self._mode == 'fwd':
            b_vecs = system._vectors['residual']
        else:  # rev
            b_vecs = system._vectors['output']

        norm = 0
        for vec_name in system._lin_rel_vec_name_list:
            b_vecs[vec_name]._data -= self._rhs_vecs[vec_name]
            norm += b_vecs[vec_name].get_norm()**2

        return norm ** 0.5

    def solve(self, vec_names, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        """
        self._vec_names = vec_names
        self._rel_systems = rel_systems
        self._mode = mode
        self._solve()
