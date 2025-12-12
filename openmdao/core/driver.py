"""Define a base class for all Drivers in OpenMDAO."""
from fnmatch import fnmatchcase
import functools
from itertools import chain
import pprint
import sys
import time
import os
import weakref

import numpy as np
import scipy.sparse as sp

from openmdao.core.group import Group
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.core.constants import INT_DTYPE, _SetupStatus
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.record_util import create_local_meta, check_path, has_match
from openmdao.utils.general_utils import _src_name_iter, DriverMetaclass
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.array_utils import sizes2offsets
from openmdao.vectors.vector import _full_slice, _flat_full_indexer
from openmdao.utils.indexer import indexer
from openmdao.utils.om_warnings import issue_warning, DerivativesWarning, \
    DriverWarning, OMDeprecationWarning, warn_deprecation


class DriverResult():
    """
    A container that stores information pertaining to the result of a driver execution.

    Parameters
    ----------
    driver : Driver
        The Driver associated with this DriverResult.

    Attributes
    ----------
    _driver : weakref to Driver
        A weakref to the Driver associated with this DriverResult.
    runtime : float
        The time required to execute the driver, in seconds.
    iter_count : int
        The number of iterations used by the optimizer.
    model_evals : int
        The number of times the objective function was evaluated (model solve_nonlinear calls).
    model_time : float
        The time spent in model solve_nonlinear evaluations.
    deriv_evals : int
        The number of times the total jacobian was computed.
    deriv_time : float
        The time spent computing the total jacobian.
    exit_status : str
        A string that may provide more detail about the results of the driver run.
    success : bool
        A boolean that dictates whether or not the driver was successful.
    """

    def __init__(self, driver):
        """
        Initialize the DriverResult object.
        """
        self._driver = weakref.ref(driver)
        self.runtime = 0.0
        self.iter_count = 0
        self.model_evals = 0
        self.model_time = 0.0
        self.deriv_evals = 0
        self.deriv_time = 0.0
        self.exit_status = 'NOT_RUN'
        self.success = False

    def reset(self):
        """
        Set the driver result attributes back to their default value.
        """
        self.runtime = 0.0
        self.iter_count = 0
        self.model_evals = 0
        self.model_time = 0.0
        self.deriv_evals = 0
        self.deriv_time = 0.0
        self.exit_status = 'NOT_RUN'
        self.success = False

    def __getitem__(self, s):
        """
        Provide key access to the attributes of DriverResult.

        This is included for backward compatibility with some
        tests which require dictionary-like access.

        Parameters
        ----------
        s : str
            The name of the attribute.

        Returns
        -------
        object
            The value of the attribute
        """
        return getattr(self, s)

    def __repr__(self):
        """
        Return a string representation of the DriverResult.

        Returns
        -------
        str
            A string-representation of the DriverResult object
        """
        driver = self._driver()
        prob = driver._problem()
        s = (f'Problem: {prob._name}\n'
             f'Driver:  {driver.__class__.__name__}\n'
             f'  success     : {self.success}\n'
             f'  iterations  : {self.iter_count}\n'
             f'  runtime     : {self.runtime:-10.4E} s\n'
             f'  model_evals : {self.model_evals}\n'
             f'  model_time  : {self.model_time:-10.4E} s\n'
             f'  deriv_evals : {self.deriv_evals}\n'
             f'  deriv_time  : {self.deriv_time:-10.4E} s\n'
             f'  exit_status : {self.exit_status}')
        return s

    def __bool__(self):
        """
        Mimick the behavior of the previous `failed` return value of run_driver.

        The return value is True if the driver was NOT successful.
        An OMDeprecationWarning is currently issued so users know to change their code.
        Users should utilize the `success` attribute to test for driver success.

        Returns
        -------
        bool
            True if the Driver was NOT successful.

        """
        issue_warning(msg='boolean evaluation of DriverResult is temporarily implemented '
                      'to mimick the previous `failed` return behavior of run_driver.\n'
                      'Use the `success` attribute of the returned DriverResult '
                      'object to test for successful driver completion.',
                      category=OMDeprecationWarning)
        return not self.success

    @staticmethod
    def track_stats(kind):
        """
        Decorate methods to track the model solve_nonlinear or deriv time and count.

        This decorator should be applied to the _objfunc or _gradfunc (or equivalent) methods
        of drivers. It will either accumulate the elapsed time in driver.result.model_time or
        driver.result.deriv_time, based on the value of kind.

        Parameters
        ----------
        kind : str
            One of 'model' or 'deriv', specifying which statistics should be accumulated.

        Returns
        -------
        Callable
            A wrapped version of the decorated function such that it accumulates the time and
            call count for either the objective or derivatives.
        """
        if kind not in ('model', 'deriv'):
            raise AttributeError('time_type must be one of "model" or "deriv".')

        def _track_time(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                ret = func(*args, **kwargs)
                end_time = time.perf_counter()
                result = args[0].result

                if kind == 'model':
                    result.model_time += end_time - start_time
                    result.model_evals += 1
                else:
                    result.deriv_time += end_time - start_time
                    result.deriv_evals += 1
                return ret
            return wrapper
        return _track_time


class Driver(object, metaclass=DriverMetaclass):
    """
    Top-level container for the systems and drivers.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    iter_count : int
        Keep track of iterations for case recording.
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
    recording_options : <OptionsDictionary>
        Dictionary with driver recording options.
    cite : str
        Listing of relevant citations that should be referenced when
        publishing work that uses this class.
    _problem : weakref to <Problem>
        Pointer to the containing problem.
    supports : <OptionsDictionary>
        Provides a consistent way for drivers to declare what features they support.
    _designvars : dict
        Contains all design variable info.
    _designvars_discrete : list
        List of design variables that are discrete.
    _dist_driver_vars : dict
        Dict of constraints that are distributed outputs. Key is a 'user' variable name,
        typically promoted name or an alias. Values are (local indices, local sizes).
    _exc_info : 3 item tuple
        Storage for exception and traceback information.
    _cons : dict
        Contains all constraint info.
    _objs : dict
        Contains all objective info.
    _responses : dict
        Contains all response info.
    _lin_dvs : dict
        Contains design variables relevant to linear constraints.
    _nl_dvs : dict
        Contains design variables relevant to nonlinear constraints.
    _remote_dvs : dict
        Dict of design variables that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_cons : dict
        Dict of constraints that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_objs : dict
        Dict of objectives that are remote on at least one proc. Values are
        (owning rank, size).
    _rec_mgr : <RecordingManager>
        Object that manages all recorders added to this driver.
    _coloring_info : dict
        Metadata pertaining to total coloring.
    _total_jac_format : str
        Specifies the format of the total jacobian. Allowed values are 'flat_dict', 'dict', and
        'array'.
    _con_subjacs : dict
        Dict of sparse subjacobians for use with certain optimizers, e.g. pyOptSparseDriver.
        Keyed by sources and aliases.
    _total_jac : _TotalJacInfo or None
        Cached total jacobian handling object.
    _total_jac_linear : _TotalJacInfo or None
        Cached linear total jacobian handling object.
    result : DriverResult
        DriverResult object containing information for use in the optimization report.
    _has_scaling : bool
        If True, scaling has been set for this driver.
    _filtered_vars_to_record : dict or None
        Variables to record based on recording options.
    _in_find_feasible : bool
        True if the driver is currently executing find_feasible.
    """

    def __init__(self, **kwargs):
        """
        Initialize the driver.
        """
        self._rec_mgr = RecordingManager()

        self._exc_info = None
        self._problem = None
        self._designvars = None
        self._designvars_discrete = []
        self._cons = None
        self._objs = None
        self._responses = None
        self._lin_dvs = None
        self._nl_dvs = None
        self._in_find_feasible = False

        # Driver options
        self.options = OptionsDictionary(parent_name=type(self).__name__)

        self.options.declare('debug_print', types=list,
                             values=['desvars', 'nl_cons', 'ln_cons', 'objs', 'totals'],
                             desc="List of what type of Driver variables to print at each "
                                  "iteration.",
                             default=[])

        default_desvar_behavior = os.environ.get('OPENMDAO_INVALID_DESVAR_BEHAVIOR', 'warn').lower()

        self.options.declare('invalid_desvar_behavior', values=('warn', 'raise', 'ignore'),
                             desc='Behavior of driver if the initial value of a design '
                                  'variable exceeds its bounds. The default value may be'
                                  'set using the `OPENMDAO_INVALID_DESVAR_BEHAVIOR` environment '
                                  'variable to one of the valid options.',
                             default=default_desvar_behavior)

        # Case recording options
        self.recording_options = OptionsDictionary(parent_name=type(self).__name__)

        self.recording_options.declare('record_desvars', types=bool, default=True,
                                       desc='Set to True to record design variables at the '
                                            'driver level')
        self.recording_options.declare('record_responses', types=bool, default=False,
                                       desc='Set True to record constraints and objectives at the '
                                            'driver level')
        self.recording_options.declare('record_objectives', types=bool, default=True,
                                       desc='Set to True to record objectives at the driver level')
        self.recording_options.declare('record_constraints', types=bool, default=True,
                                       desc='Set to True to record constraints at the '
                                            'driver level')
        self.recording_options.declare('includes', types=list, default=[],
                                       desc='Patterns for variables to include in recording. '
                                            'Uses fnmatch wildcards')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                            '(processed post-includes). Uses fnmatch wildcards')
        self.recording_options.declare('record_derivatives', types=bool, default=False,
                                       desc='Set to True to record derivatives at the driver '
                                            'level')
        self.recording_options.declare('record_inputs', types=bool, default=True,
                                       desc='Set to True to record inputs at the driver level')
        self.recording_options.declare('record_outputs', types=bool, default=True,
                                       desc='Set True to record outputs at the '
                                            'driver level.')
        self.recording_options.declare('record_residuals', types=bool, default=False,
                                       desc='Set True to record residuals at the '
                                            'driver level.')

        # What the driver supports.
        self.supports = OptionsDictionary(parent_name=type(self).__name__)
        self.supports.declare('optimization', types=bool, default=False)
        self.supports.declare('inequality_constraints', types=bool, default=False)
        self.supports.declare('equality_constraints', types=bool, default=False)
        self.supports.declare('linear_constraints', types=bool, default=False)
        self.supports.declare('linear_only_designvars', types=bool, default=False)
        self.supports.declare('two_sided_constraints', types=bool, default=False)
        self.supports.declare('multiple_objectives', types=bool, default=False)
        self.supports.declare('integer_design_vars', types=bool, default=True)
        self.supports.declare('gradients', types=bool, default=False)
        self.supports.declare('active_set', types=bool, default=False)
        self.supports.declare('simultaneous_derivatives', types=bool, default=False)
        self.supports.declare('total_jac_sparsity', types=bool, default=False)
        self.supports.declare('distributed_design_vars', types=bool, default=True)

        self.iter_count = 0
        self.cite = ""

        self._coloring_info = coloring_mod.ColoringMeta()

        self._total_jac_format = 'flat_dict'
        self._con_subjacs = {}
        self._total_jac = None
        self._total_jac_linear = None

        self._declare_options()
        self.options.update(kwargs)
        self.result = DriverResult(self)
        self._has_scaling = False
        self._filtered_vars_to_record = None

    def _get_inst_id(self):
        if self._problem is None:
            return None
        return f"{self._problem()._get_inst_id()}.driver"

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        return type(self).__name__

    def add_recorder(self, recorder):
        """
        Add a recorder to the driver.

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

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Driver.
        """
        pass

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        return comm

    def _set_problem(self, problem):
        """
        Set a reference to the containing Problem.

        Parameters
        ----------
        problem : <Problem>
            Reference to the containing problem.
        """
        self._problem = weakref.ref(problem)

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        model = problem.model

        self._total_jac = None

        # Determine if any design variables are discrete.
        self._designvars_discrete = [name for name, meta in self._designvars.items()
                                     if meta['source'] in model._discrete_outputs]
        if not self.supports['integer_design_vars'] and len(self._designvars_discrete) > 0:
            msg = "Discrete design variables are not supported by this driver: "
            msg += '.'.join(self._designvars_discrete)
            raise RuntimeError(msg)

        self._split_dvs(model)

        self._remote_dvs = {}
        self._remote_cons = {}
        self._dist_driver_vars = {}
        self._remote_objs = {}

        # Only allow distributed design variables on drivers that support it.
        if self.supports['distributed_design_vars'] is False:
            dist_vars = []
            abs2meta_in = model._var_allprocs_abs2meta['input']
            discrete_in = model._var_allprocs_discrete['input']
            for dv, meta in self._designvars.items():

                # For Auto-ivcs, we need to check the distributed metadata on the target instead.
                if meta['source'].startswith('_auto_ivc.'):
                    for abs_name in model._resolver.absnames(dv, 'input'):
                        # we can use abs name to check for discrete vars here because
                        # relative names are absolute names at the model level.
                        if abs_name in discrete_in:
                            # Discrete vars aren't distributed.
                            break

                        if abs2meta_in[abs_name]['distributed']:
                            dist_vars.append(dv)
                            break
                elif meta['distributed']:
                    dist_vars.append(dv)

            if dist_vars:
                dstr = ', '.join(dist_vars)
                msg = "Distributed design variables are not supported by this driver, but the "
                msg += f"following variables are distributed: [{dstr}]"
                raise RuntimeError(msg)

        # Now determine if later we'll need to allgather cons, objs, or desvars.
        if model.comm.size > 1:
            loc_vars = set(model._outputs)
            # some of these lists could have duplicate src names if aliases are used. We'll
            # fix that when we convert to sets after the allgather.
            remote_dvs = [n for n in _src_name_iter(self._designvars) if n not in loc_vars]
            remote_cons = [n for n in _src_name_iter(self._cons) if n not in loc_vars]
            remote_objs = [n for n in _src_name_iter(self._objs) if n not in loc_vars]

            con_set = set()
            obj_set = set()
            dv_set = set()

            all_remote_vois = model.comm.allgather((remote_dvs, remote_cons, remote_objs))
            for rem_dvs, rem_cons, rem_objs in all_remote_vois:
                con_set.update(rem_cons)
                obj_set.update(rem_objs)
                dv_set.update(rem_dvs)

            # If we have remote VOIs, pick an owning rank for each and use that
            # to bcast to others later
            owning_ranks = model._owning_rank
            abs2idx = model._var_allprocs_abs2idx
            abs2meta_out = model._var_allprocs_abs2meta['output']
            sizes = model._var_sizes['output']
            rank = model.comm.rank
            nprocs = model.comm.size

            dist_dict = self._dist_driver_vars

            # Loop over all VOIs.
            for vname, voimeta in chain(self._responses.items(), self._designvars.items()):
                # vname may be a promoted name or an alias

                indices = voimeta['indices']
                vsrc = voimeta['source']

                meta = abs2meta_out[vsrc]
                i = abs2idx[vsrc]

                if meta['distributed']:

                    dist_sizes = sizes[:, i]
                    tot_size = np.sum(dist_sizes)

                    # Determine which indices are on our proc.
                    offsets = sizes2offsets(dist_sizes)

                    if indices is not None:
                        indices = indices.shaped_array()
                        true_sizes = np.zeros(nprocs, dtype=INT_DTYPE)
                        for irank in range(nprocs):
                            dist_inds = indices[np.logical_and(indices >= offsets[irank],
                                                               indices < (offsets[irank] +
                                                                          dist_sizes[irank]))]
                            true_sizes[irank] = dist_inds.size
                            if irank == rank:
                                local_indices = dist_inds - offsets[rank]
                                distrib_indices = dist_inds

                        ind = indexer(local_indices, src_shape=(tot_size,), flat_src=True)
                        dist_dict[vname] = (ind, true_sizes, distrib_indices)
                    else:
                        dist_dict[vname] = (_flat_full_indexer, dist_sizes,
                                            slice(offsets[rank], offsets[rank] + dist_sizes[rank]))

                else:
                    owner = owning_ranks[vsrc]
                    sz = sizes[owner, i]

                    if vsrc in dv_set:
                        self._remote_dvs[vname] = (owner, sz)
                    if vsrc in con_set:
                        self._remote_cons[vname] = (owner, sz)
                    if vsrc in obj_set:
                        self._remote_objs[vname] = (owner, sz)

        self._remote_responses = self._remote_cons.copy()
        self._remote_responses.update(self._remote_objs)

        # set up simultaneous deriv coloring
        if coloring_mod._use_total_sparsity:
            # reset the coloring
            if self._coloring_info.dynamic or self._coloring_info.static is not None:
                self._coloring_info.coloring = None

            coloring = self._get_static_coloring()
            if coloring is not None and self.supports['simultaneous_derivatives']:
                if model._owns_approx_jac:
                    coloring._check_config_partial(model)
                else:
                    coloring._check_config_total(self, model)

                if not problem.model._use_derivatives:
                    issue_warning("Derivatives are turned off.  Skipping simul deriv coloring.",
                                  category=DerivativesWarning)

    def _split_dvs(self, model):
        """
        Determine which design vars are relevant to linear constraints vs nonlinear constraints.

        For some optimizers, this information will be used to determine the columns of the total
        linear jacobian vs. the total nonlinear jacobian.

        Parameters
        ----------
        model : <Group>
            The model being used in the optimization problem.
        """
        lin_cons = tuple([meta['source'] for meta in self._cons.values() if meta['linear']])
        if lin_cons:
            relevance = model._relevance
            dvs = tuple([meta['source'] for meta in self._designvars.values()])

            with relevance.seeds_active(fwd_seeds=dvs, rev_seeds=lin_cons):
                self._lin_dvs = {dv: meta for dv, meta in self._designvars.items()
                                 if relevance.is_relevant(meta['source'])}

            nl_resps = [meta['source'] for meta in self._cons.values() if not meta['linear']]
            nl_resps.extend([meta['source'] for meta in self._objs.values()])

            with relevance.seeds_active(fwd_seeds=dvs, rev_seeds=tuple(nl_resps)):
                self._nl_dvs = {dv: meta for dv, meta in self._designvars.items()
                                if relevance.is_relevant(meta['source'])}

        else:
            self._lin_dvs = {}
            self._nl_dvs = self._designvars

    def _get_lin_dvs(self):
        """
        Get the design variables relevant to linear constraints.

        If the driver does not support linear-only design variables, this will return all design
        variables.

        Returns
        -------
        dict
            Dictionary containing design variables relevant to linear constraints.
        """
        return self._lin_dvs if self.supports['linear_only_designvars'] else self._designvars

    def _get_nl_dvs(self):
        """
        Get the design variables relevant to nonlinear constraints.

        If the driver does not support linear-only design variables, this will return all design
        variables.

        Returns
        -------
        dict
            Dictionary containing design variables relevant to nonlinear constraints.
        """
        return self._nl_dvs if self.supports['linear_only_designvars'] else self._designvars

    def _check_for_missing_objective(self):
        """
        Check for missing objective and raise error if no objectives found.
        """
        if len(self._objs) == 0:
            msg = "Driver requires objective to be declared"
            raise RuntimeError(msg)

    def _check_for_invalid_desvar_values(self):
        """
        Check for design variable values that exceed their bounds.

        This method's behavior is controlled by the OPENMDAO_INVALID_DESVAR environment variable,
        which may take on values 'ignore', 'raise'', 'warn'.
        - 'ignore' : Proceed without checking desvar bounds.
        - 'warn' : Issue a warning if one or more desvar values exceed bounds.
        - 'raise' : Raise an exception if one or more desvar values exceed bounds.

        These options are case insensitive.
        """
        if self.options['invalid_desvar_behavior'] != 'ignore':
            invalid_desvar_data = []
            for var, meta in self._designvars.items():
                _val = self._problem().get_val(var, units=meta['units'], get_remote=True)
                val = np.array([_val]) if np.ndim(_val) == 0 else _val  # Handle discrete desvars
                idxs = meta['indices']() if meta['indices'] else None
                flat_idxs = meta['flat_indices']
                scaler = meta['scaler'] if meta['scaler'] is not None else 1.
                adder = meta['adder'] if meta['adder'] is not None else 0.
                lower = meta['lower'] / scaler - adder
                upper = meta['upper'] / scaler - adder
                flat_val = val.ravel()[idxs] if flat_idxs else val[idxs].ravel()

                if (flat_val < lower).any() or (flat_val > upper).any():
                    invalid_desvar_data.append((var, val, lower, upper))
            if invalid_desvar_data:
                s = 'The following design variable initial conditions are out of their ' \
                    'specified bounds:'
                for var, val, lower, upper in invalid_desvar_data:
                    s += f'\n  {var}\n    val: {val.ravel()}' \
                         f'\n    lower: {lower}\n    upper: {upper}'
                s += '\nSet the initial value of the design variable to a valid value or set ' \
                     'the driver option[\'invalid_desvar_behavior\'] to \'ignore\'.'
                if self.options['invalid_desvar_behavior'] == 'raise':
                    raise ValueError(s)
                else:
                    issue_warning(s, category=DriverWarning)

    def _get_vars_to_record(self, obj=None):
        """
        Get variables to record based on recording options.

        Parameters
        ----------
        obj : Problem or Driver
            Parent object which has recording options.

        Returns
        -------
        dict
           Dictionary containing lists of variables to record.
        """
        if obj is None:
            obj = self

        recording_options = obj.recording_options

        problem = self._problem()
        model = problem.model
        resolver = model._resolver
        incl = recording_options['includes']
        excl = recording_options['excludes']

        # includes and excludes for outputs are specified using promoted names
        # includes and excludes for inputs are specified using _absolute_ names

        # set of promoted output names and absolute input and residual names
        # used for matching includes/excludes
        match_names = set()

        # 1. If record_outputs is True, get the set of outputs
        # 2. Filter those using includes and excludes to get the baseline set of variables to record
        # 3. Add or remove from that set any desvars, objs, and cons based on the recording
        #    options of those

        # includes and excludes for outputs are specified using _promoted_ names
        # vectors are keyed on absolute name, discretes on relative/promoted name
        myinputs = set()
        myoutputs = set()
        myresiduals = set()

        if recording_options['record_outputs']:
            match_names.update(resolver.prom_iter('output'))
            myoutputs = {n for n, prom in resolver.abs2prom_iter('output')
                         if check_path(prom, incl, excl)}

        if recording_options['record_residuals']:
            match_names.update(model._residuals)
            myresiduals = [n for n in model._residuals
                           if check_path(resolver.abs2prom(n, 'output'), incl, excl)]

        if recording_options['record_desvars']:
            myoutputs.update(_src_name_iter(self._designvars))
        if recording_options['record_objectives'] or recording_options['record_responses']:
            myoutputs.update(_src_name_iter(self._objs))
        if recording_options['record_constraints'] or recording_options['record_responses']:
            myoutputs.update(_src_name_iter(self._cons))

        # inputs (if in options). inputs use _absolute_ names for includes/excludes
        if 'record_inputs' in recording_options:
            if recording_options['record_inputs']:
                match_names.update(resolver.abs_iter('input'))
                myinputs = {n for n in resolver.abs_iter('input') if check_path(n, incl, excl)}

                match_names.update(model._resolver.prom_iter('input'))
                for p in model._resolver.prom_iter('input'):
                    if check_path(p, incl, excl):
                        myoutputs.add(model._resolver.source(p))

        # check that all exclude/include globs have at least one matching output or input name
        for pattern in excl:
            if not has_match(pattern, match_names):
                issue_warning(f"{obj.msginfo}: No matches for pattern '{pattern}' in "
                              "recording_options['excludes'].")
        for pattern in incl:
            if pattern != '*' and not has_match(pattern, match_names):
                issue_warning(f"{obj.msginfo}: No matches for pattern '{pattern}' in "
                              "recording_options['includes'].")

        # sort lists to ensure that vars are iterated over in the same order on all procs
        vars2record = {
            'input': sorted(myinputs),
            'output': sorted(myoutputs),
            'residual': sorted(myresiduals)
        }

        return vars2record

    def _setup_recording(self):
        """
        Set up case recording.
        """
        if self._rec_mgr.has_recorders():
            self._filtered_vars_to_record = self._get_vars_to_record()
            self._rec_mgr.startup(self, self._problem().comm)

    def _run(self):
        """
        Execute this driver.

        This calls the run() method, which should be overriden by the subclass.

        Returns
        -------
        DriverResult
            DriverResult object, containing information about the run.
        """
        problem = self._problem()
        model = problem.model

        if self.supports['optimization'] and problem.options['group_by_pre_opt_post']:
            if model._pre_components:
                with model._relevance.nonlinear_active('pre'):
                    self._run_solve_nonlinear()

            with SaveOptResult(self):
                with model._relevance.nonlinear_active('iter'):
                    self.result.success = not self.run()

            if model._post_components:
                with model._relevance.nonlinear_active('post'):
                    self._run_solve_nonlinear()

        else:
            with SaveOptResult(self):
                self.result.success = not self.run()

        return self.result

    def _get_voi_val(self, name, meta, remote_vois, driver_scaling=True,
                     get_remote=True, rank=None):
        """
        Get the value of a variable of interest (objective, constraint, or design var).

        This will retrieve the value if the VOI is remote.

        Parameters
        ----------
        name : str
            Name of the variable of interest.
        meta : dict
            Metadata for the variable of interest.
        remote_vois : dict
            Dict containing (owning_rank, size) for all remote vois of a particular
            type (design var, constraint, or objective).
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.
        get_remote : bool or None
            If True, retrieve the value even if it is on a remote process.  Note that if the
            variable is remote on ANY process, this function must be called on EVERY process
            in the Problem's MPI communicator.
            If False, only retrieve the value if it is on the current process, or only the part
            of the value that's on the current process for a distributed variable.
        rank : int or None
            If not None, gather value to this rank only.

        Returns
        -------
        float or ndarray
            The value of the named variable of interest.
        """
        model = self._problem().model
        comm = model.comm
        get = model._outputs._abs_get_val
        indices = meta['indices']
        src_name = meta['source']

        if MPI:
            distributed = comm.size > 0 and name in self._dist_driver_vars
        else:
            distributed = False

        if name in remote_vois:
            owner, size = remote_vois[name]
            # if var is distributed or only gathering to one rank
            # TODO - support distributed var under a parallel group.
            if owner is None or rank is not None:
                val = model.get_val(src_name, get_remote=get_remote, rank=rank, flat=True)
                if indices is not None:
                    val = val[indices.flat()]
            else:
                if owner == comm.rank:
                    if indices is None:
                        val = get(src_name, flat=True).copy()
                    else:
                        val = get(src_name, flat=True)[indices.as_array()]
                else:
                    if indices is not None:
                        size = indices.indexed_src_size
                    val = np.empty(size)

                if get_remote:
                    comm.Bcast(val, root=owner)

        elif distributed:
            local_val = model.get_val(src_name, get_remote=False, flat=True)
            local_indices, sizes, _ = self._dist_driver_vars[name]
            if local_indices is not _full_slice:
                local_val = local_val[local_indices()]

            if get_remote:
                local_val = np.ascontiguousarray(local_val)
                offsets = np.zeros(sizes.size, dtype=INT_DTYPE)
                offsets[1:] = np.cumsum(sizes[:-1])
                val = np.zeros(np.sum(sizes))
                comm.Allgatherv(local_val, [val, sizes, offsets, MPI.DOUBLE])
            else:
                val = local_val

        else:
            if src_name in model._discrete_outputs:
                val = model._discrete_outputs[src_name]
                if name in self._designvars_discrete:
                    # At present, only integers are supported by OpenMDAO drivers.
                    # We check the values here.
                    if not ((np.isscalar(val) and isinstance(val, (int, np.integer))) or
                            (isinstance(val, np.ndarray) and np.issubdtype(val[0], np.integer))):
                        if np.isscalar(val):
                            suffix = f"A value of type '{type(val).__name__}' was specified."
                        elif isinstance(val, np.ndarray):
                            suffix = f"An array of type '{val.dtype.name}' was specified."
                        else:
                            suffix = ''
                        raise ValueError("Only integer scalars or ndarrays are supported as values "
                                         "for discrete variables when used as a design variable. "
                                         + suffix)
            elif indices is None:
                val = get(src_name, flat=True).copy()
            else:
                val = get(src_name, flat=True)[indices.as_array()]

        if self._has_scaling and driver_scaling:
            # Scale design variable values
            adder = meta['total_adder']
            if adder is not None:
                val += adder

            scaler = meta['total_scaler']
            if scaler is not None:
                val *= scaler

        return val

    def get_driver_objective_calls(self):
        """
        Return number of objective evaluations made during a driver run.

        Returns
        -------
        int
            Number of objective evaluations made during a driver run.
        """
        warn_deprecation('get_driver_objective_calls is deprecated. '
                         'Use `driver.result.model_evals`')
        return self.result.model_evals

    def get_driver_derivative_calls(self):
        """
        Return number of derivative evaluations made during a driver run.

        Returns
        -------
        int
            Number of derivative evaluations made during a driver run.
        """
        warn_deprecation('get_driver_derivative_calls is deprecated. '
                         'Use `driver.result.deriv_evals`')
        return self.result.deriv_evals

    def get_design_var_values(self, get_remote=True, driver_scaling=True):
        """
        Return the design variable values.

        Parameters
        ----------
        get_remote : bool or None
            If True, retrieve the value even if it is on a remote process.  Note that if the
            variable is remote on ANY process, this function must be called on EVERY process
            in the Problem's MPI communicator.
            If False, only retrieve the value if it is on the current process, or only the part
            of the value that's on the current process for a distributed variable.
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.

        Returns
        -------
        dict
           Dictionary containing values of each design variable.
        """
        return {n: self._get_voi_val(n, dvmeta, self._remote_dvs, get_remote=get_remote,
                                     driver_scaling=driver_scaling)
                for n, dvmeta in self._designvars.items()}

    def set_design_var(self, name, value, set_remote=True):
        """
        Set the value of a design variable.

        'name' can be a promoted output name or an alias.

        Parameters
        ----------
        name : str
            Global pathname of the design variable.
        value : float or ndarray
            Value for the design variable.
        set_remote : bool
            If True, set the global value of the variable (value must be of the global size).
            If False, set the local value of the variable (value must be of the local size).
        """
        problem = self._problem()
        meta = self._designvars[name]

        src_name = meta['source']

        # if the value is not local, don't set the value
        if (src_name in self._remote_dvs and
                problem.model._owning_rank[src_name] != problem.comm.rank):
            return

        if name in self._designvars_discrete:

            # Note, drivers set values here and generally should know it is setting an integer.
            # However, the DOEdriver may pull a non-integer value from its generator, so we
            # convert it.
            if isinstance(value, float):
                value = int(value)
            elif isinstance(value, np.ndarray):
                if isinstance(problem.model._discrete_outputs[src_name], int):
                    # Setting an integer value with a 1D array - don't want to convert to array.
                    value = int(value.item())
                else:
                    value = value.astype(int)

            problem.model._discrete_outputs[src_name] = value

        elif problem.model._outputs._contains_abs(src_name):
            desvar = problem.model._outputs._abs_get_val(src_name)
            if name in self._dist_driver_vars:
                loc_idxs, _, dist_idxs = self._dist_driver_vars[name]
                loc_idxs = loc_idxs()  # don't use indexer here
            else:
                loc_idxs = meta['indices']
                if loc_idxs is None:
                    loc_idxs = _full_slice
                else:
                    loc_idxs = loc_idxs()
                dist_idxs = _full_slice

            if set_remote:
                # provided value is the global value, use indices for this proc
                desvar[loc_idxs] = np.atleast_1d(value)[dist_idxs]
            else:
                # provided value is the local value
                desvar[loc_idxs] = np.atleast_1d(value)

            # Undo driver scaling when setting design var values into model.
            if self._has_scaling:
                scaler = meta['total_scaler']
                if scaler is not None:
                    desvar[loc_idxs] *= 1.0 / scaler

                adder = meta['total_adder']
                if adder is not None:
                    desvar[loc_idxs] -= adder

    def get_objective_values(self, driver_scaling=True):
        """
        Return objective values.

        Parameters
        ----------
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.

        Returns
        -------
        dict
           Dictionary containing values of each objective.
        """
        return {n: self._get_voi_val(n, obj, self._remote_objs,
                                     driver_scaling=driver_scaling)
                for n, obj in self._objs.items()}

    def get_constraint_values(self, ctype='all', lintype='all', driver_scaling=True,
                              viol=False):
        """
        Return constraint values.

        Parameters
        ----------
        ctype : str
            Default is 'all'. Optionally return just the inequality constraints
            with 'ineq' or the equality constraints with 'eq'.
        lintype : str
            Default is 'all'. Optionally return just the linear constraints
            with 'linear' or the nonlinear constraints with 'nonlinear'.
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.
        viol : bool
            If True, return the constraint violation rather than the actual value. This
            is used when minimizing the constraint violation. For equality constraints
            this is the (optionally scaled) absolute value of deviation for the desired
            value. For inequality constraints, this is the (optionally scaled) absolute
            value of deviation beyond the upper or lower bounds, or zero if it is within
            bounds.

        Returns
        -------
        dict
           Dictionary containing values of each constraint.
        """
        con_dict = {}
        it = self._cons.items()
        if lintype == 'linear':
            it = filter_by_meta(it, 'linear')
        elif lintype == 'nonlinear':
            it = filter_by_meta(it, 'linear', exclude=True)
        if ctype == 'eq':
            it = filter_by_meta(it, 'equals', chk_none=True)
        elif ctype == 'ineq':
            it = filter_by_meta(it, 'equals', chk_none=True, exclude=True)

        for name, meta in it:
            if viol:
                con_val = self._get_voi_val(name, meta, self._remote_cons,
                                            driver_scaling=True)
                size = con_val.size
                con_dict[name] = np.zeros(size)
                if meta['equals'] is not None:
                    con_dict[name][...] = con_val - meta['equals']
                else:
                    lower_viol_idxs = np.where(con_val < meta['lower'])[0]
                    upper_viol_idxs = np.where(con_val > meta['upper'])[0]
                    con_dict[name][lower_viol_idxs] = con_val[lower_viol_idxs] - meta['lower']
                    con_dict[name][upper_viol_idxs] = con_val[upper_viol_idxs] - meta['upper']

                # We got the voi value in driver-scaled units.
                # Unscale if necessary.
                if not driver_scaling:
                    scaler = meta['total_scaler']
                    if scaler is not None:
                        con_dict[name] /= scaler

            else:
                con_dict[name] = self._get_voi_val(name, meta, self._remote_cons,
                                                   driver_scaling=driver_scaling)

        return con_dict

    def _get_ordered_nl_responses(self):
        """
        Return the names of nonlinear responses in the order used by the driver.

        Default order is objectives followed by nonlinear constraints.  This is used for
        simultaneous derivative coloring and sparsity determination.

        Returns
        -------
        list of str
            The nonlinear response names in order.
        """
        order = list(self._objs)
        order.extend(n for n, meta in self._cons.items() if not meta['linear'])
        return order

    def _update_voi_meta(self, model, responses, desvars):
        """
        Collect response and design var metadata from the model and size desvars and responses.

        Parameters
        ----------
        model : System
            The System that represents the entire model.
        responses : dict
            Response metadata dictionary.
        desvars : dict
            Design variable metadata dictionary.

        Returns
        -------
        int
            Total size of responses, with linear constraints excluded.
        int
            Total size of design vars.
        """
        self._objs = objs = {}
        self._cons = cons = {}

        self._responses = responses
        self._designvars = desvars

        # driver _responses are keyed by either the alias or the promoted name
        response_size = 0
        for name, meta in responses.items():
            if meta['type'] == 'con':
                cons[name] = meta
                if meta['linear']:
                    continue  # don't add to response size
            else:
                objs[name] = meta

            response_size += meta['global_size']

        desvar_size = sum(meta['global_size'] for meta in desvars.values())

        self._has_scaling = model._setup_driver_units()

        return response_size, desvar_size

    def get_exit_status(self):
        """
        Return exit status of driver run.

        Returns
        -------
        str
            String indicating result of driver run.
        """
        return 'SUCCESS' if self.result.success else 'FAIL'

    def check_relevance(self):
        """
        Check if there are constraints that don't depend on any design vars.

        This usually indicates something is wrong with the problem formulation.
        """
        # relevance not relevant if not using derivatives
        if not self.supports['gradients']:
            return

        if 'singular_jac_behavior' in self.options:
            singular_behavior = self.options['singular_jac_behavior']
            if singular_behavior == 'ignore':
                return
        else:
            singular_behavior = 'warn'

        problem = self._problem()

        # Do not perform this check if any subgroup uses approximated partials.
        # This causes the relevance graph to be invalid.
        for system in problem.model.system_iter(include_self=True, recurse=True, typ=Group):
            if system._has_approx:
                return

        bad = {n for n in self._problem().model._relevance._no_dv_responses
               if n not in self._designvars}
        if bad:
            bad_conns = [n for n, m in self._cons.items() if m['source'] in bad]
            bad_objs = [n for n, m in self._objs.items() if m['source'] in bad]
            badmsg = []
            if bad_conns:
                badmsg.append(f"constraint(s) {bad_conns}")
            if bad_objs:
                badmsg.append(f"objective(s) {bad_objs}")
            bad = ' and '.join(badmsg)
            # Note: There is a hack in ScipyOptimizeDriver for older versions of COBYLA that
            #       implements bounds on design variables by adding them as constraints.
            #       These design variables as constraints will not appear in the wrt list.
            msg = f"{self.msginfo}: {bad} do not depend on any " \
                  "design variables. Please check your problem formulation."
            if singular_behavior == 'error':
                raise RuntimeError(msg)
            else:
                issue_warning(msg, category=DriverWarning)

    def run(self):
        """
        Execute this driver.

        The base `Driver` just runs the model. All other drivers overload
        this method.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        self.result.reset()
        with RecordingDebugging(self._get_name(), self.iter_count, self):
            self._run_solve_nonlinear()

        self.iter_count += 1

        return False

    @property
    def _recording_iter(self):
        return self._problem()._metadata['recording_iter']

    @DriverResult.track_stats(kind='model')
    def _run_solve_nonlinear(self):
        return self._problem().model.run_solve_nonlinear()

    @DriverResult.track_stats(kind='deriv')
    def _compute_totals(self, of=None, wrt=None, return_format='flat_dict', driver_scaling=True):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        All derivatives are returned using driver scaling.

        Parameters
        ----------
        of : list of variable name str or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name str or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : str
            Format to return the derivatives. Default is a 'flat_dict', which
            returns them in a dictionary whose keys are tuples of form (of, wrt). For
            the scipy optimizer, 'array' is also supported.
        driver_scaling : bool
            If True (default), scale derivative values by the quantities specified when the desvars
            and responses were added. If False, leave them unscaled.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        problem = self._problem()
        debug_print = 'totals' in self.options['debug_print'] and (not MPI or
                                                                   problem.comm.rank == 0)

        if debug_print:
            header = 'Driver total derivatives for iteration: ' + str(self.iter_count)
            print(header)
            print(len(header) * '-' + '\n')

        if self._total_jac is None:
            total_jac = _TotalJacInfo(problem, of, wrt, return_format,
                                      approx=problem.model._owns_approx_jac,
                                      debug_print=debug_print,
                                      driver_scaling=driver_scaling)

            if total_jac.has_lin_cons and self.supports['linear_constraints']:
                self._total_jac_linear = total_jac
            else:
                self._total_jac = total_jac
        else:
            total_jac = self._total_jac

        totals = total_jac.compute_totals()

        if self.recording_options['record_derivatives']:
            self.record_derivatives()

        return totals

    def record_derivatives(self):
        """
        Record the current total jacobian.
        """
        if self._total_jac is not None and self._rec_mgr._recorders:
            metadata = create_local_meta(self._get_name())
            self._total_jac.record_derivatives(self, metadata)

    def record_iteration(self):
        """
        Record an iteration of the current Driver.
        """
        status = -1 if self._problem is None else self._problem()._metadata['setup_status']
        if status >= _SetupStatus.POST_FINAL_SETUP:
            record_iteration(self, self._problem(), self._get_name())
        else:
            raise RuntimeError(f'{self.msginfo} attempted to record iteration but '
                               'driver has not been initialized; `run_model()`, '
                               '`run_driver()`, or `final_setup()` must be called '
                               'before recording.')

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

    def _get_name(self):
        """
        Get name of current Driver.

        Returns
        -------
        str
            Name of current Driver.
        """
        return "Driver"

    def declare_coloring(self, num_full_jacs=coloring_mod._DEF_COMP_SPARSITY_ARGS['num_full_jacs'],
                         tol=coloring_mod._DEF_COMP_SPARSITY_ARGS['tol'],
                         orders=coloring_mod._DEF_COMP_SPARSITY_ARGS['orders'],
                         perturb_size=coloring_mod._DEF_COMP_SPARSITY_ARGS['perturb_size'],
                         min_improve_pct=coloring_mod._DEF_COMP_SPARSITY_ARGS['min_improve_pct'],
                         show_summary=coloring_mod._DEF_COMP_SPARSITY_ARGS['show_summary'],
                         show_sparsity=coloring_mod._DEF_COMP_SPARSITY_ARGS['show_sparsity'],
                         use_scaling=coloring_mod._DEF_COMP_SPARSITY_ARGS['use_scaling'],
                         randomize_subjacs=True, randomize_seeds=False, direct=True):
        """
        Set options for total deriv coloring.

        Parameters
        ----------
        num_full_jacs : int
            Number of times to repeat partial jacobian computation when computing sparsity.
        tol : float
            Tolerance used to determine if an array entry is nonzero during sparsity determination.
        orders : int
            Number of orders above and below the tolerance to check during the tolerance sweep.
        perturb_size : float
            Size of input/output perturbation during generation of sparsity.
        min_improve_pct : float
            If coloring does not improve (decrease) the number of solves more than the given
            percentage, coloring will not be used.
        show_summary : bool
            If True, display summary information after generating coloring.
        show_sparsity : bool
            If True, display sparsity with coloring info after generating coloring.
        use_scaling : bool
            If True, use driver scaling when generating the sparsity.
        randomize_subjacs : bool
            If True, use random subjacobians corresponding to their declared sparsity patterns.
        randomize_seeds : bool
            If True, use random seeds when computing the sparsity.
        direct : bool
            If using bidirectional coloring, use the direct method when computing the column
            adjacency matrix instead of the substitution method.
        """
        self._coloring_info.coloring = None
        self._coloring_info.num_full_jacs = num_full_jacs
        self._coloring_info.tol = tol
        self._coloring_info.orders = orders
        self._coloring_info.perturb_size = perturb_size
        self._coloring_info.min_improve_pct = min_improve_pct
        if self._coloring_info.static is None:
            self._coloring_info.dynamic = True
        else:
            self._coloring_info.dynamic = False
        self._coloring_info.show_summary = show_summary
        self._coloring_info.show_sparsity = show_sparsity
        self._coloring_info.use_scaling = use_scaling
        self._coloring_info.randomize_subjacs = randomize_subjacs
        self._coloring_info.randomize_seeds = randomize_seeds
        self._coloring_info.direct = direct

    def use_fixed_coloring(self, coloring=coloring_mod.STD_COLORING_FNAME()):
        """
        Tell the driver to use a precomputed coloring.

        Parameters
        ----------
        coloring : str or Coloring
            A coloring filename or a Coloring object.  If no arg is passed, filename will be
            determined automatically.
        """
        if self.supports['simultaneous_derivatives']:
            if coloring_mod._force_dyn_coloring and isinstance(coloring,
                                                               coloring_mod.STD_COLORING_FNAME):
                # force the generation of a dynamic coloring this time
                self._coloring_info.dynamic = True
                self._coloring_info.static = None
            else:
                self._coloring_info.static = coloring
                self._coloring_info.dynamic = False

            self._coloring_info.coloring = None
        else:
            raise RuntimeError("Driver '%s' does not support simultaneous derivatives." %
                               self._get_name())

    def _setup_tot_jac_sparsity(self, coloring=None):
        """
        Set up total jacobian subjac sparsity.

        Drivers that can use subjac sparsity should override this.

        Parameters
        ----------
        coloring : Coloring or None
            Current coloring.
        """
        pass

    def _get_static_coloring(self):
        """
        Get the Coloring for this driver.

        If necessary, load the Coloring from a file.

        Returns
        -------
        Coloring or None
            The pre-existing or loaded Coloring, or None
        """
        coloring = None
        info = self._coloring_info
        static = info.static
        model = self._problem().model

        if isinstance(static, coloring_mod.Coloring):
            coloring = static
            info.coloring = coloring
        else:
            coloring = info.coloring

            if coloring is None and isinstance(static, (str, coloring_mod.STD_COLORING_FNAME)):
                if isinstance(static, str):
                    fname = static
                else:
                    fname = self.get_coloring_fname(mode='input')

                print(f"loading total coloring from file {fname}")
                coloring = info.coloring = coloring_mod.Coloring.load(fname)
                info.update(coloring._meta)

                ofname = self.get_coloring_fname(mode='output')
                if ((model._full_comm is not None and model._full_comm.rank == 0) or
                        (model._full_comm is None and model.comm.rank == 0)):
                    coloring.save(ofname)

        if coloring is not None and info.static is not None:
            problem = self._problem()
            if coloring._rev and problem._orig_mode not in ('rev', 'auto'):
                revcol = coloring._rev[0][0]
                if revcol:
                    raise RuntimeError("Simultaneous coloring does reverse solves but mode has "
                                       f"been set to '{problem._orig_mode}'")
            if coloring._fwd and problem._orig_mode not in ('fwd', 'auto'):
                fwdcol = coloring._fwd[0][0]
                if fwdcol:
                    raise RuntimeError("Simultaneous coloring does forward solves but mode has "
                                       f"been set to '{problem._orig_mode}'")

        return coloring

    def get_coloring_fname(self, mode='output'):
        """
        Get the filename for the coloring file.

        Parameters
        ----------
        mode : str
            'input' or 'output'.

        Returns
        -------
        str
            The filename for the coloring file.
        """
        return self._problem().model.get_coloring_fname(mode)

    def scaling_report(self, outfile='driver_scaling_report.html', title=None, show_browser=True,
                       jac=True):
        """
        Generate a self-contained html file containing a detailed connection viewer.

        Optionally pops up a web browser to view the file.

        Parameters
        ----------
        outfile : str, optional
            The name of the output html file.  Defaults to 'driver_scaling_report.html'.
        title : str, optional
            Sets the title of the web page.
        show_browser : bool, optional
            If True, pop up a browser to view the generated html file. Defaults to True.
        jac : bool
            If True, show jacobian information.

        Returns
        -------
        dict
            Data used to create html file.
        """
        from openmdao.visualization.scaling_viewer.scaling_report import view_driver_scaling

        # Run the model if it hasn't been run yet.
        status = -1 if self._problem is None else self._problem()._metadata['setup_status']
        if status < _SetupStatus.POST_FINAL_SETUP:
            raise RuntimeError("Either 'run_model' or 'final_setup' must be called before the "
                               "scaling report can be generated.")

        prob = self._problem()
        if prob._run_counter < 0:
            prob.run_model()

        return view_driver_scaling(self, outfile=outfile, show_browser=show_browser, jac=jac,
                                   title=title)

    def _pre_run_model_debug_print(self):
        """
        Optionally print some debugging information before the model runs.
        """
        debug_opt = self.options['debug_print']
        rank = self._problem().comm.rank
        if not debug_opt or debug_opt == ['totals']:
            return

        if not MPI or rank == 0:
            header = 'Driver debug print for iter coord: {}'.format(
                self._recording_iter.get_formatted_iteration_coordinate())
            print(header)
            print(len(header) * '-')

        if 'desvars' in debug_opt:
            model = self._problem().model
            desvar_vals = {n: model.get_val(n, get_remote=True, rank=0) for n in self._designvars}
            if not MPI or rank == 0:
                print("Design Vars")
                if desvar_vals:
                    pprint.pprint(desvar_vals)
                else:
                    print("None")
                print()

        sys.stdout.flush()

    def _post_run_model_debug_print(self):
        """
        Optionally print some debugging information after the model runs.
        """
        rank = self._problem().comm.rank

        if 'nl_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='nonlinear', driver_scaling=False)
            if not MPI or rank == 0:
                print("Nonlinear constraints")
                if cons:
                    pprint.pprint(cons)
                else:
                    print("None")
                print()

        if 'ln_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='linear', driver_scaling=False)
            if not MPI or rank == 0:
                print("Linear constraints")
                if cons:
                    pprint.pprint(cons)
                else:
                    print("None")
                print()

        if 'objs' in self.options['debug_print']:
            objs = self.get_objective_values(driver_scaling=False)
            if not MPI or rank == 0:
                print("Objectives")
                if objs:
                    pprint.pprint(objs)
                else:
                    print("None")
                print()

        sys.stdout.flush()

    def get_reports_dir(self):
        """
        Get the path to the directory where the report files should go.

        If it doesn't exist, it will be created.

        Returns
        -------
        str
            The path to the directory where reports should be written.
        """
        return self._problem().get_reports_dir()

    def _get_coloring(self, run_model=None):
        """
        Get the total coloring for this driver.

        If necessary, dynamically generate it.

        Parameters
        ----------
        run_model : bool or None
            If False, don't run model, else use problem _run_counter to decide.
            This is ignored if the coloring has already been computed.

        Returns
        -------
        Coloring or None
            Coloring object, possible loaded from a file or dynamically generated, or None
        """
        if coloring_mod._use_total_sparsity:
            if run_model and self._coloring_info.coloring is not None:
                issue_warning("The 'run_model' argument is ignored because the coloring has "
                              "already been computed.")

            if self._coloring_info.dynamic and self._coloring_info.do_compute_coloring():
                ofname = self.get_coloring_fname(mode='output')
                self._coloring_info.coloring = \
                    coloring_mod.dynamic_total_coloring(self,
                                                        run_model=run_model,
                                                        fname=ofname)

            return self._coloring_info.coloring

    def _update_result(self, result):
        """
        Set additional attributes and information to the DriverResult.
        """
        pass

    def _get_active_cons_and_dvs(self, feas_atol=1.e-6, feas_rtol=1.e-6, assume_dvs_active=False):
        """
        Obtain the constraints and design varaibles which are active.

        Active means the constraint or design variable is on the bound (or close enough
        that it satisfies np.isclose(val, bound, atol=feas_atol, rtol=feas_rtol))

        Parameters
        ----------
        feas_atol : float
            Feasibility absolute tolerance
        feas_rtol : float
            Feasibility relative tolerance
        assume_dvs_active : bool
            Override to force design variables to be treated as active.

        Returns
        -------
        active_cons : dict[str: dict]
            The names of the active constraints. For each active constraint,
            a dict of the active indices and the active bound (0='equals',
            -1='lower', 1='upper') is provided. These are the active indices
            _of_ the constraint "indices".
        active_dvs : list[str]
            The names of the active design variables. For each active design
            variable, a dict of the active indices and the active bound
            (0='equals', -1='lower', 1='upper') is provided. An active
            design variable bound of 'equal' is only possible when
            assume_dvs_active is True, and the design variables are
            returned as if they are on an active equality constraint.
            These are the active indices _of_ the design var indices.
        """
        active_cons = {}
        active_dvs = {}
        des_vars = self._designvars
        constraints = self._cons

        # We obtain the driver scaled values so that feasibility check is performed
        # with driver scaling.
        dv_vals = self.get_design_var_values(driver_scaling=True)
        con_vals = self.get_constraint_values(driver_scaling=True)

        for constraint, con_options in constraints.items():
            constraint_value = con_vals[constraint]
            con_size = con_options['size']

            if con_options.get('equals', None) is not None:
                # Equality constraint, all indices active
                active_cons[constraint] = {'indices': np.arange(con_size, dtype=int),
                                           'active_bounds': np.zeros(con_size, dtype=int)}
            else:
                # Inequality constraint, determine active indices and bounds
                constraint_upper = con_options.get("upper", np.inf)
                constraint_lower = con_options.get("lower", -np.inf)

                if np.all(np.isinf(constraint_upper)):
                    upper_idxs = np.empty()
                else:
                    upper_mask = np.logical_or(constraint_value > constraint_upper,
                                               np.isclose(constraint_value, constraint_upper,
                                                          atol=feas_atol, rtol=feas_rtol))
                    upper_idxs = np.where(upper_mask)[0]

                if np.all(np.isinf(constraint_lower)):
                    lower_idxs = np.empty()
                else:
                    lower_mask = np.logical_or(constraint_value < constraint_lower,
                                               np.isclose(constraint_value, constraint_lower,
                                                          atol=feas_atol, rtol=feas_rtol))
                    lower_idxs = np.where(lower_mask)[0]

                active_idxs = sorted(np.concatenate((upper_idxs, lower_idxs)))
                active_bounds = [1 if idx in upper_idxs else -1 for idx in active_idxs]
                if active_idxs:
                    active_cons[constraint] = {'indices': active_idxs,
                                               'active_bounds': active_bounds}

        for des_var, des_var_options in des_vars.items():
            des_var_value = dv_vals[des_var]
            des_var_size = des_var_options['size']
            des_var_upper = np.ravel(des_var_options.get("upper", np.inf))
            des_var_lower = np.ravel(des_var_options.get("lower", -np.inf))

            if assume_dvs_active:
                active_dvs[des_var] = {'indices': np.arange(des_var_size, dtype=int),
                                       'active_bounds': np.zeros(des_var_size, dtype=int)}
            else:
                upper_mask = np.logical_or(des_var_value > des_var_upper,
                                           np.isclose(des_var_value, des_var_upper,
                                                      atol=feas_atol, rtol=feas_rtol))
                upper_idxs = np.where(upper_mask)[0]
                lower_mask = np.logical_or(des_var_value < des_var_lower,
                                           np.isclose(des_var_value, des_var_lower,
                                                      atol=feas_atol, rtol=feas_rtol))
                lower_idxs = np.where(lower_mask)[0]

                active_idxs = sorted(np.concatenate((upper_idxs, lower_idxs)))
                active_bounds = [1 if idx in upper_idxs else -1 for idx in active_idxs]
                if active_idxs:
                    active_dvs[des_var] = {'indices': np.asarray(active_idxs, dtype=int),
                                           'active_bounds': np.asarray(active_bounds, dtype=int)}

        return active_cons, active_dvs

    def _unscale_lagrange_multipliers(self, multipliers, assume_dv=False):
        """
        Unscale the Lagrange multipliers from optimizer scaling to physical/model scaling.

        This method assumes that the optimizer is in a converged state, satisfying both the
        primal constraints as well as the optimality conditions.

        Parameters
        ----------
        active_constraints : Sequence[str]
            Active constraints/dvs in the optimization, determined using the
            get_active_cons_and_dvs method.
        multipliers : dict[str: ArrayLike]
            The Lagrange multipliers, in Driver-scaled units.
        assume_dv : bool
            This function can unscale the multipliers of either design variables or constraints.
            Since variables can be both a design variable and a constraint, this flag
            disambiguates the type of multiplier we're handling so the appropriate scaling
            factors can be used.

        Returns
        -------
        dict
            The Lagrange multipliers in model/physical units.
        """
        if len(self._objs) != 1:
            raise ValueError('Lagrange Multplier estimation requires that there '
                             f'be a single objective, but there are {len(self._objs)}.')

        obj_meta = list(self._objs.values())[0]
        obj_ref = obj_meta['ref']
        obj_ref0 = obj_meta['ref0']

        if obj_ref is None:
            obj_ref = 1.0
        if obj_ref0 is None:
            obj_ref0 = 0.0

        obj_scaler = obj_meta['total_scaler'] or 1.0

        unscaled_multipliers = {}

        for name, val in multipliers.items():
            if name in self._designvars and assume_dv:
                scaler = self._designvars[name]['total_scaler']
            else:
                scaler = self._responses[name]['total_scaler']
            scaler = scaler or 1.0

            unscaled_multipliers[name] = val * scaler / obj_scaler

        return unscaled_multipliers

    def compute_lagrange_multipliers(self, driver_scaling=False, feas_tol=1.0E-6,
                                     use_sparse_solve=True):
        """
        Get the approximated Lagrange multipliers of one or more constraints.

        This method assumes that the optimizer is in a converged state, satisfying both the
        primal constraints as well as the optimality conditions.

        The estimation of which constraints are active depends upon the feasibility tolerance
        specified. This applies to the driver-scaled values of the constraints, and should be
        the same as that used by the optimizer, if available.

        Parameters
        ----------
        driver_scaling : bool
            If False, return the Lagrange multipliers estimates in their physical units.
            If True, return the Lagrange multiplier estimates in a driver-scaled state.
        feas_tol : float or None
            The feasibility tolerance under which the optimization was run. If None, attempt
            to determine this automatically based on the specified optimizer settings.
        use_sparse_solve : bool
            If True, use scipy.sparse.linalg.lstsq to solve for the multipliers. Otherwise, numpy
            will be used with dense arrays.

        Returns
        -------
        active_desvars : dict[str: dict]
            A dictionary with an entry for each active design variable.
            For each active design variable, the corresponding dictionary
            provides the 'multipliers', active 'indices', and 'active_bounds'.
        active_cons : dict[str: dict]
            A dictionary with an entry for each active constraint.
            For each active constraint, the corresponding dictionary
            provides the 'multipliers', active 'indices', and 'active_bounds'.
        """
        if not self.supports['optimization']:
            raise NotImplementedError('Lagrange multipliers are only available for '
                                      'drivers which support optimization.')

        prob = self._problem()

        obj_name = list(self._objs.keys())[0]
        constraints = self._cons
        des_vars = self._designvars

        of_totals = {obj_name, *constraints.keys()}

        active_cons, active_dvs = self._get_active_cons_and_dvs(feas_atol=feas_tol,
                                                                feas_rtol=feas_tol)

        # Active cons and dvs provide the active indices in the design vars and constraints.
        # But these design vars and constraints may themselves be indices of a larger
        # variable.
        totals = prob.compute_totals(list(of_totals),
                                     list(des_vars),
                                     driver_scaling=True)

        grad_f = {inp: totals[obj_name, inp] for inp in des_vars.keys()}

        n = sum([grad_f_val.size for grad_f_val in grad_f.values()])

        grad_f_vec = np.zeros((n))
        offset = 0
        for grad_f_val in grad_f.values():
            inp_size = grad_f_val.size
            grad_f_vec[offset:offset + inp_size] = grad_f_val
            offset += inp_size

        active_jac_blocks = []

        if not active_cons and not active_dvs:
            return {}, {}

        for (dv_name, active_meta) in active_dvs.items():
            # For active design variable bounds, the constraint gradient
            # wrt des vars is just an identity matrix sized by the number of
            # active elements in the design variable.
            active_idxs = active_meta['indices']

            size = des_vars[dv_name]['size']
            con_grad = {(dv_name, inp): np.eye(size)[active_idxs, ...] if inp == dv_name
                        else np.zeros((size, dv_meta['size']))[active_idxs, ...]
                        for (inp, dv_meta) in des_vars.items()}

            if use_sparse_solve:
                active_jac_blocks.append([sp.csr_matrix(cg) for cg in con_grad.values()])
            else:
                active_jac_blocks.append(list(con_grad.values()))

        for (con_name, active_meta) in active_cons.items():
            # If the constraint is a design variable, the constraint gradient
            # wrt des vars is just an identity matrix sized by the number of
            # active elements in the design variable.
            active_idxs = active_meta['indices']
            if con_name in des_vars.keys():
                size = des_vars[con_name]['size']
                con_grad = {(con_name, inp): np.eye(size)[active_idxs, ...] if inp == con_name
                            else np.zeros((size, dv_meta['size']))[active_idxs, ...]
                            for (inp, dv_meta) in des_vars.items()}
            else:
                con_grad = {(con_name, inp): totals[con_name, inp][active_idxs, ...]
                            for inp in des_vars.keys()}
            if use_sparse_solve:
                active_jac_blocks.append([sp.csr_matrix(cg) for cg in con_grad.values()])
            else:
                active_jac_blocks.append(list(con_grad.values()))

        if use_sparse_solve:
            active_cons_mat = sp.block_array(active_jac_blocks)
        else:
            active_cons_mat = np.block(active_jac_blocks)

        if use_sparse_solve:
            lstsq_sol = sp.linalg.lsqr(active_cons_mat.T, -grad_f_vec)
        else:
            lstsq_sol = np.linalg.lstsq(active_cons_mat.T, -grad_f_vec, rcond=None)
        multipliers_vec = lstsq_sol[0]

        dv_multipliers = dict()
        con_multipliers = dict()
        offset = 0

        dv_vals = self.get_design_var_values()
        con_vals = self.get_constraint_values()

        for desvar, act_info in active_dvs.items():
            act_idxs = act_info['indices']
            active_size = len(act_idxs)
            mult_vals = multipliers_vec[offset:offset + active_size]
            dv_multipliers[desvar] = np.zeros_like(dv_vals[desvar])
            dv_multipliers[desvar].flat[act_idxs] = mult_vals
            offset += active_size

        for constraint, act_info in active_cons.items():
            act_idxs = act_info['indices']
            active_size = len(act_idxs)
            mult_vals = multipliers_vec[offset:offset + active_size]
            if constraint in des_vars:
                con_multipliers[constraint] = np.zeros_like(dv_vals[constraint])
            else:
                con_multipliers[constraint] = np.zeros_like(con_vals[constraint])
            con_multipliers[constraint].flat[act_idxs] = mult_vals
            offset += active_size

        if not driver_scaling:
            dv_multipliers = self._unscale_lagrange_multipliers(dv_multipliers, assume_dv=True)
            con_multipliers = self._unscale_lagrange_multipliers(con_multipliers, assume_dv=False)

        for key, val in dv_multipliers.items():
            active_dvs[key]['multipliers'] = val

        for key, val in con_multipliers.items():
            active_cons[key]['multipliers'] = val

        return active_dvs, active_cons

    def _reraise(self):
        """
        Reraise any exception encountered when scipy calls back into our methods.
        """
        exc_info = self._exc_info
        self._exc_info = None  # clear since we're done with it
        raise exc_info[1].with_traceback(exc_info[2])

    def _scipy_update_design_vars(self, x_new, desvar_names=None):
        """
        Update the design variables in the model.

        This interface is used
        by scipy minimize and least_squares.

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.
        desvar_names : Sequence[str] or None
            If given, the names of the design variables represented in x_new.
            For the Driver.find_feasible excludes argument, one or more design
            variables may be excluded from the feasibility search. If None,
            assume all design variables are present in x_new.
        """
        if desvar_names is None:
            desvar_names = self._designvars.keys()

        i = 0
        for name in desvar_names:
            meta = self._designvars[name]
            size = meta['size']
            self.set_design_var(name, x_new[i:i + size])
            i += size

    def _compute_con_viol(self, x_new, desvar_names, driver_scaling=True):
        """
        Compute the constraint violations.

        Used in minimizing the constraint violation via least squares.

        Parameters
        ----------
        x_new : array
            The design variable vector.
        desvar_names : Sequence[str]
            The names of the design variables contained in x_new. This omits
            the ones excluded in find_feasible.
        driver_scaling : bool
            If True, compute the constraint violation in driver-scaled units.

        Returns
        -------
        array
            A flat vector of constraint violations, ordered with the linear constraints first.
        """
        model = self._problem().model

        try:
            # Pass in new inputs
            if MPI and model.comm.size > 1:
                model.comm.Bcast(x_new, root=0)

            self._scipy_update_design_vars(x_new, desvar_names)

            with RecordingDebugging(self._get_name(), self.iter_count, self):
                self.iter_count += 1
                with model._relevance.nonlinear_active('iter'):
                    self._run_solve_nonlinear()

            # Sort the constraints with the linear contributions first to make it easier to
            # apply the cached linear constraint gradient.
            lin_con_viol_dict = self.get_constraint_values(lintype='linear',
                                                           driver_scaling=driver_scaling,
                                                           viol=True)

            nl_con_viol_dict = self.get_constraint_values(lintype='nonlinear',
                                                          driver_scaling=driver_scaling,
                                                          viol=True)

            return np.concatenate([v.ravel() for v in
                                   list(lin_con_viol_dict.values()) +
                                   list(nl_con_viol_dict.values())])

        except Exception:
            if self._exc_info is None:  # only record the first one
                self._exc_info = sys.exc_info()
            return np.zeros(np.sum([c['size'] for c in self._cons.values()]))

    def _compute_con_viol_grad(self, x_new, desvar_names, con_row_map,
                               driver_scaling=True, lin_con_grad=None):
        """
        Compute the jacobian of the constraint violations wrt the design variables.

        Parameters
        ----------
        x_new : array
            The design variable vector.
        desvar_names : Sequence[str]
            The names of the design variables not excluded in find_feasible.
        con_row_map : dict[str: slice]
            A dict which maps a constraint name to its corresponding rows in
            the jacobian matrix.
        driver_scaling : bool
            If True, assume driver-scaling when computing the gradients,
            otherwise assume model scaling.
        lin_con_grad : array or None
            The cached value of the linear portion of the constraint gradient.

        Returns
        -------
        jac : array-like
            A 2D array of the sensitivities of the constraints wrt the design variables.
        """
        nlcons = [name for name, meta in self._cons.items() if not meta.get('linear')]

        # only need the gradient of the active constraints
        active_cons, _ = self._get_active_cons_and_dvs(feas_atol=1.0E-8, feas_rtol=1.0E-8)

        if nlcons:
            nl_con_grad = self._compute_totals(of=nlcons, wrt=desvar_names,
                                               return_format='array',
                                               driver_scaling=driver_scaling)
        else:
            nl_con_grad = np.empty((0, x_new.size))

        g = np.vstack((lin_con_grad, nl_con_grad))

        # Inactive constraints contribute nothing to the gradient.
        for con_name, idxs in con_row_map.items():
            if con_name not in active_cons:
                g[idxs] = 0.0

        return g

    def _find_feasible(self, driver_scaling=True, exclude_desvars=None,
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
            Failure flag; True if the infeasibility minimization failed to converge.
        """
        from scipy.optimize import Bounds, least_squares
        from scipy.optimize._constraints import old_bound_to_new

        self._in_find_feasible = True

        problem = self._problem()
        model = problem.model

        self._check_for_invalid_desvar_values()

        exclude_desvars = [exclude_desvars] if isinstance(exclude_desvars, str) \
            else exclude_desvars or []

        status = -1 if problem is None else problem._metadata['setup_status']
        if status < _SetupStatus.POST_FINAL_SETUP:
            problem.final_setup()

        desvar_vals = {dv: val for dv, val in self.get_design_var_values().items()
                       if not any(fnmatchcase(dv, pat) for pat in exclude_desvars)}

        # Size Problem
        ndesvar = 0
        for name in desvar_vals.keys():
            meta = self._designvars[name]
            size = meta['global_size'] if meta['distributed'] else meta['size']
            ndesvar += size
        x_init = np.empty(ndesvar)

        if ndesvar == 0:
            raise RuntimeError('Problem has no design variables or '
                               'all design variables are excluded.')

        i = 0
        for name, val in desvar_vals.items():
            meta = self._designvars[name]
            size = meta['global_size'] if meta['distributed'] else meta['size']
            x_init[i:i + size] = val
            i += size

        # Initial Design Vars bounds
        if method == 'lm':
            bounds = (-np.inf, np.inf)

            if any(meta['lower'] > -1.0E16 or
                   meta['upper'] < 1.0E16 for meta in self._designvars.values()):
                issue_warning("find_feasible method is 'lm' which ignores bounds "
                              "but one or more design variables have bounds.")
        else:
            i = 0
            bounds = []

            for name, val in desvar_vals.items():
                meta = self._designvars[name]
                size = meta['global_size'] if meta['distributed'] else meta['size']

                meta_low = meta['lower']
                meta_high = meta['upper']
                for j in range(size):

                    if isinstance(meta_low, np.ndarray):
                        p_low = meta_low[j]
                    else:
                        p_low = meta_low

                    if isinstance(meta_high, np.ndarray):
                        p_high = meta_high[j]
                    else:
                        p_high = meta_high

                    p_low = -np.inf if p_low < -1.0E16 else p_low
                    p_high = np.inf if p_high > 1.0E16 else p_high

                    # If lower and upper are equal at any indices, add some slack
                    equal_idxs = np.where(np.atleast_1d(np.abs(p_high - p_low)) < 1.0E-16)[0]

                    # Releive bounds if they are pinched
                    # TODO: Handle this more generically in all drivers
                    if np.isscalar(p_high):
                        p_high += 1.0E-16
                    else:
                        p_high[equal_idxs] += 1.0E-16

                    bounds.append((p_low, p_high))

            # Convert "old-style" bounds to "new_style" bounds
            lower, upper = old_bound_to_new(bounds)  # tuple, tuple
            bounds = Bounds(lb=lower, ub=upper, keep_feasible=[True] * x_init.size)

        lincons = {name: meta for name, meta in self._cons.items() if meta.get('linear')}
        nl_cons = {name: meta for name, meta in self._cons.items() if not meta.get('linear')}

        # Save the rows in the constraint vector that apply to each constrained output
        con_row_map = {}
        i = 0
        for name, meta in chain(lincons.items(), nl_cons.items()):
            size = meta['global_size'] if meta['distributed'] else meta['size']
            con_row_map[name] = slice(i, i + size)
            i += size

        # Compute and save the gradient of the linear constraints
        if lincons:
            lincongrad_cache = self._compute_totals(of=list(lincons.keys()),
                                                    wrt=desvar_vals.keys(),
                                                    driver_scaling=driver_scaling,
                                                    return_format='array')
        else:
            lincongrad_cache = np.empty((0, x_init.size))

        # Provide the jac with cached linear grad and mapping of constraint names to rows.
        jacfun = functools.partial(self._compute_con_viol_grad, desvar_names=desvar_vals.keys(),
                                   driver_scaling=driver_scaling, lin_con_grad=lincongrad_cache,
                                   con_row_map=con_row_map)

        # Wrap the actual least squares call so that we don't need to duplicate calls below'
        if MPI and problem.comm.rank != 0:
            iprint = 0

        f_lsq = functools.partial(least_squares, self._compute_con_viol,
                                  kwargs={'driver_scaling': driver_scaling,
                                          'desvar_names': list(desvar_vals.keys())},
                                  x0=x_init, bounds=bounds, verbose=2 if iprint == 2 else 0,
                                  method=method, ftol=ftol, xtol=xtol, gtol=gtol,
                                  x_scale=x_scale, loss=loss, max_nfev=max_nfev,
                                  f_scale=f_scale, tr_solver=tr_solver,
                                  tr_options=tr_options or {},
                                  jac=jacfun)

        if self._exc_info is not None:
            self._reraise()

        if iprint == 2:
            print()
            print('-------------------------')
            print('Finding feasible point...')

        self.result.reset()
        if problem.options['group_by_pre_opt_post']:
            if model._pre_components:
                with model._relevance.nonlinear_active('pre'):
                    self._run_solve_nonlinear()

            with SaveOptResult(self):
                with model._relevance.nonlinear_active('iter'):
                    res = f_lsq()
                    self.result.success = res.success and res.cost <= loss_tol

            if model._post_components:
                with model._relevance.nonlinear_active('post'):
                    self._run_solve_nonlinear()

        else:
            with SaveOptResult(self):
                res = f_lsq()
                self.result.success = res.success and res.cost <= loss_tol

        if iprint >= 1:
            if res.success:
                if res.cost <= loss_tol:
                    print('--------------------')
                    print('Feasible point found')
                    print('--------------------')
                else:
                    print('-------------------------')
                    print('Infeasibilities minimized')
                    print('-------------------------')
            else:
                print('----------------------------------')
                print('Failed to minimize infeasibilities')
                print('----------------------------------')

            print(f'    loss({loss}): {res.cost:.8f}')
            print(f'    iterations: {self.result.iter_count}')
            print(f'    model evals: {res.nfev}')
            print(f'    gradient evals: {res.njev}')
            print(f'    elapsed time: {self.result.model_time + self.result.deriv_time:.8f} s')
            if not res.success or res.cost >= loss_tol:
                max_idx = np.argmax(np.abs(res.fun))
                max_viol = res.fun[max_idx]
                for con_name, sl in con_row_map.items():
                    if sl.start <= max_idx < sl.stop:
                        max_viol_con = con_name
                        max_viol_idx_in_con = max_idx - sl.start
                        break
                else:
                    max_viol_con = con_name
                    max_viol_idx_in_con = -1
                max_viol_str = f'{max_viol_con}[{max_viol_idx_in_con}] = {max_viol:.8f}'
            else:
                max_viol_str = 'N/A'

            print(f'    max violation: {max_viol_str}')
            if not res.success:
                print(f'    message: {res.message}')
            print()

        self.result.exit_status = res.message
        self._in_find_feasible = False

        return not self.result.success


class SaveOptResult(object):
    """
    A context manager that saves details about a driver run.

    Parameters
    ----------
    driver : Driver
        The driver.

    Attributes
    ----------
    _driver : Driver
        The driver for which we are saving results.
    _start_time : float
        The start time used to compute the run time.
    """

    def __init__(self, driver):
        """
        Initialize attributes.
        """
        self._driver = driver

    def __enter__(self):
        """
        Set start time for the driver run.

        This uses 'perf_counter()' which gives "the value (in fractional seconds)
        of a performance counter, i.e. a clock with the highest available resolution
        to measure a short duration. It does include time elapsed during sleep and
        is system-wide."

        Returns
        -------
        self : object
            self
        """
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        """
        Save driver run information in the 'opt_result' attribute.

        Parameters
        ----------
        *args : array
            Solver recording requires extra args.
        """
        driver = self._driver

        # The standard driver results
        driver.result.runtime = time.perf_counter() - self._start_time
        driver.result.iter_count = driver.iter_count
        driver.result.exit_status = driver.get_exit_status()

        # The custom driver results
        driver._update_result(driver.result)


class RecordingDebugging(Recording):
    """
    A class that acts as a context manager.

    Handles doing the case recording and also the Driver
    debugging printing.

    Parameters
    ----------
    name : str
        Name of object getting recorded.
    iter_count : int
        Current counter of iterations completed.
    recording_requester : object
        Object to which this recorder is attached.
    """

    def __enter__(self):
        """
        Do things before the code inside the 'with RecordingDebugging' block.

        Returns
        -------
        self : object
            self
        """
        super().__enter__()
        self.recording_requester()._pre_run_model_debug_print()
        return self

    def __exit__(self, *args):
        """
        Do things after the code inside the 'with RecordingDebugging' block.

        Parameters
        ----------
        *args : array
            Solver recording requires extra args.
        """
        self.recording_requester()._post_run_model_debug_print()
        super().__exit__()


def record_iteration(requester, prob, case_name):
    """
    Record an iteration of the current Problem or Driver.

    Parameters
    ----------
    requester : Problem or Driver
        The recording requester.
    prob : Problem
        The Problem.
    case_name : str
        The name of this case.
    """
    rec_mgr = requester._rec_mgr
    if not rec_mgr._recorders:
        return

    # Get the data to record (collective calls that get across all ranks)
    model = prob.model
    parallel = rec_mgr._check_parallel() if model.comm.size > 1 else False
    do_gather = rec_mgr._check_gather()
    local = parallel and not do_gather

    inputs, outputs, residuals = model.get_nonlinear_vectors()
    discrete_inputs = model._discrete_inputs
    discrete_outputs = model._discrete_outputs

    opts = requester.recording_options
    data = {'input': {}, 'output': {}, 'residual': {}}
    filt = requester._filtered_vars_to_record
    if filt is None:  # recorder is not initialized
        # this will raise the proper exception
        rec_mgr.record_iteration(requester, data, requester._get_recorder_metadata(case_name))
        return

    if opts['record_inputs'] and (inputs._names or len(discrete_inputs) > 0):
        data['input'] = model._retrieve_data_of_kind(filt, 'input', 'nonlinear', local)

    if opts['record_outputs'] and (outputs._names or len(discrete_outputs) > 0):
        data['output'] = model._retrieve_data_of_kind(filt, 'output', 'nonlinear', local)

    if opts['record_residuals'] and residuals._names:
        data['residual'] = model._retrieve_data_of_kind(filt, 'residual', 'nonlinear', local)

    from openmdao.core.problem import Problem
    if isinstance(requester, Problem):
        # Record total derivatives
        if opts['record_derivatives'] and prob.driver._designvars and prob.driver._responses:
            data['totals'] = requester.compute_totals(return_format='flat_dict_structured_key')

        # Record solver info
        if opts['record_abs_error'] or opts['record_rel_error']:
            norm = residuals.get_norm()
        if opts['record_abs_error']:
            data['abs'] = norm
        if opts['record_rel_error']:
            solver = model.nonlinear_solver
            norm0 = solver._norm0 if solver._norm0 != 0.0 else 1.0  # runonce never sets _norm0
            data['rel'] = norm / norm0

    rec_mgr.record_iteration(requester, data, requester._get_recorder_metadata(case_name))


def filter_by_meta(metadict_items, key, chk_none=False, exclude=False):
    """
    Filter metadata items based on their value.

    Parameters
    ----------
    metadict_items : iter of (name, meta)
        Iterable of (name, meta) tuples.
    key : str
        Metadata key.
    chk_none : bool
        If True, compare items to None. If False, check if items are truthy.
    exclude : bool
        If True, exclude matching items rather than yielding them.

    Yields
    ------
    tuple
        Tuple of the form (name, meta) for each item in metadict_items that satisfies the condition.
    """
    if chk_none:
        for tup in metadict_items:
            none = tup[1][key] is None
            if exclude:
                if none:
                    yield tup
            elif not none:
                yield tup
    else:
        for tup in metadict_items:
            if exclude:
                if not tup[1][key]:
                    yield tup
            elif tup[1][key]:
                yield tup
