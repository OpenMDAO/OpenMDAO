"""Define a base class for all Drivers in OpenMDAO."""
from collections import OrderedDict
import pprint
import sys
import os
import weakref

import numpy as np

from openmdao.core.total_jac import _TotalJacInfo
from openmdao.core.constants import INT_DTYPE
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.record_util import create_local_meta, check_path
from openmdao.utils.general_utils import simple_warning, warn_deprecation, _prom2ivc_src_dict, \
    _prom2ivc_src_name_iter
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.array_utils import sizes2offsets, convert_neg
from openmdao.vectors.vector import _full_slice


def _check_debug_print_opts_valid(name, opts):
    """
    Check validity of debug_print option for Driver.

    Parameters
    ----------
    name : str
        The name of the option.
    opts : list
        The value of the debug_print option set by the user.
    """
    if not isinstance(opts, list):
        raise ValueError("Option '%s' with value %s is not a list." % (name, opts))

    _valid_opts = ['desvars', 'nl_cons', 'ln_cons', 'objs', 'totals']
    for opt in opts:
        if opt not in _valid_opts:
            raise ValueError("Option '%s' contains value '%s' which is not one of %s." %
                             (name, opt, _valid_opts))


class Driver(object):
    """
    Top-level container for the systems and drivers.

    Attributes
    ----------
    fail : bool
        Reports whether the driver ran successfully.
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
        Dict of constraints that are distributed outputs. Key is abs variable name, values are
        (local indices, local sizes).
    _cons : dict
        Contains all constraint info.
    _objs : dict
        Contains all objective info.
    _responses : dict
        Contains all response info.
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
    _total_jac_sparsity : dict, str, or None
        Specifies sparsity of sub-jacobians of the total jacobian. Only used by pyOptSparseDriver.
    _res_jacs : dict
        Dict of sparse subjacobians for use with certain optimizers, e.g. pyOptSparseDriver.
    _total_jac : _TotalJacInfo or None
        Cached total jacobian handling object.
    """

    def __init__(self, **kwargs):
        """
        Initialize the driver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        self._rec_mgr = RecordingManager()

        self._problem = None
        self._designvars = None
        self._designvars_discrete = []
        self._cons = None
        self._objs = None
        self._responses = None

        # Driver options
        self.options = OptionsDictionary(parent_name=type(self).__name__)

        self.options.declare('debug_print', types=list, check_valid=_check_debug_print_opts_valid,
                             desc="List of what type of Driver variables to print at each "
                                  "iteration. Valid items in list are 'desvars', 'ln_cons', "
                                  "'nl_cons', 'objs', 'totals'",
                             default=[])

        # Case recording options
        self.recording_options = OptionsDictionary(parent_name=type(self).__name__)

        self.recording_options.declare('record_model_metadata', types=bool, default=True,
                                       desc='Deprecated. Recording of model metadata will always '
                                       'be done',
                                       deprecation="The recording option, record_model_metadata, "
                                       "on Driver is "
                                       "deprecated. Recording of model metadata will always "
                                       "be done",
                                       )
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
        self.supports.declare('inequality_constraints', types=bool, default=False)
        self.supports.declare('equality_constraints', types=bool, default=False)
        self.supports.declare('linear_constraints', types=bool, default=False)
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

        self._coloring_info = coloring_mod._get_coloring_meta()

        self._total_jac_sparsity = None
        self._res_jacs = {}
        self._total_jac = None

        self.fail = False

        self._declare_options()
        self.options.update(kwargs)

    def _get_inst_id(self):
        if self._problem is None:
            return None
        probid = self._problem()._get_inst_id()
        if probid is None:
            return "driver"
        return f"{probid}.driver"

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

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        self._problem = weakref.ref(problem)
        model = problem.model

        self._total_jac = None

        self._has_scaling = (
            np.any([r['total_scaler'] is not None for r in self._responses.values()]) or
            np.any([dv['total_scaler'] is not None for dv in self._designvars.values()])
        )

        # Determine if any design variables are discrete.
        self._designvars_discrete = [name for name, meta in self._designvars.items()
                                     if meta['ivc_source'] in model._discrete_outputs]
        if not self.supports['integer_design_vars'] and len(self._designvars_discrete) > 0:
            msg = "Discrete design variables are not supported by this driver: "
            msg += '.'.join(self._designvars_discrete)
            raise RuntimeError(msg)

        con_set = set()
        obj_set = set()
        dv_set = set()

        self._remote_dvs = remote_dv_dict = {}
        self._remote_cons = remote_con_dict = {}
        self._dist_driver_vars = dist_dict = {}
        self._remote_objs = remote_obj_dict = {}

        # Only allow distributed design variables on drivers that support it.
        if self.supports['distributed_design_vars'] is False:
            dist_vars = []
            abs2meta_in = model._var_allprocs_abs2meta['input']
            discrete_in = model._var_allprocs_discrete['input']
            for dv, meta in self._designvars.items():

                # For Auto-ivcs, we need to check the distributed metadata on the target instead.
                if meta['ivc_source'].startswith('_auto_ivc.'):
                    for abs_name in model._var_allprocs_prom2abs_list['input'][dv]:
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
        if model.comm.size > 1 and model._subsystems_allprocs:
            src_design_vars = _prom2ivc_src_dict(self._designvars)
            responses = _prom2ivc_src_dict(self._responses)

            local_out_vars = set(model._outputs._abs_iter())
            remote_dvs = set(src_design_vars) - local_out_vars
            remote_cons = set(_prom2ivc_src_name_iter(self._cons)) - local_out_vars
            remote_objs = set(_prom2ivc_src_name_iter(self._objs)) - local_out_vars

            all_remote_vois = model.comm.allgather(
                (remote_dvs, remote_cons, remote_objs))
            for rem_dvs, rem_cons, rem_objs in all_remote_vois:
                con_set.update(rem_cons)
                obj_set.update(rem_objs)
                dv_set.update(rem_dvs)

            # If we have remote VOIs, pick an owning rank for each and use that
            # to bcast to others later
            owning_ranks = model._owning_rank
            sizes = model._var_sizes['nonlinear']['output']
            rank = model.comm.rank
            nprocs = model.comm.size
            for i, (vname, meta) in enumerate(model._var_allprocs_abs2meta['output'].items()):
                if vname in responses:
                    indices = responses[vname].get('indices')
                elif vname in src_design_vars:
                    indices = src_design_vars[vname].get('indices')
                else:
                    continue

                if meta['distributed']:

                    idx = model._var_allprocs_abs2idx['nonlinear'][vname]
                    dist_sizes = model._var_sizes['nonlinear']['output'][:, idx]
                    total_dist_size = np.sum(dist_sizes)

                    # Determine which indices are on our proc.
                    offsets = sizes2offsets(dist_sizes)

                    if indices is not None:
                        indices = convert_neg(indices, total_dist_size)
                        true_sizes = np.zeros(nprocs, dtype=INT_DTYPE)
                        for irank in range(nprocs):
                            dist_inds = indices[np.logical_and(indices >= offsets[irank],
                                                               indices < (offsets[irank] +
                                                                          dist_sizes[irank]))]
                            if irank == rank:
                                local_indices = dist_inds - offsets[rank]
                                distrib_indices = dist_inds

                            true_sizes[irank] = dist_inds.size
                        dist_dict[vname] = (local_indices, true_sizes, distrib_indices)
                    else:
                        dist_dict[vname] = (_full_slice, dist_sizes,
                                            slice(offsets[rank], offsets[rank] + dist_sizes[rank]))

                else:
                    owner = owning_ranks[vname]
                    sz = sizes[owner, i]

                    if vname in dv_set:
                        remote_dv_dict[vname] = (owner, sz)
                    if vname in con_set:
                        remote_con_dict[vname] = (owner, sz)
                    if vname in obj_set:
                        remote_obj_dict[vname] = (owner, sz)

        self._remote_responses = self._remote_cons.copy()
        self._remote_responses.update(self._remote_objs)

        # set up simultaneous deriv coloring
        if coloring_mod._use_total_sparsity:
            # reset the coloring
            if self._coloring_info['dynamic'] or self._coloring_info['static'] is not None:
                self._coloring_info['coloring'] = None

            coloring = self._get_static_coloring()
            if coloring is not None and self.supports['simultaneous_derivatives']:
                if model._owns_approx_jac:
                    coloring._check_config_partial(model)
                else:
                    coloring._check_config_total(self)
                self._setup_simul_coloring()

    def _check_for_missing_objective(self):
        """
        Check for missing objective and raise error if no objectives found.
        """
        if len(self._objs) == 0:
            msg = "Driver requires objective to be declared"
            raise RuntimeError(msg)

    def _get_vars_to_record(self, recording_options):
        """
        Get variables to record based on recording options.

        Parameters
        ----------
        recording_options : <OptionsDictionary>
            Dictionary with recording options.

        Returns
        -------
        dict
           Dictionary containing lists of variables to record.
        """
        problem = self._problem()
        model = problem.model

        incl = recording_options['includes']
        excl = recording_options['excludes']

        # includes and excludes for outputs are specified using promoted names
        abs2prom = model._var_allprocs_abs2prom['output']

        # 1. If record_outputs is True, get the set of outputs
        # 2. Filter those using includes and excludes to get the baseline set of variables to record
        # 3. Add or remove from that set any desvars, objs, and cons based on the recording
        #    options of those

        # includes and excludes for outputs are specified using _promoted_ names
        # vectors are keyed on absolute name, discretes on relative/promoted name
        myinputs = myoutputs = myresiduals = []

        if recording_options['record_outputs']:
            myoutputs = sorted([n for n, prom in abs2prom.items() if check_path(prom, incl, excl)])

            model_outs = model._outputs

            if model._var_discrete['output']:
                # if we have discrete outputs then residual name set doesn't match output one
                if recording_options['record_residuals']:
                    myresiduals = [n for n in myoutputs if model_outs._contains_abs(n)]
            elif recording_options['record_residuals']:
                myresiduals = myoutputs

        elif recording_options['record_residuals']:
            myresiduals = [n for n in model._residuals._abs_iter()
                           if check_path(abs2prom[n], incl, excl)]

        myoutputs = set(myoutputs)
        if recording_options['record_desvars']:
            myoutputs.update(self._designvars)
        if recording_options['record_objectives'] or recording_options['record_responses']:
            myoutputs.update(self._objs)
        if recording_options['record_constraints'] or recording_options['record_responses']:
            myoutputs.update(self._cons)

        # inputs (if in options). inputs use _absolute_ names for includes/excludes
        if 'record_inputs' in recording_options:
            if recording_options['record_inputs']:
                # sort the results since _var_allprocs_abs2prom isn't ordered
                myinputs = sorted([n for n in model._var_allprocs_abs2prom['input']
                                  if check_path(n, incl, excl)])

        vars2record = {
            'input': myinputs,
            'output': list(myoutputs),
            'residual': myresiduals
        }

        return vars2record

    def _setup_recording(self):
        """
        Set up case recording.
        """
        self._filtered_vars_to_record = self._get_vars_to_record(self.recording_options)

        self._rec_mgr.startup(self)

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
        distributed_vars = self._dist_driver_vars
        indices = meta['indices']

        if meta.get('ivc_source') is not None:
            src_name = meta['ivc_source']
        else:
            src_name = name

        if MPI:
            distributed = comm.size > 0 and src_name in distributed_vars
        else:
            distributed = False

        if src_name in remote_vois:
            owner, size = remote_vois[src_name]
            # if var is distributed or only gathering to one rank
            # TODO - support distributed var under a parallel group.
            if owner is None or rank is not None:
                val = model.get_val(src_name, get_remote=get_remote, rank=rank, flat=True)
                if indices is not None:
                    val = val[indices]
            else:
                if owner == comm.rank:
                    if indices is None:
                        val = get(name).copy()
                    else:
                        val = get(name)[indices]
                else:
                    if indices is not None:
                        size = len(indices)
                    val = np.empty(size)

                if get_remote:
                    comm.Bcast(val, root=owner)

        elif distributed:
            local_val = model.get_val(src_name, get_remote=False, flat=True)
            local_indices, sizes, _ = distributed_vars[src_name]
            if local_indices is not None:
                local_val = local_val[local_indices]

            if get_remote:
                if not local_val.flags['C_CONTIGUOUS']:
                    local_val = np.ascontiguousarray(local_val)
                offsets = np.zeros(sizes.size, dtype=INT_DTYPE)
                offsets[1:] = np.cumsum(sizes[:-1])
                val = np.zeros(np.sum(sizes))
                comm.Allgatherv(local_val, [val, sizes, offsets, MPI.DOUBLE])
            else:
                val = local_val

        else:
            if name in self._designvars_discrete:
                val = model._discrete_outputs[src_name]

                # At present, only integers are supported by OpenMDAO drivers.
                # We check the values here.
                msg = "Only integer scalars or ndarrays are supported as values for " + \
                      "discrete variables when used as a design variable. "
                if np.isscalar(val) and not isinstance(val, (int, np.integer)):
                    msg += "A value of type '{}' was specified.".format(val.__class__.__name__)
                    raise ValueError(msg)
                elif isinstance(val, np.ndarray) and not np.issubdtype(val[0], np.integer):
                    msg += "An array of type '{}' was specified.".format(val[0].__class__.__name__)
                    raise ValueError(msg)

            elif indices is None:
                val = get(src_name).copy()
            else:
                val = get(src_name)[indices]

        if self._has_scaling and driver_scaling:
            # Scale design variable values
            adder = meta['total_adder']
            if adder is not None:
                val += adder

            scaler = meta['total_scaler']
            if scaler is not None:
                val *= scaler

        return val

    def get_design_var_values(self, get_remote=True):
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

        Returns
        -------
        dict
           Dictionary containing values of each design variable.
        """
        return {n: self._get_voi_val(n, dv, self._remote_dvs, get_remote=get_remote)
                for n, dv in self._designvars.items()}

    def set_design_var(self, name, value, set_remote=True):
        """
        Set the value of a design variable.

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

        src_name = meta['ivc_source']

        # if the value is not local, don't set the value
        if (src_name in self._remote_dvs and
                problem.model._owning_rank[src_name] != problem.comm.rank):
            return

        indices = meta['indices']
        if indices is None:
            indices = _full_slice

        if name in self._designvars_discrete:

            # Note, drivers set values here and generally should know it is setting an integer.
            # However, the DOEdriver may pull a non-integer value from its generator, so we
            # convert it.
            if isinstance(value, float):
                value = int(value)
            elif isinstance(value, np.ndarray):
                if isinstance(problem.model._discrete_outputs[src_name], int):
                    # Setting an integer value with a 1D array - don't want to convert to array.
                    value = int(value)
                else:
                    value = value.astype(np.int)

            problem.model._discrete_outputs[src_name] = value

        elif problem.model._outputs._contains_abs(src_name):
            desvar = problem.model._outputs._abs_get_val(src_name)
            if src_name in self._dist_driver_vars:
                loc_idxs, _, dist_idxs = self._dist_driver_vars[src_name]
            else:
                loc_idxs = indices
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

    def get_constraint_values(self, ctype='all', lintype='all', driver_scaling=True):
        """
        Return constraint values.

        Parameters
        ----------
        ctype : string
            Default is 'all'. Optionally return just the inequality constraints
            with 'ineq' or the equality constraints with 'eq'.
        lintype : string
            Default is 'all'. Optionally return just the linear constraints
            with 'linear' or the nonlinear constraints with 'nonlinear'.
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.

        Returns
        -------
        dict
           Dictionary containing values of each constraint.
        """
        con_dict = {}
        for name, meta in self._cons.items():
            if lintype == 'linear' and not meta['linear']:
                continue

            if lintype == 'nonlinear' and meta['linear']:
                continue

            if ctype == 'eq' and meta['equals'] is None:
                continue

            if ctype == 'ineq' and meta['equals'] is not None:
                continue

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
        order.extend(n for n, meta in self._cons.items()
                     if not ('linear' in meta and meta['linear']))
        return order

    def _update_voi_meta(self, model):
        """
        Collect response and design var metadata from the model and size desvars and responses.

        Parameters
        ----------
        model : System
            The System that represents the entire model.

        Returns
        -------
        int
            Total size of responses, with linear constraints excluded.
        int
            Total size of design vars.
        """
        self._objs = objs = OrderedDict()
        self._cons = cons = OrderedDict()

        model._setup_driver_units()

        self._responses = resps = model.get_responses(recurse=True, use_prom_ivc=True)
        for name, data in resps.items():
            if data['type'] == 'con':
                cons[name] = data
            else:
                objs[name] = data

        response_size = sum(resps[n]['global_size'] for n in self._get_ordered_nl_responses())

        # Gather up the information for design vars.
        self._designvars = designvars = model.get_design_vars(recurse=True, use_prom_ivc=True)
        desvar_size = sum(data['global_size'] for data in designvars.values())

        return response_size, desvar_size

    def run(self):
        """
        Execute this driver.

        The base `Driver` just runs the model. All other drivers overload
        this method.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        with RecordingDebugging(self._get_name(), self.iter_count, self):
            self._problem().model.run_solve_nonlinear()

        self.iter_count += 1
        return False

    @property
    def _recording_iter(self):
        return self._problem()._metadata['recording_iter']

    def _compute_totals(self, of=None, wrt=None, return_format='flat_dict', global_names=None,
                        use_abs_names=True):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        All derivatives are returned using driver scaling.

        Parameters
        ----------
        of : list of variable name strings or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : string
            Format to return the derivatives. Default is a 'flat_dict', which
            returns them in a dictionary whose keys are tuples of form (of, wrt). For
            the scipy optimizer, 'array' is also supported.
        global_names : bool
            Deprecated.  Use 'use_abs_names' instead.
        use_abs_names : bool
            Set to True when passing in absolute names to skip some translation steps.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        problem = self._problem()
        total_jac = self._total_jac
        debug_print = 'totals' in self.options['debug_print'] and (not MPI or
                                                                   problem.comm.rank == 0)

        if debug_print:
            header = 'Driver total derivatives for iteration: ' + str(self.iter_count)
            print(header)
            print(len(header) * '-' + '\n')

        if global_names is not None:
            warn_deprecation("'global_names' is deprecated in calls to _compute_totals. "
                             "Use 'use_abs_names' instead.")
            use_abs_names = global_names

        if problem.model._owns_approx_jac:
            self._recording_iter.push(('_compute_totals_approx', 0))

            try:
                if total_jac is None:
                    total_jac = _TotalJacInfo(problem, of, wrt, use_abs_names,
                                              return_format, approx=True, debug_print=debug_print)

                    # Don't cache linear constraint jacobian
                    if not total_jac.has_lin_cons:
                        self._total_jac = total_jac

                    totals = total_jac.compute_totals_approx(initialize=True)
                else:
                    totals = total_jac.compute_totals_approx()
            finally:
                self._recording_iter.pop()

        else:
            if total_jac is None:
                total_jac = _TotalJacInfo(problem, of, wrt, use_abs_names, return_format,
                                          debug_print=debug_print)

                # don't cache linear constraint jacobian
                if not total_jac.has_lin_cons:
                    self._total_jac = total_jac

            self._recording_iter.push(('_compute_totals', 0))

            try:
                totals = total_jac.compute_totals()
            finally:
                self._recording_iter.pop()

        if self._rec_mgr._recorders and self.recording_options['record_derivatives']:
            metadata = create_local_meta(self._get_name())
            total_jac.record_derivatives(self, metadata)

        return totals

    def record_iteration(self):
        """
        Record an iteration of the current Driver.
        """
        record_iteration(self, self._problem(), self._get_name())

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
                         show_sparsity=coloring_mod._DEF_COMP_SPARSITY_ARGS['show_sparsity']):
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
        """
        self._coloring_info['num_full_jacs'] = num_full_jacs
        self._coloring_info['tol'] = tol
        self._coloring_info['orders'] = orders
        self._coloring_info['perturb_size'] = perturb_size
        self._coloring_info['min_improve_pct'] = min_improve_pct
        if self._coloring_info['static'] is None:
            self._coloring_info['dynamic'] = True
        else:
            self._coloring_info['dynamic'] = False
        self._coloring_info['coloring'] = None
        self._coloring_info['show_summary'] = show_summary
        self._coloring_info['show_sparsity'] = show_sparsity

    def use_fixed_coloring(self, coloring=coloring_mod._STD_COLORING_FNAME):
        """
        Tell the driver to use a precomputed coloring.

        Parameters
        ----------
        coloring : str
            A coloring filename.  If no arg is passed, filename will be determined
            automatically.

        """
        if self.supports['simultaneous_derivatives']:
            if coloring_mod._force_dyn_coloring and coloring is coloring_mod._STD_COLORING_FNAME:
                # force the generation of a dynamic coloring this time
                self._coloring_info['dynamic'] = True
                self._coloring_info['static'] = None
            else:
                self._coloring_info['static'] = coloring
                self._coloring_info['dynamic'] = False

            self._coloring_info['coloring'] = None
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
        info = self._coloring_info
        static = info['static']

        if isinstance(static, coloring_mod.Coloring):
            coloring = static
            info['coloring'] = coloring
        else:
            coloring = info['coloring']

        if coloring is not None:
            return coloring

        if static is coloring_mod._STD_COLORING_FNAME or isinstance(static, str):
            if static is coloring_mod._STD_COLORING_FNAME:
                fname = self._get_total_coloring_fname()
            else:
                fname = static
            print("loading total coloring from file %s" % fname)
            coloring = info['coloring'] = coloring_mod.Coloring.load(fname)
            info.update(coloring._meta)
            return coloring

    def _get_total_coloring_fname(self):
        return os.path.join(self._problem().options['coloring_dir'], 'total_coloring.pkl')

    def _setup_simul_coloring(self):
        """
        Set up metadata for coloring of total derivative solution.

        If set_coloring was called with a filename, load the coloring file.
        """
        # command line simul_coloring uses this env var to turn pre-existing coloring off
        if not coloring_mod._use_total_sparsity:
            return

        problem = self._problem()
        if not problem.model._use_derivatives:
            simple_warning("Derivatives are turned off.  Skipping simul deriv coloring.")
            return

        total_coloring = self._get_static_coloring()

        if total_coloring._rev and problem._orig_mode not in ('rev', 'auto'):
            revcol = total_coloring._rev[0][0]
            if revcol:
                raise RuntimeError("Simultaneous coloring does reverse solves but mode has "
                                   "been set to '%s'" % problem._orig_mode)
        if total_coloring._fwd and problem._orig_mode not in ('fwd', 'auto'):
            fwdcol = total_coloring._fwd[0][0]
            if fwdcol:
                raise RuntimeError("Simultaneous coloring does forward solves but mode has "
                                   "been set to '%s'" % problem._orig_mode)

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


class RecordingDebugging(Recording):
    """
    A class that acts as a context manager.

    Handles doing the case recording and also the Driver
    debugging printing.
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

    inputs, outputs, residuals = model.get_nonlinear_vectors()
    discrete_inputs = model._discrete_inputs
    discrete_outputs = model._discrete_outputs

    opts = requester.recording_options
    data = {'input': {}, 'output': {}, 'residual': {}}
    filt = requester._filtered_vars_to_record

    if opts['record_inputs'] and (inputs._names or len(discrete_inputs) > 0):
        data['input'] = model._retrieve_data_of_kind(filt, 'input', 'nonlinear', parallel)

    if opts['record_outputs'] and (outputs._names or len(discrete_outputs) > 0):
        data['output'] = model._retrieve_data_of_kind(filt, 'output', 'nonlinear', parallel)

    if opts['record_residuals'] and residuals._names:
        data['residual'] = model._retrieve_data_of_kind(filt, 'residual', 'nonlinear', parallel)

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
