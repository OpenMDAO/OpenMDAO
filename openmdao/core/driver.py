"""Define a base class for all Drivers in OpenMDAO."""
from __future__ import print_function

from collections import OrderedDict
import pprint
import sys
import os

from six import iteritems, itervalues, string_types

import numpy as np

from openmdao.core.total_jac import _TotalJacInfo
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.record_util import create_local_meta, check_path
from openmdao.utils.general_utils import simple_warning, warn_deprecation
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
import openmdao.utils.coloring as coloring_mod


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
    _problem : <Problem>
        Pointer to the containing problem.
    supports : <OptionsDictionary>
        Provides a consistent way for drivers to declare what features they support.
    _designvars : dict
        Contains all design variable info.
    _designvars_discrete : list
        List of design variables that are discrete.
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
                                       desc='Record metadata for all Systems in the model')
        self.recording_options.declare('record_desvars', types=bool, default=True,
                                       desc='Set to True to record design variables at the '
                                            'driver level')
        self.recording_options.declare('record_objectives', types=bool, default=True,
                                       desc='Set to True to record objectives at the driver level')
        self.recording_options.declare('record_constraints', types=bool, default=True,
                                       desc='Set to True to record constraints at the '
                                            'driver level')
        self.recording_options.declare('includes', types=list, default=[],
                                       desc='Patterns for variables to include in recording')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                            '(processed post-includes)')
        self.recording_options.declare('record_derivatives', types=bool, default=False,
                                       desc='Set to True to record derivatives at the driver '
                                            'level')
        self.recording_options.declare('record_inputs', types=bool, default=True,
                                       desc='Set to True to record inputs at the driver level')

        # What the driver supports.
        self.supports = OptionsDictionary(parent_name=type(self).__name__)
        self.supports.declare('inequality_constraints', types=bool, default=False)
        self.supports.declare('equality_constraints', types=bool, default=False)
        self.supports.declare('linear_constraints', types=bool, default=False)
        self.supports.declare('two_sided_constraints', types=bool, default=False)
        self.supports.declare('multiple_objectives', types=bool, default=False)
        self.supports.declare('integer_design_vars', types=bool, default=False)
        self.supports.declare('gradients', types=bool, default=False)
        self.supports.declare('active_set', types=bool, default=False)
        self.supports.declare('simultaneous_derivatives', types=bool, default=False)
        self.supports.declare('total_jac_sparsity', types=bool, default=False)

        self.iter_count = 0
        self.cite = ""

        self._coloring_info = coloring_mod._DEF_COMP_SPARSITY_ARGS.copy()
        self._coloring_info['coloring'] = None
        self._coloring_info['dynamic'] = False
        self._coloring_info['static'] = None

        self._total_jac_sparsity = None
        self._res_jacs = {}
        self._total_jac = None

        self.fail = False

        self._declare_options()
        self.options.update(kwargs)

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
        self._problem = problem
        self._recording_iter = problem._recording_iter
        model = problem.model

        self._total_jac = None

        self._has_scaling = (
            np.any([r['scaler'] is not None for r in itervalues(self._responses)]) or
            np.any([dv['scaler'] is not None for dv in itervalues(self._designvars)])
        )

        # Determine if any design variables are discrete.
        self._designvars_discrete = [dv for dv in self._designvars
                                     if dv in model._discrete_outputs]
        if not self.supports['integer_design_vars'] and len(self._designvars_discrete) > 0:
            msg = "Discrete design variables are not supported by this driver: "
            msg += '.'.join(self._designvars_discrete)
            raise RuntimeError(msg)

        con_set = set()
        obj_set = set()
        dv_set = set()

        self._remote_dvs = dv_dict = {}
        self._remote_cons = con_dict = {}
        self._remote_objs = obj_dict = {}

        # Now determine if later we'll need to allgather cons, objs, or desvars.
        if model.comm.size > 1 and model._subsystems_allprocs:
            local_out_vars = set(model._outputs._views)
            remote_dvs = set(self._designvars) - local_out_vars
            remote_cons = set(self._cons) - local_out_vars
            remote_objs = set(self._objs) - local_out_vars
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
            abs2meta = model._var_allprocs_abs2meta
            for i, vname in enumerate(model._var_allprocs_abs_names['output']):
                if abs2meta[vname]['distributed']:
                    owner = None
                else:
                    owner = owning_ranks[vname]
                if vname in dv_set:
                    dv_dict[vname] = (owner, sizes[owner, i])
                if vname in con_set:
                    con_dict[vname] = (owner, sizes[owner, i])
                if vname in obj_set:
                    obj_dict[vname] = (owner, sizes[owner, i])

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
           Dictionary containing sets of variables to record.
        """
        problem = self._problem
        model = problem.model
        rrank = problem.comm.rank

        incl = recording_options['includes']
        excl = recording_options['excludes']

        # includes and excludes for outputs are specified using promoted names
        # NOTE: only local var names are in abs2prom, all will be gathered later
        abs2prom = model._var_allprocs_abs2prom['output']

        allvars = []
        skip = set()
        if recording_options['record_desvars']:
            allvars.extend(self._designvars)
        else:
            skip.update(self._designvars)
        if recording_options['record_objectives']:
            allvars.extend(self._objs)
        else:
            skip.update(self._objs)
        if recording_options['record_constraints']:
            allvars.extend(self._cons)
        else:
            skip.update(self._cons)

        vars2record = {
            'out': [n for n in allvars
                    if n in abs2prom and check_path(abs2prom[n], incl, excl, True)]
        }

        # inputs (if in options)
        if 'record_inputs' in recording_options:
            if recording_options['record_inputs']:
                vars2record['in'] = [n for n in model._var_allprocs_abs_names['input']
                                     if check_path(n, incl, excl)]
            else:
                vars2record['in'] = []

        if incl:
            vars2record['out'].extend(n for n in model._var_allprocs_abs_names['output']
                                      if n in abs2prom and check_path(abs2prom[n], incl, excl)
                                      and n not in skip)
            # remove dups and make sure order is the same on all procs
            vars2record['out'] = sorted(set(vars2record['out']))

        return vars2record

    def _setup_recording(self):
        """
        Set up case recording.
        """
        self._filtered_vars_to_record = self._get_vars_to_record(self.recording_options)

        self._rec_mgr.startup(self)

        # record the system metadata to the recorders attached to this Driver
        if self.recording_options['record_model_metadata']:
            for sub in self._problem.model.system_iter(recurse=True, include_self=True):
                self._rec_mgr.record_metadata(sub)

    def _get_voi_val(self, name, meta, remote_vois, driver_scaling=True, ignore_indices=False,
                     rank=None):
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
        ignore_indices : bool
            Set to True if the full array is desired, not just those indicated by indices.
        rank : int or None
            If not None, gather value to this rank only.

        Returns
        -------
        float or ndarray
            The value of the named variable of interest.
        """
        model = self._problem.model
        comm = model.comm
        vec = model._outputs._views_flat
        indices = meta['indices']

        if name in remote_vois:
            owner, size = remote_vois[name]
            # if var is distributed or only gathering to one rank
            if owner is None or rank is not None:
                val = self._problem.get_val(name, get_remote=True, rank=rank)
                if indices is not None:
                    val = val[indices]
            else:
                if owner == comm.rank:
                    if indices is None or ignore_indices:
                        val = vec[name].copy()
                    else:
                        val = vec[name][indices]
                else:
                    if not (indices is None or ignore_indices):
                        size = len(indices)
                    val = np.empty(size)

                comm.Bcast(val, root=owner)

        else:
            if name in self._designvars_discrete:
                val = model._discrete_outputs[name]

                # At present, only integers are supported by OpenMDAO drivers.
                # We check the values here.
                valid = True
                msg = "Only integer scalars or ndarrays are supported as values for " + \
                      "discrete variables when used as a design variable. "
                if np.isscalar(val) and not isinstance(val, int):
                    msg += "A value of type '{}' was specified.".format(val.__class__.__name__)
                    raise ValueError(msg)
                elif isinstance(val, np.ndarray) and not np.issubdtype(val[0], int):
                    msg += "An array of type '{}' was specified.".format(val[0].__class__.__name__)
                    raise ValueError(msg)

            elif indices is None or ignore_indices:
                val = vec[name].copy()
            else:
                val = vec[name][indices]

        if self._has_scaling and driver_scaling:
            # Scale design variable values
            adder = meta['adder']
            if adder is not None:
                val += adder

            scaler = meta['scaler']
            if scaler is not None:
                val *= scaler

        return val

    def get_design_var_values(self, filter=None, driver_scaling=True, ignore_indices=False,
                              get_remote=True):
        """
        Return the design variable values.

        This is called to gather the initial design variable state.

        Parameters
        ----------
        filter : set
            Set of desvar names used by recorders.
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.
        ignore_indices : bool
            Set to True if the full array is desired, not just those indicated by indices.
        get_remote : bool
            If True, retrieve remote values as well.

        Returns
        -------
        dict
           Dictionary containing values of each design variable.
        """
        if filter:
            dvs = filter
        else:
            # use all the designvars
            dvs = self._designvars

        if get_remote:
            remote_dvs = self._remote_dvs
        else:
            remote_dvs = ()

        return {n: self._get_voi_val(n, self._designvars[n], remote_dvs,
                                     driver_scaling=driver_scaling,
                                     ignore_indices=ignore_indices) for n in dvs}

    def set_design_var(self, name, value):
        """
        Set the value of a design variable.

        Parameters
        ----------
        name : str
            Global pathname of the design variable.
        value : float or ndarray
            Value for the design variable.
        """
        problem = self._problem

        if (name in self._remote_dvs and
                problem.model._owning_rank[name] != problem.comm.rank):
            return

        meta = self._designvars[name]
        indices = meta['indices']
        if indices is None:
            indices = slice(None)

        if name in self._designvars_discrete:
            problem.model._discrete_outputs[name] = int(value)

        else:
            desvar = problem.model._outputs._views_flat[name]
            desvar[indices] = value

            # Undo driver scaling when setting design var values into model.
            if self._has_scaling:
                scaler = meta['scaler']
                if scaler is not None:
                    desvar[indices] *= 1.0 / scaler

                adder = meta['adder']
                if adder is not None:
                    desvar[indices] -= adder

    def get_response_values(self, filter=None):
        """
        Return response values.

        Parameters
        ----------
        filter : set
            Set of response names used by recorders.

        Returns
        -------
        dict
           Dictionary containing values of each response.
        """
        if filter:
            resps = filter
        else:
            resps = self._responses

        return {n: self._get_voi_val(n, self._responses[n], self._remote_objs) for n in resps}

    def get_objective_values(self, driver_scaling=True, filter=None, ignore_indices=False,
                             get_remote=True):
        """
        Return objective values.

        Parameters
        ----------
        driver_scaling : bool
            When True, return values that are scaled according to either the adder and scaler or
            the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is True.
        filter : set
            Set of objective names used by recorders.
        ignore_indices : bool
            Set to True if the full array is desired, not just those indicated by indices.
        get_remote : bool
            If True, retrieve remote values as well.

        Returns
        -------
        dict
           Dictionary containing values of each objective.
        """
        if filter:
            objs = filter
        else:
            objs = self._objs

        if get_remote:
            remote_objs = self._remote_objs
        else:
            remote_objs = ()

        return {n: self._get_voi_val(n, self._objs[n], remote_objs,
                                     driver_scaling=driver_scaling,
                                     ignore_indices=ignore_indices)
                for n in objs}

    def get_constraint_values(self, ctype='all', lintype='all', driver_scaling=True, filter=None,
                              ignore_indices=False, get_remote=True):
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
        filter : set
            Set of constraint names used by recorders.
        ignore_indices : bool
            Set to True if the full array is desired, not just those indicated by indices.
        get_remote : bool
            If True, retrieve remote values as well.

        Returns
        -------
        dict
           Dictionary containing values of each constraint.
        """
        if filter is not None:
            cons = filter
        else:
            cons = self._cons

        if get_remote:
            remote_cons = self._remote_cons
        else:
            remote_cons = ()

        con_dict = {}
        for name in cons:
            meta = self._cons[name]

            if lintype == 'linear' and not meta['linear']:
                continue

            if lintype == 'nonlinear' and meta['linear']:
                continue

            if ctype == 'eq' and meta['equals'] is None:
                continue

            if ctype == 'ineq' and meta['equals'] is not None:
                continue

            con_dict[name] = self._get_voi_val(name, meta, remote_cons,
                                               driver_scaling=driver_scaling,
                                               ignore_indices=ignore_indices)

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
        order.extend(n for n, meta in iteritems(self._cons)
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

        self._responses = resps = model.get_responses(recurse=True)
        for name, data in iteritems(resps):
            if data['type'] == 'con':
                cons[name] = data
            else:
                objs[name] = data

        response_size = sum(resps[n]['size'] for n in self._get_ordered_nl_responses())

        # Gather up the information for design vars.
        self._designvars = designvars = model.get_design_vars(recurse=True)
        desvar_size = sum(data['size'] for data in itervalues(designvars))

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
            self._problem.model.run_solve_nonlinear()

        self.iter_count += 1
        return False

    def _compute_totals(self, of=None, wrt=None, return_format='flat_dict', global_names=True):
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
            Set to True when passing in global names to skip some translation steps.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        problem = self._problem
        total_jac = self._total_jac
        debug_print = 'totals' in self.options['debug_print'] and (not MPI or
                                                                   MPI.COMM_WORLD.rank == 0)

        if debug_print:
            header = 'Driver total derivatives for iteration: ' + str(self.iter_count)
            print(header)
            print(len(header) * '-' + '\n')

        if problem.model._owns_approx_jac:
            self._recording_iter.stack.append(('_compute_totals_approx', 0))

            try:
                if total_jac is None:
                    total_jac = _TotalJacInfo(problem, of, wrt, global_names,
                                              return_format, approx=True, debug_print=debug_print)

                    # Don't cache linear constraint jacobian
                    if not total_jac.has_lin_cons:
                        self._total_jac = total_jac

                    totals = total_jac.compute_totals_approx(initialize=True)
                else:
                    totals = total_jac.compute_totals_approx()
            finally:
                self._recording_iter.stack.pop()

        else:
            if total_jac is None:
                total_jac = _TotalJacInfo(problem, of, wrt, global_names, return_format,
                                          debug_print=debug_print)

                # don't cache linear constraint jacobian
                if not total_jac.has_lin_cons:
                    self._total_jac = total_jac

            self._recording_iter.stack.append(('_compute_totals', 0))

            try:
                totals = total_jac.compute_totals()
            finally:
                self._recording_iter.stack.pop()

        if self._rec_mgr._recorders and self.recording_options['record_derivatives']:
            metadata = create_local_meta(self._get_name())
            total_jac.record_derivatives(self, metadata)

        return totals

    def record_iteration(self):
        """
        Record an iteration of the current Driver.
        """
        record_iteration(self, self._problem, self._get_name())

        # if not self._rec_mgr._recorders:
        #     return

        # # Get the data to record (collective calls that get across all ranks)
        # opts = self.recording_options
        # filt = self._filtered_vars_to_record

        # prob = self._problem
        # model = self._problem.model
        # rank = model.comm.rank
        # nrank = model.comm.size
        # owns = model._owning_rank
        # views = model._outputs._views
        # meta = model._var_allprocs_abs2meta

        # parallel = False
        # if nrank > 1:
        #     for r in self._rec_mgr._recorders:
        #         if r._parallel:
        #             parallel = True
        #             break

        # outs = {}
        # if filt['out']:
        #     if nrank == 1:
        #         outs = {n: views[n] for n in filt['out']}
        #     elif parallel:
        #         sizes = model._var_sizes['nonlinear']['output']
        #         abs2idx = model._var_allprocs_abs2idx['nonlinear']
        #         outs = {n: views[n] for n in filt['out'] if sizes[n][abs2idx[n]] > 0}
        #     else:
        #         for name in filt['out']:
        #             if owns[name] == 0 and not meta[name]['distributed']:
        #                 if rank == 0:
        #                     outs[name] = views[name]
        #             else:
        #                 outs[name] = prob.get_val(name, get_remote=True, rank=0)

        # ins = {}
        # if 'in' in filt and filt['in']:
        #     views = model._inputs._views
        #     names = model._inputs._names
        #     if nrank == 1:
        #         ins = {n: views[n] for n in filt['in'] if n in names}
        #     elif parallel:
        #         sizes = model._var_sizes['nonlinear']['input']
        #         abs2idx = model._var_allprocs_abs2idx['nonlinear']
        #         ins = {n: views[n] for n in filt['in'] if sizes[n][abs2idx[n]] > 0}
        #     else:
        #         for name in filt['in']:
        #             if name not in names:
        #                 continue
        #             if owns[name] == 0 and not meta[name]['distributed']:
        #                 if rank == 0:
        #                     ins[name] = views[name]
        #             else:
        #                 ins[name] = prob.get_val(name, get_remote=True, rank=0)

        # data = {
        #     'out': outs,
        #     'in': ins
        # }

        # if case_name is None:
        #     case_name = self._get_name()

        # self._rec_mgr.record_iteration(self, data, create_local_meta(case_name))

    def _gather_vars(self, root, local_vars):
        """
        Gather and return only variables listed in `local_vars` from the `root` System.

        Parameters
        ----------
        root : <System>
            the root System for the Problem
        local_vars : dict
            local variable names and values

        Returns
        -------
        dict
            variable names and values.
        """
        # if trace:
        #     debug("gathering vars for recording in %s" % root.pathname)
        all_vars = root.comm.gather(local_vars, root=0)
        # if trace:
        #     debug("DONE gathering rec vars for %s" % root.pathname)

        if root.comm.rank == 0:
            dct = all_vars[0]
            for d in all_vars[1:]:
                dct.update(d)
            return dct

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

    def set_simul_deriv_color(self, coloring):
        """
        See use_fixed_coloring. This method is deprecated.

        Parameters
        ----------
        coloring : str or Coloring
            Information about simultaneous coloring for design vars and responses.  If a
            string, then coloring is assumed to be the name of a file that contains the
            coloring information in pickle format. Otherwise it must be a Coloring object.
            See the docstring for Coloring for details.

        """
        warn_deprecation("set_simul_deriv_color is deprecated.  Use use_fixed_coloring instead.")
        self.use_fixed_coloring(coloring)

    def _setup_tot_jac_sparsity(self):
        """
        Set up total jacobian subjac sparsity.

        Drivers that can use subjac sparsity should override this.
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

        if static is coloring_mod._STD_COLORING_FNAME or isinstance(static, string_types):
            if static is coloring_mod._STD_COLORING_FNAME:
                fname = self._get_total_coloring_fname()
            else:
                fname = static
            print("loading total coloring from file %s" % fname)
            coloring = info['coloring'] = coloring_mod.Coloring.load(fname)
            info.update(coloring._meta)
            return coloring

    def _get_total_coloring_fname(self):
        return os.path.join(self._problem.options['coloring_dir'], 'total_coloring.pkl')

    def _setup_simul_coloring(self):
        """
        Set up metadata for coloring of total derivative solution.

        If set_coloring was called with a filename, load the coloring file.
        """
        # command line simul_coloring uses this env var to turn pre-existing coloring off
        if not coloring_mod._use_total_sparsity:
            return

        problem = self._problem
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

    def _pre_run_model_debug_print(self):
        """
        Optionally print some debugging information before the model runs.
        """
        debug_opt = self.options['debug_print']
        if not debug_opt or debug_opt == ['totals']:
            return

        if not MPI or MPI.COMM_WORLD.rank == 0:
            header = 'Driver debug print for iter coord: {}'.format(
                self._recording_iter.get_formatted_iteration_coordinate())
            print(header)
            print(len(header) * '-')

        if 'desvars' in debug_opt:
            desvar_vals = self.get_design_var_values(driver_scaling=False, ignore_indices=True)
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Design Vars")
                if desvar_vals:

                    # Print desvars in non_flattened state.
                    meta = self._problem.model._var_allprocs_abs2meta
                    for name in desvar_vals:
                        shape = meta[name]['shape']
                        desvar_vals[name] = desvar_vals[name].reshape(shape)
                    pprint.pprint(desvar_vals)
                else:
                    print("None")
                print()

        sys.stdout.flush()

    def _post_run_model_debug_print(self):
        """
        Optionally print some debugging information after the model runs.
        """
        if 'nl_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='nonlinear', driver_scaling=False)
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Nonlinear constraints")
                if cons:
                    pprint.pprint(cons)
                else:
                    print("None")
                print()

        if 'ln_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='linear', driver_scaling=False)
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Linear constraints")
                if cons:
                    pprint.pprint(cons)
                else:
                    print("None")
                print()

        if 'objs' in self.options['debug_print']:
            objs = self.get_objective_values(driver_scaling=False)
            if not MPI or MPI.COMM_WORLD.rank == 0:
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
        super(RecordingDebugging, self).__enter__()
        self.recording_requester._pre_run_model_debug_print()
        return self

    def __exit__(self, *args):
        """
        Do things after the code inside the 'with RecordingDebugging' block.

        Parameters
        ----------
        *args : array
            Solver recording requires extra args.
        """
        self.recording_requester._post_run_model_debug_print()
        super(RecordingDebugging, self).__exit__()


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
    if not requester._rec_mgr._recorders:
        return

    # Get the data to record (collective calls that get across all ranks)
    opts = requester.recording_options
    filt = requester._filtered_vars_to_record

    model = prob.model
    rank = model.comm.rank
    nrank = model.comm.size
    owns = model._owning_rank
    views = model._outputs._views
    meta = model._var_allprocs_abs2meta

    parallel = False
    if nrank > 1:
        for r in requester._rec_mgr._recorders:
            if r._parallel:
                parallel = True
                break

    outs = {}
    if filt['out']:
        if nrank == 1:
            outs = {n: views[n] for n in filt['out']}
        elif parallel:
            sizes = model._var_sizes['nonlinear']['output']
            abs2idx = model._var_allprocs_abs2idx['nonlinear']
            outs = {n: views[n] for n in filt['out'] if sizes[rank, abs2idx[n]] > 0}
        else:
            for name in filt['out']:
                if owns[name] == 0 and not meta[name]['distributed']:
                    if rank == 0:
                        outs[name] = views[name]
                else:
                    outs[name] = prob.get_val(name, get_remote=True, rank=0)

    ins = {}
    if 'in' in filt and filt['in']:
        views = model._inputs._views
        names = model._inputs._names
        if nrank == 1:
            ins = {n: views[n] for n in filt['in'] if n in names}
        elif parallel:
            sizes = model._var_sizes['nonlinear']['input']
            abs2idx = model._var_allprocs_abs2idx['nonlinear']
            ins = {n: views[n] for n in filt['in'] if sizes[n][abs2idx[n]] > 0}
        else:
            for name in filt['in']:
                if name not in names:
                    continue
                if owns[name] == 0 and not meta[name]['distributed']:
                    if rank == 0:
                        ins[name] = views[name]
                else:
                    ins[name] = prob.get_val(name, get_remote=True, rank=0)

    data = {
        'out': outs,
        'in': ins
    }

    import pprint
    print(model.comm.rank)
    pprint.pprint(data)
    requester._rec_mgr.record_iteration(requester, data, create_local_meta(case_name))
