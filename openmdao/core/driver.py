"""Define a base class for all Drivers in OpenMDAO."""
from __future__ import print_function

import os
import json
from collections import OrderedDict
import warnings

from six import iteritems, itervalues, string_types

import numpy as np

from openmdao.recorders.recording_manager import RecordingManager
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.record_util import create_local_meta, check_path
from openmdao.utils.mpi import MPI
from openmdao.recorders.recording_iteration_stack import get_formatted_iteration_coordinate
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.coloring import _use_simul_coloring


def _is_debug_print_opts_valid(opts):
    """
    Check validity of debug_print option for Driver.

    Parameters
    ----------
    opts : list
        The value of the debug_print option set by the user.

    Returns
    -------
    bool
        True if the option is valid. Otherwise, False.
    """
    if not isinstance(opts, list):
        return False
    _valid_opts = ['desvars', 'nl_cons', 'ln_cons', 'objs']
    for opt in opts:
        if opt not in _valid_opts:
            return False
    return True


class Driver(object):
    """
    Top-level container for the systems and drivers.

    Options
    -------
    options['debug_print'] :  list of strings([])
        Indicates what variables to print at each iteration. The valid options are:
            'desvars','ln_cons','nl_cons',and 'objs'.
    recording_options['record_metadata'] :  bool(True)
        Tells recorder whether to record variable attribute metadata.
    recording_options['record_desvars'] :  bool(True)
        Tells recorder whether to record the desvars of the Driver.
    recording_options['record_responses'] :  bool(False)
        Tells recorder whether to record the responses of the Driver.
    recording_options['record_objectives'] :  bool(False)
        Tells recorder whether to record the objectives of the Driver.
    recording_options['record_constraints'] :  bool(False)
        Tells recorder whether to record the constraints of the Driver.
    recording_options['includes'] :  list of strings("*")
        Patterns for variables to include in recording.
    recording_options['excludes'] :  list of strings('')
        Patterns for variables to exclude in recording (processed after includes).


    Attributes
    ----------
    fail : bool
        Reports whether the driver ran successfully.
    iter_count : int
        Keep track of iterations for case recording.
    metadata : list
        List of metadata
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
    recording_options : <OptionsDictionary>
        Dictionary with driver recording options.
    debug_print : <OptionsDictionary>
        Dictionary with debugging printing options.
    cite : str
        Listing of relevant citataions that should be referenced when
        publishing work that uses this class.
    _problem : <Problem>
        Pointer to the containing problem.
    supports : <OptionsDictionary>
        Provides a consistant way for drivers to declare what features they support.
    _designvars : dict
        Contains all design variable info.
    _cons : dict
        Contains all constraint info.
    _objs : dict
        Contains all objective info.
    _responses : dict
        Contains all response info.
    _rec_mgr : <RecordingManager>
        Object that manages all recorders added to this driver.
    _vars_to_record: dict
        Dict of lists of var names indicating what to record
    _model_viewer_data : dict
        Structure of model, used to make n2 diagram.
    _remote_dvs : dict
        Dict of design variables that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_cons : dict
        Dict of constraints that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_objs : dict
        Dict of objectives that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_responses : dict
        A combined dict containing entries from _remote_cons and _remote_objs.
    _simul_coloring_info : tuple of dicts
        A data structure describing coloring for simultaneous derivs.
    _res_jacs : dict
        Dict of sparse subjacobians for use with certain optimizers, e.g. pyOptSparseDriver.
    """

    def __init__(self):
        """
        Initialize the driver.
        """
        self._rec_mgr = RecordingManager()
        self._vars_to_record = {
            'desvarnames': set(),
            'responsenames': set(),
            'objectivenames': set(),
            'constraintnames': set(),
            'sysinclnames': set(),
        }

        self._problem = None
        self._designvars = None
        self._cons = None
        self._objs = None
        self._responses = None
        self.options = OptionsDictionary()
        self.recording_options = OptionsDictionary()

        ###########################
        self.options.declare('debug_print', types=list, is_valid=_is_debug_print_opts_valid,
                             desc="List of what type of Driver variables to print at each "
                             "iteration. Valid items in list are 'desvars', 'ln_cons', "
                             "'nl_cons', 'objs'",
                             default=[])

        ###########################
        self.recording_options.declare('record_metadata', types=bool, desc='Record metadata',
                                       default=True)
        self.recording_options.declare('record_desvars', types=bool, default=True,
                                       desc='Set to True to record design variables at the \
                                       driver level')
        self.recording_options.declare('record_responses', types=bool, default=False,
                                       desc='Set to True to record responses at the driver level')
        self.recording_options.declare('record_objectives', types=bool, default=True,
                                       desc='Set to True to record objectives at the \
                                       driver level')
        self.recording_options.declare('record_constraints', types=bool, default=True,
                                       desc='Set to True to record constraints at the \
                                       driver level')
        self.recording_options.declare('includes', types=list, default=['*'],
                                       desc='Patterns for variables to include in recording')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                       '(processed post-includes)')
        self.recording_options.declare('record_derivatives', types=bool, default=False,
                                       desc='Set to True to record derivatives at the driver \
                                       level')
        ###########################

        # What the driver supports.
        self.supports = OptionsDictionary()
        self.supports.declare('inequality_constraints', types=bool, default=False)
        self.supports.declare('equality_constraints', types=bool, default=False)
        self.supports.declare('linear_constraints', types=bool, default=False)
        self.supports.declare('two_sided_constraints', types=bool, default=False)
        self.supports.declare('multiple_objectives', types=bool, default=False)
        self.supports.declare('integer_design_vars', types=bool, default=False)
        self.supports.declare('gradients', types=bool, default=False)
        self.supports.declare('active_set', types=bool, default=False)
        self.supports.declare('simultaneous_derivatives', types=bool, default=False)

        # Debug printing.
        self.debug_print = OptionsDictionary()
        self.debug_print.declare('debug_print', types=bool, default=False,
                                 desc='Overall option to turn on Driver debug printing')
        self.debug_print.declare('debug_print_desvars', types=bool, default=False,
                                 desc='Print design variables')
        self.debug_print.declare('debug_print_nl_con', types=bool, default=False,
                                 desc='Print nonlinear constraints')
        self.debug_print.declare('debug_print_ln_con', types=bool, default=False,
                                 desc='Print linear constraints')
        self.debug_print.declare('debug_print_objective', types=bool, default=False,
                                 desc='Print objectives')

        self.iter_count = 0
        self.metadata = None
        self._model_viewer_data = None
        self.cite = ""

        # TODO, support these in OpenMDAO
        self.supports.declare('integer_design_vars', types=bool, default=False)

        self._simul_coloring_info = None
        self._res_jacs = {}

        self.fail = False

    def add_recorder(self, recorder):
        """
        Add a recorder to the driver.

        Parameters
        ----------
        recorder : BaseRecorder
           A recorder instance.
        """
        self._rec_mgr.append(recorder)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        self._rec_mgr.close()

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
        model = problem.model

        self._objs = objs = OrderedDict()
        self._cons = cons = OrderedDict()
        self._responses = model.get_responses(recurse=True)
        response_size = 0
        for name, data in iteritems(self._responses):
            if data['type'] == 'con':
                cons[name] = data
            else:
                objs[name] = data
            response_size += data['size']

        # Gather up the information for design vars.
        self._designvars = model.get_design_vars(recurse=True)
        desvar_size = np.sum(data['size'] for data in itervalues(self._designvars))

        if ((problem._mode == 'fwd' and desvar_size > response_size) or
                (problem._mode == 'rev' and response_size > desvar_size)):
            warnings.warn("Inefficient choice of derivative mode.  You chose '%s' for a "
                          "problem with %d design variables and %d response variables "
                          "(objectives and constraints)." %
                          (problem._mode, desvar_size, response_size), RuntimeWarning)

        self._has_scaling = (
            np.any([r['scaler'] is not None for r in self._responses.values()]) or
            np.any([dv['scaler'] is not None for dv in self._designvars.values()])
        )

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
            owning_ranks = model._owning_rank['output']
            sizes = model._var_sizes['nonlinear']['output']
            for i, vname in enumerate(model._var_allprocs_abs_names['output']):
                owner = owning_ranks[vname]
                if vname in dv_set:
                    dv_dict[vname] = (owner, sizes[owner, i])
                if vname in con_set:
                    con_dict[vname] = (owner, sizes[owner, i])
                if vname in obj_set:
                    obj_dict[vname] = (owner, sizes[owner, i])

        self._remote_responses = self._remote_cons.copy()
        self._remote_responses.update(self._remote_objs)

        # Case recording setup
        mydesvars = myobjectives = myconstraints = myresponses = set()
        mysystem_outputs = set()
        incl = self.recording_options['includes']
        excl = self.recording_options['excludes']
        rec_desvars = self.recording_options['record_desvars']
        rec_objectives = self.recording_options['record_objectives']
        rec_constraints = self.recording_options['record_constraints']
        rec_responses = self.recording_options['record_responses']

        all_desvars = {n for n in self._designvars
                       if check_path(n, incl, excl, True)}
        all_objectives = {n for n in self._objs
                          if check_path(n, incl, excl, True)}
        all_constraints = {n for n in self._cons
                           if check_path(n, incl, excl, True)}
        if rec_desvars:
            mydesvars = all_desvars

        if rec_objectives:
            myobjectives = all_objectives

        if rec_constraints:
            myconstraints = all_constraints

        if rec_responses:
            myresponses = {n for n in self._responses
                           if check_path(n, incl, excl, True)}

        # get the includes that were requested for this Driver recording
        if incl:
            prob = self._problem
            root = prob.model
            # The my* variables are sets

            # First gather all of the desired outputs
            # The following might only be the local vars if MPI
            mysystem_outputs = {n for n in root._outputs
                                if check_path(n, incl, excl)}

            # If MPI, and on rank 0, need to gather up all the variables
            #    even those not local to rank 0
            if MPI:
                all_vars = root.comm.gather(mysystem_outputs, root=0)
                if MPI.COMM_WORLD.rank == 0:
                    mysystem_outputs = all_vars[-1]
                    for d in all_vars[:-1]:
                        mysystem_outputs.update(d)

            # de-duplicate mysystem_outputs
            mysystem_outputs = mysystem_outputs.difference(all_desvars, all_objectives,
                                                           all_constraints)

        if MPI:  # filter based on who owns the variables
            # TODO Eventually, we think we can get rid of this next check. But to be safe,
            #       we are leaving it in there.
            if not model.is_active():
                raise RuntimeError(
                    "RecordingManager.startup should never be called when "
                    "running in parallel on an inactive System")
            rrank = self._problem.comm.rank  # root ( aka model ) rank.
            rowned = model._owning_rank['output']
            mydesvars = [n for n in mydesvars if rrank == rowned[n]]
            myresponses = [n for n in myresponses if rrank == rowned[n]]
            myobjectives = [n for n in myobjectives if rrank == rowned[n]]
            myconstraints = [n for n in myconstraints if rrank == rowned[n]]
            mysystem_outputs = [n for n in mysystem_outputs if rrank == rowned[n]]

        self._filtered_vars_to_record = {
            'des': mydesvars,
            'obj': myobjectives,
            'con': myconstraints,
            'res': myresponses,
            'sys': mysystem_outputs,
        }

        self._rec_mgr.startup(self)
        if self._rec_mgr._recorders:
            from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data
            self._model_viewer_data = _get_viewer_data(problem)
        if self.recording_options['record_metadata']:
            self._rec_mgr.record_metadata(self)

        # set up simultaneous deriv coloring
        if self._simul_coloring_info and self.supports['simultaneous_derivatives']:
            if problem._mode == 'fwd':
                self._setup_simul_coloring(problem._mode)
            else:
                raise RuntimeError("simultaneous derivs are currently not supported in rev mode.")

    def _get_voi_val(self, name, meta, remote_vois):
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
            if owner == comm.rank:
                if indices is None:
                    val = vec[name].copy()
                else:
                    val = vec[name][indices]
            else:
                if indices is not None:
                    size = len(indices)
                val = np.empty(size)
            comm.Bcast(val, root=owner)
        else:
            if indices is None:
                val = vec[name].copy()
            else:
                val = vec[name][indices]

        if self._has_scaling:
            # Scale design variable values
            adder = meta['adder']
            if adder is not None:
                val += adder

            scaler = meta['scaler']
            if scaler is not None:
                val *= scaler

        return val

    def get_design_var_values(self, filter=None):
        """
        Return the design variable values.

        This is called to gather the initial design variable state.

        Parameters
        ----------
        filter : list
            List of desvar names used by recorders.

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

        return {n: self._get_voi_val(n, self._designvars[n], self._remote_dvs) for n in dvs}

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
        if (name in self._remote_dvs and
                self._problem.model._owning_rank['output'][name] != self._problem.comm.rank):
            return

        meta = self._designvars[name]
        indices = meta['indices']
        if indices is None:
            indices = slice(None)

        desvar = self._problem.model._outputs._views_flat[name]
        desvar[indices] = value

        if self._has_scaling:
            # Scale design variable values
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
        filter : list
            List of response names used by recorders.

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

    def get_objective_values(self, filter=None):
        """
        Return objective values.

        Parameters
        ----------
        filter : list
            List of objective names used by recorders.

        Returns
        -------
        dict
           Dictionary containing values of each objective.
        """
        if filter:
            objs = filter
        else:
            objs = self._objs

        return {n: self._get_voi_val(n, self._objs[n], self._remote_objs) for n in objs}

    def get_constraint_values(self, ctype='all', lintype='all', filter=None):
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
        filter : list
            List of constraint names used by recorders.

        Returns
        -------
        dict
           Dictionary containing values of each constraint.
        """
        if filter is not None:
            cons = filter
        else:
            cons = self._cons

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

            con_dict[name] = self._get_voi_val(name, meta, self._remote_cons)

        return con_dict

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
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            failure_flag, _, _ = self._problem.model._solve_nonlinear()

        self.iter_count += 1
        return failure_flag

    def _dict2array_jac(self, derivs):
        osize = 0
        isize = 0
        do_wrt = True
        islices = {}
        oslices = {}
        for okey, oval in iteritems(derivs):
            if do_wrt:
                for ikey, val in iteritems(oval):
                    istart = isize
                    isize += val.shape[1]
                    islices[ikey] = slice(istart, isize)
                do_wrt = False
            ostart = osize
            osize += oval[ikey].shape[0]
            oslices[okey] = slice(ostart, osize)

        new_derivs = np.zeros((osize, isize))

        relevant = self._problem.model._relevant

        for okey, odict in iteritems(derivs):
            for ikey, val in iteritems(odict):
                if okey in relevant[ikey] or ikey in relevant[okey]:
                    new_derivs[oslices[okey], islices[ikey]] = val

        return new_derivs

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
        prob = self._problem

        # Compute the derivatives in dict format...
        if prob.model._owns_approx_jac:
            derivs = prob._compute_totals_approx(of=of, wrt=wrt, return_format='dict',
                                                 global_names=global_names)
        else:
            derivs = prob._compute_totals(of=of, wrt=wrt, return_format='dict',
                                          global_names=global_names)

        # ... then convert to whatever the driver needs.
        if return_format in ('dict', 'array'):
            if self._has_scaling:
                for okey, odict in iteritems(derivs):
                    for ikey, val in iteritems(odict):

                        iscaler = self._designvars[ikey]['scaler']
                        oscaler = self._responses[okey]['scaler']

                        # Scale response side
                        if oscaler is not None:
                            val[:] = (oscaler * val.T).T

                        # Scale design var side
                        if iscaler is not None:
                            val *= 1.0 / iscaler
        else:
            raise RuntimeError("Derivative scaling by the driver only supports the 'dict' and "
                               "'array' formats at present.")

        if return_format == 'array':
            derivs = self._dict2array_jac(derivs)

        return derivs

    def record_iteration(self):
        """
        Record an iteration of the current Driver.
        """
        if not self._rec_mgr._recorders:
            return

        metadata = create_local_meta(self._get_name())

        # Get the data to record
        data = {}
        if self.recording_options['record_desvars']:
            # collective call that gets across all ranks
            desvars = self.get_design_var_values()
        else:
            desvars = {}

        if self.recording_options['record_responses']:
            # responses = self.get_response_values() # not really working yet
            responses = {}
        else:
            responses = {}

        if self.recording_options['record_objectives']:
            objectives = self.get_objective_values()
        else:
            objectives = {}

        if self.recording_options['record_constraints']:
            constraints = self.get_constraint_values()
        else:
            constraints = {}

        desvars = {name: desvars[name]
                   for name in self._filtered_vars_to_record['des']}
        # responses not working yet
        # responses = {name: responses[name] for name in self._filtered_vars_to_record['res']}
        objectives = {name: objectives[name]
                      for name in self._filtered_vars_to_record['obj']}
        constraints = {name: constraints[name]
                       for name in self._filtered_vars_to_record['con']}

        if self.recording_options['includes']:
            root = self._problem.model
            outputs = root._outputs
            # outputsinputs, outputs, residuals = root.get_nonlinear_vectors()
            sysvars = {}
            views = outputs._views
            for name in outputs._names:
                if name in self._filtered_vars_to_record['sys']:
                    sysvars[name] = views[name]
        else:
            sysvars = {}

        if MPI:
            root = self._problem.model
            desvars = self._gather_vars(root, desvars)
            responses = self._gather_vars(root, responses)
            objectives = self._gather_vars(root, objectives)
            constraints = self._gather_vars(root, constraints)
            sysvars = self._gather_vars(root, sysvars)

        data['des'] = desvars
        data['res'] = responses
        data['obj'] = objectives
        data['con'] = constraints
        data['sys'] = sysvars

        self._rec_mgr.record_iteration(self, data, metadata)

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
        dct : dict
            variable names and values.
        """
        # if trace:
        #     debug("gathering vars for recording in %s" % root.pathname)
        all_vars = root.comm.gather(local_vars, root=0)
        # if trace:
        #     debug("DONE gathering rec vars for %s" % root.pathname)

        if root.comm.rank == 0:
            dct = all_vars[-1]
            for d in all_vars[:-1]:
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

    def set_simul_deriv_color(self, simul_info):
        """
        Set the coloring for simultaneous derivatives.

        Parameters
        ----------
        simul_info : str or ({dv1: colors, ...}, {resp1: {dv1: {0: [res_idxs, dv_idxs]} ...} ...})
            Information about simultaneous coloring for design vars and responses.  If a string,
            then simul_info is assumed to be the name of a file that contains the coloring
            information in JSON format.
        """
        if self.supports['simultaneous_derivatives']:
            self._simul_coloring_info = simul_info
        else:
            raise RuntimeError("Driver '%s' does not support simultaneous derivatives." %
                               self._get_name())

    def _setup_simul_coloring(self, mode='fwd'):
        """
        Set up metadata for simultaneous derivative solution.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.
        """
        if mode == 'rev':
            raise NotImplementedError("Simultaneous derivatives are currently not supported "
                                      "in 'rev' mode")

        # command line simul_coloring uses this env var to turn pre-existing coloring off
        if not _use_simul_coloring:
            return

        prom2abs = self._problem.model._var_allprocs_prom2abs_list['output']

        if isinstance(self._simul_coloring_info, string_types):
            with open(self._simul_coloring_info, 'r') as f:
                self._simul_coloring_info = json.load(f)

        coloring, maps = self._simul_coloring_info
        for dv, colors in iteritems(coloring):
            if dv not in self._designvars:
                # convert name from promoted to absolute
                dv = prom2abs[dv][0]
            self._designvars[dv]['simul_deriv_color'] = colors

        for res, dvdict in iteritems(maps):
            if res not in self._responses:
                # convert name from promoted to absolute
                res = prom2abs[res][0]
            self._responses[res]['simul_map'] = dvdict

            for dv, col_dict in dvdict.items():
                col_dict = {int(k): v for k, v in iteritems(col_dict)}
                if dv not in self._designvars:
                    # convert name from promoted to absolute and replace dictionary key
                    del dvdict[dv]
                    dv = prom2abs[dv][0]
                dvdict[dv] = col_dict

    def _pre_run_model_debug_print(self):
        """
        Optionally print some debugging information before the model runs.
        """
        if not self.options['debug_print']:
            return

        if not MPI or MPI.COMM_WORLD.rank == 0:
            header = 'Driver debug print for iter coord: {}'.format(
                get_formatted_iteration_coordinate())
            print(header)
            print(len(header) * '-')

        if 'desvars' in self.options['debug_print']:
            desvar_vals = self.get_design_var_values()
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Design Vars")
                if desvar_vals:
                    for name, value in iteritems(desvar_vals):
                        print("{}: {}".format(name, repr(value)))
                else:
                    print("None")
                print()

    def _post_run_model_debug_print(self):
        """
        Optionally print some debugging information after the model runs.
        """
        if 'nl_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='nonlinear')
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Nonlinear constraints")
                if cons:
                    for name, value in iteritems(cons):
                        print("{}: {}".format(name, repr(value)))
                else:
                    print("None")
                print()

        if 'ln_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='linear')
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Linear constraints")
                if cons:
                    for name, value in iteritems(cons):
                        print("{}: {}".format(name, repr(value)))
                else:
                    print("None")
                print()

        if 'objs' in self.options['debug_print']:
            objs = self.get_objective_values()
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Objectives")
                if objs:
                    for name, value in iteritems(objs):
                        print("{}: {}".format(name, repr(value)))
                else:
                    print("None")
                print()


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
