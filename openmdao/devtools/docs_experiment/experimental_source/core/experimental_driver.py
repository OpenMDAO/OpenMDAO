"""Define a base class for all Drivers in OpenMDAO."""
from collections import OrderedDict
import warnings

import numpy as np

from openmdao.recorders.recording_manager import RecordingManager
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.record_util import create_local_meta, check_path
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary


class ExperimentalDriver(object):
    """
    A fake driver class used for doc generation testing.

    Attributes
    ----------
    fail : bool
        Reports whether the driver ran successfully.
    iter_count : int
        Keep track of iterations for case recording.
    options : list
        List of options
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
    _total_coloring : tuple of dicts
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
        self.recording_options.declare('includes', types=list, default=[],
                                       desc='Patterns for variables to include in recording. \
                                       Uses fnmatch wildcards')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                       '(processed post-includes). Uses fnmatch wildcards')
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

        self.iter_count = 0
        self.options = None
        self._model_viewer_data = None
        self.cite = ""

        # TODO, support these in OpenMDAO
        self.supports.declare('integer_design_vars', types=bool, default=False)

        self._res_jacs = {}

        self.fail = False

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
        pass

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
        with Recording(self._get_name(), self.iter_count, self) as rec:
            self._problem.model.run_solve_nonlinear()

        self.iter_count += 1
        return False

    def _dict2array_jac(self, derivs):
        osize = 0
        isize = 0
        do_wrt = True
        islices = {}
        oslices = {}
        for okey, oval in derivs.items():
            if do_wrt:
                for ikey, val in oval.items():
                    istart = isize
                    isize += val.shape[1]
                    islices[ikey] = slice(istart, isize)
                do_wrt = False
            ostart = osize
            osize += oval[ikey].shape[0]
            oslices[okey] = slice(ostart, osize)

        new_derivs = np.zeros((osize, isize))

        relevant = self._problem.model._relevant

        for okey, odict in derivs.items():
            for ikey, val in odict.items():
                if okey in relevant[ikey] or ikey in relevant[okey]:
                    new_derivs[oslices[okey], islices[ikey]] = val

        return new_derivs

    def _compute_totals(self, of=None, wrt=None, return_format='flat_dict', use_abs_names=True):
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
        use_abs_names : bool
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
                                                 use_abs_names=use_abs_names)
        else:
            derivs = prob._compute_totals(of=of, wrt=wrt, return_format='dict',
                                          use_abs_names=use_abs_names)

        # ... then convert to whatever the driver needs.
        if return_format in ('dict', 'array'):
            if self._has_scaling:
                for okey, odict in derivs.items():
                    for ikey, val in odict.items():

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
            for name, value in outputs._names.items():
                if name in self._filtered_vars_to_record['sys']:
                    sysvars[name] = value
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

