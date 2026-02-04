"""Define the base class for all optimization drivers in OpenMDAO."""
from abc import abstractmethod

import numpy as np

from openmdao.core.driver import Driver
from openmdao.utils.om_warnings import issue_warning, DriverWarning


class OptimizationDriverBase(Driver):
    """
    Base class for all optimization drivers in OpenMDAO.

    This class provides common functionality for optimization drivers including
    design variable management, objective evaluation, constraint handling, and
    optimization-specific recording options. All optimization drivers (ScipyOptimizeDriver,
    pyOptSparseDriver, SimpleGADriver, DifferentialEvolutionDriver) inherit from this class.

    Optimization drivers that inherit from OptimizationDriverBase must implement
    the abstract run() method which executes the optimization algorithm.

    For analysis drivers (AnalysisDriver, DOEDriver) that don't perform optimization,
    inherit directly from Driver instead.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    _designvars : dict
        Dictionary of design variable metadata keyed by promoted name or alias.
    _designvars_discrete : list
        List of design variables that are discrete.
    _cons : dict
        Dictionary of constraint metadata keyed by promoted name or alias.
    _objs : dict
        Dictionary of objective metadata keyed by promoted name or alias.
    _lin_dvs : dict
        Design variables that affect linear constraints.
    _nl_dvs : dict
        Design variables that affect nonlinear constraints.
    _remote_dvs : dict
        Dict of design variables that are remote on at least one proc.
    _remote_cons : dict
        Dict of constraints that are remote on at least one proc.
    _remote_objs : dict
        Dict of objectives that are remote on at least one proc.
    _dist_driver_vars : dict
        Dict of constraints that are distributed outputs.
    _con_subjacs : dict
        Dict of sparse subjacobians for use with certain optimizers.
    _in_find_feasible : bool
        True if the driver is currently executing find_feasible.
    """

    def __init__(self, **kwargs):
        """
        Initialize the optimization driver.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments that will be mapped into the Driver options.
        """
        super().__init__(**kwargs)

        # Optimization-specific attributes
        self._designvars = None
        self._designvars_discrete = []
        self._cons = None
        self._objs = None
        self._lin_dvs = None
        self._nl_dvs = None
        self._remote_dvs = {}
        self._remote_cons = {}
        self._remote_objs = {}
        self._dist_driver_vars = {}
        self._con_subjacs = {}
        self._in_find_feasible = False

        # Set optimization support to True by default
        # Individual drivers can override this if needed
        self.supports['optimization'] = True

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        # Get base Driver options first
        super()._declare_options()

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
            from openmdao.vectors.vector import _full_slice
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
        from openmdao.core.driver import filter_by_meta
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

    def check_relevance(self):
        """
        Check if there are constraints that don't depend on any design vars.

        This usually indicates something is wrong with the problem formulation.
        """
        from openmdao.core.group import Group

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

    @abstractmethod
    def run(self):
        """
        Execute the optimization algorithm.

        This method must be implemented by all optimization drivers.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False if successful.
        """
        pass
