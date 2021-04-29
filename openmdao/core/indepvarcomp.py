"""Define the IndepVarComp class."""

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.array_utils import shape_to_len
from openmdao.utils.general_utils import make_set, ensure_compatible
from openmdao.warnings import warn_deprecation


class IndepVarComp(ExplicitComponent):
    """
    Class to use when all output variables are independent.
    """

    def __init__(self, name=None, val=1.0, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        name : str or None
            name of the variable.
            If None, variables should be defined external to this class by calling add_output.
        val : float or ndarray
            value of the variable if a single variable is being defined.
        **kwargs : dict
            keyword arguments.
        """
        super().__init__(**kwargs)

        if 'tags' not in kwargs:
            kwargs['tags'] = {'indep_var'}
        else:
            kwargs['tags'] = make_set(kwargs['tags'], name='tags') | {'indep_var'}

        # A single variable is declared during instantiation
        if isinstance(name, str):
            super().add_output(name, val, **kwargs)

        elif name is None:
            pass

        else:
            raise ValueError(
                "first argument to IndepVarComp init must be either of type "
                "`str` or an iterable of tuples of the form (name, value) or "
                "(name, value, keyword_dict).")

        for illegal in ('promotes', 'promotes_inputs', 'promotes_outputs'):
            if illegal in kwargs:
                raise ValueError("IndepVarComp init: '%s' is not supported "
                                 "in IndepVarComp." % illegal)

        self._no_check_partials = True

    def initialize(self):
        """
        Declare options.
        """
        opt = self.options
        opt.declare('name', types=str,
                    desc="Name of the variable in this component's namespace.")
        opt.declare('val', types=(float, list, tuple, np.ndarray), default=1.0,
                    desc="The initial value of the variable being added in user-defined units.")
        opt.declare('shape', types=(int, tuple, list), default=None,
                    desc="Shape of this variable, only required if val is not an array.")
        opt.declare('units', types=str, default=None,
                    desc="Units in which the output variables will be provided to the "
                         "component during execution.")
        opt.declare('res_units', types=str, default=None,
                    desc="Units in which the residuals of this output will be given to "
                         "the user when requested.")
        opt.declare('desc', types=str, default=None,
                    desc="Description of the variable")
        opt.declare('lower', types=(int, float, list, tuple, np.ndarray), default=None,
                    desc="Lower bound(s) in user-defined units. It can be (1) a float, "
                         "(2) an array_like consistent with the shape arg (if given), or "
                         "(3) an array_like matching the shape of val, if val is array_like. "
                         "A value of None means this output has no lower bound.")
        opt.declare('upper', types=(int, float, list, tuple, np.ndarray), default=None,
                    desc="Upper bound(s) in user-defined units. It can be (1) a float, "
                         "(2) an array_like consistent with the shape arg (if given), or "
                         "(3) an array_like matching the shape of val, if val is array_like. "
                         "A value of None means this output has no upper bound.")
        opt.declare('ref', types=float, default=1.,
                    desc="Scaling parameter. The value in the user-defined units of this output "
                         "variable when the scaled value is 1")
        opt.declare('ref0', types=float, default=0.,
                    desc="Scaling parameter. The value in the user-defined units of this output "
                         "variable when the scaled value is 0.")
        opt.declare('res_ref', types=float, default=None,
                    desc="Scaling parameter. The value in the user-defined res_units of this "
                         "output's residual when the scaled value is 1. Default is None, which "
                         "means residual scaling matches output scaling.")
        opt.declare('tags', types=(str, list), default=None,
                    desc="User defined tags that can be used to filter what gets listed when "
                         "calling list_outputs.")

    def _configure_check(self):
        """
        Do any error checking on i/o configuration.
        """
        if len(self._static_var_rel2meta) + len(self._var_rel2meta) == 0:
            raise RuntimeError(f"{self.msginfo}: No outputs (independent variables) have been "
                               "declared. They must either be declared during instantiation or "
                               "by calling add_output or add_discrete_output afterwards.")

        super()._configure_check()

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=None, ref0=None, res_ref=None, tags=None,
                   shape_by_conn=False, copy_shape=None, distributed=None):
        """
        Add an independent variable to this component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        res_units : None
            This argument is deprecated because it was unused.
        desc : str
            description of the variable
        lower : None
            This argument is deprecated because it was unused.
        upper : None
            This argument is deprecated because it was unused.
        ref : None
            This argument is deprecated because it was unused.
        ref0 : None
            This argument is deprecated because it was unused.
        res_ref : None
            This argument is deprecated because it was unused.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed when calling
            list_outputs.
        shape_by_conn : bool
            If True, shape this output to match its connected input(s).
        copy_shape : str or None
            If a str, that str is the name of a variable. Shape this output to match that of
            the named variable.
        distributed : bool
            If True, this variable is a distributed variable, so it can have different sizes/values
            across MPI processes.
        """
        if res_units is not None:
            warn_deprecation(f"{self.msginfo}: The 'res_units' argument was used when adding "
                             f"output '{name}'. This argument has been deprecated and will be "
                             "removed in a future version.")
        if lower is not None:
            warn_deprecation(f"{self.msginfo}: The 'lower' argument was used when adding "
                             f"output '{name}'. This argument has been deprecated and will be "
                             "removed in a future version.")
        if upper is not None:
            warn_deprecation(f"{self.msginfo}: The 'upper' argument was used when adding "
                             f"output '{name}'. This argument has been deprecated and will be "
                             "removed in a future version.")
        if ref0 is not None:
            warn_deprecation(f"{self.msginfo}: The 'ref0' argument was used when adding "
                             f"output '{name}'. This argument has been deprecated and will be "
                             "removed in a future version.")
        if res_ref is not None:
            warn_deprecation(f"{self.msginfo}: The 'res_ref' argument was used when adding "
                             f"output '{name}'. This argument has been deprecated and will be "
                             "removed in a future version.")
        if ref is not None:
            warn_deprecation(f"{self.msginfo}: The 'ref' argument was used when adding "
                             f"output '{name}'. This argument has been deprecated and will be "
                             "removed in a future version.")

        ref = 1.0
        ref0 = 0.0

        if res_ref is None:
            res_ref = ref

        if tags is None:
            tags = {'indep_var'}
        else:
            tags = make_set(tags) | {'indep_var'}

        kwargs = {'shape': shape, 'units': units, 'res_units': res_units, 'desc': desc,
                  'lower': lower, 'upper': upper, 'ref': ref, 'ref0': ref0,
                  'res_ref': res_ref, 'tags': tags, 'shape_by_conn': shape_by_conn,
                  'copy_shape': copy_shape, 'distributed': distributed,
                  }
        super().add_output(name, val, **kwargs)

    def add_discrete_output(self, name, val, desc='', tags=None):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        desc : str
            description of the variable.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed when calling
            list_outputs.
        """
        if tags is None:
            tags = {'indep_var'}
        else:
            tags = make_set(tags, name='tags') | {'indep_var'}

        kwargs = {'desc': desc, 'tags': tags}
        super().add_discrete_output(name, val, **kwargs)

    def _linearize(self, jac=None, sub_do_ln=False):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        sub_do_ln : boolean
            Flag indicating if the children should call linearize on their linear solvers.
        """
        # define this as empty for IndepVarComp to avoid overhead of ExplicitComponent._linearize.
        pass


class _AutoIndepVarComp(IndepVarComp):
    """
    Class to use when all output variables are independent.

    Attributes
    ----------
    _remotes : set
        Set of var names connected to remote inputs.
    """

    def __init__(self, name=None, val=1.0, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        name : str or None
            name of the variable.
            If None, variables should be defined external to this class by calling add_output.
        val : float or ndarray
            value of the variable if a single variable is being defined.
        **kwargs : dict
            keyword arguments.
        """
        super().__init__(name, val, **kwargs)
        self._remotes = set()

    def _add_remote(self, name):
        self._remotes.add(name)

    def _set_vector_class(self):
        if self.comm.size > 1:
            all_remotes = set()
            for remotes in self.comm.allgather(self._remotes):
                all_remotes.update(remotes)

            if all_remotes:
                self._has_distrib_vars = True

                self._remotes = all_remotes
                for name in all_remotes:
                    self._static_var_rel2meta[name]['distributed'] = True

        super()._set_vector_class()

    def add_output(self, name, val=1.0, units=None):
        """
        Add an independent variable to this component.

        This should never be called by a user, as it skips all checks.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        """
        # Add the output quickly.
        # We don't need to check for errors because we get the value straight from a
        # source, and ivc metadata is minimal.
        value, shape, _ = ensure_compatible(name, val, None)
        metadata = {
            'value': value,
            'shape': shape,
            'size': shape_to_len(shape),
            'units': units,
            'res_units': None,
            'desc': '',
            'distributed': False,
            'tags': set(),
            'ref': 1.0,
            'ref0': 0.0,
            'res_ref': 1.0,
            'lower': None,
            'upper': None,
            'shape_by_conn': False,
            'copy_shape': None
        }

        self._static_var_rel2meta[name] = metadata
        self._static_var_rel_names['output'].append(name)
        self._var_added(name)
