"""Define the IndepVarComp class."""

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.array_utils import shape_to_len
from openmdao.utils.general_utils import make_set, ensure_compatible
from openmdao.recorders.recording_iteration_stack import Recording


class IndepVarComp(ExplicitComponent):
    """
    Class to use when all output variables are independent.

    Parameters
    ----------
    name : str, list, tuple, or None
        Name of the variable or list/tuple of variables.
        If a string, defines a single variable with the specified name.
        If a list or tuple, each element should be a tuple with the format (name, value) or
        (name, value, kwargs) where `name` is a string, `value` can be any type compatible with val,
        and `kwargs` is a dictionary of keyword arguments.
        If None, variables should be defined external to this class by calling add_output.
    val : float or ndarray
        Initial value of the variable if a single variable is being defined.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, name=None, val=1.0, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        if 'tags' not in kwargs:
            kwargs['tags'] = {'openmdao:indep_var', 'openmdao:allow_desvar'}
        else:
            kwargs['tags'] = make_set(kwargs['tags'], name='tags') | {'openmdao:indep_var',
                                                                      'openmdao:allow_desvar'}

        # A single variable is declared during instantiation
        if isinstance(name, str):
            super().add_output(name, val, **kwargs)

        elif isinstance(name, (list, tuple)):
            for tup in name:
                if not isinstance(tup, tuple):
                    raise TypeError("Each entry in the list of tuples must be of type tuple.")
                if len(tup) == 2:
                    if not isinstance(tup[0], str):
                        raise TypeError("The first element of the tuple must be a "
                                        "string representing the variable name.")
                    if not isinstance(tup[1], (int, float, list, tuple, np.ndarray)):
                        raise TypeError("The second element of the tuple must be "
                                        "the initial value of the variable.")
                    super().add_output(tup[0], tup[1], **kwargs)
                elif len(tup) == 3:
                    if not isinstance(tup[0], str):
                        raise TypeError("The first element of the tuple must be a "
                                        "string representing the variable name.")
                    if not isinstance(tup[1], (int, float, list, tuple, np.ndarray)):
                        raise TypeError("The second element of the tuple must be "
                                        "the initial value of the variable.")
                    if not isinstance(tup[2], dict):
                        raise TypeError("The third element of the tuple must be a "
                                        "dictionary of keyword arguments.")
                    super().add_output(tup[0], tup[1], **tup[2])
                else:
                    raise ValueError("Each entry in the list of tuples must be of the form "
                                     "(name, value) or (name, value, keyword_dict).")

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
        errs = self._get_saved_errors()
        if len(self._static_var_rel2meta) + len(self._var_rel2meta) == 0 and not errs:
            raise RuntimeError(f"{self.msginfo}: No outputs (independent variables) have been "
                               "declared. They must either be declared during instantiation or "
                               "by calling add_output or add_discrete_output afterwards.")

        super()._configure_check()

    def add_input(self, name, val=1.0, **kwargs):
        """
        Add an input variable to the component.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : float or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        **kwargs : named args
            Remaining args.
        """
        raise RuntimeError(f"Can't add input '{name}' to IndepVarComp '{self.name}'. IndepVarComps "
                           "are not allowed to have inputs. If you want IndepVarComp-like behavior"
                           " for some outputs of a component that has inputs, you can tag those "
                           "outputs with 'openmdao:indep_var' and 'openmdao:allow_desvar' and they "
                           "will be treated as independent variables.")

    def add_output(self, name, val=1.0, **kwargs):
        """
        Add an independent variable to this component.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : float or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        **kwargs : named args
            Remaining args passed to the base class add_output.

        Returns
        -------
        dict
            Metadata for added variable.
        """
        tags = {'openmdao:indep_var', 'openmdao:allow_desvar'}
        if 'tags' in kwargs and kwargs['tags'] is not None:
            tags.update(make_set(kwargs['tags'], name='tags'))
        kwargs['tags'] = tags
        return super().add_output(name, val=val, **kwargs)

    def add_discrete_output(self, name, val, desc='', tags=None):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        desc : str
            Description of the variable.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed when calling list_outputs.

        Returns
        -------
        dict
            Metadata for added variable.
        """
        if tags is None:
            tags = {'openmdao:indep_var'}
        else:
            tags = make_set(tags, name='tags') | {'openmdao:indep_var'}

        kwargs = {'desc': desc, 'tags': tags}
        return super().add_discrete_output(name, val, **kwargs)

    def _linearize(self, jac=None, sub_do_ln=False):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        """
        # define this for IndepVarComp to avoid overhead of ExplicitComponent._linearize.
        pass

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        # define this for IndepVarComp to avoid overhead of ExplicitComponent._apply_nonlinear.
        self.iter_count_apply += 1

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.
        """
        # define this for IndepVarComp to avoid overhead of ExplicitComponent._solve_nonlinear.
        with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
            pass


class _AutoIndepVarComp(IndepVarComp):
    """
    IndepVarComp whose outputs are automatically connected to all unconnected inputs.

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
            Name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.

        Returns
        -------
        dict
            Metadata for added variable.

        """
        # Add the output quickly.
        # We don't need to check for errors because we get the value straight from a
        # source, and ivc metadata is minimal.
        value, shape = ensure_compatible(name, val, None)
        metadata = {
            'val': value,
            'shape': shape,
            'size': shape_to_len(shape),
            'units': units,
            'res_units': None,
            'desc': '',
            'distributed': False,
            'tags': set(['openmdao:allow_desvar', 'openmdao:indep_var']),
            'ref': 1.0,
            'ref0': 0.0,
            'res_ref': 1.0,
            'lower': None,
            'upper': None,
            'shape_by_conn': False,
            'compute_shape': None,
            'copy_shape': None,
        }

        self._static_var_rel2meta[name] = metadata
        self._static_var_rel_names['output'].append(name)
        self._var_added(name)
        return metadata
