"""Define the IndepVarComp class."""

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.general_utils import make_set


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
        super(IndepVarComp, self).__init__()

        if 'tags' not in kwargs:
            kwargs['tags'] = {'indep_var'}
        else:
            kwargs['tags'] = make_set(kwargs['tags'], name='tags') | {'indep_var'}

        # A single variable is declared during instantiation
        if isinstance(name, str):
            super(IndepVarComp, self).add_output(name, val, **kwargs)

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

    def _configure_check(self):
        """
        Do any error checking on i/o configuration.
        """
        if len(self._static_var_rel2meta) + len(self._var_rel2meta) == 0:
            raise RuntimeError("{}: No outputs (independent variables) have been declared. "
                               "They must either be declared during "
                               "instantiation or by calling add_output or add_discrete_output "
                               "afterwards.".format(self.msginfo))

        super(IndepVarComp, self)._configure_check()

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=None, tags=None):
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
        res_units : str or None
            Units in which the residuals of this output will be given to the user when requested.
            Default is None, which means it has no units.
        desc : str
            description of the variable
        lower : float or list or tuple or ndarray or None
            lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or None
            upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        ref : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        res_ref : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is None, which means residual scaling matches
            output scaling.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed when calling
            list_outputs.
        """
        if res_ref is None:
            res_ref = ref

        if tags is None:
            tags = {'indep_var'}
        else:
            tags = make_set(tags) | {'indep_var'}

        kwargs = {'shape': shape, 'units': units, 'res_units': res_units, 'desc': desc,
                  'lower': lower, 'upper': upper, 'ref': ref, 'ref0': ref0,
                  'res_ref': res_ref, 'tags': tags
                  }
        super(IndepVarComp, self).add_output(name, val, **kwargs)

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
        super(IndepVarComp, self).add_discrete_output(name, val, **kwargs)

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
