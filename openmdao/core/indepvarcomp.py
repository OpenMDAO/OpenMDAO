"""Define the IndepVarComp class."""

from __future__ import division

import collections

from six import string_types

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.general_utils import warn_deprecation


class IndepVarComp(ExplicitComponent):
    """
    Class to use when all output variables are independent.

    Attributes
    ----------
    _indep : tuple
        List of tuples of the form [(str, value, kwargs), ...].
        The value can be float or ndarray, and kwargs is a dictionary
    _indep_external : list
        list of this component's independent variables that are declared externally
        via the add_output method.
    _indep_external_discrete : list
        list of this component's discrete independent variables that are declared externally
        via the add_discrete_output method.
    """

    def __init__(self, name=None, val=1.0, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        name : str or None or [(str, value), ...] or [(str, value, kwargs), ...]
            name of the variable.
            If None, variables should be defined external to this class by calling add_output.
            For backwards compatibility with OpenMDAO v1, this can also be a list of tuples
            in the case of declaring multiple variables at once.
        val : float or ndarray
            value of the variable if a single variable is being defined.
        **kwargs : dict
            keyword arguments.
        """
        super(IndepVarComp, self).__init__()
        self._indep = []
        self._indep_external = []
        self._indep_external_discrete = []

        # A single variable is declared during instantiation
        if isinstance(name, string_types):
            self._indep.append((name, val, kwargs))
        # Mutiple variables are declared during instantiation (deprecated)
        elif isinstance(name, collections.Iterable):
            warn_deprecation('Declaring multiple variables in this way is deprecated. '
                             'In OpenMDAO 2.x or later, multiple variables should be declared '
                             'as separate add_output calls.')

            # Loop through each variable (i.e., each tuple)
            for tup in name:
                # If valid tuple, assign to (name, val, kwargs); otherwise, raise an exception
                if isinstance(tup, tuple) and len(tup) == 3:
                    name_, val, kwargs = tup
                elif isinstance(tup, tuple) and len(tup) == 2:
                    name_, val = tup
                    kwargs = {}
                else:
                    raise ValueError(
                        "IndepVarComp init: arg %s must be a tuple of the "
                        "form (name, value) or (name, value, keyword_dict)." %
                        str(tup))
                self._indep.append((name_, val, kwargs))
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

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        for (name, val, kwargs) in self._indep + self._indep_external:
            super(IndepVarComp, self).add_output(name, val, **kwargs)

        for (name, val, kwargs) in self._indep_external_discrete:
            super(IndepVarComp, self).add_discrete_output(name, val, **kwargs)

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=1.0):
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
            when the scaled value is 1. Default is 1.
        """
        kwargs = {'shape': shape, 'units': units, 'res_units': res_units, 'desc': desc,
                  'lower': lower, 'upper': upper, 'ref': ref, 'ref0': ref0,
                  'res_ref': res_ref}
        self._indep_external.append((name, val, kwargs))

    def add_discrete_output(self, name, val, desc=''):
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
        """
        kwargs = {'desc': desc}
        self._indep_external_discrete.append((name, val, kwargs))

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
