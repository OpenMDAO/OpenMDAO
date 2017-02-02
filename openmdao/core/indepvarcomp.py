"""Define the IndepVarComp class."""

from __future__ import division

import collections

import numpy
from six import string_types

from openmdao.core.explicitcomponent import ExplicitComponent


class IndepVarComp(ExplicitComponent):
    """Class to inherit from when all output variables are independent.

    Attributes
    ----------
    _indep : tuple
        Tuple (arg1, arg2), where arg1 is str or [(str, value), ...]
        or [(str, value, kwargs), ...] and arg 2 is value.
        The value can be float or ndarray
    _indep_external : list
        list of this component's independent variables that are declared externally
        via the add_var method.
    """

    def __init__(self, name=None, val=1.0, **kwargs):
        """Initialize all attributes.

        Args
        ----
        name : str or [(str, value), ...] or [(str, value, kwargs), ...] or None
            name of the variable or list of variables. If None, variables will be defined
            external to this class by calling the add_var method.
        val : float or ndarray
            value of the variable if a single variable is being defined.
        kwargs : dict
            keyword arguments.
        """
        super(IndepVarComp, self).__init__(**kwargs)
        self._indep = (name, val)
        self._indep_external = []

        for illegal in ('promotes', 'promotes_inputs', 'promotes_outputs'):
            if illegal in kwargs:
                raise ValueError("IndepVarComp init: '%s' is not supported "
                                 "in IndepVarComp." % illegal)

    def initialize_variables(self):
        """Define the independent variables as output variables."""
        name, val = self._indep
        kwargs = self.metadata._dict

        if isinstance(name, string_types):
            self.add_output(name, val, **kwargs)

        elif isinstance(name, collections.Iterable):
            for tup in name:
                badtup = None
                if isinstance(tup, tuple):
                    if len(tup) == 3:
                        n, v, kw = tup
                    elif len(tup) == 2:
                        n, v = tup
                        kw = {}
                    else:
                        badtup = tup
                else:
                    badtup = tup
                if badtup:
                    if isinstance(badtup, string_types):
                        badtup = name
                    raise ValueError(
                        "IndepVarComp init: arg %s must be a tuple of the "
                        "form (name, value) or (name, value, keyword_dict)." %
                        str(badtup))
                self.add_output(n, v, **kw)
        elif name is None:
            pass
        else:
            raise ValueError(
                "first argument to IndepVarComp init must be either of type "
                "`str` or an iterable of tuples of the form (name, value) or "
                "(name, value, keyword_dict).")

        for (name, val, shape, units, res_units, desc, lower, upper,
                ref, ref0, res_ref, res_ref0, var_set) in self._indep_external:
            self.add_output(name, val=val, shape=shape, units=units, res_units=res_units,
                            desc=desc, lower=lower, upper=upper, ref=ref, ref0=ref0,
                            res_ref=res_ref, res_ref0=res_ref0, var_set=var_set)

    def add_var(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                lower=None, upper=None, ref=1.0, ref0=0.0,
                res_ref=1.0, res_ref0=0.0, var_set=0):
        """Add an independent variable to this component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if indices not provided and val is not an array.
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
        res_ref0 : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 0. Default is 0.
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.
        """
        self._indep_external.append((name, val, shape, units, res_units, desc, lower, upper,
                                     ref, ref0, res_ref, res_ref0, var_set))
