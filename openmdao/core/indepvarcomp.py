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
    """

    def __init__(self, name, val=1.0, **kwargs):
        """Initialize all attributes.

        Args
        ----
        name : str or [(str, value), ...] or [(str, value, kwargs), ...]
            name of the variable or list of variables.
        val : float or ndarray
            value of the variable if a single variable is being defined.
        kwargs : dict
            keyword arguments.
        """
        super(IndepVarComp, self).__init__(**kwargs)
        self._indep = (name, val)

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
        else:
            raise ValueError(
                "first argument to IndepVarComp init must be either of type "
                "`str` or an iterable of tuples of the form (name, value) or "
                "(name, value, keyword_dict).")
