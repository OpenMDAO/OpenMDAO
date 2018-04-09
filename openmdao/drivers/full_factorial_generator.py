"""
Full Factorial case generator for Design-of-Experiments Driver.
"""
import itertools
from six import iteritems

import numpy as np

from collections import OrderedDict

from openmdao.drivers.doe_driver import DOEGenerator


class FullFactorialGenerator(DOEGenerator):
    """
    DOE case generator implementing the Full Factorial method.

    Attributes
    ----------
    _levels : int
        The number of evenly spaced levels between each design variable
        lower and upper bound.
    """

    def __init__(self, levels=1):
        """
        Initialize the FullFactorialGenerator.

        Parameters
        ----------
        levels : int, optional
            The number of evenly spaced levels between each design variable
            lower and upper bound. Defaults to 1.
        """
        super(FullFactorialGenerator, self).__init__()
        self._levels = levels

    def __call__(self, design_vars):
        """
        Generate case.

        Parameters
        ----------
        design_vars : dict
            Dictionary of design variables for which to generate values.

        Yields
        ------
        list
            list of name, value tuples for the design variables.
        """
        values = OrderedDict()

        for name, meta in iteritems(design_vars):
            values[name] = []

            size = meta['size']

            for k in range(size):
                lower = meta['lower']
                if isinstance(lower, np.ndarray):
                    lower = lower[k]

                upper = meta['upper']
                if isinstance(upper, np.ndarray):
                    upper = upper[k]

                values[name].append(np.linspace(lower, upper, num=self._levels).tolist())

        keys = values.keys()

        for name in keys:
            values[name] = [np.array(x) for x in itertools.product(*values[name])]

        for combination in itertools.product(*values.values()):
            yield zip(keys, combination)
