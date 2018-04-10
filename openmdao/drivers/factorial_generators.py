"""
Factorial Case generators for Design-of-Experiments Driver using pyDOE.
"""

from six import iteritems, itervalues
from six.moves import range, zip

import numpy as np

import itertools
from six import iteritems, itervalues

from collections import OrderedDict

from openmdao.drivers.doe_driver import DOEGenerator
import pyDOE

_methods = ['fullfact', 'ff2n', 'fractfract', 'pbdesign', 'bbdesign', 'ccdesign', 'lhs']


class _FactorialGenerator(DOEGenerator):
    """
    Base class for DOE case generators implementing the pyDOE factorial methods.

    Attributes
    ----------
    _levels : int
        The number of evenly spaced levels between each design variable
        lower and upper bound.
    _supported_methods : list
        supported pyDOE function names.
    _method : string
        the pyDOE function to use.
    """

    _supported_methods = ['fullfact', 'pbdesign']

    def __init__(self, method, levels):
        """
        Initialize the FullFactorialGenerator.

        Parameters
        ----------
        levels : int, optional
            The number of evenly spaced levels between each design variable
            lower and upper bound. Defaults to 2.
        """
        super(_FactorialGenerator, self).__init__()

        if method not in ['fullfact', 'pbdesign']:
            raise RuntimeError('Invalid method specified for generator: %s. '
                               'Method must be one of %s.' %
                               (method, _supported_methods))

        self._method = method
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
        names = design_vars.keys()

        size = sum([meta['size'] for name, meta in iteritems(design_vars)])

        # generate indices
        if self._method is 'fullfact':
            ff = pyDOE.fullfact([self._levels]*size)
        elif self._method is 'pbdesign':
            ff = pyDOE.pbdesign(size)
            ff[ff < 0] = 0  # replace -1 with zero
        else:
            raise RuntimeError("Invalid method for _FactorialGenerator.")

        # generate values for each level for each design variable
        # over the range of that varable's lower to upper bound

        # rows = vars (# rows/var = var size), cols = levels
        values = np.zeros((size, self._levels))

        row = 0
        for name, meta in iteritems(design_vars):
            size = meta['size']

            for k in range(size):
                lower = meta['lower']
                if isinstance(lower, np.ndarray):
                    lower = lower[k]

                upper = meta['upper']
                if isinstance(upper, np.ndarray):
                    upper = upper[k]

                values[row][:] = np.linspace(lower, upper, num=self._levels)
                row += 1

        # yield values for ff generated indices
        for idxs in ff.astype('int'):
            retval = []
            var = row = 0
            for name, meta in iteritems(design_vars):
                size = meta['size']
                val = np.zeros(size)
                for k in range(size):
                    idx = idxs[var+k]
                    val[k] = values[row+k][idx]
                retval.append((name, val))
                var += 1
                row += size

            yield retval

class FullFactorialGenerator(_FactorialGenerator):
    """
    DOE case generator implementing the Full Factorial method.
    """

    def __init__(self, levels=2):
        """
        Initialize the FullFactorialGenerator.

        Parameters
        ----------
        levels : int, optional
            The number of evenly spaced levels between each design variable
            lower and upper bound. Defaults to 2.
        """
        super(FullFactorialGenerator, self).__init__('fullfact', levels)


class PlackettBurmanGenerator(_FactorialGenerator):
    """
    DOE case generator implementing the Plackett-Burman method.
    """

    def __init__(self):
        """
        Initialize the PlackettBurmanGenerator.
        """
        super(PlackettBurmanGenerator, self).__init__('pbdesign', 2)
