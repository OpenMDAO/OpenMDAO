"""
Case generators for Design-of-Experiments Driver using pyDOE.
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


class FullFactorialGenerator(DOEGenerator):
    """
    DOE case generator implementing the Full Factorial method.

    Attributes
    ----------
    _levels : int
        The number of evenly spaced levels between each design variable
        lower and upper bound.
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
        names = design_vars.keys()

        # generate values for each level of each design variable
        # over the range of that varable's lower to upper bound
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

        from pprint import pprint
        pprint(values)
        # convert to arrays
        for name in names:
            values[name] = [np.array(x) for x in itertools.product(*values[name])]
        pprint(values)

        # generate indices
        size = sum([meta['size'] for name, meta in iteritems(design_vars)])
        print('size:', size)
        ff = pyDOE.fullfact([self._levels]*size)
        print(ff)

        # yield values for ff generated indices
        for row in ff:

            yield [(name, values[name][i]) for name, i in zip(names, row.astype('int'))]


class pyDOEGenerator(DOEGenerator):
    """
    DOE case generator using the pyDOE package.

    See: https://pythonhosted.org/pyDOE/index.html

    Attributes
    ----------
    _num_samples : int
        The number of samples to run.
    _seed : int or None
        Random seed.
    """

    def __init__(self, num_samples=1, seed=None):
        """
        Initialize the LatinHypercubeGenerator.

        Parameters
        ----------
        num_samples : int, optional
            The number of samples to run. Defaults to 1.

        seed : int or None, optional
            Random seed. Defaults to None.
        """
        super(pyDOEGenerator, self).__init__()

        self.options.declare('design', _methods,
                             desc='The design function to use.')

        self._num_samples = num_samples
        self._seed = seed
