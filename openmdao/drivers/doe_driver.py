"""
Design-of-Experiments Driver.
"""
from __future__ import print_function

import sys
import traceback
import inspect
import itertools
from itertools import chain

from six.moves import zip
from six import PY3, iteritems

import numpy as np

from collections import OrderedDict

from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.record_util import create_local_meta


class DOEGenerator(object):
    """
    Base class for a callable object that generates cases for a DOEDriver.
    """

    def __call__(self, desvars):
        """
        Generate case.

        Parameters
        ----------
        desvars : dict
            Dictionary of design variables for which to generate values.

        Returns
        -------
        dict
            Dictionary of input values for the case.
        """
        pass


class FullFactorialGenerator(DOEGenerator):
    """
    DOE case generator implementing the Full Factorial method.

    Attributes
    ----------
    _num_levels : int
        The number of evenly spaced levels between each design variable
        lower and upper bound.
    """

    def __init__(self, num_levels=1):
        """
        Constructor.

        Parameters
        ----------
        num_levels : int, optional
            The number of evenly spaced levels between each design variable
            lower and upper bound. Defaults to 1.
        """
        super(FullFactorialGenerator, self).__init__()
        self._num_levels = num_levels

    def __call__(self, desvars):
        """
        Generate case.

        Parameters
        ----------
        desvars : dict
            Dictionary of design variables for which to generate values.

        Yields
        ------
        dict
            Dictionary of input values for the case.
        """
        values = OrderedDict()

        for name, meta in iteritems(desvars):
            values[name] = []

            size = meta['size']

            for k in range(size):
                low = meta['lower']
                if isinstance(low, np.ndarray):
                    low = low[k]

                high = meta['upper']
                if isinstance(high, np.ndarray):
                    high = high[k]

                values[name].append(np.linspace(low, high, num=self._num_levels).tolist())

        keys = values.keys()

        for name in keys:
            values[name] = [np.array(x) for x in itertools.product(*values[name])]

        for combination in itertools.product(*values.values()):
            yield dict(zip(keys, combination))


class DOEDriver(Driver):
    """
    Design-of-Experiments Driver.

    Attributes
    ----------
    _generator : DOEGenerator
        The case generator
    """

    def __init__(self, generator):
        """
        Constructor.

        Parameters
        ----------
        generator : DOEGenerator
            The case generator
        """
        if not isinstance(generator, DOEGenerator):
            if inspect.isclass(generator):
                raise TypeError("DOEDriver requires an instance of DOEGenerator, "
                                "but a class object was found: %s"
                                % generator.__name__)
            else:
                raise TypeError("DOEDriver requires an instance of DOEGenerator, "
                                "but an instance of %s was found."
                                % type(generator).__name__)

        self._generator = generator

        super(DOEDriver, self).__init__()

    def run(self):
        """
        Generate cases and run the model for each set of generated input values.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        self.iter_count = 0

        # This runs a DOE in serial on a single process.
        model = self._problem.model

        for case in self._generator(self._designvars):
            metadata = self._prep_case(case, self.iter_count)

            terminate, exc = self._try_case(metadata)

            if exc is not None:
                if PY3:
                    raise exc[0].with_traceback(exc[1], exc[2])
                else:
                    # exec needed here since otherwise python3 will
                    # barf with a syntax error  :(
                    exec('raise exc[0], exc[1], exc[2]')

            self.iter_count += 1

        return False

    def _prep_case(self, case, iter_count):
        """
        Create metadata for the case and set design variables.

        Parameters
        ----------
        case : dict
            Dictionary of input values for the case.
        iter_count : int
            Keep track of iterations for case recording.

        Returns
        -------
        dict
            Information about the running of this case.
        """
        metadata = create_local_meta('DOEDriver')
        metadata['coord'] = (iter_count,)

        for dv_name in case:
            self.set_design_var(dv_name, case[dv_name])

        return metadata

    def _try_case(self, metadata):
        """
        Run case, save exception info and mark the metadata if the case fails.

        Parameters
        ----------
        metadata : dict
            Information about the running of this case.

        Returns
        -------
        bool
            Flag indicating whether or not to terminate execution
        exc_info
            Information about any Exception that occurred in running this case
        """
        terminate = False
        exc = None

        metadata['terminate'] = 0

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            try:
                failure_flag, _, _ = self._problem.model._solve_nonlinear()
                metadata['success'] = not failure_flag
            except AnalysisError:
                metadata['msg'] = traceback.format_exc()
                metadata['success'] = 0
            except Exception:
                metadata['success'] = 0
                metadata['terminate'] = 1  # tell master to stop sending cases in lb case
                metadata['msg'] = traceback.format_exc()

                print(metadata['msg'])

                # if not self._load_balance:
                exc = sys.exc_info()
                terminate = True

        return terminate, exc
