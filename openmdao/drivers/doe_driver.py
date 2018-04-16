"""
Design-of-Experiments Driver.
"""
from __future__ import print_function

import sys
import traceback
import inspect

from six import PY3

import numpy as np

from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.analysis_error import AnalysisError


class DOEGenerator(object):
    """
    Base class for a callable object that generates cases for a DOEDriver.

    Attributes
    ----------
    _num_samples : int
        The number of samples generated (available after generator has been called).
    """

    def __init__(self):
        """
        Initialize the DOEGenerator.
        """
        self._num_samples = 0

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
        pass


class DOEDriver(Driver):
    """
    Design-of-Experiments Driver.

    Options
    -------
    options['run_parallel'] :  bool
        If True and running under MPI, cases will run in parallel. Default is False.

    Attributes
    ----------
    _generator : DOEGenerator
        The case generator
    _name : str
        The name used to identify this driver in recorded cases.
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

        super(DOEDriver, self).__init__()

        self._generator = generator
        self._name = 'DOEDriver_' + type(generator).__name__.replace('Generator', '')

        self.options.declare('run_parallel', default=False,
                             desc='Set to True to run the cases in parallel.')

    def _get_name(self):
        """
        Get the type name of current DOE generator.

        Returns
        -------
        str
            The the type name of the current DOE generator.
        """
        return self._name

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super(DOEDriver, self)._setup_driver(problem)

        if self.options['run_parallel']:
            self._comm = self._problem.comm
        else:
            self._comm = None

    def run(self):
        """
        Generate cases and run the model for each set of generated input values.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        self.iter_count = 0

        if self._comm:
            case_gen = self._parallel_generator
        else:
            case_gen = self._generator

        for case in case_gen(self._designvars):
            metadata = self._try_case(case)
            self.iter_count += 1

        return False

    def _try_case(self, case=[]):
        """
        Run case, save exception info and mark the metadata if the case fails.

        Parameters
        ----------
        case : list
            list of name, value tuples for the design variables.

        Returns
        -------
        metadata : dict
            Information about the running of this case.
        """
        print('----------------------')
        print('running case:', case)
        sys.stdout.flush()

        terminate = False
        exc = None

        metadata = {}
        metadata['terminate'] = 0

        for dv_name, dv_val in case:
            self.set_design_var(dv_name, dv_val)

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

        print(metadata)
        # print('----------------------')
        # sys.stdout.flush()
        return metadata

    def _parallel_generator(self, design_vars):
        """
        Generate case for this processor when running under MPI.

        Parameters
        ----------
        design_vars : dict
            Dictionary of design variables for which to generate values.

        Yields
        ------
        list
            list of name, value tuples for the design variables.
        """
        rank = self._comm.rank
        size = self._comm.size

        for i, case in enumerate(self._generator(design_vars)):
            if rank == i % size:
                print('returning Case', i, 'of', self._generator._num_samples)
                sys.stdout.flush()
                yield case
            else:
                print('skipping Case', i, 'of', self._generator._num_samples)
                sys.stdout.flush()

        print('iter exhausted, i =', i, 'case =', case)
        extra_procs = size - (self._generator._num_samples % size)
        print('extra_procs =', extra_procs)
        if rank >= (size - extra_procs):
            print('I am extra')
            print(i + 1, 'of', self._generator._num_samples, 'duplicating last case')
            sys.stdout.flush()
            # duplicate last case on extra procs
            yield case
        else:
            print(i + 1, 'of', self._generator._num_samples, 'stopping iter')
            sys.stdout.flush()
            raise StopIteration
