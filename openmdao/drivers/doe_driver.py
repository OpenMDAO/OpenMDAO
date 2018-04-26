"""
Design-of-Experiments Driver.
"""
from __future__ import print_function

import sys
import traceback
import inspect

import numpy as np

from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.analysis_error import AnalysisError

from openmdao.utils.mpi import MPI


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

        Returns
        -------
        list
            list of name, value tuples for the design variables.
        """
        return []


class DOEDriver(Driver):
    """
    Design-of-Experiments Driver.

    Options
    -------
    options['parallel'] :  bool or int
        True or number of cases to run in parallel. Default is False.
        If True, cases will be run on all available processors.

    Attributes
    ----------
    _generator : DOEGenerator
        The case generator
    _name : str
        The name used to identify this driver in recorded cases.
    """

    def __init__(self, generator=None):
        """
        Constructor.

        Parameters
        ----------
        generator : DOEGenerator or None
            The case generator. If None, no cases will be generated.
        """
        if generator and not isinstance(generator, DOEGenerator):
            if inspect.isclass(generator):
                raise TypeError("DOEDriver requires an instance of DOEGenerator, "
                                "but a class object was found: %s"
                                % generator.__name__)
            else:
                raise TypeError("DOEDriver requires an instance of DOEGenerator, "
                                "but an instance of %s was found."
                                % type(generator).__name__)

        super(DOEDriver, self).__init__()

        if generator is None:
            self._generator = DOEGenerator()
            self._name = 'DOEDriver_None'
        else:
            self._generator = generator
            self._name = 'DOEDriver_' + type(generator).__name__.replace('Generator', '')

        self.options.declare('parallel', default=False,
                             desc='True or number of cases to run in parallel. '
                                  'If True, cases will be run on all available processors.')

    def _get_name(self):
        """
        Get the name of this DOE driver and case generator.

        Returns
        -------
        str
            The name of this DOE driver and case generator.
        """
        return self._name

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        parallel = self.options['parallel']
        if MPI and parallel:
            self._comm = comm

            size = comm.size // parallel
            color = self._color = comm.rank % size

            model_comm = comm.Split(color)
        else:
            self._comm = None
            model_comm = comm

        return model_comm

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
            self._run_case(case)
            self.iter_count += 1

        return False

    def _run_case(self, case):
        """
        Run case, save exception info and mark the metadata if the case fails.

        Parameters
        ----------
        case : list
            list of name, value tuples for the design variables.
        """
        metadata = {}

        for dv_name, dv_val in case:
            self.set_design_var(dv_name, dv_val)

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            try:
                failure_flag, _, _ = self._problem.model._solve_nonlinear()
                metadata['success'] = not failure_flag
                metadata['msg'] = ''
            except AnalysisError:
                metadata['success'] = 0
                metadata['msg'] = traceback.format_exc()
            except Exception:
                metadata['success'] = 0
                metadata['msg'] = traceback.format_exc()
                print(metadata['msg'])

            # save reference to metadata for use in record_iteration
            self._metadata = metadata

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
        size = self._comm.size // self.options['parallel']
        color = self._color

        for i, case in enumerate(self._generator(design_vars)):
            if i % size == color:
                yield case

    def record_iteration(self):
        """
        Record an iteration of the current Driver.
        """
        if not self._rec_mgr._recorders:
            return

        # Get the data to record (collective calls that get across all ranks)
        opts = self.recording_options
        filt = self._filtered_vars_to_record

        if opts['record_desvars']:
            des_vars = self.get_design_var_values(filt['des'])
        else:
            des_vars = {}

        if opts['record_objectives']:
            obj_vars = self.get_objective_values(filt['obj'])
        else:
            obj_vars = {}

        if opts['record_constraints']:
            con_vars = self.get_constraint_values(filt['con'])
        else:
            con_vars = {}

        if opts['record_responses']:
            # res_vars = self.get_response_values(filt['res'])  # not really working yet
            res_vars = {}
        else:
            res_vars = {}

        model = self._problem.model

        if opts['includes']:
            outputs = model._outputs
            views = outputs._views
            sys_vars = {name: views[name] for name in outputs._names if name in filt['sys']}
        else:
            sys_vars = {}

        data = {}
        data['des'] = des_vars
        data['res'] = res_vars
        data['obj'] = obj_vars
        data['con'] = con_vars
        data['sys'] = sys_vars

        self._rec_mgr.record_iteration(self, data, self._metadata)
