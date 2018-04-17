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
        Get the name of this DOE driver and case generator.

        Returns
        -------
        str
            The name of this DOE driver and case generator.
        """
        return self._name

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        if self.options['run_parallel']:
            self._comm = problem.comm
        else:
            self._comm = None

        super(DOEDriver, self)._setup_driver(problem)

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

    def _run_case(self, case=[]):
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
                sys.stdout.flush()

            # so we can put the proper rank even when recording on rank 0
            if self._comm:
                metadata['override_rank'] = self._comm.rank

            self._metadata = metadata
            print('_run_case returning:', self._metadata)
            sys.stdout.flush()

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
                yield case

        # duplicate last case on extra procs
        extra_procs = size - (self._generator._num_samples % size)
        if rank >= (size - extra_procs):
            yield case
        else:
            raise StopIteration

    def _setup_recording(self):
        """
        Setup case recording.

        We want to gather the same variables from all processors when running
        in parallel.
        """
        super(DOEDriver, self)._setup_recording()

        if self._comm:
            # record on all procs
            # for recorder in self._rec_mgr._recorders:
            #     recorder._parallel = True
            # record all requested vars on all procs
            self._filtered_vars_to_record = \
                self._comm.bcast(self._filtered_vars_to_record, root=0)

        print('vars_to_record:', self._filtered_vars_to_record)
        sys.stdout.flush()

    def record_iteration(self):
        """
        Record an iteration of the current Driver.
        """
        if not self._rec_mgr._recorders:
            return

        print('=====================\nrecord_iteration')
        print('=====================')
        sys.stdout.flush()

        metadata = self._metadata
        print(metadata)
        sys.stdout.flush()

        # Get the data to record
        data = {}
        if self.recording_options['record_desvars']:
            # collective call that gets across all ranks
            desvars = self.get_design_var_values()
        else:
            desvars = {}

        if self.recording_options['record_responses']:
            # responses = self.get_response_values() # not really working yet
            responses = {}
        else:
            responses = {}

        if self.recording_options['record_objectives']:
            objectives = self.get_objective_values()
        else:
            objectives = {}

        if self.recording_options['record_constraints']:
            constraints = self.get_constraint_values()
        else:
            constraints = {}

        desvars = {name: desvars[name]
                   for name in self._filtered_vars_to_record['des']}
        # responses not working yet
        # responses = {name: responses[name] for name in self._filtered_vars_to_record['res']}
        objectives = {name: objectives[name]
                      for name in self._filtered_vars_to_record['obj']}
        constraints = {name: constraints[name]
                       for name in self._filtered_vars_to_record['con']}

        model = self._problem.model

        if self.recording_options['includes']:
            outputs = model._outputs
            # outputsinputs, outputs, residuals = root.get_nonlinear_vectors()
            sysvars = {}
            views = outputs._views
            for name in outputs._names:
                if name in self._filtered_vars_to_record['sys']:
                    sysvars[name] = views[name]
        else:
            sysvars = {}

        if self._comm:
            # gather all the case data to proc 0
            desvars = self._comm.gather(desvars, root=0)
            print('desvars:', desvars)
            sys.stdout.flush()
            responses = self._comm.gather(responses, root=0)
            print('responses:', responses)
            sys.stdout.flush()
            objectives = self._comm.gather(objectives, root=0)
            print('objectives:', objectives)
            sys.stdout.flush()
            constraints = self._comm.gather(constraints, root=0)
            print('constraints:', constraints)
            sys.stdout.flush()
            sysvars = self._comm.gather(sysvars, root=0)
            print('sysvars:', sysvars)
            sys.stdout.flush()
            metadata = self._comm.gather(metadata, root=0)
            print('metadata:', metadata)
            sys.stdout.flush()

            # on proc 0, record the data for all cases
            if self._comm.rank == 0:
                for i in range(self._comm.size):
                    data['des'] = desvars[i]
                    data['res'] = responses[i]
                    data['obj'] = objectives[i]
                    data['con'] = constraints[i]
                    data['sys'] = sysvars[i]

                    print(i, 'data to record:')
                    from pprint import pprint
                    pprint(data)
                    pprint(metadata[i])
                    sys.stdout.flush()

                    self._rec_mgr.record_iteration(self, data, metadata[i])
                    print(i, 'data recorded')
                    sys.stdout.flush()

        else:
            data['des'] = desvars
            data['res'] = responses
            data['obj'] = objectives
            data['con'] = constraints
            data['sys'] = sysvars

            print('data to record:', data, metadata)
            sys.stdout.flush()
            self._rec_mgr.record_iteration(self, data, metadata)
