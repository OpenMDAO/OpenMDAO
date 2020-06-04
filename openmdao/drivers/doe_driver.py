"""
Design-of-Experiments Driver.
"""

import traceback
import inspect

from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.analysis_error import AnalysisError
from openmdao.drivers.doe_generators import DOEGenerator, ListGenerator

from openmdao.utils.mpi import MPI

from openmdao.recorders.sqlite_recorder import SqliteRecorder


class DOEDriver(Driver):
    """
    Design-of-Experiments Driver.

    Attributes
    ----------
    _name : str
        The name used to identify this driver in recorded cases.
    _recorders : list
        List of case recorders that have been added to this driver.
    _comm : MPI.Comm or None
        MPI communicator object.
    _color : int or None
        In MPI, the cached color is used to determine which cases to run on this proc.
    """

    def __init__(self, generator=None, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        generator : DOEGenerator, list or None
            The case generator or a list of DOE cases.

        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        # if given a list, create a ListGenerator
        if isinstance(generator, list):
            generator = ListGenerator(generator)

        elif generator and not isinstance(generator, DOEGenerator):
            if inspect.isclass(generator):
                raise TypeError("DOEDriver requires an instance of DOEGenerator, "
                                "but a class object was found: %s"
                                % generator.__name__)
            else:
                raise TypeError("DOEDriver requires an instance of DOEGenerator, "
                                "but an instance of %s was found."
                                % type(generator).__name__)

        super(DOEDriver, self).__init__(**kwargs)

        if generator is not None:
            self.options['generator'] = generator

        self._name = ''
        self._recorders = []
        self._comm = None
        self._color = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('generator', types=(DOEGenerator), default=DOEGenerator(),
                             desc='The case generator. If default, no cases are generated.')
        self.options.declare('run_parallel', types=bool, default=False,
                             desc='Set to True to execute cases in parallel.')
        self.options.declare('procs_per_model', types=int, default=1, lower=1,
                             desc='Number of processors to give each model under MPI.')

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
        if MPI and self.options['run_parallel']:
            self._comm = comm
            procs_per_model = self.options['procs_per_model']

            full_size = comm.size
            size = full_size // procs_per_model
            if full_size != size * procs_per_model:
                raise RuntimeError("The total number of processors is not evenly divisible by the "
                                   "specified number of processors per model.\n Provide a "
                                   "number of processors that is a multiple of %d, or "
                                   "specify a number of processors per model that divides "
                                   "into %d." % (procs_per_model, full_size))
            color = self._color = comm.rank % size

            model_comm = comm.Split(color)
        else:
            self._comm = None
            model_comm = comm

        return model_comm

    def _set_name(self):
        """
        Set the name of this DOE driver and its case generator.

        Returns
        -------
        str
            The name of this DOE driver and its case generator.
        """
        generator = self.options['generator']

        gen_type = type(generator).__name__.replace('Generator', '')
        if gen_type == 'DOEGenerator':
            self._name = 'DOEDriver'  # Empty generator
        else:
            self._name = 'DOEDriver_' + gen_type

        return self._name

    def _get_name(self):
        """
        Get the name of this DOE driver and its case generator.

        Returns
        -------
        str
            The name of this DOE driver and its case generator.
        """
        return self._name

    def run(self):
        """
        Generate cases and run the model for each set of generated input values.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        self.iter_count = 0

        # set driver name with current generator
        self._set_name()

        if self._comm:
            case_gen = self._parallel_generator
        else:
            case_gen = self.options['generator']

        for case in case_gen(self._designvars, self._problem().model):
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
            try:
                msg = None
                self.set_design_var(dv_name, dv_val)
            except ValueError as err:
                msg = "Error assigning %s = %s: " % (dv_name, dv_val) + str(err)
            finally:
                if msg:
                    raise(ValueError(msg))

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            try:
                self._problem().model.run_solve_nonlinear()
                metadata['success'] = 1
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

    def _parallel_generator(self, design_vars, model=None):
        """
        Generate case for this processor when running under MPI.

        Parameters
        ----------
        design_vars : dict
            Dictionary of design variables for which to generate values.

        model : Group
            The model containing the design variables (used by some generators).

        Yields
        ------
        list
            list of name, value tuples for the design variables.
        """
        size = self._comm.size // self.options['procs_per_model']
        color = self._color

        generator = self.options['generator']
        for i, case in enumerate(generator(design_vars, model)):
            if i % size == color:
                yield case

    def add_recorder(self, recorder):
        """
        Add a recorder to the driver.

        Parameters
        ----------
        recorder : CaseRecorder
           A recorder instance.
        """
        # keep track of recorders so we can flag them as parallel
        # if we end up running in parallel
        self._recorders.append(recorder)

        super(DOEDriver, self).add_recorder(recorder)

    def _setup_recording(self):
        """
        Set up case recording.
        """
        if MPI:
            procs_per_model = self.options['procs_per_model']

            for recorder in self._recorders:
                recorder._parallel = True

                # if SqliteRecorder, write cases only on procs up to the number
                # of parallel DOEs (i.e. on the root procs for the cases)
                if isinstance(recorder, SqliteRecorder):
                    if procs_per_model == 1:
                        recorder._record_on_proc = True
                    else:
                        size = self._comm.size // procs_per_model
                        if self._comm.rank < size:
                            recorder._record_on_proc = True
                        else:
                            recorder._record_on_proc = False

        super(DOEDriver, self)._setup_recording()

    def _get_recorder_metadata(self, case_name):
        """
        Return metadata from the latest iteration for use in the recorder.

        Parameters
        ----------
        case_name : str
            Name of current case.

        Returns
        -------
        dict
            Metadata dictionary for the recorder.
        """
        self._metadata['name'] = case_name
        return self._metadata
