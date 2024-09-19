"""
Design-of-Experiments Driver.
"""
from collections.abc import Iterable
import asyncio
import itertools
import traceback

from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.analysis_error import AnalysisError

from openmdao.utils.mpi import MPI


class AnalysisDriver(Driver):
    """
    Design-of-Experiments Driver.

    Parameters
    ----------
    cases : Sequence or None
        If given, provides a Sequence of cases (variable names and values to be tested). If None,
        cases is an empty list which may be appended.
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    cases : Sequence
        A list o cases to be executed by the AnalysisDriver.
    _name : str
        The name used to identify this driver in recorded cases.
    _problem_comm : MPI.Comm or None
        The MPI communicator for the Problem.
    _color : int or None
        In MPI, the cached color is used to determine which cases to run on this proc.
    _indep_list : list
        List of design variables, used to compute derivatives.
    _quantities : list
        Contains the objectives plus nonlinear constraints, used to compute derivatives.
    """

    def __init__(self, cases=None, **kwargs):
        """
        Construct A DOEDriver.
        """
        if cases is None:
            self.cases = []
        elif isinstance(cases, Iterable) and not isinstance(cases, str):
            self.cases = list(cases)
        else:
            raise ValueError(f'If given, cases must be Iterable but got {type(cases)}')

        super().__init__(**kwargs)

        # What we support
        self.supports['integer_design_vars'] = True

        # What we don't support
        self.supports['distributed_design_vars'] = False
        self.supports['optimization'] = False
        self.supports._read_only = True

        self._name = 'AnalysisDriver'
        self._problem_comm = None
        self._color = None

        self._indep_list = []
        self._quantities = []
        self._total_jac_format = 'dict'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('run_parallel', types=bool, default=True,
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
        self._problem_comm = comm

        if not MPI:
            return comm
        else:
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
            return comm.Split(color)

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

        Rank 0 will both manage the distribution of cases to the other procs and
        serve as a worker running the caess.

        All other procs just run cases.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        comm = self._problem_comm
        self.result.reset()
        self.iter_count = 0

        
        # Run all tasks concurrently
        if MPI and comm.size > 1:
            loop = asyncio.get_event_loop()
            if comm.rank == 0:
                loop.run_until_complete(asyncio.gather(self._distribute_cases(), self._case_worker()))
            else:
                loop.run_until_complete(self._case_worker())
        else:
            for case in self.cases:
                self._run_case(case, iter_count=self.iter_count)
                self.iter_count += 1
        
        return False
    
    async def _distribute_cases(self):
        """
        Distribute cases to the workers, which includes rank 0.
        """
        comm = self._problem_comm
        # rank_cycler = itertools.cycle(range(comm.size))
        if comm.rank == 0:
            rank_cycler = itertools.cycle(range(comm.size))
            for i, case in enumerate(self.cases):
                r = next(rank_cycler)
                print(f'distributing case {i} to rank {r}')
                comm.send((case, self.iter_count), dest=r)
                self.iter_count += 1
            # Cases exhausted, terminate all workers
            for i in range(comm.size):
                comm.send((None, self.iter_count), dest=i)

    async def _case_worker(self):
        """
        Wait for cases from the root proc and run them as they are received.


        """
        comm = self._problem_comm
        while True:
            print('waiting for case')
            # Worker receives jobs from rank 0
            case, iter_count = comm.recv(source=0)
            print(f'got case to run on rank {comm.rank}', case)
            if case is None:
                break
            else:
                self._run_case(case, iter_count)
                # print('finished running the case')

    def _run_case(self, case, iter_count):
        """
        Run case, save exception info and mark the metadata if the case fails.

        Parameters
        ----------
        case : dict
            A dictionary keyed by variable name with each value being a dictionary with a 'val' key, and optionally
            keys for 'units' and 'indices'.
        iter_count : int
            The iteration of the AnalysisDriver to which this case corresponds.
        """
        metadata = {}
                
        for var, meta in case.items():
            val = meta['val']
            units = meta.get('units', None)
            idxs = meta.get('indices', None)
            self._problem().model.set_val(var, val, units, idxs)
                
        with RecordingDebugging(self._get_name(), iter_count, self):
            try:
                self._run_solve_nonlinear()
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

        if self.recording_options['record_derivatives']:
            self._compute_totals(of=self._quantities,
                                 wrt=self._indep_list,
                                 return_format=self._total_jac_format,
                                 driver_scaling=False)
        
        print(f'on rank {self._problem_comm.rank} the result is {self._problem().get_val("f_xy")}')

    # def _parallel_generator(self, design_vars, model=None):
    #     """
    #     Generate case for this processor when running under MPI.

    #     Parameters
    #     ----------
    #     design_vars : dict
    #         Dictionary of design variables for which to generate values.

    #     model : Group
    #         The model containing the design variables (used by some generators).

    #     Yields
    #     ------
    #     list
    #         list of name, value tuples for the design variables.
    #     """
    #     size = self._problem_comm.size // self.options['procs_per_model']
    #     color = self._color

    #     generator = self.options['generator']
    #     for i, case in enumerate(generator(design_vars, model)):
    #         if i % size == color:
    #             yield case

    def _setup_recording(self):
        """
        Set up case recording.
        """
        if MPI:
            run_parallel = self.options['run_parallel']
            procs_per_model = self.options['procs_per_model']

            for recorder in self._rec_mgr:
                if run_parallel:
                    # write cases only on procs up to the number of parallel models
                    # (i.e. on the root procs for the cases)
                    if procs_per_model == 1:
                        recorder.record_on_process = True
                    else:
                        size = self._problem_comm.size // procs_per_model
                        if self._problem_comm.rank < size:
                            recorder.record_on_process = True

                elif self._problem_comm.rank == 0:
                    # if not running cases in parallel, then just record on proc 0
                    recorder.record_on_process = True

        super()._setup_recording()

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
