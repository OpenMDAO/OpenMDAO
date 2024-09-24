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
from openmdao.utils.om_warnings import issue_warning, DriverWarning


class AnalysisDriver(Driver):
    """
    Design-of-Experiments Driver.

    Parameters
    ----------
    samples : Sequence or None
        If given, provides a Sequence of samples (variable names and values to be tested). If None,
        samples is an empty list which may be appended.
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    _samples : Sequence
        A list of samples to be executed by the AnalysisDriver.
    _name : str
        The name used to identify this driver in recorded samples.
    _problem_comm : MPI.Comm or None
        The MPI communicator for the Problem.
    _color : int or None
        In MPI, the cached color is used to determine which samples to run on this proc.
    _indep_list : list
        List of design variables, used to compute derivatives.
    _quantities : list
        Contains the objectives plus nonlinear constraints, used to compute derivatives.
    """

    def __init__(self, samples=None, **kwargs):
        """
        Construct an AnalysisDriver.
        """
        if samples is None:
            self._samples = []
        elif isinstance(samples, Iterable) and not isinstance(samples, str):
            self._samples = samples
        else:
            raise ValueError(f'If given, samples must be Iterable but got {type(samples)}')

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
                             desc='Set to True to execute samples in parallel.')
        self.options.declare('procs_per_model', types=int, default=1, lower=1,
                             desc='Number of processors to give each model under MPI.')

    def add_response(self, name, indices=None, units=None,
                     linear=False, parallel_deriv_color=None,
                     cache_linear_solution=False, flat_indices=None, alias=None):
        r"""
        Add a response variable to the System associated with this AnalysisDriver.

        For AnalysisDriver, a response is an "output of interest" that we want to monitor
        as a result of changes made in the various samples.

        The AnalysisDriver.add_response interface does not support any optimization-centric
        arguments associated with constraints or objectives, such as scaling.

        Parameters
        ----------
        name : str
            Promoted name of the response variable in the system.
        indices : sequence of int, optional
            If variable is an array, these indicate which entries are of
            interest for this particular response.
        index : int, optional
            If variable is an array, this indicates which entry is of
            interest for this particular response.
        units : str, optional
            Units to convert to before applying scaling.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        linear : bool
            Set to True if constraint is linear. Default is False.
        parallel_deriv_color : str
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.
        flat_indices : bool
            If True, interpret specified indices as being indices into a flat source array.
        alias : str or None
            Alias for this response. Necessary when adding multiple responses on different
            indices of the same variable.
        """
        model = self._problem().model
        model.add_response(name=name, type_='obj', indices=indices,
                           linear=linear, units=units,
                           parallel_deriv_color=parallel_deriv_color,
                           cache_linear_solution=cache_linear_solution,
                           flat_indices=flat_indices, alias=alias)

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
        Generate samples and run the model for each set of generated input values.

        Rank 0 will both manage the distribution of samples to the other procs and
        serve as a worker running the samples.

        All other procs just run samples.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        comm = self._problem_comm
        model = self._problem().model
        self.result.reset()
        self.iter_count = 0

        # The variables that may be included in samples are the inputs or implicit outputs in the model.
        # Non-model-inputs would just have their value overridden when evaluating the model.
        # Implicit outputs can override the value given in the case, but it might be a useful mechanism
        # for providing an initial guess.
        model_inputs =  {meta['prom_name'] for var, meta in model.list_inputs(is_indep_var=True, out_stream=None)}
        model_implicit_outputs = {meta['prom_name'] for var, meta in model.list_outputs(explicit=False, out_stream=None)}
        self._allowable_vars = model_inputs | model_implicit_outputs

        # Run all tasks concurrently
        if MPI and comm.size > 1:
            loop = asyncio.get_event_loop()
            if comm.rank == 0:
                loop.run_until_complete(asyncio.gather(self._distribute_samples(), self._case_worker()))
            else:
                loop.run_until_complete(self._case_worker())
        else:
            for case in self._samples:
                self._run_case(case, iter_count=self.iter_count)
                self.iter_count += 1
        
        return False
    
    async def _distribute_samples(self):
        """
        Distribute samples to the workers, which includes rank 0.
        """
        comm = self._problem_comm
        # rank_cycler = itertools.cycle(range(comm.size))
        if comm.rank == 0:
            rank_cycler = itertools.cycle(range(comm.size))
            for i, case in enumerate(self._samples):
                r = next(rank_cycler)
                comm.send((case, self.iter_count), dest=r)
                self.iter_count += 1
            # samples exhausted, terminate all workers
            for i in range(comm.size):
                comm.send((None, self.iter_count), dest=i)

    async def _case_worker(self):
        """
        Wait for samples from the root proc and run them as they are received.
        """
        comm = self._problem_comm
        while True:
            # Worker receives case and iter count from rank 0
            # The iter_count is necessary to print the correct
            # iteration coordinate in the recorder.
            case, iter_count = comm.recv(source=0)
            if case is None:
                # if rank0 has sent us None, we are finished.
                break
            else:
                self._run_case(case, iter_count)

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
            if var not in self._allowable_vars:
                issue_warning(msg='Variable `{var}` is neither an independent variable\n'
                              'nor an implicit output in the model.\n'
                              'Setting its value in the case data will have no\n'
                              'impact on the outputs of the model after execution.',
                              category=DriverWarning)
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

    def _setup_recording(self):
        """
        Set up case recording.
        """
        # We don't necessarily know a-priori what variables are in our case generators.
        # Tee the samples and add the variables defined within to be recorded.
        self._samples, temp_samples = itertools.tee(self._samples)
        implicit_outputs = {meta['prom_name'] for _, meta in 
                            self._problem().model.list_outputs(explicit=False, implicit=True)}

        for case in temp_samples:
            for prom_name in case:
                if prom_name in implicit_outputs and prom_name not in self.recding_options['includes']:
                    self.recording_options['includes'].append(prom_name)
                for model_abs_name, model_prom_name in self._problem().model._var_allprocs_abs2prom['input'].items():
                    if model_prom_name == prom_name:
                        if model_abs_name not in self.recording_options['includes']:
                            self.recording_options['includes'].append(model_abs_name)

        if MPI:
            run_parallel = self.options['run_parallel']
            procs_per_model = self.options['procs_per_model']

            for recorder in self._rec_mgr:
                if run_parallel:
                    # write samples only on procs up to the number of parallel models
                    # (i.e. on the root procs for the samples)
                    if procs_per_model == 1:
                        recorder.record_on_process = True
                    else:
                        size = self._problem_comm.size // procs_per_model
                        if self._problem_comm.rank < size:
                            recorder.record_on_process = True

                elif self._problem_comm.rank == 0:
                    # if not running samples in parallel, then just record on proc 0
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
