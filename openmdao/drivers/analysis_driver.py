"""
AnalysisDriver definition.
"""
from collections import deque
from collections.abc import Iterable
import itertools
import traceback

from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.analysis_error import AnalysisError

from openmdao.drivers.analysis_generator import AnalysisGenerator, SequenceGenerator
from openmdao.utils.mpi import MPI
from openmdao.utils.om_warnings import issue_warning, DriverWarning


class AnalysisDriver(Driver):
    """
    A driver for repeatedly running the model with a list of sampled data.

    Samples may be provided as a Sequence of dictionaries, where each entry
    in the sequence is a dictionary keyed by the variable names to be set
    for that specific execution.

    For instance, the following sequence of samples provides 3 executions,
    testing (x=0, y=4), (x=1, y=5), and (x=2, y=6).  Units may be optionally
    provided.

    Alternatively, samples can be provided as an instance of AnalysisGenerator,
    which will provide each sample in a lazily-evaluated way.

    Parameters
    ----------
    samples : list, tuple, or AnalysisGenerator
        If given, provides a list or tuple of samples (variable names and values to be tested), or
        an AnalysisGenerator which provides samples.
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    _name : str
        The name used to identify this driver in recorded samples.
    _problem_comm : MPI.Comm or None
        The MPI communicator for the Problem.
    _color : int or None
        In MPI, the cached color is used to determine which samples to run on this proc.
    _num_colors : int
        The number of total MPI colors for the run.
    _prev_sample_vars : set
        The set of variables seen in the previous iteration of the driver on this rank.
    _generator : AnalysisGenerator
        The internal AnalysisGenerator providing samples.
    """

    def __init__(self, samples=None, **kwargs):
        """
        Construct an AnalysisDriver.
        """
        if isinstance(samples, (list, tuple)):
            self._generator = SequenceGenerator(samples)
        elif isinstance(samples, AnalysisGenerator):
            self._generator = samples
        elif samples is not None:
            raise ValueError('samples must be a list, tuple, '
                             f'or derived from AnalysisGenerator but got {type(samples)}')

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
        self._num_colors = 1
        self._prev_sample_vars = set()
        self._total_jac_format = 'dict'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('run_parallel', types=bool, default=False,
                             desc='Set to True to execute samples in parallel.')
        self.options.declare('batch_size', types=int, default=1000,
                             desc='Number of samples to distribute among the processors '
                             'at a time when run_parallel is True. This should be limited when '
                             'the memory required to store the batch size of samples grows too '
                             'large.')
        self.options.declare('procs_per_model', types=int, default=1, lower=1,
                             desc='Number of processors to give each model under MPI.')

    def add_response(self, name, indices=None, units=None,
                     linear=False, parallel_deriv_color=None,
                     cache_linear_solution=False, flat_indices=None, alias=None):
        r"""
        Add a response variable to the model associated with this AnalysisDriver.

        For AnalysisDriver, a response is an "output of interest" that we want to monitor
        as a result of changes made in the various samples.

        The AnalysisDriver.add_response interface does not support any optimization-centric
        arguments associated with constraints or objectives, such as scaling.

        Internally, the driver does add this as an 'objective' to the model for the purposes
        of tracking derivatives.

        Parameters
        ----------
        name : str
            Promoted name of the response variable in the system.
        indices : sequence of int, optional
            If variable is an array, these indicate which entries are of
            interest for this particular response.
        units : str, optional
            Units to convert to before applying scaling.
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

    def add_responses(self, responses):
        """
        Add multiple responses to be recorded by the AnalysisDriver.

        Parameters
        ----------
        responses : Sequence or dict or str
            A sequence of response names to be recorded.  If more
            metadata needs to be specified, reponses can be provided
            as a dictionary whose keys are the variables to be recorded,
            and whose associated values are dictionaries of metadata to
            be passed on as keyword arguments to add_response.
        """
        if isinstance(responses, str):
            self.add_response(responses)
        elif isinstance(responses, dict):
            for var, meta in responses.items():
                self.add_response(var, **meta)
        elif isinstance(responses, Iterable):
            for res in responses:
                if isinstance(res, str):
                    self.add_response(res)

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
        self._prev_sample_vars.clear()

        self._problem_comm = comm

        if not MPI:
            return comm
        else:
            procs_per_model = self.options['procs_per_model']

            full_size = comm.size
            self._num_colors = size = full_size // procs_per_model
            if full_size != size * procs_per_model:
                raise RuntimeError("The total number of processors is not evenly divisible by the "
                                   "specified number of processors per model.\n Provide a "
                                   f"number of processors that is a multiple of {procs_per_model}, "
                                   "or specify a number of processors per model that divides "
                                   f"into {full_size}.")

            color = self._color = comm.rank % size
            new_comm = comm.Split(color)
            return new_comm

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

        # Variables allowed samples are the inputs or implicit outputs in the model.
        # Outputs from sources other than IndepVarComps would just have their value
        # overridden when evaluating the model.
        # Implicit outputs can override the value given in the case, but it might be a
        # useful mechanism for providing an initial guess for a nonlinear solver.
        model_inputs = {meta['prom_name'] for _, meta in
                        model.list_inputs(is_indep_var=True, out_stream=None)}
        model_implicit_outputs = {meta['prom_name'] for _, meta
                                  in model.list_outputs(explicit=False, out_stream=None)}
        self._allowable_vars = model_inputs | model_implicit_outputs
        n_procs = 1 if comm is None else comm.size

        if self.options['run_parallel'] and MPI and n_procs > 1:
            batch_size = self.options['batch_size']
            color_cycler = itertools.cycle(range(self._num_colors))
            samples_complete = False
            sample_num = 0
            job_queues = None
            colors = comm.gather(self._color, root=0)

            if comm.rank == 0:
                color_to_rank_map = {num: [i for i, x in enumerate(colors)
                                     if x == num] for num in set(colors)}

            while not samples_complete:
                if comm.rank == 0:
                    job_queues = [deque() for _ in range(n_procs)]
                    # Rank 0 pushes batch_size jobs to the ranks in job_queues
                    batch_i = 0
                    while True:
                        try:
                            sample = next(self._generator)
                        except StopIteration:
                            samples_complete = True
                            break

                        color_idx = next(color_cycler)
                        for rank_idx in color_to_rank_map[color_idx]:
                            job_queues[rank_idx].appendleft((sample_num, sample))
                        if batch_i >= batch_size:
                            break
                        batch_i += 1
                        sample_num += 1

                # Broadcast the samples_complete signal from root to all ranks
                samples_complete = comm.bcast(samples_complete, root=0)

                # Scatter the job list to each rank
                q = comm.scatter(job_queues, root=0)

                # Now each proc does the jobs in its queue
                while q:
                    sample_num, sample = q.pop()
                    self._run_sample(sample, sample_num)

                # Wait for all processors to run their jobs.
                # Then repeat until samples are exhausted.
                comm.barrier()

        else:
            # Not under MPI
            for sample_num, sample in enumerate(self._generator):
                self._run_sample(sample, sample_num)

        return False

    def _run_sample(self, sample, sample_num):
        """
        Run case, save exception info and mark the metadata if the case fails.

        Parameters
        ----------
        sample : dict
            A dictionary keyed by variable name with each value being
            a dictionary with a 'val' key, and optionally keys for
            'units' and 'indices'.
        sample_num : int
            The iteration of the AnalysisDriver to which this case corresponds.
        """
        comm = self._problem_comm
        rank = 0 if self._problem_comm is None else comm.rank
        self.iter_count = sample_num
        metadata = {}

        sample_vars = set()

        for var, meta in sample.items():
            sample_vars.add(var)
            val = meta['val']
            units = meta.get('units', None)
            idxs = meta.get('indices', None)
            # If we've given the model more procs than necessary,
            # then it will not have inputs/implicit outputs on some ranks.
            # Check that self._allowable_vars is not empty before we warn.
            if self._allowable_vars and var not in self._allowable_vars:
                issue_warning(msg=f'Variable `{var}` is neither an independent variable\n'
                              f'nor an implicit output in the model on rank {rank}.\n'
                              'Setting its value in the case data will have no\n'
                              'impact on the outputs of the model after execution.',
                              category=DriverWarning)
            self._problem().model.set_val(var, val, units, idxs)

        if self._prev_sample_vars and sample_vars != self._prev_sample_vars:
            new_vars = self._prev_sample_vars - sample_vars
            missing_vars = sample_vars - self._prev_sample_vars
            info = f'Missing variables: {missing_vars}\n' if missing_vars else ''
            info += f'New variables: {new_vars}\n' if new_vars else ''
            issue_warning(msg=f'The variables in sample {sample_num} differ from\n'
                          f'the previous sample\'s variables.\n{info}',
                          category=DriverWarning)
        self._prev_sample_vars = sample_vars

        with RecordingDebugging(self._get_name(), self.iter_count, self):
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
            self._compute_totals(of=list(self._responses.keys()),
                                 wrt=list(self._get_sampled_vars()),
                                 return_format=self._total_jac_format,
                                 driver_scaling=False)

    def _get_sampled_vars(self):
        """
        Return all of the variables (promoted name) to be sampled by this driver.
        """
        if hasattr(self._generator, '_get_sampled_vars'):
            return set(self._generator._get_sampled_vars())
        elif isinstance(self._generator, (list, tuple)):
            try:
                return set(self._generator[0].keys())
            except IndexError:
                pass
        raise AttributeError('The samples for AnalysisDriver must be a list, tuple, '
                             'or an AnalysisGenerator that provides a _get_sampled_vars() method')

    def _setup_recording(self):
        """
        Set up case recording.
        """
        # We don't necessarily know a-priori what variables are in our case generators.
        # Tee the samples and add the variables defined within to be recorded.
        comm = self._problem_comm
        model = self._problem().model
        rec_includes = self.recording_options['includes']
        implicit_outputs = {meta['prom_name'] for _, meta in
                            model.list_outputs(explicit=False, implicit=True, out_stream=None)}
        resolver = model._resolver

        # Responses are recorded by default, add the inputs to be recorded.
        for prom_name in self._get_sampled_vars():
            if prom_name in implicit_outputs and prom_name not in rec_includes:
                self.recording_options['includes'].append(prom_name)
            elif resolver.is_prom(prom_name, 'input'):
                self.recording_options['includes'].append(prom_name)

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

                elif comm is None or comm.rank == 0:
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
