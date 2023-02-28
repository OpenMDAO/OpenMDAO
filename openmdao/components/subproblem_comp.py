"""Define the SubproblemComp class for evaluating OpenMDAO systems within problems."""

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.problem import Problem
from openmdao.core.constants import _UNDEFINED
from openmdao.utils.om_warnings import issue_warning
from openmdao.core.driver import Driver


def _get_model_vars(varType, vars, model_vars):
    """
    Get the requested IO variable data from model's list of IO.

    Parameters
    ----------
    varType : str
        Specifies whether inputs or outputs are being extracted.
    vars : list of str or tuple
        List of provided variable names in str or tuple form. If an element is a str,
        then it should be the var name in its promoted name. If it is a tuple, then the
        first element should be the promoted name, and the second element should be the
        var name you wish to refer to it by within the subproblem [e.g. (prom_name, var_name)].
    model_vars : list of tuples
        List of model's variable names and meta data.

    Returns
    -------
    dict
        Dict to update `self.options` with desired IO data in `SubproblemComp`.
    """
    var_dict = {varType: {}}

    for var in vars:
        if isinstance(var, tuple):
            # check if variable already exists in var_dict[varType]
            # i.e. no repeated variable names
            if var[1] in var_dict[varType]:
                raise Exception(f'Variable {var[1]} already exists. Rename variable'
                                ' or delete copy of variable.')

            # make dict with given var name as key and meta data from model_vars
            # check if name == var[0] -> var[0] is abs name and var[1] is alias
            # check if meta['prom_name'] == var[0] -> var[0] is prom name and var[1] is alias
            tmp_dict = {var[1]: meta for name, meta in model_vars
                        if name == var[0] or meta['prom_name'] == var[0]}

            # check if dict is empty (no vars added)
            if len(tmp_dict) == 0:
                raise Exception(f'Promoted name {var[0]} does not'
                                ' exist in model.')

            var_dict[varType].update(tmp_dict)

        elif isinstance(var, str):
            # check if variable already exists in var_dict[varType]
            # i.e. no repeated variable names
            if var in var_dict[varType]:
                raise Exception(f'Variable {var} already exists. Rename variable'
                                ' or delete copy of variable.')

            # make dict with given var name as key and meta data from model_vars
            # check if name == var -> given var is abs name
            # check if meta['prom_name'] == var -> given var is prom_name
            # check if name.endswith('.' + var) -> given var is last part of abs name
            tmp_dict = {var: meta for name, meta in model_vars
                        if name == var or meta['prom_name'] == var or name.endswith('.' + var)}

            # check if provided variable appears more than once in model
            if len(tmp_dict) > 1:
                raise Exception(f'Ambiguous variable {var}. To'
                                ' specify which one is desired, use a tuple'
                                ' with the promoted name and variable name'
                                ' instead [e.g. (prom_name, var_name)].')

            # checks if provided variable doesn't exist in model
            elif len(tmp_dict) == 0:
                raise Exception(f'Variable {var} does not exist in model.')

            var_dict[varType].update(tmp_dict)

        else:
            raise Exception(f'Type {type(var)} is invalid. Must be'
                            ' string or tuple.')

    return var_dict


class SubproblemComp(ExplicitComponent):
    """
    System level container for systems and drivers.

    Parameters
    ----------
    model : <System>
        The system-level <System>.
    inputs : list of str or tuple
        List of desired inputs to subproblem. If an element is a str, then it should be
        the var name in its promoted name. If it is a tuple, then the first element
        should be the promoted name, and the second element should be the var name
        you wish to refer to it by within the subproblem [e.g. (prom_name, var_name)].
    outputs : list of str or tuple
        List of desired outputs from subproblem. If an element is a str, then it should be
        the var name in its promoted name. If it is a tuple, then the first element
        should be the promoted name, and the second element should be the var name
        you wish to refer to it by within the subproblem [e.g. (prom_name, var_name)].
    driver : <Driver> or None
        The driver for the problem. If not specified, a simple "Run Once" driver will be used.
    comm : MPI.Comm or <FakeComm> or None
        The global communicator.
    name : str or None
        Problem name. Can be used to specify a Problem instance when multiple Problems
        exist.
    reports : str, bool, None, _UNDEFINED
        If _UNDEFINED, the OPENMDAO_REPORTS variable is used. Defaults to _UNDEFINED.
        If given, reports overrides OPENMDAO_REPORTS. If boolean, enable/disable all reports.
        Since none is acceptable in the environment variable, a value of reports=None
        is equivalent to reports=False. Otherwise, reports may be a sequence of
        strings giving the names of the reports to run.
    prob_options : dict or None
        Remaining named args for problem that are converted to options.
    **kwargs : named args
        All remaining named args that become options for `SubproblemComp`.

    Attributes
    ----------
    _prev_complex_step : bool
        Flag to determine if the system will need to switch to use complex IO
        or to switch away from using complex IO.
    prob_args : dict
        Extra arguments to be passed to the problem instantiation.
    model : <System>
        The system being analyzed in subproblem.
    model_input_names : list of str or tuple
        List of inputs requested by user to be used as inputs in the
        subproblem's system.
    model_output_names : list of str or tuple
        List of outputs requested by user to be used as inputs in the
        subproblem's system.
    """

    def __init__(self, model, inputs, outputs, driver=None, comm=None,
                 name=None, reports=_UNDEFINED, prob_options=None, **kwargs):
        """
        Initialize all attributes.
        """
        # check for driver and issue warning about its current use
        # in subproblem
        if driver is not None:
            issue_warning('Driver results may not be accurate if'
                          ' derivatives are needed. Set driver to'
                          ' None if your subproblem isn\'t reliant on'
                          ' a driver.')

        # make `prob_options` empty dict to be passed as **options to problem
        # instantiation
        if prob_options is None:
            prob_options = {}

        # call base class to set kwargs
        super().__init__(**kwargs)

        # store inputs and outputs in options
        self.options.declare('inputs', {}, types=dict,
                             desc='Subproblem Component inputs')
        self.options.declare('outputs', {}, types=dict,
                             desc='Subproblem Component outputs')

        # set other variables necessary for subproblem
        self._prev_complex_step = False

        self.prob_args = {'driver': driver,
                          'comm': comm,
                          'name': name,
                          'reports': reports}

        self.prob_args.update(prob_options)

        self.model = model
        self.model_input_names = inputs
        self.model_output_names = outputs

    def setup(self):
        """
        Perform some final setup and checks.
        """
        p = self._subprob = Problem(**self.prob_args)
        p.model.add_subsystem('subsys', self.model, promotes=['*'])

        p.setup(force_alloc_complex=self._problem_meta['force_alloc_complex'])
        p.final_setup()

        model_inputs = p.model.list_inputs(out_stream=None, prom_name=True,
                                           units=True, shape=True, desc=True)
        model_outputs = p.model.list_outputs(out_stream=None, prom_name=True,
                                             units=True, shape=True, desc=True)

        self.options.update(_get_model_vars('inputs', self.model_input_names, model_inputs))
        self.options.update(_get_model_vars('outputs', self.model_output_names, model_outputs))

        inputs = self.options['inputs']
        outputs = self.options['outputs']

        # instantiate input/output name list for use in compute and
        # compute partials
        self._input_names = []
        self._output_names = []

        # remove the `prom_name` from the metadata and then store it for each
        # input and output
        for var, meta in inputs.items():
            prom_name = meta.pop('prom_name')
            self.add_input(var, **meta)
            meta['prom_name'] = prom_name
            self._input_names.append(var)

        for var, meta in outputs.items():
            prom_name = meta.pop('prom_name')
            self.add_output(var, **meta)
            meta['prom_name'] = prom_name
            self._output_names.append(var)

            for ip in self._input_names:
                self.declare_partials(of=var, wrt=ip)

    def _set_complex_step_mode(self, active):
        super()._set_complex_step_mode(active)
        self._subprob.set_complex_step_mode(active)

    def compute(self, inputs, outputs):
        """
        Perform the subproblem system computation at run time.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        """
        p = self._subprob

        # switch subproblem to use complex IO if in complex step mode
        self._set_complex_step_mode(self.under_complex_step)

        # setup input values
        for inp in self._input_names:
            p.set_val(self.options['inputs'][inp]['prom_name'], inputs[inp])

        if not isinstance(p.driver, Driver):
            p.run_driver()
        else:
            p.run_model()

        # store output vars
        for op in self._output_names:
            outputs[op] = p.get_val(self.options['outputs'][op]['prom_name'])

    def compute_partials(self, inputs, partials):
        """
        Collect computed partial derivatives and return them.

        Checks if the needed derivatives are cached already based on the
        inputs vector. Refreshes the cache by re-computing the current point
        if necessary.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Sub-jac components written to partials[output_name, input_name].
        """
        p = self._subprob
        for inp in self._input_names:
            p.set_val(self.options['inputs'][inp]['prom_name'], inputs[inp])

        # compute total derivatives for now... assuming every output is sensitive
        # to every input. Will be changed in a future version
        tots = p.compute_totals(of=self._output_names, wrt=self._input_names,
                                use_abs_names=False)

        # store derivatives in Jacobian
        for of in self._output_names:
            for wrt in self._input_names:
                partials[of, wrt] = tots[of, wrt]
