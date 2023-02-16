import openmdao.api as om
from openmdao.core.constants import _UNDEFINED
from openmdao.utils.om_warnings import issue_warning
from openmdao.core.driver import Driver


class SubproblemComp(om.ExplicitComponent):
    """
    System level container for systems and drivers.
    
    Parameters
    ----------
    model : <System> or None
        The system-level <System>. Required for subproblem.
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
    inputs : list of str or tuple or None
        List of desired inputs to subproblem. If an element is a str, then it should be
        the var name in its promoted name. If it is a tuple, then the first element 
        should be the promoted name, and the second element should be the var name
        you wish to refer to it by within the subproblem [e.g. (prom_name, var_name)].
        If None, then it will automatically use all inputs in the model.
    outputs : list of str or tuple or None
        List of desired outputs from subproblem. If an element is a str, then it should be
        the var name in its promoted name. If it is a tuple, then the first element 
        should be the promoted name, and the second element should be the var name
        you wish to refer to it by within the subproblem [e.g. (prom_name, var_name)].
        If None, then it will automatically use all outputs in the model.
    **kwargs : named args
        All remaining named args that become options for `SubproblemComp`.
    """
    def __init__(self, model=None, driver=None, comm=None, name=None,
                 reports=_UNDEFINED, prob_options=None, inputs=None,
                 outputs=None, **kwargs):

        if model is None:
            raise Exception('`model` arg is required for SubproblemComp.')

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

        p = self._subprob = om.Problem(driver=driver, comm=comm, name=name,
                                       reports=reports, **prob_options)
        p.model.add_subsystem('subsys', model, promotes=['*'])

        p.setup(force_alloc_complex=False)
        p.final_setup()

        model_inputs = p.model.list_inputs(out_stream=None, prom_name=True,
                                           units=True, shape=True, desc=True)
        model_outputs = p.model.list_outputs(out_stream=None, prom_name=True,
                                             units=True, shape=True, desc=True)

        # store model inputs/outputs as dictionary with keys as the promoted name
        model_inputs = {meta['prom_name']: meta for _, meta in model_inputs}
        model_outputs = {meta['prom_name']: meta for _, meta in model_outputs}

        # use all inputs in model if none are provided or if it's requested
        if inputs is None or inputs == ['*']:
            self.options['inputs'].update(model_inputs)

        else:
            # loop through inputs and make sure they're valid for use
            for inp in inputs:
                if type(inp) is tuple:
                    # check if variable already exists in self.options['inputs']
                    # i.e. no repeated input variable names
                    inp_vars = list(self.options['inputs'].keys())
                    if inp[1] in inp_vars:
                        raise Exception(f'Variable {inp[1]} already exists. Rename variable'
                                        ' or delete copy of variable.')

                    # make dict with given var name as key and meta data from model_inputs
                    inp_dict = {inp[1]: meta for _,meta in model_inputs.items()
                                if meta['prom_name'] == inp[0]}

                    # check if dict is empty (no inputs added)
                    if len(inp_dict) == 0:
                        raise Exception(f'Promoted name {inp[0]} does not'
                                        ' exist in model.')

                    # update options inputs dict with new input dict
                    self.options['inputs'].update(inp_dict)

                elif type(inp) is str:
                    # check if variable already exists in self.options['inputs']
                    # i.e. no repeated input variable names
                    inp_vars = list(self.options['inputs'].keys())
                    if inp in inp_vars:
                        raise Exception(f'Variable {inp} already exists. Rename variable'
                                        ' or delete copy of variable.')

                    # make dict with given var name as key and meta data from model_inputs
                    inp_dict = {inp: meta for _,meta in model_inputs.items()
                                if meta['prom_name'].endswith(inp)}

                    # check if provided variable appears more than once in model
                    if len(inp_dict) > 1:
                        raise Exception(f'Ambiguous variable {inp} in inputs. To'
                                        ' specify which one is desired, use a tuple'
                                        ' with the promoted name and variable name'
                                        ' instead [e.g. (prom_name, var_name)].')

                    # checks if provided variable doesn't exist in model
                    elif len(inp_dict) == 0:
                        raise Exception(f'Variable {inp} does not exist in model.')

                    # update options inputs dict with new input dict
                    self.options['inputs'].update(inp_dict)

                else:
                    raise Exception(f'Type {type(inp)} is invalid for input. Must be'
                                    ' string or tuple.')

        # use all outputs in model if none are provided or if it's requested
        if outputs is None or outputs == ['*']:
            self.options['outputs'].update(model_outputs)

        else:
            # loop through outputs and make sure they're valid for use
            for out in outputs:
                if type(out) is tuple:
                    # check if variable already exists in self.options['outputs']
                    # i.e. no repeated output variable names
                    out_vars = list(self.options['outputs'].keys())
                    if out[1] in out_vars:
                        raise Exception(f'Variable {out[1]} already exists. Rename variable'
                                        ' or delete copy of variable.')

                    # make dict with given var name as key and meta data from model_outputs
                    out_dict = {out[1]: meta for _,meta in model_outputs.items()
                                if meta['prom_name'] == out[0]}

                    # checks if provided variable doesn't exist in model
                    if len(out_dict) == 0:
                        raise Exception(f'Variable {out[0]} does not exist in model.')

                    # update options outputs dict with new output dict
                    self.options['outputs'].update(out_dict)

                elif type(out) is str:
                    # check if variable already exists in self.options['outputs']
                    # i.e. no repeated output variable names
                    out_vars = list(self.options['outputs'].keys())
                    if out in out_vars:
                        raise Exception(f'Variable {out} already exists. Rename variable'
                                        ' or delete copy of variable.')

                    # make dict with given var name as key and meta data from model_outputs
                    out_dict = {out: meta for _,meta in model_outputs.items()
                                if meta['prom_name'].endswith(out)}

                    # check if provided variable appears more than once in model
                    if len(out_dict) > 1:
                        raise Exception(f'Ambiguous variable {out} in outputs. To'
                                        ' specify which one is desired, use a tuple'
                                        ' with the promoted name and variable name'
                                        ' instead [e.g. (prom_name, var_name)].')

                    # checks if provided variable doesn't exist in model
                    elif len(out_dict) == 0:
                        raise Exception(f'Variable {out} does not exist in model.')

                    # update options outputs dict with new output dict
                    self.options['outputs'].update(out_dict)

                else:
                    raise Exception(f'Type {type(out)} is invalid for output. Must be'
                                    ' string or tuple.')

    def setup(self):
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
            # add prom_name back
            meta['prom_name'] = prom_name
            self._input_names.append(var)

        for var, meta in outputs.items():
            prom_name = meta.pop('prom_name')
            self.add_output(var, **meta)
            # add prom_name back
            meta['prom_name'] = prom_name
            self._output_names.append(var)

            for ip in self._input_names:
                self.declare_partials(of=var, wrt=ip)
    
    def compute(self, inputs, outputs):
        p = self._subprob

        # switch subproblem to use complex IO if in complex step mode
        if self.under_complex_step != self._prev_complex_step:
            if self.under_complex_step:
                p.setup(force_alloc_complex=True)
                p.final_setup()
                p.set_complex_step_mode(True)
            else:
                p.setup()
                p.final_setup()
                p.set_complex_step_mode(False)
            self._prev_complex_step = self.under_complex_step

        # setup input values
        for inp in self._input_names:
            p.set_val(self.options['inputs'][inp]['prom_name'], inputs[inp])
            # p.set_val(inp, inputs[inp])

        # use driver if it is provided
        # if driver is None when problem is instantiated, then problem.driver is
        # an instance of Driver(). Two Driver instances are not equal, so it's
        # necessary to check the type
        if type(p.driver) is not type(Driver()):
            p.run_driver()
        else:
            p.run_model()

        # store output vars
        for op in self._output_names:
            # outputs[op] = p.get_val(self.options['outputs'][op]['prom_name'])
            outputs[op] = p.get_val(op)

    def compute_partials(self, inputs, partials):
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