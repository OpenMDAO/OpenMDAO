import openmdao.api as om
from openmdao.core.constants import _UNDEFINED
# from openmdao.utils.om_warnings import issue_warning


class SubproblemComp(om.ExplicitComponent):
    """
    Subproblem component implementation
    """
    def __init__(self, model=None, driver=None, comm=None, name=None,
                 reports=_UNDEFINED, prob_kwargs=None, inputs=None,
                 outputs=None, **kwargs):
        
        if driver is not None:
            raise Exception('Drivers are not yet permitted for this operation'
                            ' and should be set to `None`.')
        prob_kwargs = {} if prob_kwargs is None else prob_kwargs
                
        super().__init__(**kwargs)
        
        # set options to cache input and output data
        self.options.declare('inputs', {}, types=dict,
                             desc='Subproblem Component inputs')
        self.options.declare('outputs', {}, types=dict,
                             desc='Subproblem Component outputs')
        
        p = om.Problem(model=model)
        p.setup()
        p.final_setup()
        
        # model.setup()
        # model.final_setup()
        
        model_inputs = p.model.list_inputs(out_stream=None, prom_name=True,
                                         units=True, shape=True, desc=True)
        model_outputs = p.model.list_outputs(out_stream=None, prom_name=True,
                                           units=True, shape=True, desc=True)
        
        model_inputs = {meta['prom_name']: meta for _, meta in model_inputs}
        model_outputs = {meta['prom_name']: meta for _, meta in model_outputs}

        for inp in inputs:
            if type(inp) is tuple:
                # check if variable already exists in self.options['inputs']
                inp_vars = list(self.options['inputs'].keys())
                if inp[1] in inp_vars:
                    raise Exception(f'Variable {inp[1]} already exists. Rename variable'
                                    ' or delete copy of variable.')
                
                inp_dict = {inp[1]: meta for _,meta in model_inputs.items()
                            if meta['prom_name'] == inp[0]}
                
                # if len(inp_dict) > 1:
                #     raise Exception('aklsdfjla')
                # ^^^ shouldn't be a case if promoted name is given
                #
                # check if there aren't any inputs with the given 
                # promoted name
                if len(inp_dict) == 0:
                    raise Exception(f'Promoted name {inp[0]} does not'
                                    ' exist in model.')                
                
                self.options['inputs'].update(inp_dict)
            
            elif type(inp) is str:
                # check if variable already exists in self.options['inputs']
                inp_vars = list(self.options['inputs'].keys())
                if inp in inp_vars:
                    raise Exception(f'Variable {inp} already exists. Rename variable'
                                    ' or delete copy of variable.')                
                inp_dict = {inp: meta for _,meta in model_inputs.items()
                            if meta['prom_name'].endswith(inp)}
                
                # Check if provided variable appears more than once in model
                if len(inp_dict) > 1:
                    raise Exception(f'Ambiguous variable {inp} in inputs. To'
                                    ' specify which one is desired, use a tuple'
                                    ' with the promoted name and variable name'
                                    ' instead [e.g. (prom_name, var_name)].')
                # checks if provided variable doesn't exist in model
                elif len(inp_dict) == 0:
                    raise Exception(f'Variable {inp} does not exist in model.')
                              
                self.options['inputs'].update(inp_dict)
            
            # wrong type
            else:
                raise Exception('Wrong dataype used for input.')

        for out in outputs:
            if type(out) is tuple:
                # check if variable already exists in self.options['outputs']
                out_vars = list(self.options['outputs'].keys())
                if out[1] in out_vars:
                    raise Exception(f'Variable {out[1]} already exists. Rename variable'
                                    ' or delete copy of variable.')
                
                out_dict = {out[1]: meta for _,meta in model_outputs.items()
                            if meta['prom_name'] == out[0]}

                # if len(out_dict) > 1:
                #     raise Exception('aklsdfjla')
                # ^^^ shouldn't be a case if promoted name is given
                
                # check if there aren't any outputs with the given promoted
                # name
                if len(out_dict) == 0:
                    raise Exception('aldkfjalsdj')               
                
                self.options['outputs'].update(out_dict)
            
            elif type(out) is str:
                # check if variable already exists in self.options['outputs']
                out_vars = list(self.options['outputs'].keys())
                if out in out_vars:
                    raise Exception(f'Variable {out} already exists. Rename variable'
                                    ' or delete copy of variable.')
                
                out_dict = {out: meta for _,meta in model_outputs.items()
                            if meta['prom_name'].endswith(out)}
                
                if len(out_dict) > 1:
                    raise Exception(f'Ambiguous variable {out} in outputs. To'
                                    ' specify which one is desired, use a tuple'
                                    ' with the promoted name and variable name'
                                    ' instead [e.g. (prom_name, var_name)].')
                elif len(out_dict) == 0:
                    raise Exception(f'Variable {out} does not exist in model.')
                
                self.options['outputs'].update(out_dict)
            
            # wrong type
            else:
                raise Exception('Wrong dataype used for output.')
                
        self._prev_complex_step = False
        self.prob_args = {'model' : model, 'driver' : driver, 'comm' : comm,
                          'name' : name, 'reports' : reports,'kwargs' : prob_kwargs}


    def setup(self):
        model = self.prob_args['model']
        driver = self.prob_args['driver']
        comm = self.prob_args['comm']
        name = self.prob_args['name']
        reports= self.prob_args['reports']
        prob_kwargs = self.prob_args['kwargs']
        inputs = self.options['inputs']
        outputs = self.options['outputs']
        
        p = self._subprob = om.Problem(model=model, driver=driver, comm=comm,
                                       name=name, reports=reports, **prob_kwargs)
        
        # p.model.add_subsystem('subsys', self.model, promotes=['*'])
        
        p.setup(force_alloc_complex=False)
        p.final_setup()
        
        # Lets interrogate the subsystem to figure out what its inputs and outputs are
        # inputs = p.model.list_inputs(out_stream=None, prom_name=True,
        #                              units=True, shape=True, desc=True)

        # outputs = p.model.list_outputs(out_stream=None, prom_name=True,
        #                                units=True, shape=True, desc=True)

        # Store the inputs and outputs with promoted path instead of absolute
        # inputs = {meta['prom_name']: meta for _, meta in inputs}
        # outputs = {meta['prom_name']: meta for _, meta in outputs}

        self._input_names = []
        self._output_names = []

        # Remove the `prom_name` from the metadata and then store it for each
        # input and output.
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
    
    def compute(self, inputs, outputs):
        p = self._subprob

        # Switch subproblem to use complex IO if in complex step mode.
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

        for inp in self._input_names:
            p.set_val(self.options['inputs'][inp]['prom_name'], inputs[inp])

        p.run_model()

        # do we need prom_name for get_val too?
        for op in self._output_names:
            outputs[op] = p.get_val(self.options['outputs'][op]['prom_name'])
            # outputs[op] = p.get_val(op)

    def compute_partials(self, inputs, partials):
        p = self._subprob
        for inp in self._input_names:
            p.set_val(self.options['inputs'][inp]['prom_name'], inputs[inp])

        # tots = p.compute_totals(of=self._output_names, wrt=self._input_names,
        #                         use_abs_names=False)
        # tots = p.compute_totals()

        # for of in self._output_names:
        #     for wrt in self._input_names:
        #         partials[of, wrt] = tots[of, wrt]
    
        