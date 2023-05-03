"""Define the SubmodelComp class for evaluating OpenMDAO systems within components."""

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.general_utils import find_matches
from openmdao.core.constants import _SetupStatus, INF_BOUND
# from openmdao.utils.indexer import slicer


class SubmodelComp(ExplicitComponent):
    """
    System level container for systems.

    Parameters
    ----------
    model : <System>
        The system-level <System>.
    problem : object
        Instantiated problem to use for the model.
    inputs : list of str or tuple or None
        List of provided input names in str or tuple form. If an element is a str,
        then it should be the promoted name in its group. If it is a tuple,
        then the first element should be the group's promoted name, and the
        second element should be the var name you wish to refer to it within the subproblem
        [e.g. (path.to.var, desired_name)].
    outputs : list of str or tuple or None
        List of provided output names in str or tuple form. If an element is a str,
        then it should be the promoted name in its group. If it is a tuple,
        then the first element should be the group's promoted name, and the
        second element should be the var name you wish to refer to it within the subproblem
        [e.g. (path.to.var, desired_name)].
    **kwargs : named args
        All remaining named args that become options for `SubmodelComp`.

    Attributes
    ----------
    model : <System>
        The system being analyzed in subproblem.
    _subprob : object
        Instantiated problem used to run the model.
    submodel_inputs : list of tuple
        List of inputs requested by user to be used as inputs in the
        subproblem's system.
    submodel_outputs : list of tuple
        List of outputs requested by user to be used as outputs in the
        subproblem's system.
    is_set_up : bool
        Flag to determne if subproblem is set up. Used for add_input/add_output to
        determine how to add the io.
    """

    def __init__(self, model, problem, inputs=None, outputs=None, **kwargs):
        """
        Initialize all attributes.
        """
        # call base class to set kwargs
        super().__init__(**kwargs)

        self.model = model
        self._subprob = problem

        # need to make submodel_inputs/outputs be lists of tuples
        # so, if a str is provided, that becomes the iface_name and
        # the prom_name
        self.submodel_inputs = {}
        self.submodel_outputs = {}

        if inputs is not None:
            for inp in inputs:
                if isinstance(inp, str):
                    self.submodel_inputs[inp] = inp
                    # if '*' in inp:
                    #     # TODO account for other wildcard chars like '?'
                    #     # NOTE special case... will take care of when we get boundary inputs
                    #     self.submodel_inputs.append(inp)
                    # else:
                    #     self.submodel_inputs.append((inp, inp))
                elif isinstance(inp, tuple):
                    # self.submodel_inputs.append(inp)
                    self.submodel_inputs[inp[0]] = inp[1]
                else:
                    raise Exception(f'Expected input of type str or tuple, got {type(inp)}.')

        if outputs is not None:
            for out in outputs:
                if isinstance(out, str):
                    self.submodel_outputs[out] = out
                    # if '*' in out:
                    #     # NOTE special case... will take care of when we get all outputs
                    #     self.submodel_outputs.append(out)
                    # else:
                    #     self.submodel_outputs.append((out, out))
                elif isinstance(out, tuple):
                    # self.submodel_outputs.append(out)
                    self.submodel_outputs[out[0]] = out[1]
                else:
                    raise Exception(f'Expected output of type str or tuple, got {type(out)}.')

        self.is_set_up = False
        # self.is_final_set_up = False

    def add_input(self, path, name=None):
        """
        Add input to model before or after setup.

        Parameters
        ----------
        path : str
            Absolute path name of input.
        name : str or None
            Name of input to be added. If none, it will default to the var name after
            the last '.'.
        """
        if name is None:
            # name = path.split('.')[-1]
            name = path.replace('.', ':')

        # self.submodel_inputs.append((path, name))
        self.submodel_inputs[path] = name

        if not self.is_set_up:
            return

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_input after configure.')

        # meta = next(data for _, data in self.boundary_inputs if data['prom_name'] == path)
        meta = self.boundary_inputs[path]
        meta.pop('prom_name')
        abs_name = meta.pop('abs_name')
        super().add_input(name, **meta)
        meta['prom_name'] = path
        meta['abs_name'] = abs_name

    def add_output(self, path, name=None):
        """
        Add output to model before or after setup.

        Parameters
        ----------
        path : str
            Absolute path name of output.
        name : str or None
            Name of output to be added. If none, it will default to the var name after
            the last '.'.
        """
        if name is None:
            # name = path.split('.')[-1]
            name = path.replace('.', ':')

        # self.submodel_outputs.append((path, name))
        self.submodel_outputs[path] = name

        if not self.is_set_up:
            return

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_output after configure.')

        # meta = next(data for _, data in self.all_outputs if data['prom_name'] == path)
        meta = self.all_outputs[path]

        meta.pop('prom_name')
        abs_name = meta.pop('abs_name')
        super().add_output(name, **meta)
        meta['prom_name'] = path
        meta['abs_name'] = abs_name
    
    def _reset_driver_vars(self):
        p = self._subprob

        p.driver._designvars = {}
        p.driver._cons = {}
        p.driver._objs = {}
        p.driver._responses = {}

    def setup(self):
        """
        Perform some final setup and checks.
        """
        p = self._subprob

        # need to make sure multiple subsystems aren't added if setup more than once
        if not self.is_set_up:
            p.model.add_subsystem('subsys', self.model, promotes=['*'])

        # if subprob.setup is called before the top problem setup, we can't rely
        # on using the problem meta data, so default to False
        if self._problem_meta is None:
            p.setup(force_alloc_complex=False)
        else:
            p.setup(force_alloc_complex=self._problem_meta['force_alloc_complex'])
        p.final_setup()

        # store boundary_inputs as dict so lookup is quicker and make keys prom_names
        self.boundary_inputs = {}
        indep_vars = p.list_indep_vars(out_stream=None)
        for name, meta in indep_vars:
            if name in p.model._var_abs2prom['input']:
                meta['prom_name'] = p.model._var_abs2prom['input'][name]
            elif name in p.model._var_abs2prom['output']:
                meta['prom_name'] = p.model._var_abs2prom['output'][name]
            elif p.model.get_source(name).startswith('_auto_ivc.'):
                meta['prom_name'] = name
                # self.boundary_inputs[p.model.get_source(name)] = meta
                # continue
                # indep_vars.pop(name)
                # indep_vars.append((p.model.get_source(name), meta))
                # indep_vars[p.model.get_source(name)] = meta
            else:
                raise Exception(f'var {name} not in meta data')
            # self.boundary_inputs[meta['prom_name']] = meta
            self.boundary_inputs[name] = meta
            # meta['prom_name'] = name
            # meta['abs_name'] = name

        # self.boundary_inputs = {name: meta for name, meta in p.list_indep_vars(out_stream=None)}#, options=['name'])
        # for name, meta in self.boundary_inputs.items():
        #     meta['prom_name'] = name

        self.all_outputs = {}
        outputs = p.model.list_outputs(out_stream=None, prom_name=True,
                                       units=True, shape=True, desc=True)#,
                                    #    is_indep_var=False)
        for name, meta in outputs:
            # self.all_outputs[meta['prom_name']] = meta
            self.all_outputs[name] = meta
            # meta['abs_name'] = name
        # want all outputs from the `SubmodelComp`, including ivcs/design vars
        # self.all_outputs = {name: meta for name, meta in 
        #                     p.model.list_outputs(out_stream=None, prom_name=True,
        #                                          units=True, shape=True, desc=True,
        #                                          is_indep_var=False)}
        # add vars that follow patterns to io
        # boundary_input_prom_names = [meta['prom_name'] for _, meta in self.boundary_inputs.items()]
        # all_outputs_prom_names = [meta['prom_name'] for _, meta in self.all_outputs.items()]

        wildcard_inputs = [var for var in self.submodel_inputs.items()
                           if isinstance(var, str) and '*' in var]
        wildcard_outputs = [var for var in self.submodel_outputs.items()
                            if isinstance(var, str) and '*' in var]

        for inp in wildcard_inputs:
            matches = find_matches(inp, list(self.boundary_inputs.keys()))
            if len(matches) == 0:
                raise Exception(f'Pattern {inp} not found in model')
            for match in matches:
                # self.submodel_inputs.append((match, match))
                self.submodel_inputs[match] = match.replace('.', ':')
            # self.submodel_inputs.remove(inp)
            self.submodel_inputs.pop(inp)

        for out in wildcard_outputs:
            matches = find_matches(out, list(self.all_outputs.keys()))
            if len(matches) == 0:
                raise Exception(f'Pattern {out} not found in model')
            for match in matches:
                self.submodel_outputs.append((match, match))
            self.submodel_outputs.pop(out)

        # NOTE iface_name is what user refers to var as
        # iface_name is different from prom_name because sometimes prom_name
        # can have illegal chars in it like '.'
        for var in self.submodel_inputs.items():
            iface_name = var[1]
            prom_name = var[0]
            try:
                # meta = next(data for _, data in self.boundary_inputs
                #             if data['prom_name'] == prom_name)
                meta = self.boundary_inputs[p.model.get_source(prom_name)] \
                                if not p.model.get_source(prom_name).startswith('_auto_ivc.') \
                                else self.boundary_inputs[prom_name]
            except Exception:
                raise Exception(f'Variable {prom_name} not found in model')
            meta.pop('prom_name')
            # abs_name = meta.pop('abs_name')
            super().add_input(iface_name, **meta)
            meta['prom_name'] = prom_name
            # meta['abs_name'] = abs_name

        for var in self.submodel_outputs.items():
            iface_name = var[1]
            prom_name = var[0]
            try:
                # meta = next(data for _, data in self.all_outputs if data['prom_name'] == prom_name)
                meta = self.all_outputs[p.model.get_source(prom_name)]
            except Exception:
                raise Exception(f'Variable {prom_name} not found in model')
            meta.pop('prom_name')
            # abs_name = meta.pop('abs_name')
            super().add_output(iface_name, **meta)
            meta['prom_name'] = prom_name
            # meta['abs_name'] = abs_name

        driver_vars = p.list_problem_vars(out_stream=None,
                                          desvar_opts = ['lower', 'upper', 'ref', 'ref0',
                                                         'indices', 'adder', 'scaler',
                                                         'parallel_deriv_color',
                                                         'cache_linear_solution', 'units',
                                                         'min', 'max'],
                                          cons_opts = ['lower', 'upper', 'equals', 'ref',
                                                       'ref0', 'indices', 'adder', 'scaler',
                                                       'linear', 'parallel_deriv_color',
                                                       'cache_linear_solution', 'units',
                                                       'min', 'max'],
                                          objs_opts = ['ref', 'ref0', 'indices', 'adder',
                                                      'scaler', 'units', 'parallel_deriv_color',
                                                      'cache_linear_solution'])

        self.driver_dvs = driver_vars['design_vars']
        self.driver_cons = driver_vars['constraints']
        self.driver_objs = driver_vars['objectives']
        # self.driver_objs = {meta['name']: meta for _, meta in driver_vars['objectives']}

        self._reset_driver_vars()

        for name, dv_meta in self.driver_dvs:
            prom_name = self.boundary_inputs[name]['prom_name']
            # vvv do I need this for multiple setups? vvv
            # if prom_name in self._design_vars:
            #     continue
            iface_name = prom_name.replace('.', ':')
            self.submodel_inputs[prom_name] = iface_name

            meta = self.boundary_inputs[name]
            meta.pop('prom_name')
            super().add_input(iface_name, **meta)
            meta['prom_name'] = prom_name
            meta['abs_name'] = name
            
            dv_meta.pop('size')
            dv_meta.pop('val')
            dv_meta['indices'] = dv_meta['indices'].as_array() if dv_meta['indices'] is not None else None
            dv_meta['name'] = prom_name
            self.add_design_var(**dv_meta)

        for name, con_meta in self.driver_cons:
            prom_name = self.all_outputs[name]['prom_name']
            iface_name = prom_name.replace('.', ':')
            self.submodel_outputs[prom_name] = iface_name
            
            meta = self.all_outputs[name]
            meta.pop('prom_name')
            super().add_output(iface_name, **meta)
            meta['prom_name'] = prom_name
            meta['abs_name'] = name
            
            con_meta.pop('size')
            con_meta.pop('val')
            con_meta['indices'] = con_meta['indices'].as_array() if con_meta['indices'] is not None else None
            con_meta['lower'] = None if con_meta['lower'] == -INF_BOUND else con_meta['lower']
            con_meta['upper'] = None if con_meta['upper'] == INF_BOUND else con_meta['upper']
            con_meta['name'] = prom_name
            self.add_constraint(**con_meta)

        for name, obj_meta in self.driver_objs: #.items():
            prom_name = self.all_outputs[name]['prom_name']
            iface_name = prom_name.replace('.', ':')
            self.submodel_outputs[prom_name] = iface_name
            
            meta = self.all_outputs[name]
            meta.pop('prom_name')
            super().add_output(iface_name, **meta)
            meta['prom_name'] = prom_name
            meta['abs_name'] = name
            
            obj_meta.pop('size')
            obj_meta.pop('val')
            obj_meta['index'] = int(obj_meta.pop('indices').as_array()[0]) if obj_meta['indices'] is not None else None
            obj_meta['name'] = prom_name
            self.add_objective(**obj_meta)

        if not self.is_set_up:
            self.is_set_up = True

    def _setup_var_data(self):
        super()._setup_var_data()

        p = self._subprob
        inputs = self._var_rel_names['input']
        outputs = self._var_rel_names['output']

        if len(inputs) == 0 or len(outputs) == 0:
            return

        for prom_name in self.submodel_inputs.keys():
            # changed this for consistency
            # if prom_name in [meta['name'] for _, meta in p.driver._designvars.items()]:
            #     continue
            p.model.add_design_var(prom_name)

        for prom_name in self.submodel_outputs.keys():
            # got abs name back for self._cons key for some reason in `test_multiple_setups`
            # TODO look into this
            # if prom_name in [meta['name'] for _, meta in p.driver._cons.items()]:
            #     continue
            # alias = prom_name+'_con' if prom_name in self.driver_objs else None
            # indices = slicer[:-1] if prom_name in self.driver_objs else None
            # alias = None
            # indices = None
            p.model.add_constraint(prom_name)#, alias=alias, indices=indices)

        # setup again to compute coloring
        p.set_solver_print(-1)
        if self._problem_meta is None:
            p.setup(force_alloc_complex=False)
        else:
            p.setup(force_alloc_complex=self._problem_meta['force_alloc_complex'])
        p.final_setup()
        
        # self._reset_driver_vars()

        self.coloring = p.driver._get_coloring(run_model=True)
        if self.coloring is not None:
            self.coloring._col_vars = list(p.driver._designvars)

        if self.coloring is None:
            self.declare_partials(of='*', wrt='*')
        else:
            for of, wrt, nzrows, nzcols, _, _, _, _ in self.coloring._subjac_sparsity_iter():
                self.declare_partials(of=of, wrt=wrt, rows=nzrows, cols=nzcols)

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

        for prom_name, iface_name in self.submodel_inputs.items():
            p.set_val(prom_name, inputs[iface_name])

        # TODO figure out naming issues when using `p.driver.run()`
        # if not self.is_final_set_up:
        #     p.final_setup()
        #     self.is_final_set_up = True
        # p.final_setup()
        p.driver.run()
        # p.run_driver()

        for prom_name, iface_name in self.submodel_outputs.items():
            outputs[iface_name] = p.get_val(prom_name)

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

        for prom_name, iface_name in self.submodel_inputs.items():
            p.set_val(prom_name, inputs[iface_name])

        # wrt = [input_name for input_name, _ in self.submodel_inputs]
        # of = [output_name for output_name, _ in self.submodel_outputs]
        wrt = list(self.submodel_inputs.keys())
        of = list(self.submodel_outputs.keys())

        tots = p.driver._compute_totals(wrt=wrt,
                                        of=of,
                                        use_abs_names=False, driver_scaling=False)

        if self.coloring is None:
            for (tot_output, tot_input), tot in tots.items():
                # input_iface_name = next(iface_name for prom_name, iface_name
                #                         in self.submodel_inputs if prom_name == tot_input)
                # output_iface_name = next(iface_name for prom_name, iface_name
                #                          in self.submodel_outputs if prom_name == tot_output)
                input_iface_name = self.submodel_inputs[tot_input]
                output_iface_name = self.submodel_outputs[tot_output]
                partials[output_iface_name, input_iface_name] = tot
        else:
            for of, wrt, nzrows, nzcols, _, _, _, _ in self.coloring._subjac_sparsity_iter():
                partials[of, wrt] = tots[of, wrt][nzrows, nzcols].ravel()
