"""Define the SubmodelComp class for evaluating OpenMDAO systems within components."""

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.problem import Problem
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.constants import _UNDEFINED
from openmdao.utils.general_utils import find_matches
from openmdao.core.constants import _SetupStatus
from openmdao.utils.om_warnings import issue_warning


class SubmodelComp(ExplicitComponent):
    """
    System level container for systems.

    Parameters
    ----------
    model : <System>
        The system-level <System>.
    problem : object
        Instantiated problem to use for the model
    inputs : list of str or tuple or None
        List of provided input names in str or tuple form. If an element is a str,
        then it should be the absolute name or the promoted name in its group. If it is a tuple,
        then the first element should be the absolute name or group's promoted name, and the
        second element should be the var name you wish to refer to it within the subproblem
        [e.g. (path.to.var, desired_name)].
    outputs : list of str or tuple or None
        List of provided output names in str or tuple form. If an element is a str,
        then it should be the absolute name or the promoted name in its group. If it is a tuple,
        then the first element should be the absolute name or group's promoted name, and the
        second element should be the var name you wish to refer to it within the subproblem
        [e.g. (path.to.var, desired_name)].
    **kwargs : named args
        All remaining named args that become options for `SubmodelComp`.

    Attributes
    ----------
    model : <System>
        The system being analyzed in subproblem.
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
        self.submodel_inputs = []
        self.submodel_outputs = []
        
        if inputs is not None:
            for inp in inputs:
                if isinstance(inp, str):
                    if '*' in inp:
                        # TODO account for other wildcard chars like '?'
                        # NOTE special case... will take care of when we get boundary inputs
                        self.submodel_inputs.append(inp)
                    else:
                        self.submodel_inputs.append((inp, inp))
                elif isinstance(inp, tuple):
                    # TODO add check for if tuple len > 2 or < 2
                    self.submodel_inputs.append(inp)
                else:
                    raise Exception(f'Expected input of type str or tuple, got {type(inp)}.')
        
        if outputs is not None:
            for out in outputs:
                if isinstance(out, str):
                    if '*' in out:
                        # NOTE special case... will take care of when we get all outputs
                        self.submodel_outputs.append(out)
                    else:
                        self.submodel_outputs.append((out, out))
                elif isinstance(out, tuple):
                    # TODO add check for if tuple len > 2 or < 2
                    self.submodel_outputs.append(out)
                else:
                    raise Exception(f'Expected output of type str or tuple, got {type(out)}.')

        self.is_set_up = False

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
            name = path.split('.')[-1]

        self.submodel_inputs.append((path, name))

        if not self.is_set_up:
            return

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_input after configure.')

        meta = next(data for _, data in self.boundary_inputs if data['prom_name'] == path)
        meta.pop('prom_name')
        super().add_input(name, **meta)
        meta['prom_name'] = path

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
        if name == None:
            name = path.split('.')[-1]

        self.submodel_outputs.append((path, name))

        if not self.is_set_up:
            return

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_output after configure.')

        meta = next(data for _, data in self.all_outputs if data['prom_name'] == path)
        meta.pop('prom_name')
        super().add_output(name, **meta)
        meta['prom_name'] = path

    def setup(self):
        """
        Perform some final setup and checks.
        """
        p = self._subprob

        # need to make sure multiple subsystems aren't added if setup more than once
        if not self.is_set_up:
            p.model.add_subsystem('subsys', self.model, promotes=['*'])

        # perform first setup to be able to get inputs and outputs
        # if subprob.setup is called before the top problem setup, we can't rely
        # on using the problem meta data, so default to False
        if self._problem_meta is None:
            p.setup(force_alloc_complex=False)
        else:
            p.setup(force_alloc_complex=self._problem_meta['force_alloc_complex'])
        p.final_setup()
        
        self.boundary_inputs = p.list_indep_vars(out_stream=None, options=['name'])
        for name, meta in self.boundary_inputs:
            meta['prom_name'] = name
        
        # want all outputs from the `SubmodelComp`, including ivcs/design vars
        self.all_outputs = p.model.list_outputs(out_stream=None, prom_name=True,
                                                units=True, shape=True, desc=True,
                                                is_indep_var=False)

        boundary_input_prom_names = [meta['prom_name'] for _, meta in self.boundary_inputs]
        all_outputs_prom_names = [meta['prom_name'] for _, meta in self.all_outputs]

        wildcard_inputs = [var for var in self.submodel_inputs if isinstance(var, str) and '*' in var]
        wildcard_outputs = [var for var in self.submodel_outputs if isinstance(var, str) and '*' in var]

        for inp in wildcard_inputs:
            matches = find_matches(inp, boundary_input_prom_names)
            if len(matches) == 0:
                raise Exception(f'Pattern {inp} not found in model')
            for match in matches:
                self.submodel_inputs.append((match, match))
            self.submodel_inputs.remove(inp)
        
        for out in wildcard_outputs:
            matches = find_matches(out, all_outputs_prom_names)
            if len(matches) == 0:
                raise Exception(f'Pattern {out} not found in model')
            for match in matches:
                self.submodel_outputs.append((match, match))
            self.submodel_outputs.remove(out)

        # NOTE interface name -> what SubmodelComp refers to the var as
        # NOTE interior name -> what the user refers to the var as
        for var in self.submodel_inputs:
            iface_name = var[1]
            prom_name = var[0]
            try:
                meta = next(data for _, data in self.boundary_inputs if data['prom_name'] == prom_name)
            except:
                raise Exception(f'Variable {prom_name} not found in model')
            meta.pop('prom_name')
            super().add_input(iface_name, **meta)
            meta['prom_name'] = prom_name

        for var in self.submodel_outputs:
            iface_name = var[1]
            prom_name = var[0]
            try:
                meta = next(data for _, data in self.all_outputs if data['prom_name'] == prom_name)
            except:
                raise Exception(f'Variable {prom_name} not found in model')
            meta.pop('prom_name')
            super().add_output(iface_name, **meta)
            meta['prom_name'] = prom_name

        if not self.is_set_up:
            self.is_set_up = True

    def _setup_var_data(self):
        super()._setup_var_data()

        p = self._subprob
        inputs = self._var_rel_names['input']
        outputs = self._var_rel_names['output']

        # can't have coloring if there are no io declared
        if len(inputs) == 0 or len(outputs) == 0:
            return

        for prom_name, _ in self.submodel_inputs:
            # changed this for consistency
            if prom_name in [meta['name'] for _, meta in p.driver._designvars.items()]:
                continue
            p.model.add_design_var(prom_name)

        for prom_name, _ in self.submodel_outputs:
            # got abs name back for self._cons key for some reasons in `test_multiple_setups`
            # TODO look into this
            if prom_name in [meta['name'] for _, meta in p.driver._cons.items()]:
                continue
            p.model.add_constraint(prom_name)

        # p.driver.declare_coloring()

        # setup again to compute coloring
        p.set_solver_print(-1)
        if self._problem_meta is None:
            p.setup(force_alloc_complex=False)
        else:
            p.setup(force_alloc_complex=self._problem_meta['force_alloc_complex'])
        p.final_setup()
        self.coloring = p.driver._get_coloring(run_model=True)
        if self.coloring is not None:
            self.coloring._col_vars = list(p.driver._designvars)

        if self.coloring is None:
            self.declare_partials(of='*', wrt='*')
        else:
            for of, wrt, nzrows, nzcols, _, _, _, _ in self.coloring._subjac_sparsity_iter():
                # TODO ADD OF AND WRT FROM _DESIGNVARS AND _RESPONSES HERE
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

        for prom_name, iface_name in self.submodel_inputs:
            p.set_val(prom_name, inputs[iface_name])

        p.driver.run()

        for prom_name, iface_name in self.submodel_outputs:
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

        for prom_name, iface_name in self.submodel_inputs:
            p.set_val(prom_name, inputs[iface_name])

        tots = p.compute_totals(wrt=[input_name for input_name, _ in self.submodel_inputs],
                                of=[output_name for output_name, _ in self.submodel_outputs],
                                use_abs_names=False, driver_scaling=False)

        if self.coloring is None:
            for (tot_output, tot_input), tot in tots.items():
                input_iface_name = next(iface_name for prom_name, iface_name in self.submodel_inputs if prom_name == tot_input)
                output_iface_name = next(iface_name for prom_name, iface_name in self.submodel_outputs if prom_name == tot_output)
                partials[output_iface_name, input_iface_name] = tot
        else:
            for of, wrt, nzrows, nzcols, _, _, _, _ in self.coloring._subjac_sparsity_iter():             
                # TODO WANT TO USE _DESIGNVARS AND _RESPONSES HERE
                partials[of, wrt] = tots[of, wrt][nzrows, nzcols].ravel()
