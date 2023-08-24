"""Define the SubmodelComp class for evaluating OpenMDAO systems within components."""

from openmdao.core.constants import _SetupStatus, INF_BOUND
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.general_utils import find_matches
from openmdao.utils.reports_system import clear_reports
from openmdao.utils.mpi import MPI, FakeComm


class SubmodelComp(ExplicitComponent):
    """
    System level container for systems.

    Parameters
    ----------
    problem : <Problem>
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
    reports : bool
        Determines if reports should be include in subproblem. Default is False because
        submodelcomp runs faster without reports.
    **kwargs : named args
        All remaining named args that become options for `SubmodelComp`.

    Attributes
    ----------
    _subprob : <Problem>
        Instantiated problem used to run the model.
    submodel_inputs : dict
        Inputs to be used as inputs in the subproblem's system.
    submodel_outputs : dict
        Outputs to be used as outputs in the subproblem's system.
    _static_submodel_inputs : dict
        Inputs passed into __init__ to be used as inputs in the subproblem's system. These
        must be bookkept separately from submodel inputs added at setup time because setup
        can be called multiple times and the submodel inputs dict is reset each time.
    _static_submodel_outputs : dict
        Outputs passed into __init__ to be used as outputs in the subproblem's system. These
        must be bookkept separately from submodel outputs added at setup time because setup
        can be called multiple times and the submodel outputs dict is reset each time.
    """

    def __init__(self, problem, inputs=None, outputs=None, reports=False, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        if not reports:
            clear_reports(problem)
        self._subprob = problem

        self.submodel_inputs = {}
        self.submodel_outputs = {}
        self._static_submodel_inputs = {}
        self._static_submodel_outputs = {}

        if inputs is not None:
            for inp in inputs:
                if isinstance(inp, str):
                    self._add_static_input(inp)
                elif isinstance(inp, tuple):
                    self._add_static_input(inp[0], name=inp[1])
                else:
                    raise Exception(f'Expected input of type str or tuple, got {type(inp)}.')

        if outputs is not None:
            for out in outputs:
                if isinstance(out, str):
                    self._add_static_output(out)
                elif isinstance(out, tuple):
                    self._add_static_output(out[0], name=out[1])
                else:
                    raise Exception(f'Expected output of type str or tuple, got {type(out)}.')

    def _add_static_input(self, path, name=None, **kwargs):
        if name is None:
            name = path.replace('.', ':')
        self._static_submodel_inputs[path] = {'iface_name': name, **kwargs}

    def _add_static_output(self, path, name=None, **kwargs):
        if name is None:
            name = path.replace('.', ':')
        self._static_submodel_outputs[path] = {'iface_name': name, **kwargs}

    def add_input(self, path, name=None, **kwargs):
        """
        Add input to model before or after setup.

        Parameters
        ----------
        path : str
            Absolute path name of input.
        name : str or None
            Name of input to be added. If none, it will default to the var name after
            the last '.'.
        **kwargs : named args
            All remaining named args that can become options for `add_input`.
        """
        if self._static_mode:
            self._add_static_input(path, name, **kwargs)
            return

        if name is None:
            name = path.replace('.', ':')

        self.submodel_inputs[path] = {'iface_name': name, **kwargs}

        # if the submodel is not set up fully, then self._problem_meta will be None
        # in which case we only want to add inputs to self.submodel_inputs
        if self._problem_meta is None:
            return

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_input after configure.')

        meta = self.boundary_inputs[path]

        # if the user wants to change some meta data like val, units, etc. they can update it here
        for key, val in kwargs.items():
            meta[key] = val

        meta.pop('prom_name')
        super().add_input(name, **meta)
        meta['prom_name'] = path

    def add_output(self, path, name=None, **kwargs):
        """
        Add output to model before or after setup.

        Parameters
        ----------
        path : str
            Absolute path name of output.
        name : str or None
            Name of output to be added. If none, it will default to the var name after
            the last '.'.
        **kwargs : named args
            All remaining named args that can become options for `add_output`.
        """
        if self._static_mode:
            self._add_static_output(path, name, **kwargs)
            return

        if name is None:
            name = path.replace('.', ':')

        self.submodel_outputs[path] = {'iface_name': name, **kwargs}

        # if the submodel is not set up fully, then self._problem_meta will be None
        # in which case we only want to add outputs to self.submodel_outputs
        if self._problem_meta is None:
            return

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_output after configure.')

        meta = self.all_outputs[path]

        for key, val in kwargs.items():
            meta[key] = val

        meta.pop('prom_name')
        super().add_output(name, **meta)
        meta['prom_name'] = path

    def _reset_driver_vars(self):
        # NOTE driver var names can be different from those in model
        # this causes some problems, so this function is used to
        # reset the driver vars so the inner problem only uses
        # the model vars
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

        # make sure comm is correct or at least reasonable.  In cases
        # where the submodel comp setup() is being called from the parent
        # setup(), our comm will be None, and we don't want to use the
        # parent's comm because it could be too big if we're under a ParallelGroup.
        # If our comm is not None then we'll just set the problem comm to ours.
        if self.comm is None:
            p.comm = FakeComm()
        else:
            p.comm = self.comm

        # if subprob.setup is called before the top problem setup, we can't rely
        # on using the problem meta data, so default to False
        if self._problem_meta is None:
            p.setup(force_alloc_complex=False)
        else:
            p.setup(force_alloc_complex=self._problem_meta['force_alloc_complex'])
        p.final_setup()

        # store prom name differently based on if the var is an input, output, or auto_ivc
        # because list_indep_vars doesn't have prom_name as part of its meta data
        # TODO some of this might not be necessary... need to revisit
        self.boundary_inputs = {}
        for name, meta in p.list_indep_vars(out_stream=None):
            if name in p.model._var_abs2prom['input']:
                meta['prom_name'] = p.model._var_abs2prom['input'][name]
            elif name in p.model._var_abs2prom['output']:
                meta['prom_name'] = p.model._var_abs2prom['output'][name]
            elif p.model.get_source(name).startswith('_auto_ivc.'):
                meta['prom_name'] = name
            else:
                raise Exception(f'var {name} not in meta data')
            self.boundary_inputs[name] = meta

        self.all_outputs = {}
        outputs = p.model.list_outputs(out_stream=None, prom_name=True,
                                       units=True, shape=True, desc=True,
                                       all_procs=True)

        # turn outputs into dict
        for _, meta in outputs:
            self.all_outputs[meta['prom_name']] = meta

        self.submodel_inputs = {}
        self.submodel_outputs = {}
        boundary_keys = list(self.boundary_inputs.keys())

        for var, meta in self._static_submodel_inputs.items():
            if '*' in var:
                matches = find_matches(var, boundary_keys)
                if len(matches) == 0:
                    raise Exception(f"Pattern '{var}' not found in model")
                for match in matches:
                    self.submodel_inputs[match] = {'iface_name': match.replace('.', ':')}
            else:
                self.submodel_inputs[var] = meta.copy()

        for var, meta in self._static_submodel_outputs.items():
            if '*' in var:
                matches = find_matches(var, self.all_outputs)
                if len(matches) == 0:
                    raise Exception(f"Pattern '{var}' not found in model")
                for match in matches:
                    self.submodel_outputs[match] = {'iface_name': match.replace('.', ':')}
            else:
                self.submodel_outputs[var] = meta.copy()

        # NOTE iface_name is what the outer problem knows the variable to be
        # it can't be the same name as the prom name in the inner variable because
        # component var names can't include '.'
        for var, data in sorted(self.submodel_inputs.items(), key=lambda x: x[0]):
            iface_name = data['iface_name']
            if iface_name in self._static_var_rel2meta or iface_name in self._var_rel2meta:
                continue
            prom_name = var
            try:
                meta = self.boundary_inputs[p.model.get_source(prom_name)] \
                    if not p.model.get_source(prom_name).startswith('_auto_ivc.') \
                    else self.boundary_inputs[prom_name]
            except Exception:
                raise Exception(f'Variable {prom_name} not found in model')
            meta.pop('prom_name')

            for key, val in data.items():
                if key == 'iface_name':
                    continue
                meta[key] = val

            super().add_input(iface_name, **meta)
            meta['prom_name'] = prom_name

        for var, data in sorted(self.submodel_outputs.items(), key=lambda x: x[0]):
            iface_name = data['iface_name']
            if iface_name in self._static_var_rel2meta or iface_name in self._var_rel2meta:
                continue
            prom_name = var
            try:
                meta = self.all_outputs[prom_name]
            except Exception:
                raise Exception(f'Variable {prom_name} not found in model')
            meta.pop('prom_name')

            for key, val in data.items():
                if key == 'iface_name':
                    continue
                meta[key] = val

            super().add_output(iface_name, **meta)
            meta['prom_name'] = prom_name

        # NOTE to be looked at later. Trying to get variables from subsystems has been causing
        # issues and is a goal for a future version
        #
        # driver_vars = p.list_problem_vars(out_stream=None,
        #                                   desvar_opts = ['lower', 'upper', 'ref', 'ref0',
        #                                                  'indices', 'adder', 'scaler',
        #                                                  'parallel_deriv_color',
        #                                                  'cache_linear_solution', 'units',
        #                                                  'min', 'max'],
        #                                   cons_opts = ['lower', 'upper', 'equals', 'ref',
        #                                                'ref0', 'indices', 'adder', 'scaler',
        #                                                'linear', 'parallel_deriv_color',
        #                                                'cache_linear_solution', 'units',
        #                                                'min', 'max'],
        #                                   objs_opts = ['ref', 'ref0', 'indices', 'adder',
        #                                               'scaler', 'units', 'parallel_deriv_color',
        #                                               'cache_linear_solution'])
        #
        # self.driver_dvs = driver_vars['design_vars']
        # self.driver_cons = driver_vars['constraints']
        # self.driver_objs = driver_vars['objectives']
        #
        # for name, dv_meta in self.driver_dvs:
        #     prom_name = self.boundary_inputs[name]['prom_name']
        #     iface_name = prom_name.replace('.', ':')
        #     if self.is_set_up and iface_name in self._var_allprocs_prom2abs_list['input']:
        #         continue
        #     self.submodel_inputs[prom_name] = iface_name
        #     # p.model._var_allprocs_prom2abs_list[iface_name] = name
        #
        #     meta = self.boundary_inputs[name]
        #     meta.pop('prom_name')
        #     super().add_input(iface_name, **meta)
        #     meta['prom_name'] = prom_name
        #     meta['abs_name'] = name
        #
        #     size = dv_meta.pop('size')
        #     val = dv_meta.pop('val')
        #     dv_meta['indices'] = dv_meta['indices'].as_array() \
        #                              if dv_meta['indices'] is not None else None
        #     dv_meta['name'] = prom_name
        #     self.add_design_var(**dv_meta)
        #     dv_meta['size'] = size
        #     dv_meta['val'] = val
        #
        # for name, con_meta in self.driver_cons:
        #     prom_name = self.all_outputs[name]['prom_name']
        #     # prom_name = con_meta['name']
        #     iface_name = prom_name.replace('.', ':')
        #     if self.is_set_up and iface_name in self._var_allprocs_prom2abs_list['output']:
        #         continue
        #     self.submodel_outputs[prom_name] = iface_name
        #     # p.model._var_allprocs_prom2abs_list[iface_name] = name
        #
        #     meta = self.all_outputs[name]
        #     meta.pop('prom_name')
        #     super().add_output(iface_name, **meta)
        #     meta['prom_name'] = prom_name
        #     meta['abs_name'] = name
        #
        #     size = con_meta.pop('size')
        #     val = con_meta.pop('val')
        #     con_meta['indices'] = con_meta['indices'].as_array() \
        #                               if con_meta['indices'] is not None else None
        #     con_meta['lower'] = None if con_meta['lower'] == -INF_BOUND else con_meta['lower']
        #     con_meta['upper'] = None if con_meta['upper'] == INF_BOUND else con_meta['upper']
        #     con_meta['name'] = prom_name
        #     self.add_constraint(**con_meta)
        #     con_meta['size'] = size
        #     con_meta['val'] = val
        #
        # for name, obj_meta in self.driver_objs: #.items():
        #     prom_name = self.all_outputs[name]['prom_name']
        #     # prom_name = obj_meta['name']
        #     iface_name = prom_name.replace('.', ':')
        #     if self.is_set_up and iface_name in self._var_allprocs_prom2abs_list['output']:
        #         continue
        #     self.submodel_outputs[prom_name] = iface_name
        #     # p.model._var_allprocs_prom2abs_list[iface_name] = name
        #
        #     meta = self.all_outputs[name]
        #     meta.pop('prom_name')
        #     super().add_output(iface_name, **meta)
        #     meta['prom_name'] = prom_name
        #     meta['abs_name'] = name
        #
        #     size = obj_meta.pop('size')
        #     val = obj_meta.pop('val')
        #     indices = obj_meta.pop('indices')
        #     obj_meta['index'] = int(indices.as_array()[0]) if indices is not None else None
        #     obj_meta['name'] = prom_name
        #     self.add_objective(**obj_meta)
        #     obj_meta['size'] = size
        #     obj_meta['val'] = val
        #     obj_meta['indices'] = indices
        #     obj_meta.pop('index')

    def _setup_var_data(self):
        super()._setup_var_data()

        p = self._subprob
        inputs = self._var_rel_names['input']
        outputs = self._var_rel_names['output']

        if len(inputs) == 0 or len(outputs) == 0:
            return

        for prom_name in sorted(self.submodel_inputs.keys()):
            if prom_name in p.model._static_design_vars or prom_name in p.model._design_vars:
                continue
            p.model.add_design_var(prom_name)

        for prom_name in sorted(self.submodel_outputs.keys()):
            # got abs name back for self._cons key for some reason in `test_multiple_setups`
            # TODO look into this
            if prom_name in p.model._responses:
                continue
            p.model.add_constraint(prom_name)

        # setup again to compute coloring
        if self._problem_meta is None:
            p.setup(force_alloc_complex=False)
        else:
            p.setup(force_alloc_complex=self._problem_meta['force_alloc_complex'])
        p.final_setup()

        self.coloring = p.driver._get_coloring(run_model=True)
        if self.coloring is not None:
            self.coloring._col_vars = list(p.driver._designvars)

        # self._reset_driver_vars()

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

        for prom_name, meta in self.submodel_inputs.items():
            p.set_val(prom_name, inputs[meta['iface_name']])

        # set initial output vals
        for prom_name, meta in self.submodel_outputs.items():
            p.set_val(prom_name, outputs[meta['iface_name']])

        p.driver.run()

        for prom_name, meta in self.submodel_outputs.items():
            outputs[meta['iface_name']] = p.get_val(prom_name)

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

        for prom_name, meta in self.submodel_inputs.items():
            p.set_val(prom_name, inputs[meta['iface_name']])

        wrt = list(self.submodel_inputs.keys())
        of = list(self.submodel_outputs.keys())

        tots = p.driver._compute_totals(wrt=wrt,
                                        of=of,
                                        use_abs_names=False, driver_scaling=False)

        if self.coloring is None:
            for (tot_output, tot_input), tot in tots.items():
                input_iface_name = self.submodel_inputs[tot_input]['iface_name']
                output_iface_name = self.submodel_outputs[tot_output]['iface_name']
                partials[output_iface_name, input_iface_name] = tot
        else:
            for of, wrt, nzrows, nzcols, _, _, _, _ in self.coloring._subjac_sparsity_iter():
                partials[of, wrt] = tots[of, wrt][nzrows, nzcols].ravel()
