"""Define the SubmodelComp class for evaluating OpenMDAO systems within components."""

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.problem import Problem
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.constants import _UNDEFINED
from openmdao.utils.general_utils import find_matches
from openmdao.core.constants import _SetupStatus
from openmdao.utils.om_warnings import issue_warning


# def _get_model_vars(vars, model_vars):
#     """
#     Get the requested IO variable data from model's list of IO.

#     Parameters
#     ----------
#     vars : list of str or tuple
#         List of provided var names in str or tuple form. If an element is a str,
#         then it should be the absolute name or the promoted name in its group. If it is a tuple,
#         then the first element should be the absolute name or group's promoted name, and the
#         second element should be the var name you wish to refer to it within the subproblem
#         [e.g. (path.to.var, desired_name)].
#     model_vars : list of tuples
#         List of model variable absolute names and meta data.

#     Returns
#     -------
#     dict
#         Dict to update `self.options` with desired IO data in `SubmodelComp`.
#     """
#     var_dict = {}
#     tmp = {var: meta for var, meta in model_vars}

#     # check for wildcards and append them to vars list
#     patterns = [i for i in vars if isinstance(i, str)]
#     var_list = [meta['prom_name'] for _, meta in model_vars]
#     for i in patterns:
#         matches = find_matches(i, var_list)
#         if len(matches) == 0:
#             continue
#         vars.extend(matches)
#         vars.remove(i)

#     for var in vars:
#         if isinstance(var, tuple):
#             # check if user tries to use wildcard in tuple
#             if '*' in var[0] or '*' in var[1]:
#                 raise Exception('Cannot use \'*\' in tuple variable.')

#             # check if variable already exists in var_dict[varType]
#             # i.e. no repeated variable names
#             if var[1] in var_dict:
#                 continue
#                 # raise Exception(f'Variable {var[1]} already exists. Rename variable'
#                 #                 ' or delete copy.')

#             # make dict with given var name as key and meta data from model_vars
#             # check if name[7:] == var[0] -> var[0] is abs name and var[1] is alias
#             # check if meta['prom_name'] == var[0] -> var[0] is prom name and var[1] is alias
#             # NOTE name[7:] is the path name with out the 'subsys.' group level
#             tmp_dict = {var[1]: meta for name, meta in model_vars
#                         if name[7:] == var[0] or meta['prom_name'] == var[0]}

#             # check if dict is empty (no vars added)
#             if len(tmp_dict) == 0:
#                 raise Exception(f'Path name {var[0]} does not'
#                                 ' exist in model.')

#             var_dict.update(tmp_dict)

#         elif isinstance(var, str):
#             # if variable already exists in dict, it is connected so continue
#             if var in var_dict:
#                 continue
#                 # raise Exception(f'Variable {var} already exists. Rename variable'
#                 #                 ' or delete copy.')

#             # make dict with given var name as key and meta data from model_vars
#             # check if name[7:] == var -> given var is abs name
#             # check if meta['prom_name'] == var -> given var is prom_name
#             # NOTE name[7:] is the path name with out the 'subsys.' group level
#             tmp_dict = {var[1]: meta for name, meta in model_vars
#                         if name[7:] == var[0] or meta['prom_name'] == var[0]}

#             # check if provided variable appears more than once in model
#             if len(tmp_dict) > 1:
#                 raise Exception(f'Ambiguous variable {var}. To'
#                                 ' specify which one is desired, use a tuple'
#                                 ' with the promoted name and variable name'
#                                 ' instead [e.g. (prom_name, var_name)].')

#             # checks if provided variable doesn't exist in model
#             elif len(tmp_dict) == 0:
#                 raise Exception(f'Variable {var} does not exist in model.')

#             var_dict.update(tmp_dict)

#         else:
#             raise Exception(f'Type {type(var)} is invalid. Must be'
#                             ' string or tuple.')

#     return var_dict


class SubmodelComp(ExplicitComponent):
    """
    System level container for systems.

    Parameters
    ----------
    model : <System>
        The system-level <System>.
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
        All remaining named args that become options for `SubmodelComp`.

    Attributes
    ----------
    prob_args : dict
        Extra arguments to be passed to the problem instantiation.
    model : <System>
        The system being analyzed in subproblem.
    submodel_inputs : list of str or tuple
        List of inputs requested by user to be used as inputs in the
        subproblem's system.
    submodel_outputs : list of str or tuple
        List of outputs requested by user to be used as outputs in the
        subproblem's system.
    is_set_up : bool
        Flag to determne if subproblem is set up. Used for add_input/add_output to
        determine how to add the io.
    """
    # TODO change doc string for _input_names and _output_names

    def __init__(self, model, problem, inputs=None, outputs=None, # comm=None, name=None,
                 #reports=_UNDEFINED, prob_options=None, **kwargs):
                 **kwargs):
        """
        Initialize all attributes.
        """
        # call base class to set kwargs
        super().__init__(**kwargs)

        self.model = model
        self._subprob = problem
        self.submodel_inputs = inputs if inputs is not None else []
        self.submodel_outputs = outputs if outputs is not None else []
        self.is_set_up = False

    # TODO make path as name the default behavior
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

        if not self.is_set_up:
            self.submodel_inputs.append((path, name))
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

        if not self.is_set_up:
            self.submodel_outputs.append((path, name))
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
        p = self._subprob # = Problem(**self.prob_args)

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

        # NOTE interface name -> what SubmodelComp refers to the var as
        # NOTE interior name -> what the user refers to the var as
        for var in self.submodel_inputs:
            if isinstance(var, tuple):
                iface_name = var[1]
                prom_name = var[0]
            else:
                iface_name = prom_name = var
            meta = next(data for _, data in self.boundary_inputs if data['prom_name'] == prom_name)
            meta.pop('prom_name')
            super().add_input(iface_name, **meta)
            meta['prom_name'] = prom_name

        for var in self.submodel_outputs:
            if isinstance(var, tuple):
                iface_name = var[1]
                prom_name = var[0]
            else:
                iface_name = prom_name = var
            meta = next(data for _, data in self.all_outputs if data['prom_name'] == prom_name)
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
            p.model.add_design_var(prom_name)

        for prom_name, _ in self.submodel_outputs:
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
