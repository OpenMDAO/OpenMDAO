"""Define the SubmodelComp class for evaluating OpenMDAO systems within components."""

from openmdao.core.constants import _SetupStatus, INF_BOUND
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.utils.general_utils import pattern_filter
from openmdao.utils.reports_system import clear_reports
from openmdao.utils.mpi import MPI, FakeComm
from openmdao.utils.coloring import compute_total_coloring, ColoringMeta
from openmdao.utils.om_warnings import warn_deprecation


def _is_glob(name):
    return '*' in name or '?' in name or '[' in name


def _check_wild_name(name, out_name):
    if out_name is not None and _is_glob(name):
        raise NameError(f"Can't specify outer_name '{out_name}' when inner_name '{name}' has "
                        "wildcards.")


def _io_namecheck_iter(names, iotype):
    if names is not None:
        for name in names:
            if isinstance(name, str):
                yield name, None
            elif isinstance(name, tuple):
                _check_wild_name(*name)
                yield name
            else:
                raise TypeError(f'Expected {iotype} of type str or tuple, got {type(name)}.')


_allowed_add_output_args = {'val', 'shape', 'units', 'res_units', 'desc', 'lower', 'upper',
                            'ref', 'ref0', 'res_ref', 'tags', 'shape_by_conn', 'copy_shape',
                            'compute_shape', 'distributed'}

_allowed_add_input_args = {'val', 'shape', 'units', 'desc', 'tags', 'shape_by_conn',
                           'copy_shape', 'compute_shape', 'distributed'}


class SubmodelComp(ExplicitComponent):
    """
    System level container for systems.

    Parameters
    ----------
    problem : <Problem>
        Instantiated problem to use for the model.
    inputs : list of str or tuple or None
        List of provided input names in str or tuple form. If an element is a str,
        then it should be the inner promoted name. If it is a tuple,
        then the first element should be the inner promoted name, and the
        second element should be the outer name, the name used by everything outside
        the subproblem [e.g. (path.to.inner.var, outer_name)].
    outputs : list of str or tuple or None
        List of provided output names in str or tuple form. If an element is a str,
        then it should be the inner promoted name. If it is a tuple,
        then the first element should be the inner promoted name, and the
        second element should be the outer name, the name used by everything outside
        the subproblem [e.g. (path.to.inner.var, outer_name)].
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
        Mapping of inner promoted input names to outer input names and kwargs.
    submodel_outputs : dict
        Mapping of inner promoted output names to outer output names and kwargs.
    _static_submodel_inputs : dict
        Mapping of inner promoted input names to outer input names and kwargs that is populated
        outside of setup. These must be bookkept separately from submodel inputs added during setup
        because setup can be called multiple times and the submodel inputs dict is reset each time.
    _static_submodel_outputs : dict
        Mapping of inner promoted output names to outer output names and kwargs that is populated
        outside of setup. These must be bookkept separately from submodel outputs added during setup
        because setup can be called multiple times and the submodel outputs dict is reset each time.
    """

    def __init__(self, problem, inputs=None, outputs=None, reports=False, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        if not reports:
            clear_reports(problem)

        self._subprob = problem
        self.coloring = None

        self.submodel_inputs = {}
        self.submodel_outputs = {}

        self._static_submodel_inputs = {
            name: (out_name, {}) for name, out_name in _io_namecheck_iter(inputs, 'input')
        }
        self._static_submodel_outputs = {
            name: (out_name, {}) for name, out_name in _io_namecheck_iter(outputs, 'output')
        }

    def _add_static_input(self, inner_prom_name_or_pattern, outer_name=None, **kwargs):
        if outer_name is None and not _is_glob(inner_prom_name_or_pattern):
            outer_name = inner_prom_name_or_pattern.replace('.', ':')
        self._static_submodel_inputs[inner_prom_name_or_pattern] = (outer_name, kwargs)

    def _add_static_output(self, inner_prom_name_or_pattern, outer_name=None, **kwargs):
        if outer_name is None and not _is_glob(inner_prom_name_or_pattern):
            outer_name = inner_prom_name_or_pattern.replace('.', ':')
        self._static_submodel_outputs[inner_prom_name_or_pattern] = (outer_name, kwargs)

    def add_input(self, prom_in, outer_name=None, name=None, **kwargs):
        """
        Add input to model before or after setup.

        Parameters
        ----------
        prom_in : str
            Promoted inner name of input.
        outer_name : str or None
            Outer name of input to be added. If none, it will default to prom_in after replacing
            any '.'s with ':'s.
        name : str or None
            Deprecated. Use outer_name instead.
        **kwargs : named args
            All remaining named args that can become options for `add_input`.
        """
        if name is not None:
            warn_deprecation("The 'name' argument is deprecated and has been replaced by "
                             "'outer_name'.")
            if outer_name is not None:
                raise ValueError("Can't specify both 'name' and 'outer_name'.")
            outer_name = name

        if self._static_mode:
            self._add_static_input(prom_in, outer_name, **kwargs)
            return

        if _is_glob(prom_in):
            raise NameError(f"Can't add input using wildcard '{prom_in}' during setup. "
                            "Use add_static_input outside of setup instead.")

        # if we get here, our internal setup() is complete, so we can add the input to the
        # submodel immediately.

        if outer_name is None:
            outer_name = prom_in.replace('.', ':')

        self.submodel_inputs[prom_in] = (outer_name, kwargs)

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_input after configure.')

        super().add_input(outer_name, **kwargs)

    def add_output(self, prom_out, outer_name=None, name=None, **kwargs):
        """
        Add output to model before or after setup.

        Parameters
        ----------
        prom_out : str
            Promoted name of the inner output.
        outer_name : str or None
            Outer name of output to be added. If none, it will default to prom_out after replacing
            any '.'s with ':'s.
        name : str or None
            Deprecated. Use outer_name instead.
        **kwargs : named args
            All remaining named args that can become options for `add_output`.
        """
        if name is not None:
            warn_deprecation("The 'name' argument is deprecated and has been replaced by "
                             "'outer_name'.")
            if outer_name is not None:
                raise ValueError("Can't specify both 'name' and 'outer_name'.")
            outer_name = name

        if self._static_mode:
            self._add_static_output(prom_out, outer_name, **kwargs)
            return

        if _is_glob(prom_out):
            raise NameError(f"Can't add output using wildcard '{prom_out}' during setup. "
                            "Use add_static_output outside of setup instead.")

        # if we get here, our internal setup() is complete, so we can add the output to the
        # submodel immediately.

        if outer_name is None:
            outer_name = prom_out.replace('.', ':')

        self.submodel_outputs[prom_out] = (outer_name, kwargs)

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_output after configure.')

        super().add_output(outer_name, **kwargs)

    def setup(self):
        """
        Perform some final setup and checks.
        """
        self._totjacinfo = None
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

        abs2meta = p.model._var_allprocs_abs2meta['output']
        abs2meta_local = p.model._var_abs2meta['output']
        prom2abs_in = p.model._var_allprocs_prom2abs_list['input']
        prom2abs_out = p.model._var_allprocs_prom2abs_list['output']

        self.indep_vars = {}
        for prom in prom2abs_in:
            src = p.model.get_source(prom)
            if 'openmdao:indep_var' in abs2meta[src]['tags']:
                if src in abs2meta_local:
                    meta = abs2meta_local[src]
                else:
                    meta = abs2meta[src]
                self.indep_vars[prom] = meta
                self.indep_vars[src] = meta

        self.submodel_inputs = {}
        self.submodel_outputs = {}

        for var, (out_name, kwargs) in self._static_submodel_inputs.items():
            if _is_glob(var):
                found = False
                for match in pattern_filter(var, self.indep_vars):
                    self.submodel_inputs[match] = (match.replace('.', ':'), kwargs.copy())
                    found = True
                if not found:
                    raise NameError(f"Pattern '{var}' doesn't match any independent variables in "
                                    "the submodel.")
            elif var in self.indep_vars:
                if out_name is None:
                    out_name = var.replace('.', ':')
                self.submodel_inputs[var] = (out_name, kwargs.copy())
            else:
                raise NameError(f"'{var}' is not an independent variable in the submodel.")

        for var, (out_name, kwargs) in self._static_submodel_outputs.items():
            if _is_glob(var):
                found = False
                for match in pattern_filter(var, prom2abs_out):
                    if match.startswith('_auto_ivc.'):
                        continue
                    self.submodel_outputs[match] = (match.replace('.', ':'), kwargs.copy())
                    found = True
                if not found:
                    raise NameError(f"Pattern '{var}' doesn't match any outputs in the submodel.")
            elif var in prom2abs_out:
                if out_name is None:
                    out_name = var.replace('.', ':')
                self.submodel_outputs[var] = (out_name, kwargs.copy())
            else:
                raise NameError(f"'{var}' is not an output in the submodel.")

        # NOTE outer_name is what the outer problem knows the variable to be
        # it won't always be the same name as the prom name in the inner variable because
        # the inner prom name could contain '.'s and the outer name, which is the name relative
        # to this component, cannot contain '.'s.
        for prom_name, (outer_name, kwargs) in sorted(self.submodel_inputs.items(),
                                                      key=lambda x: x[0]):
            if outer_name in self._static_var_rel2meta or outer_name in self._var_rel2meta:
                raise RuntimeError("this shouldn't happen")
            try:
                meta = self.indep_vars[prom_name]
            except KeyError:
                raise KeyError(f"Independent variable '{prom_name}' not found in model")

            final_kwargs = {n: v for n, v in meta.items() if n in _allowed_add_input_args}
            final_kwargs.update(kwargs)
            if out_name is None:
                out_name = var.replace('.', ':')
            super().add_input(outer_name, **final_kwargs)
            if 'val' in kwargs:  # val in kwargs overrides internal value
                self._subprob.set_val(prom_name, kwargs['val'])

        for prom_name, (outer_name, kwargs) in sorted(self.submodel_outputs.items(),
                                                      key=lambda x: x[0]):
            if outer_name in self._static_var_rel2meta or outer_name in self._var_rel2meta:
                raise RuntimeError("this shouldn't happen")

            try:
                # look for metadata locally first, then use allprocs data if we have to
                meta = abs2meta_local[prom2abs_out[prom_name][0]]
            except KeyError:
                try:
                    meta = abs2meta[prom2abs_out[prom_name][0]]
                except KeyError:
                    raise KeyError(f"Output '{prom_name}' not found in model")

            final_kwargs = {n: v for n, v in meta.items() if n in _allowed_add_output_args}
            final_kwargs.update(kwargs)
            if out_name is None:
                out_name = var.replace('.', ':')
            super().add_output(outer_name, **final_kwargs)
            if 'val' in kwargs:  # val in kwargs overrides internal value
                self._subprob.set_val(prom_name, kwargs['val'])

    def setup_partials(self):
        p = self._subprob
        inputs = self._var_rel_names['input']
        outputs = self._var_rel_names['output']

        if len(inputs) == 0 or len(outputs) == 0:
            return

        ofs = list(self.submodel_outputs)
        wrts = list(self.submodel_inputs)

        coloring_info = ColoringMeta()
        coloring_info.coloring = compute_total_coloring(p, of=ofs, wrt=wrts, run_model=True)
        if coloring_info.coloring is not None:
            self.coloring = coloring_info.coloring

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
        for prom_name, (outer_name, _) in self.submodel_inputs.items():
            p.set_val(prom_name, inputs[outer_name])

        # set initial output vals
        for prom_name, (outer_name, _) in self.submodel_outputs.items():
            p.set_val(prom_name, outputs[outer_name])

        p.run_model()

        for prom_name, (outer_name, _) in self.submodel_outputs.items():
            outputs[outer_name] = p.get_val(prom_name)

    def compute_partials(self, inputs, partials):
        """
        Update the partials object with updated partial derivatives.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Sub-jac components written to partials[output_name, input_name].
        """
        p = self._subprob

        for prom_name, (outer_name, _) in self.submodel_inputs.items():
            p.set_val(prom_name, inputs[outer_name])

        if self._totjacinfo is None:
            self._totjacinfo = _TotalJacInfo(p, self.submodel_outputs, self.submodel_inputs,
                                             return_format='flat_dict',
                                             approx=p.model._owns_approx_jac, use_coloring=True)
        tots = self._totjacinfo.compute_totals()

        if self.coloring is None:
            for (tot_output, tot_input), tot in tots.items():
                input_outer_name = self.submodel_inputs[tot_input][0]
                output_outer_name = self.submodel_outputs[tot_output][0]
                partials[output_outer_name, input_outer_name] = tot
        else:
            for of, wrt, nzrows, nzcols, _, _, _, _ in self.coloring._subjac_sparsity_iter():
                partials[of, wrt] = tots[of, wrt][nzrows, nzcols].ravel()


    # TODO:
    # def _transfer_to_sub(self):
    #    pass

    # def _transfer_from_sub(self):
    #    pass
