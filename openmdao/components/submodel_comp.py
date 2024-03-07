"""Define the SubmodelComp class for evaluating OpenMDAO systems within components."""
from collections import defaultdict, Counter

from openmdao.core.constants import _SetupStatus, INF_BOUND
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.utils.general_utils import pattern_filter
from openmdao.utils.reports_system import clear_reports
from openmdao.utils.mpi import MPI, FakeComm
from openmdao.utils.coloring import compute_total_coloring, ColoringMeta
from openmdao.utils.om_warnings import warn_deprecation
from openmdao.utils.indexer import ranges2indexer
from openmdao.utils.iter_utils import size2range_iter, meta2item_iter
from openmdao.utils.relevance import get_relevance


def _is_glob(name):
    return '*' in name or '?' in name or '[' in name


def _check_wild_name(name, outer_name):
    if outer_name is not None and _is_glob(name):
        raise NameError(f"Can't specify outer_name '{outer_name}' when inner_name '{name}' has "
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
    _submodel_inputs : dict
        Mapping of inner promoted input names to outer input names and kwargs.
    _submodel_outputs : dict
        Mapping of inner promoted output names to outer output names and kwargs.
    _static_submodel_inputs : dict
        Mapping of inner promoted input names to outer input names and kwargs that is populated
        outside of setup. These must be bookkept separately from submodel inputs added during setup
        because setup can be called multiple times and the submodel inputs dict is reset each time.
    _static_submodel_outputs : dict
        Mapping of inner promoted output names to outer output names and kwargs that is populated
        outside of setup. These must be bookkept separately from submodel outputs added during setup
        because setup can be called multiple times and the submodel outputs dict is reset each time.
    _sub_coloring_info : ColoringMeta
        The coloring information for the submodel.
    _input_xfer_idxs : ndarray
        Index array that maps our input array into parts of the output array of the submodel.
    _output_xfer_idxs : ndarray
        Index array that maps our output array into parts of the output array of the submodel.
    _zero_partials : set
        Set of (output, input) pairs that should have zero partials.
    """

    def __init__(self, problem, inputs=None, outputs=None, reports=False, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        if not reports:
            clear_reports(problem)

        self._subprob = problem
        self._sub_coloring_info = ColoringMeta()

        self._submodel_inputs = {}
        self._submodel_outputs = {}
        self._input_xfer_idxs = None
        self._output_xfer_idxs = None
        self._zero_partials = set()

        self._static_submodel_inputs = {
            name: (outer_name, {}) for name, outer_name in _io_namecheck_iter(inputs, 'input')
        }
        self._static_submodel_outputs = {
            name: (outer_name, {}) for name, outer_name in _io_namecheck_iter(outputs, 'output')
        }

    def _declare_options(self):
        """
        Declare options.
        """
        super()._declare_options()
        self.options.declare('do_coloring', types=bool, default=True,
                             desc='If True, attempt to compute a total coloring for the submodel.')

    def _add_static_input(self, inner_prom_name_or_pattern, outer_name=None, **kwargs):
        self._static_submodel_inputs[inner_prom_name_or_pattern] = (outer_name, kwargs)

    def _add_static_output(self, inner_prom_name_or_pattern, outer_name=None, **kwargs):
        self._static_submodel_outputs[inner_prom_name_or_pattern] = (outer_name, kwargs)

    def _to_outer_output(self, inner_name):
        return self._submodel_outputs[inner_name][0]

    def _to_outer_input(self, inner_name):
        return self._submodel_inputs[inner_name][0]

    def _make_valid_name(self, name):
        """
        Make an internal, potentially dotted name into a valid component variable name.

        Parameters
        ----------
        name : str
            The name to convert.

        Returns
        -------
        str
            The converted name.
        """
        return name.replace('.', ':')

    def add_input(self, prom_in, name=None, **kwargs):
        """
        Add input to model before or after setup.

        Parameters
        ----------
        prom_in : str
            Promoted inner name of input.
        name : str or None
            Name of input relative to this component. If none, it will default to prom_in after
            replacing any '.'s with ':'s.
        **kwargs : named args
            All remaining named args that can become options for `add_input`.
        """
        if self._static_mode:
            self._add_static_input(prom_in, name, **kwargs)
            return

        if _is_glob(prom_in):
            raise NameError(f"Can't add input using wildcard '{prom_in}' during setup. "
                            "Use add_static_input outside of setup instead.")

        # if we get here, our internal setup() is complete, so we can add the input to the
        # submodel immediately.

        if name is None:
            name = self._make_valid_name(prom_in)

        self._submodel_inputs[prom_in] = (name, kwargs)

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_input after configure.')

        super().add_input(name, **kwargs)

    def add_output(self, prom_out, name=None, **kwargs):
        """
        Add output to model before or after setup.

        Parameters
        ----------
        prom_out : str
            Promoted name of the inner output.
        name : str or None
            Name of output relative to this component. If none, it will default to prom_out after
            replacing any '.'s with ':'s.
        **kwargs : named args
            All remaining named args that can become options for `add_output`.
        """
        if self._static_mode:
            self._add_static_output(prom_out, name, **kwargs)
            return

        if _is_glob(prom_out):
            raise NameError(f"Can't add output using wildcard '{prom_out}' during setup. "
                            "Use add_static_output outside of setup instead.")

        # if we get here, our internal setup() is complete, so we can add the output to the
        # submodel immediately.

        if name is None:
            name = self._make_valid_name(prom_out)

        self._submodel_outputs[prom_out] = (name, kwargs)

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_output after configure.')

        super().add_output(name, **kwargs)

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
        abs2prom_out = p.model._var_allprocs_abs2prom['output']

        # store indep vars by promoted name, excluding _auto_ivc vars because later we'll
        # add the inputs connected to _auto_ivc vars as indep vars (because that's how the user
        # would expect them to be named)
        self.indep_vars = {}
        for src, meta in abs2meta.items():
            if src.startswith('_auto_ivc.'):
                continue
            prom = abs2prom_out[src]
            if prom not in self.indep_vars and 'openmdao:indep_var' in meta['tags']:
                if src in abs2meta_local:
                    meta = abs2meta_local[src]  # get local metadata if we have it
                self.indep_vars[prom] = (src, meta)

        # add any inputs connected to _auto_ivc vars as indep vars
        for prom in prom2abs_in:
            src = p.model.get_source(prom)
            if not src.startswith('_auto_ivc.'):
                continue
            if src in abs2meta_local:
                meta = abs2meta_local[src]  # get local metadata if we have it
            else:
                meta = abs2meta[src]
            self.indep_vars[prom] = (src, meta)

        self._submodel_inputs = {}
        self._submodel_outputs = {}

        for inner_prom, (outer_name, kwargs) in self._static_submodel_inputs.items():
            if _is_glob(inner_prom):
                found = False
                for match in pattern_filter(inner_prom, self.indep_vars):
                    self._submodel_inputs[match] = (match, kwargs.copy())
                    found = True
                if not found:
                    raise NameError(f"Pattern '{inner_prom}' doesn't match any independent "
                                    "variables in the submodel.")
            elif inner_prom in self.indep_vars:
                self._submodel_inputs[inner_prom] = (outer_name, kwargs.copy())
            else:
                raise NameError(f"'{inner_prom}' is not an independent variable in the submodel.")

        for inner_prom, (outer_name, kwargs) in self._static_submodel_outputs.items():
            if _is_glob(inner_prom):
                found = False
                for match in pattern_filter(inner_prom, prom2abs_out):
                    if match.startswith('_auto_ivc.'):
                        continue
                    self._submodel_outputs[match] = (match, kwargs.copy())
                    found = True
                if not found:
                    raise NameError(f"Pattern '{inner_prom}' doesn't match any outputs in the "
                                    "submodel.")
            elif inner_prom in prom2abs_out:
                self._submodel_outputs[inner_prom] = (outer_name, kwargs.copy())
            else:
                raise NameError(f"'{inner_prom}' is not an output in the submodel.")

        for inner_prom, (outer_name, kwargs) in sorted(self._submodel_inputs.items(),
                                                       key=lambda x: x[0]):
            try:
                _, meta = self.indep_vars[inner_prom]
            except KeyError:
                raise KeyError(f"Independent variable '{inner_prom}' not found in model")

            final_kwargs = {n: v for n, v in meta.items() if n in _allowed_add_input_args}
            final_kwargs.update(kwargs)
            if outer_name is None:
                outer_name = inner_prom

            outer_name = self._make_valid_name(outer_name)
            self._submodel_inputs[inner_prom] = (outer_name, kwargs)  # in case outer_name was None

            super().add_input(outer_name, **final_kwargs)
            if 'val' in kwargs:  # val in kwargs overrides internal value
                self._subprob.set_val(inner_prom, kwargs['val'])

        for inner_prom, (outer_name, kwargs) in sorted(self._submodel_outputs.items(),
                                                       key=lambda x: x[0]):
            try:
                # look for metadata locally first, then use allprocs data if we have to
                meta = abs2meta_local[prom2abs_out[inner_prom][0]]
            except KeyError:
                try:
                    meta = abs2meta[prom2abs_out[inner_prom][0]]
                except KeyError:
                    raise KeyError(f"Output '{inner_prom}' not found in model")

            final_kwargs = {n: v for n, v in meta.items() if n in _allowed_add_output_args}
            final_kwargs.update(kwargs)
            if outer_name is None:
                outer_name = inner_prom

            outer_name = self._make_valid_name(outer_name)
            self._submodel_outputs[inner_prom] = (outer_name, kwargs)  # in case outer_name was None

            super().add_output(outer_name, **final_kwargs)
            if 'val' in kwargs:  # val in kwargs overrides internal value
                self._subprob.set_val(inner_prom, kwargs['val'])

    def setup_partials(self):
        """
        Compute a coloring and declare partials based on the coloring.
        """
        p = self._subprob
        self._do_opt = p.driver.supports['gradients']

        inputs = self._var_rel_names['input']
        outputs = self._var_rel_names['output']

        ofs = list(self._submodel_outputs)
        wrts = list(self._submodel_inputs)
        of_metadata, wrt_metadata, _ = p.model._get_totals_metadata(driver=p.driver,
                                                                    of=ofs, wrt=wrts)

        self._setup_transfer_idxs(of_metadata, wrt_metadata)

        if len(inputs) == 0 or len(outputs) == 0:
            return

        self._sub_coloring_info = coloring_info = ColoringMeta()

        if self.options['do_coloring']:
            coloring_info.set_coloring(compute_total_coloring(p, of=ofs, wrt=wrts, run_model=True,
                                                              driver=False),
                                       msginfo=self.msginfo)

            coloring_info.display()

        # save the _TotJacInfo object so we can use it in future calls to compute_partials
        self._totjacinfo = _TotalJacInfo(p, of=self._submodel_outputs, wrt=self._submodel_inputs,
                                         return_format='flat_dict',
                                         approx=p.model._owns_approx_jac,
                                         coloring_info=coloring_info,
                                         driver=False)

        coloring = coloring_info.coloring

        if coloring is None:
            # only declare partials where of and wrt are relevant to each other
            ofs = [m['source'] for m in of_metadata.values()]
            relevance = get_relevance(p.model, of_metadata, wrt_metadata)
            for wrt, wrt_meta in wrt_metadata.items():
                outer_wrt = self._to_outer_input(wrt)
                wrtsrc = wrt_meta['source']
                with relevance.seeds_active((wrtsrc,), ofs):
                    for of, of_meta in of_metadata.items():
                        outer_of = self._to_outer_output(of)
                        if relevance.is_relevant(of_meta['source']):
                            self.declare_partials(of=outer_of, wrt=outer_wrt)
                        else:
                            self._zero_partials.add((outer_of, outer_wrt))
        else:
            # get a coloring with outer names for rows and cols to be used as the partial
            # coloring for this component
            row_map = {of: self._to_outer_output(of) for of in coloring._row_vars}
            col_map = {wrt: self._to_outer_input(wrt) for wrt in coloring._col_vars}
            self._coloring_info.coloring = coloring.get_renamed_copy(row_map, col_map)
            # prevent config check that will fail due to name changes
            self._coloring_info.dynamic = True

            for of, wrt, nzrows, nzcols, _, _, _, _ in coloring._subjac_sparsity_iter():
                self.declare_partials(of=self._to_outer_output(of),
                                      wrt=self._to_outer_input(wrt),
                                      rows=nzrows, cols=nzcols)

    def _set_complex_step_mode(self, active):
        super()._set_complex_step_mode(active)
        self._subprob.set_complex_step_mode(active)

    def _update_subjac_sparsity(self, sparsity):
        # do nothing here because if the submodel has a coloring, we've already declared the
        # partials based on that coloring and they already have the correct sparsity pattern.
        pass

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

        # set our inputs and outputs into the submodel
        p.model._outputs.set_val(self._inputs.asarray(), idxs=self._input_xfer_idxs())

        inner_idxs = self._output_xfer_idxs()
        p.model._outputs.set_val(self._outputs.asarray(), idxs=inner_idxs)

        if self._do_opt:
            p.run_driver()
        else:
            p.run_model()

        # collect outputs from the submodel
        self._outputs.set_val(p.model._outputs.asarray()[inner_idxs])

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
        if self._do_opt:
            raise RuntimeError("Can't compute partial derivatives of a SubmodelComp with "
                               "an internal optimizer.")

        p = self._subprob

        # we don't need to set our inputs into the submodel here because we've already done it
        # in compute.

        tots = self._totjacinfo.compute_totals()
        coloring = self._sub_coloring_info.coloring

        if coloring is None:
            for (of, wrt), tot in tots.items():
                key = (self._to_outer_output(of), self._to_outer_input(wrt))
                if key not in self._zero_partials:
                    partials[key] = tot
        else:
            for of, wrt, nzrows, nzcols, _, _, _, _ in coloring._subjac_sparsity_iter():
                partials[(self._to_outer_output(of), self._to_outer_input(wrt))] = \
                    tots[of, wrt][nzrows, nzcols].ravel()

    def _setup_transfer_idxs(self, of_metadata, wrt_metadata):
        """
        Set up the transfer indices for input and output variables.

        These map parts of our input and output arrays to the input and output arrays of the
        submodel.
        """
        abs2meta = self._subprob.model._var_allprocs_abs2meta['output']
        prom2abs = self._subprob.model._var_allprocs_prom2abs_list['output']

        full_inner_map = {}
        for name, rng in size2range_iter(meta2item_iter(abs2meta.items(), 'size')):
            full_inner_map[name] = rng

        full_shape = (rng[1],) if full_inner_map else (0,)

        # get ranges for subodel outputs corresponding to our inputs
        inp_ranges = []
        for inner_prom in self._submodel_inputs:
            src, _ = self.indep_vars[inner_prom]
            inp_ranges.append(full_inner_map[src])

        # get ranges for submodel outputs corresponding to our outputs
        out_ranges = []
        for inner_prom in self._submodel_outputs:
            out_ranges.append(full_inner_map[prom2abs[inner_prom][0]])

        self._input_xfer_idxs = ranges2indexer(inp_ranges, src_shape=full_shape)
        self._output_xfer_idxs = ranges2indexer(out_ranges, src_shape=full_shape)
