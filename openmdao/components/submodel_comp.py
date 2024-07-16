"""Define the SubmodelComp class for evaluating OpenMDAO systems within components."""

import numpy as np

from openmdao.core.constants import _SetupStatus
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.utils.general_utils import pattern_filter
from openmdao.utils.reports_system import clear_reports
from openmdao.utils.mpi import MPI, FakeComm
from openmdao.utils.coloring import compute_total_coloring, ColoringMeta
from openmdao.utils.indexer import ranges2indexer
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
        Determines if reports should be included in subproblem. Default is False because
        submodelcomp runs faster without reports.
    **kwargs : named args
        All remaining named args that become options for `SubmodelComp`.

    Attributes
    ----------
    _subprob : <Problem>
        Instantiated problem used to run the model.
    _submodel_inputs : dict
        Mapping of inner promoted input names to outer input names.
    _submodel_outputs : dict
        Mapping of inner promoted output names to outer output names.
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
    _ins2sub_outs_idxs : ndarray
        Index array that maps our input array into parts of the output array of the submodel.
    _sub_outs_idxs : ndarray
        Index array that maps parts of the output array of the submodel into our output array.
    _zero_partials : set
        Set of (output, input) pairs that should have zero partials.
    _totjacinfo : _TotalJacInfo or None
        Object that computes the total jacobian for the submodel.
    _do_opt : bool
        True if the submodel has an optimizer.
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
        self._ins2sub_outs_idxs = None
        self._sub_outs_idxs = None
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
        self.options.declare('do_coloring', types=bool, default=False,
                             desc='If True, attempt to compute a total coloring for the submodel.')

    def _add_static_input(self, inner_prom_name_or_pattern, outer_name=None, **kwargs):
        self._static_submodel_inputs[inner_prom_name_or_pattern] = (outer_name, kwargs)

    def _add_static_output(self, inner_prom_name_or_pattern, outer_name=None, **kwargs):
        self._static_submodel_outputs[inner_prom_name_or_pattern] = (outer_name, kwargs)

    def _to_outer_output(self, inner_name, absolute=False):
        if absolute:
            return self.pathname + '.' + self._submodel_outputs[inner_name]
        return self._submodel_outputs[inner_name]

    def _to_outer_input(self, inner_name, absolute=False):
        if absolute:
            return self.pathname + '.' + self._submodel_inputs[inner_name]
        return self._submodel_inputs[inner_name]

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

    @property
    def problem(self):
        """
        Allow user read-only access to the sub-problem.

        Returns
        -------
        <Problem>
            Instantiated problem used to run the model.
        """
        return self._subprob

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

        # if we get here, our internal setup() is complete, so we can add the input immediately.

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_input after configure.')

        if prom_in not in self.indep_vars:
            raise NameError(f"'{prom_in}' is not an independent variable in the submodel or an "
                            "unconnected input variable.")

        if name is None:
            name = self._make_valid_name(prom_in)

        self._submodel_inputs[prom_in] = name

        super().add_input(name, **self._get_input_kwargs(prom_in, kwargs))

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

        self._submodel_outputs[prom_out] = name

        if self._problem_meta['setup_status'] > _SetupStatus.POST_CONFIGURE:
            raise Exception('Cannot call add_output after configure.')

        super().add_output(name, **self._get_output_kwargs(prom_out, kwargs))

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

        abs2meta_out = p.model._var_allprocs_abs2meta['output']
        abs2meta_local = p.model._var_abs2meta['output']
        prom2abs_in = p.model._var_allprocs_prom2abs_list['input']
        prom2abs_out = p.model._var_allprocs_prom2abs_list['output']
        abs2prom_out = p.model._var_allprocs_abs2prom['output']

        # indep vars is a dict containing promoted names of all indep vars belonging to
        # IndepVarComps in the submodel, along with all inputs connected to _auto_ivc vars.
        # All SubmodelComp inputs must come from the indep_vars dict, because any other inputs
        # in the submodel are dependent on other outputs and would be overwritten when the
        # submodel runs, erasing any values set by the SubmodelComp.
        self.indep_vars = indep_vars = {}
        for src, meta in abs2meta_out.items():
            if src.startswith('_auto_ivc.'):
                continue
            prom = abs2prom_out[src]
            if prom not in indep_vars and 'openmdao:indep_var' in meta['tags']:
                if src in abs2meta_local:
                    meta = abs2meta_local[src]  # get local metadata if we have it
                indep_vars[prom] = (src, meta)

        # add any inputs connected to auto_ivc vars as indep vars.  Their name will be the
        # promoted name of the input that connects to the actual indep var.
        for prom in prom2abs_in:
            src = p.model.get_source(prom)
            if src.startswith('_auto_ivc.'):
                if src in abs2meta_local:
                    meta = abs2meta_local[src]  # get local metadata if we have it
                else:
                    meta = abs2meta_out[src]
                indep_vars[prom] = (src, meta)

        submodel_inputs = {}
        for inner_prom, (outer_name, kwargs) in self._static_submodel_inputs.items():
            # outer_name could still be None here
            if _is_glob(inner_prom):
                matches = list(pattern_filter(inner_prom, indep_vars))
                if not matches:
                    raise NameError(f"Pattern '{inner_prom}' doesn't match any independent "
                                    "variables in the submodel.")
            elif inner_prom in indep_vars:
                matches = [inner_prom]
            else:
                raise NameError(f"'{inner_prom}' is not an independent variable in the submodel.")

            for match in matches:
                iname = self._make_valid_name(match if outer_name is None else outer_name)
                submodel_inputs[match] = iname

                super().add_input(iname, **self._get_input_kwargs(match, kwargs))

                if 'val' in kwargs:  # val in kwargs overrides internal value
                    self._subprob.set_val(match, kwargs['val'])

        self._submodel_inputs = dict(sorted(submodel_inputs.items(), key=lambda x: x[0]))

        submodel_outputs = {}
        for inner_prom, (outer_name, kwargs) in self._static_submodel_outputs.items():
            # outer_name could still be None here
            if _is_glob(inner_prom):
                matches = []
                for match in pattern_filter(inner_prom, prom2abs_out):
                    if match.startswith('_auto_ivc.') or match in self._submodel_inputs:
                        continue
                    matches.append(match)
                if not matches:
                    raise NameError(f"Pattern '{inner_prom}' doesn't match any outputs in the "
                                    "submodel.")
            elif inner_prom in prom2abs_out:
                matches = [inner_prom]
            else:
                raise NameError(f"'{inner_prom}' is not an output in the submodel.")

            for match in matches:

                oname = self._make_valid_name(match if outer_name is None else outer_name)
                submodel_outputs[match] = oname

                super().add_output(oname, **self._get_output_kwargs(match, kwargs))

                if 'val' in kwargs:  # val in kwargs overrides internal value
                    self._subprob.set_val(inner_prom, kwargs['val'])

        self._submodel_outputs = dict(sorted(submodel_outputs.items(), key=lambda x: x[0]))

    def _get_output_kwargs(self, prom, kwargs):
        """
        Get updated kwargs based on metadata from the submodel for the given promoted output.

        Parameters
        ----------
        prom : str
            Promoted name of the output.
        kwargs : dict
            Keyword arguments for the add_output call.

        Returns
        -------
        dict
            Updated kwargs.
        """
        prom2abs = self._subprob.model._var_allprocs_prom2abs_list['output']

        try:
            # look for local metadata first, in case it sets 'val'
            meta = self._subprob.model._var_abs2meta['output'][prom2abs[prom][0]]
        except KeyError:
            try:
                # just use global metadata
                meta = self._subprob.model._var_allprocs_abs2meta['output'][prom2abs[prom][0]]
            except KeyError:
                raise KeyError(f"Output '{prom}' not found in model")

        self._check_var_allowed(prom2abs[prom], meta)

        final_kwargs = {n: v for n, v in meta.items() if n in _allowed_add_output_args}
        final_kwargs.update(kwargs)
        return final_kwargs

    def _get_input_kwargs(self, prom, kwargs):
        """
        Get updated kwargs based on metadata from the submodel for the given promoted input.

        Parameters
        ----------
        prom : str
            Promoted name of the input.
        kwargs : dict
            Keyword arguments for the add_input call.

        Returns
        -------
        dict
            Updated kwargs.
        """
        src, meta = self.indep_vars[prom]
        self._check_var_allowed(src, meta)
        final_kwargs = {n: v for n, v in meta.items() if n in _allowed_add_input_args}
        final_kwargs.update(kwargs)
        return final_kwargs

    def _check_var_allowed(self, abs_name, meta):
        """
        Raise an exception if the variable is not allowed in a SubmodelComp.

        Parameters
        ----------
        abs_name : str
            Absolute name of the variable.
        meta : dict
            Metadata for the variable.
        """
        if self._subprob.comm.size > 1:
            if meta['distributed']:
                raise RuntimeError(f"Variable '{abs_name}' is distributed, and  distributed "
                                   "variables are not currently supported in SubmodelComp.")

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
                                         return_format='flat_dict', get_remote=True,
                                         approx=p.model._owns_approx_jac,
                                         coloring_info=coloring_info,
                                         driver=False)

        coloring = coloring_info.coloring

        if coloring is None:
            # only declare partials where of and wrt are relevant to each other
            ofs = tuple([m['source'] for m in of_metadata.values()])
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

    def _setup_vectors(self, root_vectors):
        """
        Compute all vectors for all vec names and assign excluded variables lists.

        Parameters
        ----------
        root_vectors : dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        """
        super()._setup_vectors(root_vectors)
        self._setup_transfer_idxs()

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

        # set our inputs into the submodel outputs. We don't set into the submodel inputs
        # because they are all connected to outputs which would overwrite our values when the
        # submodel runs. So the only inputs we allow from outside are those that are connected to
        # indep vars, which are outputs in the submodel that are not dependent on anything else.
        p.model._outputs.set_val(inputs.asarray()[self._ins_idxs()], idxs=self._ins2sub_outs_idxs())

        if self._do_opt:
            p.run_driver()
        else:
            p.run_model()

        # collect outputs from the submodel
        self._outputs.set_val(0.0)
        self._outputs.set_val(p.model._outputs.asarray()[self._sub_outs_idxs()],
                              idxs=self._outs_idxs())

        if self.comm.size > 1:
            self._outputs.set_val(self.comm.allreduce(self._outputs.asarray(), op=MPI.SUM))

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

    def _setup_transfer_idxs(self):
        """
        Set up the transfer indices for input and output variables.

        These map parts of our input and output arrays to the input and output arrays of the
        submodel.
        """
        submod = self._subprob.model
        sub_slices = submod._outputs.get_slice_dict()
        slices = self._inputs.get_slice_dict()
        prefix = self.pathname + '.'

        input_ranges = []
        sub_out_ranges = []
        for inner_prom, outer_name in self._submodel_inputs.items():
            sub_src = submod.get_source(inner_prom)
            if submod._owned_size(sub_src) > 0:
                sub_slc = sub_slices[sub_src]
                sub_out_ranges.append((sub_slc.start, sub_slc.stop))
                slc = slices[prefix + outer_name]
                input_ranges.append((slc.start, slc.stop))

        self._ins_idxs = ranges2indexer(input_ranges, src_shape=(len(self._inputs),))
        self._ins2sub_outs_idxs = ranges2indexer(sub_out_ranges, src_shape=(len(submod._outputs),))

        prom2abs = submod._var_allprocs_prom2abs_list['output']
        slices = self._outputs.get_slice_dict()
        sub_out_ranges = []
        out_ranges = []

        for inner_prom, outer_name in self._submodel_outputs.items():
            sub_src = prom2abs[inner_prom][0]
            if submod._owned_size(sub_src) > 0:
                sub_slc = sub_slices[sub_src]
                slc = slices[prefix + outer_name]
                out_ranges.append((slc.start, slc.stop))
                sub_out_ranges.append((sub_slc.start, sub_slc.stop))

        self._outs_idxs = ranges2indexer(out_ranges, src_shape=(len(self._outputs),))
        self._sub_outs_idxs = ranges2indexer(sub_out_ranges, src_shape=(len(submod._outputs),))
