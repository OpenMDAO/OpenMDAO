"""Define the ImplicitComponent class."""

from collections import defaultdict
import numpy as np

from openmdao.core.component import Component, _allowed_types
from openmdao.core.constants import _UNDEFINED, _SetupStatus
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.class_util import overrides_method
from openmdao.utils.array_utils import shape_to_len
from openmdao.utils.general_utils import format_as_float_or_array
from openmdao.utils.units import simplify_unit


_inst_functs = ['apply_linear', 'solve_nonlinear']
_full_slice = slice(None)


def _get_slice_shape_dict(name_shape_iter):
    """
    Return a dict of (slice, shape) tuples using provided names and shapes.

    Parameters
    ----------
    name_shape_iter : iterator
        An iterator yielding (name, shape) pairs

    Returns
    -------
    dict
        A dict of (slice, shape) tuples using provided names and shapes.
    """
    dct = {}

    start = end = 0
    for name, shape in name_shape_iter:
        size = shape_to_len(shape)
        end += size
        dct[name] = (slice(start, end), shape)
        start = end

    return dct


class ImplicitComponent(Component):
    """
    Class to inherit from when all output variables are implicit.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Component options.

    Attributes
    ----------
    _inst_functs : dict
        Dictionary of names mapped to bound methods.
    _declared_residuals : dict
        Contains local residual names mapped to metadata.
    """

    def __init__(self, **kwargs):
        """
        Store some bound methods so we can detect runtime overrides.
        """
        self._declared_residuals = {}
        super().__init__(**kwargs)

        self._inst_functs = {name: getattr(self, name, None) for name in _inst_functs}

    def _configure(self):
        """
        Configure this system to assign children settings.

        Also tag component if it provides a guess_nonlinear.
        """
        self._has_guess = overrides_method('guess_nonlinear', self, ImplicitComponent)

        solve_nl = getattr(self, 'solve_nonlinear')
        self._has_solve_nl = (overrides_method('solve_nonlinear', self, ImplicitComponent) or
                              solve_nl != self._inst_functs['solve_nonlinear'])

        new_apply_linear = getattr(self, 'apply_linear', None)

        self.matrix_free = (overrides_method('apply_linear', self, ImplicitComponent) or
                            (new_apply_linear is not None and
                             self._inst_functs['apply_linear'] != new_apply_linear))

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
            with self._call_user_function('apply_nonlinear', protect_outputs=True):
                if self._run_root_only():
                    if self.comm.rank == 0:
                        if self._discrete_inputs or self._discrete_outputs:
                            self.apply_nonlinear(self._inputs, self._outputs,
                                                 self._residuals_wrapper,
                                                 self._discrete_inputs, self._discrete_outputs)
                        else:
                            self.apply_nonlinear(self._inputs, self._outputs,
                                                 self._residuals_wrapper)
                        self.comm.bcast([self._residuals.asarray(), self._discrete_outputs], root=0)
                    else:
                        new_res, new_disc_outs = self.comm.bcast(None, root=0)
                        self._residuals.set_val(new_res)
                        if new_disc_outs:
                            for name, val in new_disc_outs.items():
                                self._discrete_outputs[name] = val
                else:
                    if self._discrete_inputs or self._discrete_outputs:
                        self.apply_nonlinear(self._inputs, self._outputs, self._residuals_wrapper,
                                             self._discrete_inputs, self._discrete_outputs)
                    else:
                        self.apply_nonlinear(self._inputs, self._outputs, self._residuals_wrapper)

        self.iter_count_apply += 1

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.
        """
        if self._nonlinear_solver is not None:
            with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
                self._nonlinear_solver._solve_with_cache_check()
        elif self._has_solve_nl:
            with self._unscaled_context(outputs=[self._outputs]):
                with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
                    with self._call_user_function('solve_nonlinear'):
                        if self._run_root_only():
                            if self.comm.rank == 0:
                                if self._discrete_inputs or self._discrete_outputs:
                                    self.solve_nonlinear(self._inputs, self._outputs,
                                                         self._discrete_inputs,
                                                         self._discrete_outputs)
                                else:
                                    self.solve_nonlinear(self._inputs, self._outputs)
                                self.comm.bcast([self._outputs.asarray(), self._discrete_outputs],
                                                root=0)
                            else:
                                new_res, new_disc_outs = self.comm.bcast(None, root=0)
                                self._outputs.set_val(new_res)
                                if new_disc_outs:
                                    for name, val in new_disc_outs.items():
                                        self._discrete_outputs[name] = val
                        else:
                            if self._discrete_inputs or self._discrete_outputs:
                                self.solve_nonlinear(self._inputs, self._outputs,
                                                     self._discrete_inputs, self._discrete_outputs)
                            else:
                                self.solve_nonlinear(self._inputs, self._outputs)

        # Iteration counter is incremented in the Recording context manager at exit.

    def _guess_nonlinear(self):
        """
        Provide initial guess for states.
        """
        if self._has_guess:
            self._apply_nonlinear()
            complex_step = self._inputs._under_complex_step

            try:
                with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
                    if complex_step:
                        self._inputs.set_complex_step_mode(False)
                        self._outputs.set_complex_step_mode(False)
                        self._residuals.set_complex_step_mode(False)

                    with self._call_user_function('guess_nonlinear', protect_residuals=True):
                        if self._discrete_inputs or self._discrete_outputs:
                            self.guess_nonlinear(self._inputs, self._outputs,
                                                 self._residuals_wrapper,
                                                 self._discrete_inputs, self._discrete_outputs)
                        else:
                            self.guess_nonlinear(self._inputs, self._outputs,
                                                 self._residuals_wrapper)
            finally:
                if complex_step:
                    self._inputs.set_complex_step_mode(True)
                    self._outputs.set_complex_step_mode(True)
                    self._residuals.set_complex_step_mode(True)

    def _apply_linear_wrapper(self, *args):
        """
        Call apply_linear based on the value of the "run_root_only" option.

        Parameters
        ----------
        *args : list
            List of positional arguments.
        """
        inputs, outputs, d_inputs, d_outputs, d_residuals, mode = args
        if self._run_root_only():
            if self.comm.rank == 0:
                self.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
                if mode == 'fwd':
                    self.comm.bcast(d_residuals.asarray(), root=0)
                else:  # rev
                    self.comm.bcast((d_inputs.asarray(), d_outputs.asarray()), root=0)
            else:
                if mode == 'fwd':
                    new_res = self.comm.bcast(None, root=0)
                    d_residuals.set_val(new_res)
                else:  # rev
                    new_ins, new_outs = self.comm.bcast(None, root=0)
                    d_inputs.set_val(new_ins)
                    d_outputs.set_val(new_outs)
        else:
            self.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, mode)

    def _apply_linear(self, jac, rel_systems, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        mode : str
            Either 'fwd' or 'rev'.
        scope_out : set or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : set or None
            Set of absolute input names in the scope of this mat-vec product.
            If None, all are in the scope.
        """
        if jac is None:
            jac = self._assembled_jac if self._assembled_jac is not None else self._jacobian

        with self._matvec_context(scope_out, scope_in, mode) as vecs:
            d_inputs, d_outputs, d_residuals = vecs
            d_residuals = self._dresiduals_wrapper

            # Jacobian and vectors are all scaled, unitless
            jac._apply(self, d_inputs, d_outputs, d_residuals, mode)

            # if we're not matrix free, we can skip the bottom of
            # this loop because apply_linear does nothing.
            if not self.matrix_free:
                return

            # Jacobian and vectors are all unscaled, dimensional
            with self._unscaled_context(
                    outputs=[self._outputs, d_outputs], residuals=[d_residuals]):

                # set appropriate vectors to read_only to help prevent user error
                if mode == 'fwd':
                    d_inputs.read_only = d_outputs.read_only = True
                elif mode == 'rev':
                    d_residuals.read_only = True

                try:
                    with self._call_user_function('apply_linear', protect_outputs=True):
                        self._apply_linear_wrapper(self._inputs, self._outputs,
                                                   d_inputs, d_outputs, d_residuals, mode)
                finally:
                    d_inputs.read_only = d_outputs.read_only = d_residuals.read_only = False

    def _solve_linear(self, mode, rel_systems, scope_out=_UNDEFINED, scope_in=_UNDEFINED):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        scope_out : set, None, or _UNDEFINED
            Outputs relevant to possible lower level calls to _apply_linear on Components.
        scope_in : set, None, or _UNDEFINED
            Inputs relevant to possible lower level calls to _apply_linear on Components.
        """
        if self._linear_solver is not None:
            self._linear_solver._set_matvec_scope(scope_out, scope_in)
            self._linear_solver.solve(mode, rel_systems)

        else:
            d_outputs = self._doutputs
            d_residuals = self._dresiduals_wrapper

            with self._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                # set appropriate vectors to read_only to help prevent user error
                if mode == 'fwd':
                    d_residuals.read_only = True
                elif mode == 'rev':
                    d_outputs.read_only = True

                try:
                    with self._call_user_function('solve_linear'):
                        self.solve_linear(d_outputs, d_residuals, mode)
                finally:
                    d_outputs.read_only = d_residuals.read_only = False

    def _approx_subjac_keys_iter(self):
        for abs_key, meta in self._subjacs_info.items():
            if 'method' in meta:
                method = meta['method']
                if method is not None and method in self._approx_schemes:
                    yield abs_key

    def _linearize_wrapper(self):
        """
        Call linearize based on the value of the "run_root_only" option.
        """
        with self._call_user_function('linearize', protect_outputs=True):
            if self._run_root_only():
                if self.comm.rank == 0:
                    if self._discrete_inputs or self._discrete_outputs:
                        self.linearize(self._inputs, self._outputs, self._jac_wrapper,
                                       self._discrete_inputs, self._discrete_outputs)
                    else:
                        self.linearize(self._inputs, self._outputs, self._jac_wrapper)
                    self.comm.bcast(list(self._jacobian.items()), root=0)
                else:
                    for key, val in self.comm.bcast(None, root=0):
                        self._jac_wrapper[key] = val
            else:
                if self._discrete_inputs or self._discrete_outputs:
                    self.linearize(self._inputs, self._outputs, self._jac_wrapper,
                                   self._discrete_inputs, self._discrete_outputs)
                else:
                    self.linearize(self._inputs, self._outputs, self._jac_wrapper)

    def _linearize(self, jac=None, sub_do_ln=True):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        """
        self._check_first_linearize()

        with self._unscaled_context(outputs=[self._outputs]):
            # Computing the approximation before the call to compute_partials allows users to
            # override FD'd values.
            for approximation in self._approx_schemes.values():
                approximation.compute_approximations(self, jac=self._jacobian)

            self._linearize_wrapper()

        if (jac is None or jac is self._assembled_jac) and self._assembled_jac is not None:
            self._assembled_jac._update(self)

    def add_output(self, name, val=1.0, **kwargs):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        **kwargs : dict
            Keyword args to store.  The value corresponding to each key is a dict containing the
            metadata for the input name that matches that key.

        Returns
        -------
        dict
            Metadata for added variable.
        """
        metadata = super().add_output(name, val, **kwargs)

        metadata['tags'].add('openmdao:allow_desvar')

        return metadata

    def add_residual(self, name, shape=(), units=None, desc='', ref=None):
        """
        Add an residual variable to the component.

        Note that the total size of the residual vector must match the total size of
        the outputs vector for this component.

        Parameters
        ----------
        name : str
            Name of the residual in this component's namespace.
        shape : int or tuple
            Shape of this residual.
        units : str or None
            Units in which this residual will be given to the user when requested.
            Default is None, which means it has no units.
        desc : str
            Description of the residual.
        ref : float or ndarray or None
            Scaling parameter. The value in the user-defined units of this residual
            when the scaled value is 1. Default is 1.

        Returns
        -------
        dict
            Metadata for the added residual.
        """
        if self._problem_meta is None:
            raise RuntimeError(f"{self.msginfo}: "
                               "A residual may only be added during component setup.")

        metadict = self._declared_residuals

        # Catch duplicated residuals
        if name in metadict:
            raise ValueError(f"{self.msginfo}: Residual name '{name}' already exists.")

        if self._problem_meta is not None:
            if self._problem_meta['setup_status'] >= _SetupStatus.POST_SETUP:
                raise RuntimeError(f"{self.msginfo}: Can't add residual '{name}' after setup.")

        # check ref shape
        if ref is not None:
            if np.isscalar(ref):
                self._has_resid_scaling |= ref != 1.0
            else:
                self._has_resid_scaling |= np.any(ref != 1.0)

                if not isinstance(ref, _allowed_types):
                    raise TypeError(f'{self.msginfo}: The ref argument should be a '
                                    'float, list, tuple, ndarray or Iterable')

                it = np.atleast_1d(ref)
                if shape is None:
                    shape = it.shape
                elif it.shape != shape:
                    raise ValueError(f"{self.msginfo}: When adding residual '{name}', expected "
                                     f"shape {shape} but got shape {it.shape} for argument 'ref'.")

        if units is not None:
            if not isinstance(units, str):
                raise TypeError(f"{self.msginfo}: The units argument should be a str or None")
            units = simplify_unit(units, msginfo=self.msginfo)

        metadict[name] = meta = {
            'shape': shape,
            'units': units,
            'desc': desc,
            'ref': format_as_float_or_array('ref', ref, flatten=True, val_if_none=None),
        }

        return meta

    def _compute_root_scale_factors(self):
        """
        Compute scale factors for all variables.

        Returns
        -------
        dict
            Mapping of each absoute var name to its corresponding scaling factor tuple.
        """
        if self._has_resid_scaling:
            scale_factors = super()._compute_root_scale_factors()
            prefix = self.pathname + '.' if self.pathname else ''
            for resid, meta in self._declared_residuals.items():
                ref = meta['ref']
                if ref is not None:
                    scale_factors[prefix + resid]['residual'] = (0.0, ref)
            return scale_factors
        else:
            return super()._compute_root_scale_factors()

    def _setup_procs(self, pathname, comm, mode, prob_meta):
        """
        Execute first phase of the setup process.

        Distribute processors, assign pathnames, and call setup on the component.

        Parameters
        ----------
        pathname : str
            Global name of the system, including the path.
        comm : MPI.Comm or <FakeComm>
            MPI communicator object.
        mode : str
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'rev'.
        prob_meta : dict
            Problem level metadata.
        """
        self._declared_residuals = {}
        super()._setup_procs(pathname, comm, mode, prob_meta)
        self._resid2out_subjac_map = {}

    def _resid_name_shape_iter(self):
        for name, meta in self._declared_residuals.items():
            yield name, meta['shape']

    def _setup_vectors(self, root_vectors):
        """
        Compute all vectors for all vec names and assign excluded variables lists.

        Parameters
        ----------
        root_vectors : dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        """
        super()._setup_vectors(root_vectors)

        if self._declared_residuals:
            self._check_res_out_overlaps()
            name2slcshape = _get_slice_shape_dict(self._resid_name_shape_iter())

            if self._use_derivatives:
                self._dresiduals_wrapper = _ResidsWrapper(self._dresiduals, name2slcshape)

            self._residuals_wrapper = _ResidsWrapper(self._residuals, name2slcshape)
            self._jac_wrapper = _JacobianWrapper(self._jacobian, self._resid2out_subjac_map)
        else:
            self._residuals_wrapper = self._residuals
            self._dresiduals_wrapper = self._dresiduals
            self._jac_wrapper = self._jacobian

    def _declare_partials(self, of, wrt, dct):
        """
        Store subjacobian metadata for later use.

        Parameters
        ----------
        of : tuple of str
            The names of the residuals that derivatives are being computed for.
            May also contain glob patterns.
        wrt : tuple of str
            The names of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain glob patterns.
        dct : dict
            Metadata dict specifying shape, and/or approx properties.
        """
        if self._declared_residuals:
            # if we have renamed resids, remap them to use output naming

            resbundle = self._find_partial_matches(of, wrt, use_resname=True)[0]

            plen = len(self.pathname) + 1 if self.pathname else 0
            rmap = self._resid2out_subjac_map
            for _, resids in resbundle:
                if not resids:
                    continue
                for tup in _overlap_range_iter(self._declared_residuals,
                                               self._var_abs2meta['output'], names1=resids):
                    resid, rstart, rend, oname, ostart, oend = tup

                    if resid not in rmap:
                        rmap[resid] = []

                    oslc, rslc = _get_overlap_slices(ostart, oend, rstart, rend)
                    rmap[resid].append((oname[plen:], wrt, dct, oslc, rslc))

            for resid, lst in self._resid2out_subjac_map.items():
                for oname, wrt, dct, _, _ in lst:
                    super()._declare_partials(oname, wrt, dct)
        else:
            super()._declare_partials(of, wrt, dct)

    def _check_res_vs_out_meta(self, resid, output):
        """
        Check for mismatch of 'ref' vs. 'res_ref' and 'units' vs. 'res_units'.

        Raises an exception if a mismatch exists.

        Parameters
        ----------
        resid : str
            Local name of residual that overlaps the output.
        output : str
            Local name of the output.
        """
        resmeta = self._declared_residuals[resid]
        outmeta = self._var_abs2meta['output'][output]
        loc_out = output[len(self.pathname) + 1:] if self.pathname else output

        ref = resmeta['ref']
        res_ref = outmeta['res_ref']

        if ref is not None and res_ref is not None:
            ref_arr = isinstance(ref, np.ndarray)
            res_ref_arr = isinstance(res_ref, np.ndarray)
            if (ref_arr != res_ref_arr or (ref_arr and not np.all(ref == res_ref) or
                                           (not ref_arr and ref != res_ref))):
                raise ValueError(f"{self.msginfo}: ({ref} != {res_ref}), 'ref' for residual "
                                 f"'{resid}' != 'res_ref' for output '{loc_out}'.")

        units = resmeta['units']
        res_units = outmeta['res_units']

        # assume units and res_units are already simplified
        if units is not None and res_units is not None and units != res_units:
            raise ValueError(f"{self.msginfo}: residual units '{units}' for residual '{resid}' != "
                             f"output res_units '{res_units}' for output '{loc_out}'.")

    def _check_res_out_overlaps(self):
        for resid, _, _, output, _, _ in _overlap_range_iter(self._declared_residuals,
                                                             self._var_abs2meta['output']):
            self._check_res_vs_out_meta(resid, output)

    def _get_partials_varlists(self, use_resname=False):
        """
        Get lists of 'of' and 'wrt' variables that form the partial jacobian.

        Parameters
        ----------
        use_resname : bool
            If True, 'of' will be a list of residual names instead of output names.

        Returns
        -------
        tuple(list, list)
            'of' and 'wrt' variable lists.
        """
        of = list(self._var_allprocs_prom2abs_list['output'])
        wrt = list(self._var_allprocs_prom2abs_list['input'])

        # filter out any discrete inputs or outputs
        if self._discrete_outputs:
            of = [n for n in of if n not in self._discrete_outputs]
        if self._discrete_inputs:
            wrt = [n for n in wrt if n not in self._discrete_inputs]

        if use_resname and self._declared_residuals:
            return list(self._declared_residuals), of + wrt

        return of, of + wrt

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None,
                        discrete_outputs=None):
        """
        Compute residuals given inputs and outputs.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        raise NotImplementedError('ImplicitComponent.apply_nonlinear() must be overridden '
                                  'by the child class.')

    def solve_nonlinear(self, inputs, outputs):
        """
        Compute outputs given inputs. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        """
        pass

    def guess_nonlinear(self, inputs, outputs, residuals,
                        discrete_inputs=None, discrete_outputs=None):
        """
        Provide initial guess for states.

        Override this method to set the initial guess for states.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        pass

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        r"""
        Compute jac-vector product. The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': (d_inputs, d_outputs) \|-> d_residuals

            'rev': d_residuals \|-> (d_inputs, d_outputs)

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        d_inputs : Vector
            See inputs; product must be computed only if var_name in d_inputs.
        d_outputs : Vector
            See outputs; product must be computed only if var_name in d_outputs.
        d_residuals : Vector
            See outputs.
        mode : str
            Either 'fwd' or 'rev'.
        """
        pass

    def solve_linear(self, d_outputs, d_residuals, mode):
        r"""
        Apply inverse jac product. The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': d_residuals \|-> d_outputs

            'rev': d_outputs \|-> d_residuals

        Note : this is not the linear solution for the implicit component. We use identity so
        that simple implicit components can function in a preconditioner under linear gauss-seidel.
        To correctly solve this component, you should slot a solver in linear_solver or override
        this method.

        Parameters
        ----------
        d_outputs : Vector
            Unscaled, dimensional quantities read via d_outputs[key].
        d_residuals : Vector
            Unscaled, dimensional quantities read via d_residuals[key].
        mode : str
            Either 'fwd' or 'rev'.
        """
        if mode == 'fwd':
            d_outputs.set_vec(d_residuals)
        else:  # rev
            d_residuals.set_vec(d_outputs)

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        """
        Compute sub-jacobian parts and any applicable matrix factorizations.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        jacobian : Jacobian
            Sub-jac components written to jacobian[output_name, input_name].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        pass

    def _list_states(self):
        """
        Return list of all states at and below this system.

        If final setup has not been performed yet, return relative names for this system only.

        Returns
        -------
        list
            List of all states.
        """
        prefix = self.pathname + '.' if self.pathname else ''
        return sorted(list(self._var_abs2meta['output']) +
                      [prefix + n for n in self._var_discrete['output']])

    def _list_states_allprocs(self):
        """
        Return list of all states for this component.

        Returns
        -------
        list
            List of all states.
        """
        return self._list_states()

    def has_declared_resids(self):
        """
        Return True if this System has declared residuals.

        Returns
        -------
        bool
            True if this System has declared residuals.
        """
        return len(self._declared_residuals) > 0


def meta2range_iter(meta_dict, names=None, shp_name='shape'):
    """
    Iterate over variables and their ranges, based on shape metadata for each variable.

    Parameters
    ----------
    meta_dict : dict
        Mapping of variable name to metadata (which contains shape information).
    names : iter of str or None
        If not None, restrict the ranges to those variables contained in names.
    shp_name : str
        Name of the shape metadata entry.  Defaults to 'shape', but could also be 'global_shape'.

    Yields
    ------
    str
        Name of variable.
    int
        Starting index.
    int
        Ending index.
    """
    start = end = 0

    if names is None:
        for name in meta_dict:
            end += shape_to_len(meta_dict[name][shp_name])
            yield name, start, end
            start = end
    else:
        if not isinstance(names, (set, dict)):
            names = set(names)

        for name in meta_dict:
            end += shape_to_len(meta_dict[name][shp_name])
            if name in names:
                yield name, start, end
            start = end


def _get_overlap_slices(ostart, oend, rstart, rend):
    """
    For an overlapping residual and output, return the slices where they overlap.
    """
    minend = min(oend, rend)
    start = max(rstart - ostart, 0)
    stop = minend - ostart
    if start == 0 and stop == (oend - ostart):
        oslc = _full_slice
    else:
        oslc = slice(start, stop)

    start = max(ostart - rstart, 0)
    stop = minend - rstart
    if start == 0 and stop == (rend - rstart):
        return oslc, _full_slice
    else:
        return oslc, slice(start, stop)


def _overlap_range_iter(meta_dict1, meta_dict2, names1=None, names2=None):
    """
    Yield names and ranges of overlapping variables from two metadata dictionaries.

    The metadata dicts are assumed to contain a 'shape' entry, and the total size of the
    variables in meta_dict1 must equal the total size of the variables in meta_dict2.
    """
    iter2 = meta2range_iter(meta_dict2, names=names2)
    start2 = end2 = -1

    for name1, start1, end1 in meta2range_iter(meta_dict1, names=names1):
        try:
            while not (start2 <= start1 < end2 or start2 <= end1 < end2):
                name2, start2, end2 = next(iter2)

            if end1 < end2:
                yield name1, start1, end1, name2, start2, end2
            else:
                while end1 >= end2:
                    yield name1, start1, end1, name2, start2, end2
                    name2, start2, end2 = next(iter2)
        except StopIteration:
            return


class _ResidsWrapper(object):
    def __init__(self, vec, name2slice_shape):
        self.__dict__['_vec'] = vec
        self.__dict__['_dct'] = name2slice_shape
        # check that vec size matches mapped resids size
        for slc, _ in name2slice_shape.values():
            pass
        if slc.stop != len(vec):
            system = vec._system()
            raise RuntimeError(f"{system.msginfo}: The number of residuals ({slc.stop}) doesn't "
                               f"match number of outputs ({len(vec)}).  If any residuals are "
                               "added using 'add_residuals', their total size must match the "
                               "total size of the outputs.")

    def __getitem__(self, name):
        arr = self._vec.asarray(copy=False)
        if name in self._dct:
            slc, shape = self._dct[name]
            view = arr[slc]
            view.shape = shape
            return view

        return self._vec.__getitem__(name)  # handles errors

    def __setitem__(self, name, val):
        arr = self._vec.asarray(copy=False)
        if name in self._dct:
            slc, _ = self._dct[name]
            arr[slc] = np.asarray(val).flat
            return

        self._vec.__setitem__(name, val)  # handles errors

    def __getattr__(self, name):
        return getattr(self._vec, name)

    def __setattr__(self, name, val):
        setattr(self._vec, name, val)


class _JacobianWrapper(object):
    def __init__(self, jac, res2outmap):
        self.__dict__['_jac'] = jac
        self.__dict__['_dct'] = res2outmap

    def __getitem__(self, key):
        res, wrt = key

        if len(self._dct) == 1:
            of, slc, _ = self._dct[res]
            return self._jac[(of, wrt)][slc]

        return np.vstack([self._jac[(of, wrt)][slc] for of, _, _, slc, _ in self._dct[res]])

    def __setitem__(self, key, val):
        res, wrt = key

        for of, _, _, outslc, resslc in self._dct[res]:
            if isinstance(val, np.ndarray):
                v = val[resslc]
            else:
                v = val
            if outslc is _full_slice:
                self._jac[of, wrt] = v
            else:
                # setting only subset of the rows in the subjac
                sjac = self._jac[of, wrt]
                sjac[outslc] = v
                self._jac[of, wrt] = sjac

    def __getattr__(self, name):
        return getattr(self._jac, name)

    def __setattr__(self, name, val):
        setattr(self._jac, name, val)
