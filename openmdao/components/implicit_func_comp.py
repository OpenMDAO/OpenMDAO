"""Define the ImplicitFuncComp class."""

import sys
import traceback
from itertools import chain
import numpy as np
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.constants import INT_DTYPE
import openmdao.func_api as omf
from openmdao.components.func_comp_common import _check_var_name, _copy_with_ignore, _add_options, \
    jac_forward, jac_reverse, _get_tangents
from openmdao.utils.array_utils import shape_to_len

try:
    import jax
    from jax import jit, jacfwd, jacrev
    from jax.config import config
    config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
except Exception:
    _, err, tb = sys.exc_info()
    if not isinstance(err, ImportError):
        traceback.print_tb(tb)
    jax = None


class ImplicitFuncComp(ImplicitComponent):
    """
    An implicit component that wraps a python function.

    Parameters
    ----------
    apply_nonlinear : function
        The function to be wrapped by this Component.
    solve_nonlinear : function or None
        Optional function to perform a nonlinear solve.
    linearize : function or None
        Optional function to compute partial derivatives.
    solve_linear : function or None
        Optional function to perform a linear solve.
    **kwargs : named args
        Args passed down to ImplicitComponent.

    Attributes
    ----------
    _apply_nonlinear_func : callable
        The function wrapper used by this component.
    _apply_nonlinear_func_jax : callable
        Function decorated to ensure use of jax numpy.
    _solve_nonlinear_func : function or None
        Optional function to do a nonlinear solve.
    solve_nonlinear : method
        Local override of _solve_nonlinear method.
    _solve_linear_func : function or None
        Optional function to do a linear solve.
    solve_linear : method
        Local override of solve_linear method.
    _linearize_func : function or None
        Optional function to compute partial derivatives.
    linearize : method
        Local override of linearize method.
    _linearize_info : object
        Some state information to compute in _linearize_func and pass to _solve_linear_func
    _tangents : tuple
        Tuple of parts of the tangent matrix cached for jax derivative computation.
    _jac2func_inds : ndarray
        Translation array from jacobian indices to function array indices.
    """

    def __init__(self, apply_nonlinear, solve_nonlinear=None, linearize=None, solve_linear=None,
                 **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._apply_nonlinear_func = omf.wrap(apply_nonlinear)
        self._solve_nonlinear_func = solve_nonlinear
        self._solve_linear_func = solve_linear
        self._linearize_func = linearize
        self._linearize_info = None
        self._tangents = None
        self._jac2func_inds = None

        if solve_nonlinear:
            self.solve_nonlinear = self._user_solve_nonlinear
            self._has_solve_nl = True
        if linearize:
            self.linearize = self._user_linearize
        if solve_linear:
            self.solve_linear = self._user_solve_linear

        if self._apply_nonlinear_func._use_jax:
            self.options['use_jax'] = True

        # setup requires an undecorated, unjitted function, so do it now
        if self._apply_nonlinear_func._call_setup:
            self._apply_nonlinear_func._setup()

        if self.options['use_jax']:
            if jax is None:
                raise RuntimeError(f"{self.msginfo}: jax is not installed. Try 'pip install jax'.")
            self._apply_nonlinear_func_jax = omf.jax_decorate(self._apply_nonlinear_func._f)

        if self.options['use_jax'] and self.options['use_jit']:
            static_argnums = [i for i, m in enumerate(self._apply_nonlinear_func._inputs.values())
                              if 'is_option' in m]
            try:
                with omf.jax_context(self._apply_nonlinear_func._f.__globals__):
                    self._apply_nonlinear_func_jax = jit(self._apply_nonlinear_func_jax,
                                                         static_argnums=static_argnums)
            except Exception as err:
                raise RuntimeError(f"{self.msginfo}: failed jit compile of solve_nonlinear "
                                   f"function: {err}")

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        _add_options(self)

    def setup(self):
        """
        Define our inputs and outputs.
        """
        optignore = {'is_option'}

        for name, meta in self._apply_nonlinear_func.get_input_meta():
            _check_var_name(self, name)
            if 'is_option' in meta and meta['is_option']:
                kwargs = _copy_with_ignore(meta, omf._allowed_declare_options_args,
                                           ignore=optignore)
                self.options.declare(name, **kwargs)
            else:
                kwargs = omf._filter_dict(meta, omf._allowed_add_input_args)
                self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._apply_nonlinear_func.get_output_meta()):
            _check_var_name(self, name)
            kwargs = _copy_with_ignore(meta, omf._allowed_add_output_args, ignore=('resid',))
            self.add_output(name, **kwargs)

    def declare_partials(self, *args, **kwargs):
        """
        Declare information about this component's subjacobians.

        Parameters
        ----------
        *args : list
            Positional args to be passed to base class version of declare_partials.
        **kwargs : dict
            Keyword args  to be passed to base class version of declare_partials.

        Returns
        -------
        dict
            Metadata dict for the specified partial(s).
        """
        if self._linearize_func is None and ('method' not in kwargs or
                                             kwargs['method'] == 'exact'):
            raise RuntimeError(f"{self.msginfo}: declare_partials must be called with method equal "
                               "to 'cs', 'fd', or 'jax'.")

        return super().declare_partials(*args, **kwargs)

    def _setup_partials(self):
        """
        Check that all partials are declared.
        """
        kwargs = self._apply_nonlinear_func.get_declare_coloring()
        if kwargs is not None:
            self.declare_coloring(**kwargs)

        for kwargs in self._apply_nonlinear_func.get_declare_partials():
            self.declare_partials(**kwargs)

        super()._setup_partials()

    def apply_nonlinear(self, inputs, outputs, residuals,
                        discrete_inputs=None, discrete_outputs=None):
        """
        R = Ax - b.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        discrete_inputs : _DictValues or None
            Dict-like object containing discrete inputs.
        discrete_outputs : _DictValues or None
            Dict-like object containing discrete outputs.
        """
        residuals.set_vals(self._apply_nonlinear_func(*self._ordered_func_invals(inputs, outputs)))

    def _user_solve_nonlinear(self, inputs, outputs):
        """
        Compute outputs. The model is assumed to be in a scaled state.
        """
        self._outputs.set_vals(self._solve_nonlinear_func(*self._ordered_func_invals(inputs,
                                                                                     outputs)))

    def _linearize(self, jac=None, sub_do_ln=False):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            Ignored.
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        """
        if self.options['use_jax']:
            self._check_first_linearize()
            self._jax_linearize()
            if (jac is None or jac is self._assembled_jac) and self._assembled_jac is not None:
                self._assembled_jac._update(self)
        else:
            super()._linearize(jac, sub_do_ln)

    def _jax_linearize(self):
        """
        Compute the jacobian using jax.

        This updates self._jacobian.
        """
        func = self._apply_nonlinear_func
        # argnums specifies which position args are to be differentiated
        inames = list(func.get_input_names())
        argnums = aa = [i for i, m in enumerate(func._inputs.values()) if 'is_option' not in m]
        if len(argnums) == len(inames):
            argnums = None  # speedup if there are no static args
        osize = len(self._outputs)
        isize = len(self._inputs) + osize
        invals = list(self._ordered_func_invals(self._inputs, self._outputs))
        coloring = self._coloring_info['coloring']

        if self._mode == 'rev':  # use reverse mode to compute derivs
            outvals = tuple(self._outputs.values())
            tangents = self._get_tangents(outvals, 'rev', coloring)
            if coloring is not None:
                j = [np.asarray(a).reshape((a.shape[0], shape_to_len(a.shape[1:])))
                     for a in jac_reverse(self._apply_nonlinear_func_jax, argnums,
                                          tangents)(*invals)]
                j = coloring.expand_jac(np.hstack(self._reorder_col_chunks(j)), 'rev')
            else:
                j = []
                for a in jac_reverse(self._apply_nonlinear_func_jax, argnums, tangents)(*invals):
                    a = np.asarray(a)
                    if a.ndim < 2:
                        a = a.reshape((a.size, 1))
                    else:
                        a = a.reshape((a.shape[0], shape_to_len(a.shape[1:])))
                    j.append(a)
                j = np.hstack(self._reorder_col_chunks(j)).reshape((osize, isize))
        else:
            if coloring is not None:
                tangents = self._get_tangents(invals, 'fwd', coloring, argnums,
                                              trans=self._get_jac2func_inds(self._inputs,
                                                                            self._outputs))
                j = [np.asarray(a).reshape((shape_to_len(a.shape[:-1]), a.shape[-1]))
                     for a in jac_forward(self._apply_nonlinear_func_jax, argnums,
                                          tangents)(*invals)]
                j = coloring.expand_jac(np.vstack(j), 'fwd')
            else:
                tangents = self._get_tangents(invals, 'fwd', coloring, argnums)
                j = []
                for a in jac_forward(self._apply_nonlinear_func_jax, argnums, tangents)(*invals):
                    a = np.asarray(a)
                    if a.ndim < 2:
                        a = a.reshape((1, a.size))
                    else:
                        a = a.reshape((shape_to_len(a.shape[:-1]), a.shape[-1]))
                    j.append(a)
                j = self._reorder_cols(np.vstack(j).reshape((osize, isize)))

        self._jacobian.set_dense_jac(self, j)

    def _user_linearize(self, inputs, outputs, jacobian):
        """
        Calculate the partials of the residual for each balance.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        jacobian : Jacobian
            Sub-jac components written to jacobian[output_name, input_name].
        """
        self._linearize_info = self._linearize_func(*chain(self._ordered_func_invals(inputs,
                                                                                     outputs),
                                                           (jacobian,)))

    def _user_solve_linear(self, d_outputs, d_residuals, mode):
        r"""
        Run solve_linear function if there is one.

        Parameters
        ----------
        d_outputs : Vector
            Unscaled, dimensional quantities read via d_outputs[key].
        d_residuals : Vector
            Unscaled, dimensional quantities read via d_residuals[key].
        mode : str
            Derivative solution direction, either 'fwd' or 'rev'.
        """
        if mode == 'fwd':
            d_outputs.set_vals(self._solve_linear_func(*chain(d_residuals.values(),
                                                              (mode, self._linearize_info))))
        else:  # rev
            d_residuals.set_vals(self._solve_linear_func(*chain(d_outputs.values(),
                                                                (mode, self._linearize_info))))

    def _ordered_func_invals(self, inputs, outputs):
        """
        Yield function input args in their proper order.

        In OpenMDAO, states are outputs, but for our some of our functions they are inputs, so
        this function yields the values of the inputs and states in the same order that they
        were originally given for the _apply_nonlinear_func.

        Parameters
        ----------
        inputs : Vector
            The input vector.
        outputs : Vector
            The output vector (contains the states).

        Yields
        ------
        float or ndarray
            Value of input or state variable.
        """
        inps = inputs.values()
        outs = outputs.values()

        for name, meta in self._apply_nonlinear_func._inputs.items():
            if 'is_option' in meta:  # it's an option
                yield self.options[name]
            elif 'resid' in meta:  # it's a state
                yield next(outs)
            else:
                yield next(inps)

    def _get_jac2func_inds(self, inputs, outputs):
        """
        Return a translation array from jac column indices into function input ordering.

        Parameters
        ----------
        inputs : Vector
            The input vector.
        outputs : Vector
            The output vector (contains the states).

        Returns
        -------
        ndarray
            Index translation array
        """
        if self._jac2func_inds is None:
            inds = np.arange(len(outputs) + len(inputs), dtype=INT_DTYPE)
            indict = {}
            start = end = 0
            for n, meta in self._apply_nonlinear_func._inputs.items():
                if 'is_option' not in meta:
                    end += shape_to_len(meta['shape'])
                    indict[n] = inds[start:end]
                    start = end

            inds = [indict[n] for n in chain(outputs, inputs)]
            self._jac2func_inds = np.concatenate(inds)

        return self._jac2func_inds

    def _reorder_col_chunks(self, col_chunks):
        """
        Return jacobian column chunks in correct OpenMDAO order (outputs first, then inputs).

        This is needed in rev mode because the return values of the jacrev function are ordered
        based on the order of the function inputs, which may be different than OpenMDAO's
        required order.

        Parameters
        ----------
        col_chunks : list of ndarray
            List of column chunks to be reordered

        Returns
        -------
        list
            Chunks in OpenMDAO jacobian order.
        """
        inps = []
        ordered_chunks = []
        chunk_iter = iter(col_chunks)
        for meta in self._apply_nonlinear_func._inputs.values():
            if 'is_option' in meta:  # it's an option
                pass  # skip it (don't include in jacobian)
            elif 'resid' in meta:  # it's a state
                ordered_chunks.append(next(chunk_iter))
            else:
                inps.append(next(chunk_iter))
        return ordered_chunks + inps

    def _reorder_cols(self, arr, coloring=None):
        """
        Reorder the columns of jacobian row chunks in fwd mode.

        Parameters
        ----------
        arr : ndarray
            Jacobian or compressed jacobian.
        coloring : Coloring or None
            Coloring object.

        Returns
        -------
        ndarray
            Reordered array.
        """
        if coloring is None:
            trans = self._get_jac2func_inds(self._inputs, self._outputs)
            return arr[:, trans]
        else:
            trans = self._get_jac2func_inds(self._inputs, self._outputs)
            J = np.zeros(coloring._shape)
            for col, nzpart, icol in coloring.colored_jac_iter(arr, 'fwd', trans):
                J[nzpart, icol] = col
            return J

    def _get_tangents(self, vals, direction, coloring=None, argnums=None, trans=None):
        """
        Return a tuple of tangents values for use with vmap.

        Parameters
        ----------
        vals : list
            List of function input values.
        direction : str
            Derivative computation direction ('fwd' or 'rev').
        coloring : Coloring or None
            If not None, the Coloring object used to compute a compressed tangent array.
        argnums : list of int or None
            Indices of dynamic (differentiable) function args.
        trans : ndarray
            Translation array from jacobian indices into function arg indices.  This is needed
            because OpenMDAO expects ordering to be outputs first, then inputs, but function args
            could be in any order.

        Returns
        -------
        tuple of ndarray or ndarray
            The tangents values to be passed to vmap.
        """
        if self._tangents is None:
            self._tangents = _get_tangents(vals, direction, coloring, argnums, trans)
        return self._tangents

    def _compute_coloring(self, recurse=False, **overrides):
        """
        Compute a coloring of the partial jacobian.

        This assumes that the current System is in a proper state for computing derivatives.
        It just calls the base class version and then resets the tangents so that after coloring
        a new set of compressed tangents values can be computed.

        Parameters
        ----------
        recurse : bool
            If True, recurse from this system down the system hierarchy.  Whenever a group
            is encountered that has specified its coloring metadata, we don't recurse below
            that group unless that group has a subsystem that has a nonlinear solver that uses
            gradients.
        **overrides : dict
            Any args that will override either default coloring settings or coloring settings
            resulting from an earlier call to declare_coloring.

        Returns
        -------
        list of Coloring
            The computed colorings.
        """
        ret = super()._compute_coloring(recurse, **overrides)
        self._tangents = None  # reset to compute new colored tangents later
        return ret
