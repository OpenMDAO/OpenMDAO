"""Define the ExplicitFuncComp class."""

import sys
import traceback
import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.constants import INT_DTYPE
import openmdao.func_api as omf
from openmdao.components.func_comp_common import _check_var_name, _copy_with_ignore, _add_options, \
    jac_forward, jac_reverse, _get_tangents
from openmdao.utils.array_utils import shape_to_len

try:
    import jax
    from jax import jit
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
except Exception:
    _, err, tb = sys.exc_info()
    if not isinstance(err, ImportError):
        traceback.print_tb(tb)
    jax = None

if jax is not None:
    try:
        from jax import Array as JaxArray
    except ImportError:
        # versions of jax before 0.3.18 do not have the jax.Array base class
        raise RuntimeError(f"An unsupported version of jax is installed. "
                           "OpenMDAO requires 'jax>=4.0' and 'jaxlib>=4.0'. "
                           "Try 'pip install openmdao[jax]' with Python>=3.8.")


class ExplicitFuncComp(ExplicitComponent):
    """
    A component that wraps a python function.

    Parameters
    ----------
    compute : function
        The function to be wrapped by this Component.
    compute_partials : function or None
        If not None, call this function when computing partials.
    **kwargs : named args
        Args passed down to ExplicitComponent.

    Attributes
    ----------
    _compute : callable
        The function wrapper used by this component.
    _compute_jax : callable
        Function decorated to ensure use of jax numpy.
    _compute_partials : function or None
        If not None, call this function when computing partials.
    _tangents : tuple
        Tuple of parts of the tangent matrix cached for jax derivative computation.
    _tangent_direction : str
        Direction of the last tangent computation.
    """

    def __init__(self, compute, compute_partials=None, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._compute = omf.wrap(compute)
        # in case we're doing jit, force setup of wrapped func because we compute output shapes
        # during setup and that won't work on a jit compiled function
        if self._compute._call_setup:
            self._compute._setup()

        if self._compute._use_jax:
            self.options['use_jax'] = True

        if self.options['use_jax']:
            if jax is None:
                raise RuntimeError(f"{self.msginfo}: jax is not installed. "
                                   "Try 'pip install openmdao[jax]' with Python>=3.8.")
            self._compute_jax = omf.jax_decorate(self._compute._f)

        self._tangents = None
        self._tangent_direction = None

        self._compute_partials = compute_partials
        if self.options['use_jax'] and self.options['use_jit']:
            static_argnums = [i for i, m in enumerate(self._compute._inputs.values())
                              if 'is_option' in m]
            try:
                self._compute_jax = jit(self._compute_jax, static_argnums=static_argnums)
            except Exception as err:
                raise RuntimeError(f"{self.msginfo}: failed jit compile of compute function: {err}")

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        _add_options(self)

    def setup(self):
        """
        Define out inputs and outputs.
        """
        optignore = {'is_option'}
        use_jax = self.options['use_jax'] and jax is not None

        for name, meta in self._compute.get_input_meta():
            _check_var_name(self, name)
            if 'is_option' in meta and meta['is_option']:
                kwargs = _copy_with_ignore(meta, omf._allowed_declare_options_args,
                                           ignore=optignore)
                self.options.declare(name, **kwargs)
            else:
                kwargs = omf._filter_dict(meta, omf._allowed_add_input_args)
                if use_jax:
                    # make sure internal openmdao values are numpy arrays and not jax Arrays
                    self._dev_arrays_to_np_arrays(kwargs)
                self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._compute.get_output_meta()):
            _check_var_name(self, name)
            kwargs = _copy_with_ignore(meta, omf._allowed_add_output_args, ignore=('resid',))
            if use_jax:
                # make sure internal openmdao values are numpy arrays and not jax Arrays
                self._dev_arrays_to_np_arrays(kwargs)
            self.add_output(name, **kwargs)

    def _dev_arrays_to_np_arrays(self, meta):
        if 'val' in meta:
            if isinstance(meta['val'], JaxArray):
                meta['val'] = np.asarray(meta['val'])

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
            if self._mode != self._tangent_direction:
                # force recomputation of coloring and tangents
                self._first_call_to_linearize = True
                self._tangents = None
            self._check_first_linearize()
            self._jax_linearize()
        else:
            super()._linearize(jac, sub_do_ln)

    def _jax_linearize(self):
        """
        Compute the jacobian using jax.

        This updates self._jacobian.
        """
        inames = list(self._compute.get_input_names())
        # argnums specifies which position args are to be differentiated
        argnums = [i for i, m in enumerate(self._compute._inputs.values()) if 'is_option' not in m]
        # keep this around for use locally even if we pass None as argnums to jax
        argidxs = argnums
        if len(argnums) == len(inames):
            argnums = None  # speedup if there are no static args
        osize = len(self._outputs)
        isize = len(self._inputs)
        invals = list(self._func_values(self._inputs))
        coloring = self._coloring_info.coloring
        func = self._compute_jax

        if self._mode == 'rev':  # use reverse mode to compute derivs
            outvals = tuple(self._outputs.values())
            tangents = self._get_tangents(outvals, 'rev', coloring)
            if coloring is None:
                j = np.empty((osize, isize), dtype=float)
                cstart = cend = 0
                for i, a in zip(argidxs, jac_reverse(func, argnums, tangents)(*invals)):
                    if isinstance(invals[i], np.ndarray):
                        cend += invals[i].size
                    else:  # must be a scalar
                        cend += 1
                    a = np.asarray(a)
                    if a.ndim < 2:
                        j[:, cstart:cend] = a.reshape((a.size, 1))
                    else:
                        j[:, cstart:cend] = a.reshape((a.shape[0], cend - cstart))
                    cstart = cend
            else:
                j = [np.asarray(a).reshape((a.shape[0], shape_to_len(a.shape[1:])))
                     for a in jac_reverse(func, argnums, tangents)(*invals)]
                j = coloring.expand_jac(np.hstack(j), 'rev')
        else:
            tangents = self._get_tangents(invals, 'fwd', coloring, argnums)
            if coloring is None:
                j = np.empty((osize, isize), dtype=float)
                start = end = 0
                for a in jac_forward(func, argnums, tangents)(*invals):
                    a = np.asarray(a)
                    if a.ndim < 2:
                        a = a.reshape((1, a.size))
                    else:
                        a = a.reshape((shape_to_len(a.shape[:-1]), a.shape[-1]))
                    end += a.shape[0]
                    if osize == 1:
                        j[0, start:end] = a
                    else:
                        j[start:end, :] = a
                    start = end
            else:
                j = [np.asarray(a).reshape((shape_to_len(a.shape[:-1]), a.shape[-1]))
                     for a in jac_forward(func, argnums, tangents)(*invals)]
                j = coloring.expand_jac(np.vstack(j), 'fwd')

        self._jacobian.set_dense_jac(self, j)

    def _get_tangents(self, vals, direction, coloring=None, argnums=None):
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

        Returns
        -------
        tuple of ndarray or ndarray
            The tangents values to be passed to vmap.
        """
        if self._tangents is None:
            self._tangents = _get_tangents(vals, direction, coloring, argnums)
            self._tangent_direction = direction
        return self._tangents

    def compute(self, inputs, outputs):
        """
        Compute the result of calling our function with the given inputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables.
        outputs : Vector
            Unscaled, dimensional output variables.
        """
        outputs.set_vals(self._compute(*self._func_values(inputs)))

    def _setup_partials(self):
        """
        Check that all partials are declared.
        """
        for kwargs in self._compute.get_declare_partials():
            self.declare_partials(**kwargs)

        kwargs = self._compute.get_declare_coloring()
        if kwargs is not None:
            self.declare_coloring(**kwargs)

        super()._setup_partials()

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Sub-jac components written to partials[output_name, input_name].
        """
        if self._compute_partials is None:
            return

        self._compute_partials(*self._func_values(inputs), partials)

    def _func_values(self, inputs):
        """
        Yield current function input args.

        Parameters
        ----------
        inputs : Vector
            The input vector.

        Yields
        ------
        object
            Value of current function input variable.
        """
        inps = inputs.values()

        for name, meta in self._compute._inputs.items():
            if 'is_option' in meta:
                yield self.options[name]
            else:
                yield next(inps)

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
