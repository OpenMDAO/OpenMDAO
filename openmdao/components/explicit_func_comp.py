"""Define the FuncComponent class."""

from functools import partial
import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.constants import INT_DTYPE
import openmdao.func_api as omf
from openmdao.components.func_comp_common import _check_var_name, _copy_with_ignore, _add_options
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.array_utils import identity_column_iter

try:
    import jax
    from jax import jit, jacfwd, jacrev, vmap, jvp, linear_util
    import jax.numpy as jnp
    from jax.numpy import DeviceArray
    from jax.config import config
    from jax.api_util import argnums_partial
    from jax._src.api import _jvp
    config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
except ImportError as err:
    jax = None


def jac_forward(fun, argnums, tangents):
    """
    Similar to the jax.jacfwd function but allows specification of the tangent matrix.

    This allows us to generate a compressed jacobian based on coloring.

    Parameters
    ----------
    fun : function
        The function to be differentiated.
    argnums : tuple of int or None
        Specifies which positional args are dynamic.  None means all positional args are dynamic.
    tangents : ndarray
        Array of 1.0's and 0's that is used to compute the value of the jacobian matrix.

    Returns
    -------
    function
        A function that returns rows of the jacobian grouped by output variable, e.g., if there were
        2 output variables of size 3 and 4, the function would return a list with two entries. The
        first entry would contain the first 3 rows of J and the second would contain the next
        4 rows of J.
    """
    if argnums is None:
        def jacfun(*args, **kwargs):
            f = linear_util.wrap_init(fun, kwargs)
            return vmap(partial(_jvp, f, args), out_axes=(None, -1))(tangents)[1]
    else:
        def jacfun(*args, **kwargs):
            f = linear_util.wrap_init(fun, kwargs)
            f_partial, dyn_args = argnums_partial(f, argnums, args)
            return vmap(partial(_jvp, f_partial, dyn_args), out_axes=(None, -1))(tangents)[1]
    return jacfun


def jacvec_prod(fun, argnums, invals, tangent):
    """
    Similar to the jvp function but gives back a flat column.

    Parameters
    ----------
    fun : function
        The function to be differentiated.
    argnums : tuple of int or None
        Specifies which positional args are dynamic.  None means all positional args are dynamic.
    tangent : ndarray
        Array of 1.0's and 0's that is used to compute a column of the jacobian matrix.

    Returns
    -------
    function
        A function to compute the jacobian vector product.
    """
    f = linear_util.wrap_init(fun)
    if argnums is not None:
        invals = list(argnums_partial(f, argnums, invals)[1])

    # compute shaped tangents to use later
    sizes = np.array([jnp.size(a) for a in invals])
    inds = np.cumsum(sizes[:-1])
    shaped_tangents = [a.reshape(s.shape) for a, s in zip(np.split(tangent, inds, axis=0), invals)]

    if argnums is None:
        def jvfun(inps):
            return _jvp(f, inps, shaped_tangents)[1]
    else:
        def jvfun(inps):
            f_partial, dyn_args = argnums_partial(f, argnums, inps)
            return _jvp(f_partial, list(dyn_args), shaped_tangents)[1]

    return jvfun


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
            self._compute_jax = omf.jax_decorate(self._compute._f)

        self._tangents = None

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
                    # make sure internal openmdao values are numpy arrays and not DeviceArrays
                    self._dev_arrays_to_np_arrays(kwargs)
                self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._compute.get_output_meta()):
            _check_var_name(self, name)
            kwargs = _copy_with_ignore(meta, omf._allowed_add_output_args, ignore=('resid',))
            if use_jax:
                # make sure internal openmdao values are numpy arrays and not DeviceArrays
                self._dev_arrays_to_np_arrays(kwargs)
            self.add_output(name, **kwargs)

    def _dev_arrays_to_np_arrays(self, meta):
        if 'val' in meta:
            if isinstance(meta['val'], DeviceArray):
                meta['val'] = np.asarray(meta['val'])

    def _linearize(self, jac=None, sub_do_ln=False):
        if self.options['use_jax']:
            self._check_first_linearize()

            # argnums specifies which position args are to be differentiated
            argnums = [i for i, m in enumerate(self._compute._inputs.values())
                       if 'is_option' not in m]
            inames = list(self._compute.get_input_names())
            onames = list(self._compute.get_output_names())
            nouts = len(onames)
            osize = len(self._outputs)
            isize = len(self._inputs)
            invals = list(self._func_values(self._inputs))
            coloring = self._coloring_info['coloring']

            # since rev mode is more expensive, adjust size comparison a little to pick fwd
            # in some cases where input size is slightly larger than output size
            if self._mode != 'fwd' and osize * 1.3 < isize:  # use reverse mode to compute derivs
                assert coloring is None, "no coloring defined yet for rev mode"
                j = jacrev(self._compute_jax, argnums)(*invals)
                if nouts == 1:
                    for col, inp in zip(argnums, inames):
                        abs_key = self._jacobian._get_abs_key((onames[0], inp))
                        if abs_key in self._jacobian:
                            self._jacobian[abs_key] = np.asarray(j[col])
                else:
                    for col, inp in zip(argnums, inames):
                        for row, out in enumerate(onames):
                            abs_key = self._jacobian._get_abs_key((out, inp))
                            if abs_key in self._jacobian:
                                self._jacobian[abs_key] = np.asarray(j[row][col])
            else:
                if len(argnums) == len(inames):
                    argnums = None  # speedup if there are no static args

                tangents = self._get_tangents(invals, coloring, argnums)
                if coloring is not None:
                    j = [np.asarray(a).reshape((np.prod(a.shape[:-1], dtype=INT_DTYPE), a.shape[-1]))
                         for a in jac_forward(self._compute_jax, argnums, tangents)(*invals)]
                    J = coloring.expand_jac(np.vstack(j), 'fwd')
                    self._jacobian.set_dense_jac(self, J)
                else:
                    j = []
                    for a in jac_forward(self._compute_jax, argnums, tangents)(*invals):
                        if a.ndim > 2:
                            j.append(np.asarray(a).reshape((np.prod(a.shape[:-1], dtype=INT_DTYPE), a.shape[-1])))
                        else:  # a scalar
                            j.append(np.atleast_2d(a))
                    J = np.vstack(j).reshape((osize, isize))
                    self._jacobian.set_dense_jac(self, J)
        else:
            super()._linearize(jac, sub_do_ln)

    # TODO: modify this to give tangents for jac_forward (uses vmap) OR tangents for jvp/vjp
    def _get_tangents(self, invals, coloring=None, argnums=None):
        if self._tangents is None:
            if argnums is None:
                ins = invals
            else:
                ins = [invals[i] for i in argnums]
            sizes = [np.size(a) for a in ins]
            inds = np.cumsum(sizes[:-1])
            axis = 1
            ndim = np.sum(sizes)
            if coloring is None:
                tangent = jnp.eye(ndim)
            else:
                tangent = coloring.tangent_matrix('fwd')
            shapes = [tangent.shape[:axis] + np.shape(v) + tangent.shape[axis + 1:] for v in ins]
            self._tangents = tuple([jnp.reshape(a, shp) for a, shp in
                                    zip(jnp.split(tangent, inds, axis=axis), shapes)])
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
        return super().declare_partials(*args, **kwargs)

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
        ret = super()._compute_coloring(recurse, **overrides)
        self._tangents = None  # reset to compute new colored tangents later
        return ret

    def _update_jac_cols(self, direction=None):
        """
        Compute a jacobian using randomized inputs to generate a sparsity matrix.

        This is called 1 or more times from compute_coloring.
        """
        coloring = self._coloring_info['coloring']
        if coloring is not None:
            for i, result in self._colored_col_iter(direction):
                self._jacobian.set_col(self, i, result)
        else:
            for i, result in self._uncolored_col_iter(direction):
                self._jacobian.set_col(self, i, result)

    def _colored_col_iter(self, direction):
        coloring = self._coloring_info['coloring']

        assert direction == 'fwd'
        # TODO: update this to use rev mode for 'wide' jacobians
        # argnums specifies which position args are to be differentiated
        argnums = [i for i, m in enumerate(self._compute._inputs.values())
                   if 'is_option' not in m]
        invals = list(self._inputs.values())
        size = len(self._inputs) if direction == 'fwd' else len(self._outputs)
        tangent = np.empty(size)
        scratch = np.empty(len(self._outputs) if direction == 'fwd' else len(self._inputs))
        jcol = jacvec_prod(self._compute_jax, argnums, invals, tangent)
        it = coloring.tangent_iter(direction, tangent)
        for i, tres in enumerate(it):
            result = np.asarray(jcol(invals)).ravel()
            _, nzs, nzparts = tres
            for i, parts in zip(nzs, nzparts):
                scratch[:] = 0.
                scratch[parts] = result[parts]
                yield i, scratch

    def _uncolored_col_iter(self, direction):
        if direction is None:
            direction = 'fwd'
        # TODO: update this to use rev mode for 'wide' jacobians
        # argnums specifies which position args are to be differentiated
        argnums = [i for i, m in enumerate(self._compute._inputs.values())
                   if 'is_option' not in m]
        invals = list(self._inputs.values())
        size = len(self._inputs) if direction == 'fwd' else len(self._outputs)
        tangent = np.empty(size)
        jcol = jacvec_prod(self._compute_jax, argnums, invals, tangent)
        for i, _ in enumerate(identity_column_iter(tangent)):
            # this updates the tangent array each iter
            yield i, np.asarray(jcol(invals)).ravel()
