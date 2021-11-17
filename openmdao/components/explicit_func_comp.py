"""Define the FuncComponent class."""

from functools import partial
import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.constants import INT_DTYPE
import openmdao.func_api as omf
from openmdao.components.func_comp_common import _check_var_name, _copy_with_ignore, _add_options

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


def jac_forward(fun, argnums, seed_mat):
    """
    Similar to the jax.jacfwd function but allows specification of the seed matrix.

    Parameters
    ----------
    fun : function
        The function to be differentiated.
    argnums : tuple of int or None
        Specifies which positional args are dynamic.  None means all positional args are dynamic.
    seed_mat : ndarray
        Array of 1.0's and 0's that is used to compute the value of the jacobian matrix.
        jax calls these 'tangents'.

    Returns
    -------
    function
        A function that returns rows of the jacobian grouped by output variable, e.g., if there were
        2 output variables of size 3 and 4, the function would return a list with two entries. The
        first entry would contain the first 3 rows of J and the second would contain the next
        4 rows of J.
    """
    def jacfun(*args, **kwargs):
        f = linear_util.wrap_init(fun, kwargs)
        if argnums is not None:
            f_partial, dyn_args = argnums_partial(f, argnums, args)
            pushfwd = partial(_jvp, f_partial, dyn_args)
        else:
            pushfwd = partial(_jvp, f, args)
        return vmap(pushfwd, out_axes=(None, -1))(seed_mat)[1]
    return jacfun


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
    _jaxseeds : tuple
        Tuple of parts of the seed matrix cached for jax derivative computation.
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

        self._jaxseeds = None

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
            # force _jvp and/or _vjp to be re-initialized the next time they are needed
            # is called
            self._jvp = None
            self._vjp = None
            # argnums specifies which position args are to be differentiated
            argnums = [i for i, m in enumerate(self._compute._inputs.values())
                       if 'is_option' not in m]
            inames = list(self._compute.get_input_names())
            onames = list(self._compute.get_output_names())
            nouts = len(onames)
            osize = len(self._outputs)
            isize = len(self._inputs)
            invals = list(self._func_values(self._inputs))

            # since rev mode is more expensive, adjust size comparison a little to pick fwd
            # in some cases where input size is slightly larger than output size
            if osize * 1.3 < isize:  # use reverse mode to compute derivs
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
                # j = jacfwd(self._compute_jax, argnums)(*invals)
                if len(argnums) == len(inames):
                    argnums = None
                j = [np.asarray(a) for a in jac_forward(self._compute_jax, argnums,
                                                        self._get_seeds(invals))(*invals)]
                start = end = 0
                for inp, meta in self._compute.get_input_meta():
                    if 'is_option' in meta:
                        continue
                    end += int(np.product(meta['shape']))
                    for out, sub in zip(onames, j):
                        abs_key = self._jacobian._get_abs_key((out, inp))
                        if abs_key in self._jacobian:
                            self._jacobian[abs_key] = sub[:, start:end]
                    start = end
        else:
            super()._linearize(jac, sub_do_ln)

    def _get_seeds(self, invals):
        if self._jaxseeds is None:
            sizes = [np.size(a) for a in invals]
            ndim = np.sum(sizes)
            seedmat = jnp.eye(ndim)
            inds = np.cumsum(sizes[:-1])
            axis = 1
            shapes = [seedmat.shape[:axis] + np.shape(v) + seedmat.shape[axis + 1:] for v in invals]
            self._jaxseeds = tuple([jnp.reshape(a, shp) for a, shp in
                                    zip(jnp.split(seedmat, inds, axis=axis), shapes)])
        return self._jaxseeds

    def _setup_jax(self):
        pass
        # self.matrix_free = True
        # self.compute_jacvec_product = self._compute_jacvec_product_

    # def _compute_jacvec_product_(self, inputs, dinputs, doutputs, mode):
    #     if mode == 'fwd':
    #         # if self._jvp is None:
    #         #     self._jvp = jax.linearize(self._compute._f, *inputs.values())[1]
    #         # doutputs.set_vals(self._jvp(*dinputs.values()))
    #         doutputs.set_val(self._jacfwd.dot(dinputs.asarray()))
    #     else:  # rev
    #         if self._vjp is None:
    #             self._vjp = vjp(self._compute._f, *inputs.values())[1]
    #         dinputs.set_vals(self._vjp(tuple(doutputs.values())))

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
        if self.options['use_jax']:
            self._setup_jax()

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
