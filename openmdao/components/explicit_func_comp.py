"""Define the FuncComponent class."""

import functools
import numpy as np
from numpy import asarray, isscalar
from itertools import chain
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.constants import INT_DTYPE
import openmdao.func_api as omf
from openmdao.components.func_comp_common import _check_var_name, _copy_with_ignore, _add_options

try:
    import jax
    from jax import jvp, vjp, vmap, random, jit
    import jax.numpy as jnp
except ImportError:
    jax = None


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
    _compute_partials : function or None
        If not None, call this function when computing partials.
    """

    def __init__(self, compute, compute_partials=None, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._compute = omf.wrap(compute)
        if self._compute._use_jax:
            self.options['use_jax'] = True
        self._compute_partials = compute_partials
        if self.options['use_jax'] and self.options['use_jit']:
            try:
                self._compute._f = jit(self._compute._f)
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

        for name, meta in self._compute.get_input_meta():
            _check_var_name(self, name)
            if 'is_option' in meta and meta['is_option']:
                kwargs = _copy_with_ignore(meta, omf._allowed_declare_options_args,
                                           ignore=optignore)
                self.options.declare(name, **kwargs)
            else:
                kwargs = _copy_with_ignore(meta, omf._allowed_add_input_args)
                self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._compute.get_output_meta()):
            if name is None:
                raise RuntimeError(f"{self.msginfo}: Can't add output corresponding to return "
                                   f"value in position {i} because it has no name.  Specify the "
                                   "name by returning a variable, for example 'return myvar', or "
                                   "include the name in the function's metadata.")
            _check_var_name(self, name)
            kwargs = _copy_with_ignore(meta, omf._allowed_add_output_args, ignore=('resid',))
            self.add_output(name, **kwargs)

    def _linearize(self, jac=None, sub_do_ln=False):
        if self.options['use_jax']:
            # self._jvp = functools.partial(jvp, self._compute._f, tuple(self._inputs.values()))
            # self._jvp((jnp.eye(N)[:, 2],))[1]
            # y, out_tangents = vmap(self._jvp, out_axes=(None, 0))((jnp.eye(N),))

            # force _jvp and _vjp to be re-initialized the next time compute_jacvec_product
            # is called
            self._jvp = None
            self._vjp = None
        else:
            super()._linearize(jac, sub_do_ln)

    def _setup_jax(self):
        self.matrix_free = True
        self.compute_jacvec_product = self._compute_jacvec_product_

    def _compute_jacvec_product_(self, inputs, dinputs, doutputs, mode):
        if mode == 'fwd':
            if self._jvp is None:
                self._jvp = jax.linearize(self._compute._f, *inputs.values())[1]
            doutputs.set_vals(self._jvp(*dinputs.values()))
        else:  # rev
            if self._vjp is None:
                self._vjp = vjp(self._compute._f, *inputs.values())[1]
            dinputs.set_vals(self._vjp(tuple(doutputs.values())))

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
        outputs.set_vals(self._compute(*inputs.values()))

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
        if self._compute_partials is None and ('method' not in kwargs or
                                               kwargs['method'] == 'exact'):
            raise RuntimeError(f"{self.msginfo}: declare_partials must be called with method equal "
                               "to 'cs', 'fd', or 'jax'.")

        return super().declare_partials(*args, **kwargs)

    def _setup_partials(self):
        """
        Check that all partials are declared.
        """
        if self.options['use_jax']:
            self._setup_jax()
        else:
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

        self._compute_partials(*chain(inputs.values(), (partials,)))
