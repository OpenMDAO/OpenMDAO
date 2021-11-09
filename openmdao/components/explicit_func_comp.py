"""Define the FuncComponent class."""

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.constants import INT_DTYPE
import openmdao.func_api as omf
from openmdao.components.func_comp_common import _check_var_name, _copy_with_ignore, _add_options

try:
    from jax import jit, jacfwd
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
        # in case we're doing jit, force setup of wrapped func because we compute output shapes
        # during setup and that won't work on a jit compiled function
        if self._compute._call_setup:
            self._compute._setup()

        if self._compute._use_jax:
            self.options['use_jax'] = True
        self._compute_partials = compute_partials
        if self.options['use_jax'] and self.options['use_jit']:
            static_argnums = np.where(np.array(['is_option' in m for m in
                                                self._compute._inputs.values()], dtype=bool))[0]
            try:
                self._compute._f = jit(self._compute._f, static_argnums=static_argnums)
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

        for name, meta in self._compute.get_input_meta():
            _check_var_name(self, name)
            if 'is_option' in meta and meta['is_option']:
                kwargs = _copy_with_ignore(meta, omf._allowed_declare_options_args,
                                           ignore=optignore)
                self.options.declare(name, **kwargs)
            else:
                kwargs = omf._filter_dict(meta, omf._allowed_add_input_args)
                self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._compute.get_output_meta()):
            _check_var_name(self, name)
            kwargs = _copy_with_ignore(meta, omf._allowed_add_output_args, ignore=('resid',))
            self.add_output(name, **kwargs)

    def _linearize(self, jac=None, sub_do_ln=False):
        if self.options['use_jax']:
            # force _jvp and/or _vjp to be re-initialized the next time they are needed
            # is called
            self._jvp = None
            self._vjp = None
            # argnums specifies which position args are to be differentiated
            argnums = np.where(
                np.array(['is_option' not in m for m in self._compute._inputs.values()],
                         dtype=bool))[0]
            onames = list(self._compute.get_output_names())
            nouts = len(onames)
            jf = jacfwd(self._compute._f, argnums)(*self._func_values(self._inputs))
            for col, inp in enumerate(self._compute.get_input_names()):
                if col in argnums:
                    if nouts == 1:
                        if (onames[0], inp) in self._jacobian:
                            self._jacobian[onames[0], inp] = np.asarray(jf[col])
                    else:
                        for row, out in enumerate(onames):
                            if (out, inp) in self._jacobian:
                                self._jacobian[out, inp] = np.asarray(jf[row][col])
        else:
            super()._linearize(jac, sub_do_ln)

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
