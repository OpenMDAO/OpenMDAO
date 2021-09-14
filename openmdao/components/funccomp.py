"""Define the FuncComponent class."""

try:
    from jax import jvp, vjp, vmap, random, jit
    import jax.numpy as jnp
except ImportError:
    jax = None

import re
import numpy as np
from numpy import asarray, isscalar, ndarray, imag, complex as npcomplex
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.func_utils import get_func_info
from openmdao.core.constants import INT_DTYPE
from openmdao.utils.units import valid_units


# regex to check for variable names.
namecheck_rgx = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')

# Names of metadata entries allowed for ExecComp variables.
_allowed_meta = {'val', 'shape', 'units', 'res_units', 'desc',
                 'ref', 'ref0', 'res_ref', 'lower', 'upper', 'src_indices',
                 'flat_src_indices', 'tags', 'shape_by_conn', 'copy_shape'}

# Names that are not allowed for input or output variables (keywords for options)
_disallowed_varnames = {'has_diag_partials', 'units', 'shape', 'shape_by_conn', 'run_root_only'}


_allowed_add_input_args = {
    'val', 'shape', 'src_indices', 'flat_src_indices', 'units', 'desc', 'tags', 'shape_by_conn',
    'copy_shape', 'distributed', 'new_style_idx',
}

_allowed_add_output_args = {
    'val', 'shape', 'units', 'res_units', 'desc' 'lower', 'upper', 'ref', 'ref0', 'res_ref', 'tags',
    'shape_by_conn', 'copy_shape', 'distributed',
}


def _check_units_option(option, value):
    if value is not None and not valid_units(value):
        raise ValueError(f"The units '{value}' are invalid.")


class ExplicitFuncComp(ExplicitComponent):
    """
    A component that wraps a python function.

    Parameters
    ----------
    func : function
        The function to be wrapped by this Component.
    **kwargs : named args
        Args passed down to ExplicitComponent.

    Attributes
    ----------
    _func : function
        The function wrapped by this component.
    _inmeta : dict
        Function input metadata.
    _outmeta : dict
        Function return value metadata.
    complex_stepsize : float
        Step size used for complex step.
    """

    def __init__(self, func, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._func = func
        self._inmeta = None
        self._outmeta = None
        # if complex step is used for derivatives, this is the stepsize
        self.complex_stepsize = 1.e-40

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('has_diag_partials', types=bool, default=False,
                             desc='If True, treat all array/array partials as diagonal if both '
                                  'arrays have size > 1. All arrays with size > 1 must have the '
                                  'same flattened size or an exception will be raised.')

        self.options.declare('units', types=str, allow_none=True, default=None,
                             desc='Units to be assigned to all variables in this component. '
                                  'Default is None, which means units may be provided for variables'
                                  ' individually.',
                             check_valid=_check_units_option)

        self.options.declare('shape', types=(int, tuple, list), allow_none=True, default=None,
                             desc='Shape to be assigned to all variables in this component. '
                                  'Default is None, which means shape may be provided for variables'
                                  ' individually.')

        self.options.declare('use_jax', types=bool, default=False,
                             desc='If True, use jax to compute derivatives.')

        # self.options.declare('shape_by_conn', types=bool, default=False,
        #                      desc='If True, shape all inputs and outputs based on their '
        #                           'connection. Default is False.')

    def setup(self):
        """
        Define out inputs and outputs.
        """
        try:
            self._inmeta, self._outmeta = get_func_info(self._func, self.options)
        except Exception as err:
            raise err.__class__(f"{self.msginfo}: {err}")
        for name, meta in self._inmeta.items():
            self._check_var_name(name)
            kwargs = {n: v for n, v in meta.items() if n in _allowed_add_input_args}
            if kwargs['val'] is None:
                kwargs['val'] = 1.0
            self.add_input(name, **kwargs)

        for i, (name, meta) in enumerate(self._outmeta.items()):
            if name is None:
                raise RuntimeError(f"{self.msginfo}: Can't add output corresponding to return "
                                   f"value in position {i} because it has no name.  Specify the "
                                   "name by returning a variable, for example 'return myvar', or "
                                   "include the name in the function's return value annotation.")
            self._check_var_name(name)
            kwargs = {n: v for n, v in meta.items() if n in _allowed_add_output_args}
            self.add_output(name, **kwargs)

    def _setup_partials(self):
        """
        Check that all partials are declared.
        """
        meta = self._var_rel2meta
        decl_partials = super().declare_partials
        hasdiag = self.options['has_diag_partials']
        for i, (out, ometa) in enumerate(sorted(self._outmeta.items(), key=lambda x: x[0])):
            oshp = ometa['shape']
            if not oshp:
                osize = 1
            else:
                osize = np.product(oshp) if isinstance(oshp, tuple) else oshp

            inds = np.arange(osize, dtype=INT_DTYPE)
            for inp, imeta in self._inmeta.items():
                if inp not in ometa['deps']:
                    continue

                if hasdiag:
                    ishp = imeta['shape']
                    if not ishp:
                        isize = 1
                    else:
                        isize = np.product(ishp) if isinstance(oshp, tuple) else ishp
                    if osize != isize:
                        raise RuntimeError(f"{self.msginfo}: has_diag_partials is True but "
                                           f"partial({out}, {inp}) is not square "
                                           f"(shape=({osize}, {isize})).")
                    # partial will be declared as diagonal
                    if osize > 1:
                        decl_partials(of=out, wrt=inp, rows=inds, cols=inds)
                    else:
                        decl_partials(of=out, wrt=inp)
                else:
                    decl_partials(of=out, wrt=inp)

        super()._setup_partials()

    def _check_var_name(self, name):
        match = namecheck_rgx.match(name)
        if match is None or match.group() != name:
            raise NameError(f"{self.msginfo}: '{name}' is not a valid variable name." )

        if name in _disallowed_varnames:
            raise NameError(f"{self.msginfo}: cannot use variable name '{name}' because "
                            "it's a reserved keyword.")

    def _compute_output_array(self, input_values, output_array):
        """
        Fill the given output array with our function result based on the given input values.

        Parameters
        ----------
        input_values : tuple of ndarrays or floats
            Unscaled, dimensional input variables.
        output_array
            The output array being filled.
        """
        outs = self._func(*input_values)
        if isinstance(outs, tuple):
            start = end = 0
            for o in outs:
                a = asarray(o) if isscalar(o) else o
                end += a.size
                output_array[start:end] = a.flat
                start = end
        else:
            if isscalar(outs):
                output_array[:] = outs
            else:
                output_array[:] = outs.flat

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
        # this will update the outputs array in place
        self._compute_output_array(inputs.values(), outputs.asarray())

    def compute_partials(self, inputs, partials):
        """
        Use complex step method to update the given Jacobian.

        Parameters
        ----------
        inputs : `VecWrapper`
            `VecWrapper` containing parameters (p).
        partials : `Jacobian`
            Contains sub-jacobians.
        """
        step = self.complex_stepsize * 1j
        inv_stepsize = 1.0 / self.complex_stepsize
        has_diag_partials = self.options['has_diag_partials']

        in_vals = [np.asarray(v, dtype=npcomplex) for v in inputs.values()]
        result = np.zeros(len(self._outputs), dtype=npcomplex)
        out_slices = self._outputs.get_slice_dict()

        for ivar, inp in enumerate(inputs._abs_iter()):

            if has_diag_partials or in_vals[ivar].size == 1:
                # set a complex input value
                in_vals[ivar] += step

                # solve with complex input value
                self._compute_output_array(in_vals, result)

                for u, slc in out_slices.items():
                    if (u, inp) in self._subjacs_info:
                        partials[(u, inp)] = imag(result[slc] * inv_stepsize)

                # restore old input value
                in_vals[ivar] -= step
            else:
                pval = in_vals[ivar]
                if np.isscalar(pval):
                    # set a complex input value
                    in_vals[ivar] += step

                    # solve with complex input value
                    self._compute_output_array(in_vals, result)

                    for u, slc in out_slices.items():
                        if (u, inp) in self._subjacs_info:
                            # set the column in the Jacobian entry
                            partials[(u, inp)][:, i] = imag(result[slc] * inv_stepsize)

                    # restore old input value
                    in_vals[ivar] -= step
                else:
                    for i in range(pval.size):
                        # set a complex input value
                        in_vals[ivar].flat[i] += step

                        # solve with complex input value
                        self._compute_output_array(in_vals, result)

                        for u, slc in out_slices.items():
                            if (u, inp) in self._subjacs_info:
                                # set the column in the Jacobian entry
                                partials[(u, inp)][:, i] = imag(result[slc] * inv_stepsize)

                        # restore old input value
                        in_vals[ivar].flat[i] -= step


if __name__ == '__main__':
    def _some_func(x=np.zeros(4), y=np.ones(4), z=3):
        foo = 2. * x + 3. * y
        bar = 2 * (x + y)
        baz = z * 3. + 1.
        return foo, bar, baz

    import openmdao.api as om

    p = om.Problem()
    p.model.add_subsystem('comp', ExplicitFuncComp(_some_func))
    p.setup()
    p['comp.x'] = np.arange(4, dtype=float) + 1.
    p['comp.y'] = np.arange(4, dtype=float) + 2.
    p['comp.z'] = 99.

    p.run_model()

    print('comp.foo', p['comp.foo'])
    print('comp.bar', p['comp.bar'])
    print('comp.baz', p['comp.baz'])
