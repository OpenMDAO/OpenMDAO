"""Define the FuncComponent class."""

try:
    from jax import jvp, vjp, vmap, random, jit
    import jax.numpy as jnp
except ImportError:
    jax = None

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.func_utils import get_func_info


_allowed_add_input_args = {
    'val', 'shape', 'src_indices', 'flat_src_indices', 'units', 'desc', 'tags', 'shape_by_conn',
    'copy_shape', 'distributed', 'new_style_idx',
}

_allowed_add_output_args = {
    'val', 'shape', 'units', 'res_units', 'desc' 'lower', 'upper', 'ref', 'ref0', 'res_ref', 'tags',
    'shape_by_conn', 'copy_shape', 'distributed',
}


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
    """

    def __init__(self, func, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._func = func
        self._inmeta = None
        self._outmeta = None

    def setup(self):
        """
        Define out inputs and outputs.
        """
        self._inmeta, self._outmeta = get_func_info(self._func)
        for name, meta in self._inmeta.items():
            kwargs = {n: v for n, v in meta.items() if n in _allowed_add_input_args}
            self.add_input(name, **kwargs)

        for name, meta in self._outmeta.items():
            kwargs = {n: v for n, v in meta.items() if n in _allowed_add_output_args}
            self.add_output(name, **kwargs)

    def setup_partials(self):
        """
        Set up our partial derivative sparsity.
        """
        pass

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
        outs = self._func(*inputs.values())
        arr = outputs.asarray()
        start = end = 0
        for o in outs:
            end += o.size
            arr[start:end] = np.asarray(o).flat
            start = end


if __name__ == '__main__':
    @jit
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
