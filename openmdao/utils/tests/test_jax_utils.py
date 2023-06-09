from functools import partial
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal


class TestJaxUtils(unittest.TestCase):

    def test_jit_stub_no_jax(self):
        """Test the functionality of the jit stub."""
        from openmdao.utils.jax_utils import jit_stub as jit
        import numpy as np

        class TestClass(object):

            @partial(jit, static_argnums=(0,))
            def f(self, x):
                return np.sqrt(x)

        x = np.linspace(0, 10, 11)
        assert_near_equal(TestClass().f(x)**2, x)

    def test_jit_with_jax(self):
        """Make sure the jit stub test case also works with jax."""
        try:
            import jax.numpy as np
            from jax import jit
            import jaxlib
        except ImportError:
            self.skipTest('jax is not available but required for this test.')

        class TestClass(object):

            @partial(jit, static_argnums=(0,))
            def f(self, x):
                return np.sqrt(x)

        x = np.linspace(0, 10, 11)
        result = TestClass().f(x)
        assert_near_equal(result**2, x)
        self.assertIsInstance(result, jaxlib.xla_extension.ArrayImpl)

    def test_register_jax_component(self):
        """Test that the registration of jax-compatible components works."""
        try:
            import jax
        except ImportError:
            self.skipTest('jax is not available but required for this test.')
        import numpy as np

        @om.register_jax_component
        class PowComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('vec_size', types=(int,))
                self.options.declare('pow', types=(int,))

            def setup(self):
                n = self.options['vec_size']
                self.add_input('x', shape=(n,))
                self.add_output('f', shape=(n,))

                # The partials are a dense row in this case (1 row x N inputs)
                # There is no need to specify a sparsity pattern.
                ar = np.arange(n, dtype=int)
                self.declare_partials(of='f', wrt='x', rows=ar, cols=ar)

            @partial(jax.jit, static_argnums=(0,))
            def _compute_partials_jacfwd(self, x):
                deriv_func = jax.jacfwd(self.compute_primal, argnums=[0])
                # Here we make sure we extract the diagonal of the computed jacobian, since we
                # know it will have the only non-zero values.
                return jax.numpy.diagonal(deriv_func(x)[0])

            @partial(jax.jit, static_argnums=(0,))
            def compute_primal(self, x):
                return x**self.options['pow']

            def compute(self, inputs, outputs):
                outputs['f'] = self.compute_primal(*inputs.values())

            def compute_partials(self, inputs, partials):
                # Since the partials are sparse and stored in a flat array, ravel
                # the resulting derivative jacobian.
                partials['f', 'x'] = self._compute_partials_jacfwd(*inputs.values()).ravel()

            def _tree_flatten(self):
                """
                Per the jax documentation, these are the attributes
                of this class that we need to reference in the jax jitted
                methods of the class.
                There are no dynamic values or arrays, only self.options is used.
                Note that we do not change the options during the evaluation of
                these methods.
                """
                children = ()  # arrays / dynamic values
                aux_data = {'options': self.options}  # static values
                return (children, aux_data)

            @classmethod
            def _tree_unflatten(cls, aux_data, children):
                """
                Per the jax documentation, this method is needed by jax.jit since
                we are referencing attributes of the class (self.options) in our
                jitted methods.
                """
                return cls(*children, **aux_data)

        p = om.Problem()
        p.model.add_subsystem('c', PowComp(vec_size=11, pow=2))
        p.setup(force_alloc_complex=True)
        p.set_val('c.x', np.linspace(0, 10, 11))
        p.run_model()
        assert_near_equal(np.sqrt(p.get_val('c.f')), p.get_val('c.x'))
        assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None))
