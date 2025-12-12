import sys
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
            import jax.numpy as jnp
            from jax import jit
            try:
                from jax import Array as JaxArray
            except ImportError:
                from jaxlib.xla_extension import ArrayImpl as JaxArray
        except ImportError:
            self.skipTest('jax is not available but required for this test.')

        class TestClass(object):

            @partial(jit, static_argnums=(0,))
            def f(self, x):
                return jnp.sqrt(x)

        x = jnp.linspace(0, 10, 11)
        result = TestClass().f(x)
        assert_near_equal(result**2, x)
        self.assertIsInstance(result, JaxArray)

    def test_jax_component_option(self):
        """Test that the registration of jax-compatible components works."""
        import numpy as np

        if sys.version_info < (3, 9):
            self.skipTest('JaxExplicitComponent requires Python 3.9+')

        from openmdao.utils.jax_utils import jax
        if jax is None:
            self.skipTest('jax is not available.')

        class PowComp(om.JaxExplicitComponent):

            def initialize(self):
                self.options.declare('vec_size', types=(int,))
                self.options.declare('pow', types=(int,))

            def setup(self):
                n = self.options['vec_size']
                self.add_input('x', shape=(n,))
                self.add_output('f', shape=(n,))

                ar = np.arange(n, dtype=int)
                self.declare_partials(of='f', wrt='x', rows=ar, cols=ar)

            def get_self_statics(self):
                return (self.options['pow'],)

            def compute_primal(self, x):
                return x**self.options['pow']

        p = om.Problem()
        powcomp = p.model.add_subsystem('c', PowComp(derivs_method='jax', vec_size=11, pow=2))
        # deriv shape: [(11, 1, 11, 1)]
        p.setup(mode='fwd', force_alloc_complex=True)
        c_x = np.linspace(1, 10, 11)#.reshape(11, 1)
        p.set_val('c.x', c_x)
        p.run_model()
        assert_near_equal(p.get_val('c.f'), c_x ** 2)
        assert_check_partials(p.check_partials(method='cs', show_only_incorrect=False))

        p.set_val('c.x', c_x)
        powcomp.options['pow'] = 3  # change the option to verify that re-jit happens
        p.run_model()
        assert_near_equal(p.get_val('c.f'), c_x ** 3)
        assert_check_partials(p.check_partials(method='cs', show_only_incorrect=False))


if __name__ == '__main__':
    unittest.main()
