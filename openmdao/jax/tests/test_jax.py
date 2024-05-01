import unittest

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
import openmdao.api as om

try:
    import jax
except (ImportError, ModuleNotFoundError):
    jax = None

if jax is not None:
    jax.config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
    import jax.numpy as jnp
    from openmdao.jax import act_tanh, smooth_abs, smooth_max, smooth_min, ks_max, ks_min


@unittest.skipIf(jax is None, 'jax is not available.')
class TestJax(unittest.TestCase):

    def test_tanh_act(self):
        f = act_tanh(6, mu=1.0E-5, z=6, a=-10, b=10)
        assert_near_equal(np.asarray(f), 0.0)

        f = act_tanh(6, mu=1.0E-5, z=6, a=-10, b=0)
        assert_near_equal(np.asarray(f), -5.0)

        f = act_tanh(-10, mu=1.0E-5, z=6, a=-10, b=0)
        assert_near_equal(np.asarray(f), -10)

        f = act_tanh(10, mu=1.0E-5, z=6, a=-10, b=20)
        assert_near_equal(np.asarray(f), 20)

    def test_smooth_max(self):
        x = np.linspace(0, 1, 1000)
        sin = np.sin(x)
        cos = np.cos(x)

        smax = smooth_max(sin, cos, mu=1.0E-6)

        idxs_sgt = np.where(sin > cos)
        idxs_cgt = np.where(sin < cos)

        assert_near_equal(smax[idxs_sgt], sin[idxs_sgt])
        assert_near_equal(smax[idxs_cgt], cos[idxs_cgt])

    def test_smooth_min(self):
        x = np.linspace(0, 1, 1000)
        sin = np.sin(x)
        cos = np.cos(x)

        smin = smooth_min(sin, cos, mu=1.0E-6)

        idxs_sgt = np.where(sin > cos)
        idxs_cgt = np.where(sin < cos)

        assert_near_equal(smin[idxs_sgt], cos[idxs_sgt])
        assert_near_equal(smin[idxs_cgt], sin[idxs_cgt])

    def test_smooth_abs(self):
        x = np.linspace(-0.5, 0.5, 1000)

        sabs = smooth_abs(x)
        abs = np.abs(x)

        idxs_compare = np.where(abs > 0.1)
        assert_near_equal(sabs[idxs_compare], abs[idxs_compare], tolerance=1.0E-9)

    def test_ks_max(self):
        x = np.random.random(1000)

        ksmax = ks_max(x, rho=1.E6)
        npmax = np.max(x)

        assert_near_equal(ksmax, npmax, tolerance=1.0E-6)

    def test_ks_min(self):
        x = np.random.random(1000)

        ksmin = ks_min(x, rho=1.E6)
        npmin = np.min(x)

        assert_near_equal(ksmin, npmin, tolerance=1.0E-6)


@unittest.skipIf(jax is None, 'jax is not available.')
class TestJaxComp(unittest.TestCase):

    def test_jax_explicit_comp(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape_by_conn=True)
                self.add_input('y', shape_by_conn=True)
                self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))

                self.declare_partials(of='z', wrt=['x', 'y'])

            def compute(self, inputs, outputs):
                outputs['z'] = np.dot(inputs['x'], inputs['y'])

        p = om.Problem()
        # create an IVC manually so we can set the shapes.  Otherwise must set shape in the component
        # itself.
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones((3, 4))))
        ivc.add_output('y', val=np.ones((4, 5)))
        p.model.add_subsystem('comp', MyComp(derivs_method='jax'))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode='rev')

        x = np.arange(1,13).reshape((3,4))
        y = np.arange(1,21).reshape((4, 5))
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        p.check_totals(of=['comp.z'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)

    def test_jax_explicit_comp2(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape_by_conn=True)
                self.add_input('y', shape_by_conn=True)
                self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
                self.add_output('zz', copy_shape='x')

                self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

            def compute(self, inputs, outputs):
                outputs['z'] = jnp.dot(inputs['x'], inputs['y'])
                outputs['zz'] = inputs['x'] * 2.0

        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones((3, 4))))
        ivc.add_output('y', val=np.ones((4, 5)))
        p.model.add_subsystem('comp', MyComp(derivs_method='jax'))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode='rev')

        x = np.arange(1,13).reshape((3,4))
        y = np.arange(1,21).reshape((4, 5))
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), x * 2.0)
        p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)

    # TODO: test with discrete vars
    # TODO: test with options
    # TODO: test with mixed np and jnp in compute

if __name__ == '__main__':
    unittest.main()
