import unittest
from functools import partial
from itertools import chain

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import openmdao.api as om

try:
    import jax
except (ImportError, ModuleNotFoundError):
    jax = None

if jax is not None:
    jax.config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
    import jax.numpy as jnp
    from openmdao.jax import act_tanh, smooth_abs, smooth_max, smooth_min, ks_max, ks_min
    from openmdao.utils.jax_utils import ExplicitCompJaxify
    from openmdao.utils.jax_utils import benchmark_component


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


class MyCompJax1(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))

        # self.declare_partials(of='z', wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])


class MyCompJax1Shaped(om.ExplicitComponent):
    def __init__(self, xshape, yshape, **kwargs):
        super().__init__(**kwargs)
        self.xshape = xshape
        self.yshape = yshape

    def setup(self):
        self.add_input('x', shape=self.xshape)
        self.add_input('y', shape=self.yshape)
        self.add_output('z', shape=(self.xshape[0], self.yshape[1]))

        # self.declare_partials(of='z', wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])


class MyCompJax2(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        # self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])
        outputs['zz'] = inputs['y'] * 2.5


class MyCompJax2Shaped(om.ExplicitComponent):
    def __init__(self, xshape, yshape, **kwargs):
        super().__init__(**kwargs)
        self.xshape = xshape
        self.yshape = yshape

    def setup(self):
        self.add_input('x', shape=self.xshape)
        self.add_input('y', shape=self.yshape)
        self.add_output('z', shape=(self.xshape[0], self.yshape[1]))
        self.add_output('zz', shape=self.yshape)

        # self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])
        outputs['zz'] = inputs['y'] * 2.5


class MyCompJax2Primal(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        # self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    @partial(jax.jit, static_argnums=(0,))
    def compute_primal(self, x, y):
        z = jnp.dot(x, y)
        zz = y * 2.5
        return z, zz


class MyCompJax2PrimalOption(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('mult', default=1.0, types=float, allow_none=False)

    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        # self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def get_static_args(self):
        return (self.options['mult'],)

    @partial(jax.jit, static_argnums=(0,1))
    def compute_primal(self, mult, x, y):
        z = jnp.dot(x, y)
        zz = y * mult
        return z, zz



class MyCompJaxWithOption(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('mult', default=2.5, desc='multiplier', types=(float,))

    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        # self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])
        outputs['zz'] = inputs['y'] * self.options['mult']


class MyCompJaxWithDiscrete(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_discrete_input('disc_in', val=2)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')
        self.add_discrete_output('disc_out', val=3)

        # self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['disc_out'] = -discrete_inputs['disc_in']
        if discrete_inputs['disc_in'] > 0:
            outputs['z'] = np.dot(inputs['x'], inputs['y'])
        else:
            outputs['z'] = -np.dot(inputs['x'], inputs['y'])

        if discrete_outputs['disc_out'] > 0:
            outputs['zz'] = inputs['y'] * 2.5
        else:
            outputs['zz'] = inputs['y'] * 3.0


x_shape = (2, 3)
y_shape = (3, 4)


class ASTContinuousCompTester(om.ExplicitComponent):
    def setup(self):
        self.add_input('in_scalar', val=7.0)
        self.add_input('in_array', val=np.ones((2, 3)))
        self.add_input('in_array2', val=np.ones((3,4)))
        self.add_output('out_scalar', val=5.0)
        self.add_output('out_array', val=np.ones((2, 3)))
        self.add_output('out_array2', val=np.ones((3, 4)))

    def compute(self, inputs, outputs):
        outputs['out_scalar'] = inputs['in_scalar'] * 2.0
        outputs['out_array'] = inputs['in_array'] * 2.0
        outputs['out_array2'] = np.dot(inputs['in_array'], inputs['in_array2'])


class ASTDiscreteCompTester(om.ExplicitComponent):
    def setup(self):
        self.add_input('in_scalar', val=7.0)
        self.add_input('in_array', val=np.ones((2, 3)))
        self.add_input('in_array2', val=np.ones((3,4)))
        self.add_output('out_scalar', val=5.0)
        self.add_output('out_array', val=np.ones((2, 3)))
        self.add_output('out_array2', val=np.ones((3, 4)))
        self.add_discrete_input('disc_in', val=2)
        self.add_discrete_output('disc_out', val=3)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        outputs['out_scalar'] = inputs['in_scalar'] * 2.0
        outputs['out_array'] = inputs['in_array'] * 2.0
        outputs['out_array2'] = np.dot(inputs['in_array'], inputs['in_array2'])
        if discrete_inputs['disc_in'] > 0:
            outputs['out_scalar'] *= 2.0
            outputs['out_array'] *= 2.0
            outputs['out_array2'] *= 2.0
        else:
            outputs['out_scalar'] *= 3.0
            outputs['out_array'] *= 3.0
            outputs['out_array2'] *= 3.0


@unittest.skipIf(jax is None, 'jax is not available.')
class TestJaxAST(unittest.TestCase):
    def test_ast_continuous(self):
        p = om.Problem()
        comp = p.model.add_subsystem('comp', ASTContinuousCompTester())
        p.setup()
        p.final_setup()

        converter = ExplicitCompJaxify(comp)

        expected = """
def compute_primal(self, in_scalar, in_array, in_array2):
    out_scalar = in_scalar * 2.0
    out_array = in_array * 2.0
    out_array2 = jnp.dot(in_array, in_array2)
    return (out_scalar, out_array, out_array2)
""".strip()

        self.assertEqual(converter.get_new_source().strip(), expected)

    def test_ast_discrete(self):
        p = om.Problem()
        comp = p.model.add_subsystem('comp', ASTDiscreteCompTester())
        p.setup()
        p.final_setup()

        converter = ExplicitCompJaxify(comp)

        expected = """
def compute_primal(self, disc_in, in_scalar, in_array, in_array2):
    disc_out, = self._discrete_outputs.values()
    out_scalar = in_scalar * 2.0
    out_array = in_array * 2.0
    out_array2 = jnp.dot(in_array, in_array2)
    if disc_in > 0:
        out_scalar *= 2.0
        out_array *= 2.0
        out_array2 *= 2.0
    else:
        out_scalar *= 3.0
        out_array *= 3.0
        out_array2 *= 3.0
    self._discrete_outputs.set_vals((disc_out,))
    return (disc_out, out_scalar, out_array, out_array2)
""".strip()

        self.assertEqual(converter.get_new_source().strip(), expected)


@unittest.skipIf(jax is None, 'jax is not available.')
class TestJaxComp(unittest.TestCase):

    def test_jax_explicit_comp(self):
        p = om.Problem()
        # create an IVC manually so we can set the shapes.  Otherwise must set shape in the component
        # itself.
        xshape = x_shape
        yshape = y_shape
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(xshape)))
        ivc.add_output('y', val=np.ones(yshape))
        p.model.add_subsystem('comp', MyCompJax1(derivs_method='jax'))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode='rev', force_alloc_complex=True)

        x = np.arange(1,np.prod(xshape)+1).reshape(xshape) * 2.0
        y = np.arange(1,np.prod(yshape)+1).reshape(yshape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        p.check_totals(of=['comp.z'], wrt=['comp.x', 'comp.y'], method='cs', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)

    def test_jax_explicit_comp2(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        p.model.add_subsystem('comp', MyCompJax2(derivs_method='jax'))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode='rev')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)

    def test_jax_explicit_comp2primal(self):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        p.model.add_subsystem('comp', MyCompJax2Primal(derivs_method='jax'))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode='rev')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)

    def test_jax_explicit_comp2primal_w_option(self):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', MyCompJax2PrimalOption(derivs_method='jax', mult=1.5))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode='rev')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape) * 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 1.5)
        p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)

        comp.options['mult'] = 3.5
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 3.5)
        p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)

    def test_jax_explicit_comp_with_option(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', MyCompJaxWithOption(mult=1.7, derivs_method='jax'))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode='rev')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 1.7)
        p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)

        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        comp.options['mult'] = 1.9
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 1.9)

    def test_jax_explicit_comp_with_discrete(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        ivc.add_discrete_output('disc_out', val=3)
        p.model.add_subsystem('comp', MyCompJaxWithDiscrete(derivs_method='jax'))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')
        p.model.connect('ivc.disc_out', 'comp.disc_in')

        p.setup(mode='rev')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 3.0)
        p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)

        p.set_val('ivc.disc_out', -2)
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), -np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        p.check_partials(show_only_incorrect=True)


@unittest.skipIf(jax is None, 'jax is not available.')
class TestJaxGroup(unittest.TestCase):
    def test_jax_group_outer_ivc(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        G = p.model.add_subsystem('G', om.Group())
        G.options['derivs_method'] = 'jax'
        G.add_subsystem('comp2', MyCompJax2())
        G.add_subsystem('comp', MyCompJax1())

        p.model.connect('ivc.x', ['G.comp.x', 'G.comp2.x'])
        p.model.connect('ivc.y', 'G.comp2.y')
        G.connect('comp2.zz', 'comp.y')

        p.setup(mode='rev')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('G.comp2.z'), np.dot(x, y))
        assert_near_equal(p.get_val('G.comp2.zz'), y * 2.5)
        assert_check_partials(p.check_partials(show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['G.comp.z','G.comp2.z', 'G.comp2.zz'],
                                             wrt=['G.comp2.x', 'G.comp2.y'], method='fd', show_only_incorrect=True))

    def test_jax_group_auto_ivc(self):
        p = om.Problem()
        G = p.model.add_subsystem('G', om.Group())
        G.options['derivs_method'] = 'jax'
        G.add_subsystem('comp2', MyCompJax2Shaped(x_shape, y_shape))
        G.add_subsystem('comp', MyCompJax1Shaped(x_shape, y_shape))

        G.connect('comp2.zz', 'comp.y')

        p.setup(mode='rev')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('G.comp.x', x)
        p.set_val('G.comp2.x', x)
        p.set_val('G.comp2.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('G.comp2.z'), np.dot(x, y))
        assert_near_equal(p.get_val('G.comp2.zz'), y * 2.5)
        assert_check_partials(p.check_partials(show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['G.comp.z','G.comp2.z', 'G.comp2.zz'],
                                             wrt=['G.comp2.x', 'G.comp2.y'], method='fd', show_only_incorrect=True))

    @unittest.skip("unskip this after inner ivc issue is fixed for fd")
    def test_jax_group_inner_ivc(self):
        p = om.Problem()
        G = p.model.add_subsystem('G', om.Group())
        ivc = G.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        G.options['derivs_method'] = 'jax'
        G.add_subsystem('comp2', MyCompJax2())
        G.add_subsystem('comp', MyCompJax1())

        p.model.connect('G.ivc.x', ['G.comp.x', 'G.comp2.x'])
        p.model.connect('G.ivc.y', 'G.comp2.y')
        G.connect('comp2.zz', 'comp.y')

        p.setup(mode='fwd')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('G.ivc.x', x)
        p.set_val('G.ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('G.comp2.z'), np.dot(x, y))
        assert_near_equal(p.get_val('G.comp2.zz'), y * 2.5)

        print('-' * 80)
        assert_check_partials(p.check_partials(show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['G.comp.z','G.comp2.z', 'G.comp2.zz'],
                                             wrt=['G.ivc.x', 'G.ivc.y'], method='fd', show_only_incorrect=True))

    def test_jax_group_top_level(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        G = p.model
        G.options['derivs_method'] = 'jax'
        G.add_subsystem('comp2', MyCompJax2())
        G.add_subsystem('comp', MyCompJax1())

        p.model.connect('ivc.x', ['comp.x', 'comp2.x'])
        p.model.connect('ivc.y', 'comp2.y')
        G.connect('comp2.zz', 'comp.y')

        p.setup(mode='rev')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape) * 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp2.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp2.zz'), y * 2.5)
        assert_check_partials(p.check_partials(show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['comp.z','comp2.z', 'comp2.zz'],
                                             wrt=['ivc.x', 'comp2.y'], method='fd', show_only_incorrect=True))


    # TODO: test with mixed np and jnp in compute

if __name__ == '__main__':
    # unittest.main()

    result = benchmark_component(MyCompJax2Shaped, methods=('jax', 'cs'),
                                 repeats=10, table_format='tabulator', xshape=(44, 330), yshape=(330, 55))
