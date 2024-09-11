import unittest
import sys
import itertools

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import openmdao.api as om

from openmdao.utils.jax_utils import jax, jnp, ExplicitCompJaxify
from openmdao.utils.testing_utils import parameterized_name

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

# if jax is None:
#     def jjit(f, *args, **kwargs):
#         return f
# else:
#     from openmdao.jax import act_tanh, smooth_abs, smooth_max, smooth_min, ks_max, ks_min
#     def jjit(f, *args, **kwargs):
#         if om.env_truthy('JAX_CPU') and 'backend' not in kwargs:
#             return jax.jit(f, *args, backend='cpu', **kwargs)
#         else:
#             return jax.jit(f, *args, **kwargs)



class MyCompJax1(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))

        self.declare_partials(of='z', wrt=['x', 'y'])

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

        self.declare_partials(of='z', wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])



class MyCompJax2(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

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

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])
        outputs['zz'] = inputs['y'] * 2.5



class MyCompJax2Primal(om.JaxExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute_primal(self, x, y):
        z = jnp.dot(x, y)
        zz = y * 2.5
        return z, zz


class MyCompJax2PrimalOption(om.JaxExplicitComponent):
    def __init__(self, stat=2., **kwargs):
        super().__init__(**kwargs)
        self.stat = stat

    def initialize(self):
        self.options.declare('mult', default=1.0, types=float, allow_none=False)

    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def get_self_statics(self):
        return (self.options['mult'], self.stat)

    def compute_primal(self, x, y):
        z = jnp.dot(x, y)
        zz = y * self.options['mult'] * self.stat
        return z, zz



class MyCompJaxWithOption(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('mult', default=2.5, desc='multiplier', types=(float,))

    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

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

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

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



class MyCompJaxWithDiscretePrimal(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_discrete_input('disc_in', val=2)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')
        self.add_discrete_output('disc_out', val=3)

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute_primal(self, disc_in, x, y):
        disc_out = -disc_in
        if disc_in > 0:
           z = jnp.dot(x, y)
        else:
           z = -jnp.dot(x, y)

        if disc_out > 0:
            zz = y * 2.5
        else:
            zz = y * 3.0

        self._discrete_outputs.set_vals((disc_out,))
        return (disc_out, z, zz)

x_shape = (2, 3)
y_shape = (3, 4)


@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
class TestJaxAST(unittest.TestCase):
    def test_ast_continuous(self):
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

        self.assertEqual(converter.get_compute_primal_src().strip(), expected)

    def test_ast_discrete(self):
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

        self.assertEqual(converter.get_compute_primal_src().strip(), expected)


@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
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
        assert_check_partials(p.check_partials(show_only_incorrect=True))

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
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'],
                                           wrt=['comp.x', 'comp.y'], method='fd',
                                           show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    def test_jax_explicit_comp2_matfree(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', MyCompJax2(derivs_method='jax'))
        comp.matrix_free = True
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
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'],
                                           wrt=['comp.x', 'comp.y'], method='fd',
                                           show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

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
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    def test_jax_explicit_comp2primal_matfree(self):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', MyCompJax2Primal(derivs_method='jax'))
        comp.matrix_free = True
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode='fwd')

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    def test_jax_explicit_comp2primal_w_option(self):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', MyCompJax2PrimalOption(derivs_method='jax', mult=1.5))
        # deriv shape: [[(2, 4, 2, 3), (2, 4, 3, 4)], [(3, 4, 2, 3), (3, 4, 3, 4)]]
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
        assert_near_equal(p.get_val('comp.zz'), y * 3.0)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        comp.options['mult'] = 3.5
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 7.0)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        comp.stat = 1./3.5
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

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
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        comp.options['mult'] = 1.9
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 1.9)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

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
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        p.set_val('ivc.disc_out', -2)
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), -np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    def test_jax_explicit_comp_with_discrete_primal(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        ivc.add_discrete_output('disc_out', val=3)
        p.model.add_subsystem('comp', MyCompJaxWithDiscretePrimal(derivs_method='jax'))
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
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                             method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        p.set_val('ivc.disc_out', -2)
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), -np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))


if sys.version_info >= (3, 9):


    class CompRetValue(om.JaxExplicitComponent):
        def __init__(self, shape, nins=1, nouts=1, **kwargs):
            super().__init__(**kwargs)
            self.shape = shape
            self.nins = nins
            self.nouts = nouts

            self.compute_primal = getattr(self, f'compute_primal_{self.nins}_{self.nouts}')

        def setup(self):
            if self.shape == ():
                for i in range(self.nins):
                    self.add_input(f'x{i}', val=0.)
                for i in range(self.nouts):
                    self.add_output(f'y{i}', val=0.)
            else:
                for i in range(self.nins):
                    self.add_input(f'x{i}', val=jnp.zeros(self.shape))
                for i in range(self.nouts):
                    self.add_output(f'y{i}', val=jnp.zeros(self.shape))

        def setup_partials(self):
            self.declare_partials('*', '*')

        def compute_primal_1_1(self, x0):
            return x0**2

        def compute_primal_2_1(self, x0, x1):
            return x0**2 + x1**2

        def compute_primal_1_2(self, x0):
            return x0**2, x0*2

        def compute_primal_2_2(self, x0, x1):
            return x0**2, x1**2


    class CompRetTuple(om.JaxExplicitComponent):
        def __init__(self, shape, nins=1, nouts=1, **kwargs):
            super().__init__(**kwargs)
            self.shape = shape
            self.nins = nins
            self.nouts = nouts

            self.compute_primal = getattr(self, f'compute_primal_{self.nins}_{self.nouts}')

        def setup(self):
            if self.shape == ():
                for i in range(self.nins):
                    self.add_input(f'x{i}', val=0.)
                for i in range(self.nouts):
                    self.add_output(f'y{i}', val=0.)
            else:
                for i in range(self.nins):
                    self.add_input(f'x{i}', val=jnp.zeros(self.shape))
                for i in range(self.nouts):
                    self.add_output(f'y{i}', val=jnp.zeros(self.shape))

        def setup_partials(self):
            self.declare_partials('*', '*')

        def compute_primal_1_1(self, x0):
            return (x0**2,)

        def compute_primal_2_1(self, x0, x1):
            return (x0**2 + x1**2,)

        def compute_primal_1_2(self, x0):
            return x0**2, x0*2

        def compute_primal_2_2(self, x0, x1):
            return x0**2, x1**2


    class TopGrp(om.Group):
        def __init__(self, shape, ret_tuple=False, nins=1, nouts=1, **kwargs):
            super().__init__(**kwargs)
            self.shape = shape
            self.ret_tuple = ret_tuple
            self.nins = nins
            self.nouts = nouts

        def setup(self):
            self.add_subsystem('ivc', om.IndepVarComp())
            if self.shape == ():
                for i in range(self.nins):
                    self.ivc.add_output(f'x{i}', 0.)

                if self.ret_tuple:
                    self.add_subsystem('comp', CompRetTuple(shape=self.shape, nins=self.nins, nouts=self.nouts))
                else:
                    self.add_subsystem('comp', CompRetValue(shape=self.shape, nins=self.nins, nouts=self.nouts))
            else:
                for i in range(self.nins):
                    self.ivc.add_output(f'x{i}', np.zeros(self.shape))
                if self.ret_tuple:
                    self.add_subsystem('comp', CompRetTuple(shape=self.shape, nins=self.nins, nouts=self.nouts))
                else:
                    self.add_subsystem('comp', CompRetValue(shape=self.shape, nins=self.nins, nouts=self.nouts))

            for io in range(self.nouts):
                for ii in range(self.nins):
                    if ii == io:
                        self.connect(f'ivc.x{io}', f'comp.x{ii}')



@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
class TestJaxShapesAndReturns(unittest.TestCase):
    @parameterized.expand(itertools.product([(), (2,), (2,3)], [(1, 1), (2, 2), (1, 2), (2, 1)],[True, False]),
                          name_func=parameterized_name)
    def test_compute_primal_return_shapes(self, shape, sizetup, ret_tuple):
        nins, nouts = sizetup
        prob = om.Problem()
        prob.model = TopGrp(shape=shape, ret_tuple=ret_tuple, nins=nins, nouts=nouts)

        prob.set_solver_print(level=0)

        ofs = [f'comp.y{i}' for i in range(nouts)]
        wrts = [f'ivc.x{i}' for i in range(nins)]

        prob.setup(force_alloc_complex=True, check=False, mode='fwd')
        prob.final_setup()
        prob.compute_totals(of=ofs, wrt=wrts)

        assert_check_partials(prob.check_partials(method='cs', out_stream=None), atol=1e-5)
        assert_check_totals(prob.check_totals(of=ofs, wrt=wrts, method='cs', out_stream=None), atol=1e-5)

    # TODO: test with mixed np and jnp in compute

if __name__ == '__main__':
    unittest.main()
