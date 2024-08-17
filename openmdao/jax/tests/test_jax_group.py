import unittest
import sys

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import openmdao.api as om

from openmdao.utils.jax_utils import jax
from openmdao.jax.tests.test_jax_implicit import QuadraticComp, JaxQuadraticCompPrimal, \
    SimpleLinearSystemComp, JaxLinearSystemCompPrimal

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

# if jax is None:
#     def jjit(f, *args, **kwargs):
#         return f
# else:
#     def jjit(f, *args, **kwargs):
#         if om.env_truthy('JAX_CPU') and 'backend' not in kwargs:
#             # have to force this to 'cpu' else wing debugger stops at GPU check exception
#             return jax.jit(f, *args, backend='cpu', **kwargs)
#         else:
#             return jax.jit(f, *args, **kwargs)


class JaxExplicitComp1(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))

        self.declare_partials(of='z', wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])


class JaxExplicitComp1Shaped(om.ExplicitComponent):
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


class JaxExplicitComp2(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])
        outputs['zz'] = inputs['y'] * 2.5


class JaxExplicitComp2Shaped(om.ExplicitComponent):
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


x_shape = (2, 3)
y_shape = (3, 4)


@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
class TestJaxGroup(unittest.TestCase):
    def test_jax_group_outer_ivc(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        G = p.model.add_subsystem('G', om.Group())
        G.options['derivs_method'] = 'jax'
        G.add_subsystem('comp2', JaxExplicitComp2())
        G.add_subsystem('comp', JaxExplicitComp1())

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
        G.add_subsystem('comp2', JaxExplicitComp2Shaped(x_shape, y_shape))
        G.add_subsystem('comp', JaxExplicitComp1Shaped(x_shape, y_shape))

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
        G.add_subsystem('comp2', JaxExplicitComp2())
        G.add_subsystem('comp', JaxExplicitComp1())

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
        G.add_subsystem('comp2', JaxExplicitComp2())
        G.add_subsystem('comp', JaxExplicitComp1())

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

    def test_jax_implicit_comp_group(self):
        p = om.Problem()
        # create an IVC manually so we can set the shapes.  Otherwise must set shape in the component
        # itself.
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('a'))
        ivc.add_output('b')
        ivc.add_output('c')
        G = p.model.add_subsystem('G', om.Group(derivs_method='jax'))
        #G = p.model.add_subsystem('G', om.Group())
        G.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        G.linear_solver = om.ScipyKrylov()
        comp = G.add_subsystem('comp', JaxQuadraticCompPrimal())
        p.model.connect('ivc.a', 'G.comp.a')
        p.model.connect('ivc.b', 'G.comp.b')
        p.model.connect('ivc.c', 'G.comp.c')
        p.setup()

        p.set_val('ivc.a', 1.0)
        p.set_val('ivc.b', -4.0)
        p.set_val('ivc.c', 3.0)

        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('G.comp.x'), 3.0)
        assert_check_totals(p.check_totals(of=['G.comp.x'],
                                           wrt=['G.comp.a', 'G.comp.b', 'G.comp.c'],
                                           method='fd',
                                           show_only_incorrect=True,
                                           abs_err_tol=3e-5,
                                           rel_err_tol=5e-6),
                            atol=3e-5, rtol=5e-6)
        assert_check_partials(p.check_partials(show_only_incorrect=True), atol=2e-6)

if __name__ == '__main__':
    unittest.main()

    # from openmdao.utils.jax_utils import benchmark_component
    # result = benchmark_component(JaxExplicitComp2Shaped, methods=('jax', 'cs'),
    #                              repeats=10, table_format='tabulator', xshape=(44, 330), yshape=(330, 55))
