import unittest
import math

import numpy as np
from numpy.testing import assert_almost_equal
from io import StringIO

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
from openmdao.utils.cs_safe import abs, arctan2
import openmdao.func_api as omf
from openmdao.utils.coloring import compute_total_coloring


try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

class TestFuncCompNoWrap(unittest.TestCase):

    def test_scalar_function(self):
        def func(x=2.0):
            y = x + 1.
            return y

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(func))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._inputs['x'], 2.0, 0.00001)
        assert_near_equal(C1._outputs['y'], 3.0, 0.00001)

    def test_array_in_scalar_out(self):
        def func(x=np.arange(10, dtype=float)):
            y = np.sum(x)
            return y

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(func))
        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._inputs['x'], np.arange(10, dtype=float), 0.00001)
        assert_near_equal(C1._outputs['y'], 45.0, 0.00001)

    def test_assumed_default_val(self):
        def func(x):  # x defaults to 1.0
            y = x + 1.
            return y

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(func))

        prob.setup()
        prob.run_model()

        assert_near_equal(C1._inputs['x'], 1.0, 0.00001)
        assert_near_equal(C1._outputs['y'], 2.0, 0.00001)

    def test_array(self):
        def func(x=np.array([1., 2., 3.])):
            y=x[1]
            return y

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(func))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 2.0, 0.00001)

    def test_complex_step(self):
        def func(x=2.0):
            y =2.0*x+1.
            return y

        f = omf.wrap(func).declare_partials(of='*', wrt='*', method='cs')

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(f))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 5.0, 0.00001)

        C1._linearize()

        assert_near_equal(C1._jacobian[('y', 'x')], [[2.0]], 0.00001)

    def test_complex_step2(self):
        def func(x=2.0):
            y=x*x + x*2.0
            return y

        f = omf.wrap(func).declare_partials(of='*', wrt='*', method='cs')

        prob = om.Problem(om.Group())
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(f))
        prob.set_solver_print(level=0)

        prob.setup(mode='fwd')
        prob.run_model()

        J = prob.compute_totals(['comp.y'], ['comp.x'], return_format='flat_dict')
        assert_near_equal(J['comp.y', 'comp.x'], np.array([[6.0]]), 0.00001)

        prob.setup(mode='rev')
        prob.run_model()

        J = prob.compute_totals(['comp.y'], ['comp.x'], return_format='flat_dict')
        assert_near_equal(J['comp.y', 'comp.x'], np.array([[6.0]]), 0.00001)

    def test_complex_step_multivars(self):
        def func(a=2.0, b=3.0, c=5.0):
            x = a**2 + c * 3.
            y = b * -1.
            z = 1.5 * a + b * b - c
            return x, y, z

        f = omf.wrap(func).declare_partials(of='*', wrt='*', method='cs')

        prob = om.Problem(om.Group())
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(f))
        prob.set_solver_print(level=0)

        prob.setup(mode='fwd')
        prob.run_model()

        J = prob.compute_totals(of=['comp.x', 'comp.y', 'comp.z'], wrt=['comp.a', 'comp.b', 'comp.c'], return_format='flat_dict')
        assert_near_equal(J['comp.x', 'comp.a'], np.array([[4.0]]), 0.00001)
        assert_near_equal(J['comp.x', 'comp.b'], np.array([[0.0]]), 0.00001)
        assert_near_equal(J['comp.x', 'comp.c'], np.array([[3.0]]), 0.00001)

        assert_near_equal(J['comp.y', 'comp.a'], np.array([[0.0]]), 0.00001)
        assert_near_equal(J['comp.y', 'comp.b'], np.array([[-1.0]]), 0.00001)
        assert_near_equal(J['comp.y', 'comp.c'], np.array([[0.0]]), 0.00001)

        assert_near_equal(J['comp.z', 'comp.a'], np.array([[1.5]]), 0.00001)
        assert_near_equal(J['comp.z', 'comp.b'], np.array([[6.0]]), 0.00001)
        assert_near_equal(J['comp.z', 'comp.c'], np.array([[-1.0]]), 0.00001)

        prob.setup(mode='rev')
        prob.run_model()

        J = prob.compute_totals(['comp.x', 'comp.y', 'comp.z'], wrt=['comp.a', 'comp.b', 'comp.c'], return_format='flat_dict')
        assert_near_equal(J['comp.x', 'comp.a'], np.array([[4.0]]), 0.00001)
        assert_near_equal(J['comp.x', 'comp.b'], np.array([[0.0]]), 0.00001)
        assert_near_equal(J['comp.x', 'comp.c'], np.array([[3.0]]), 0.00001)

        assert_near_equal(J['comp.y', 'comp.a'], np.array([[0.0]]), 0.00001)
        assert_near_equal(J['comp.y', 'comp.b'], np.array([[-1.0]]), 0.00001)
        assert_near_equal(J['comp.y', 'comp.c'], np.array([[0.0]]), 0.00001)

        assert_near_equal(J['comp.z', 'comp.a'], np.array([[1.5]]), 0.00001)
        assert_near_equal(J['comp.z', 'comp.b'], np.array([[6.0]]), 0.00001)
        assert_near_equal(J['comp.z', 'comp.c'], np.array([[-1.0]]), 0.00001)

    def test_abs_complex_step(self):
        def func(x=-2.0):
            y=2.0*abs(x)
            return y

        f = omf.wrap(func).declare_partials(of='*', wrt='*', method='cs')

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(f))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 4.0, 0.00001)

        # any positive C1.x should give a 2.0 derivative for dy/dx
        C1._inputs['x'] = 1.0e-10
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], [[2.0]], 0.00001)

        C1._inputs['x'] = -3.0
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], [[-2.0]], 0.00001)

        C1._inputs['x'] = 0.0
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], [[2.0]], 0.00001)

    def test_arctan_complex_step(self):
        def func(x=np.array([1+2j]), y=1):
            z=2.0*np.arctan2(y, x)
            return z

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(func))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['z'], np.array([1.57079633]), 1e-8)

    def test_feature_multi_output(self):

        def func(x=1.):
            y1=x+1.
            y2=x-1.
            return y1, y2

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', om.ExplicitFuncComp(func), promotes=['x'])

        prob.setup()

        prob.set_val('x', 2.0)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.y1'), 3.0, 0.00001)
        assert_near_equal(prob.get_val('comp.y2'), 1.0, 0.00001)

    def test_feature_multi_output2(self):
        # verify that expressions can have multiple LHS variables.

        def func(x=1.):
            y1, y2 = x+1., x-1.
            return y1, y2

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExplicitFuncComp(func), promotes=['x'])

        prob.setup()

        prob.set_val('x', 2.0)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.y1'), 3.0, 0.00001)
        assert_near_equal(prob.get_val('comp.y2'), 1.0, 0.00001)

    def test_feature_array(self):

        def func(x=np.array([1., 2., 3.])):
            y = x[1]
            return y

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExplicitFuncComp(func))

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.y'), 2.0, 0.00001)

    def test_feature_math(self):

        def func(x, y):
            z = np.sin(x)**2 + np.cos(y)**2
            return z

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExplicitFuncComp(func))

        prob.setup()

        prob.set_val('comp.x', np.pi/2.0)
        prob.set_val('comp.y', np.pi/2.0)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.z'), 1.0, 0.00001)

    def test_feature_numpy(self):

        def func(x=np.array([1., 2., 3.])):
            y = np.sum(x)
            return y

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExplicitFuncComp(func))

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['comp.y'], 6.0, 0.00001)


class TestFuncCompWrapped(unittest.TestCase):

    def test_units(self):
        def func(x=2.0, z=2.0):
            y=x+z+1.
            return y

        f = (omf.wrap(func)
             .add_input('x', units='m')
             .add_output('y', units='m'))

        prob = om.Problem()
        prob.model.add_subsystem('indep', om.IndepVarComp('x', 100.0, units='cm'))
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(f))
        prob.model.connect('indep.x', 'C1.x')

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 4.0, 0.00001)

    def test_units_meta(self):

        def func(x=2.0, z=2.0):
            y=x+z+1.
            return y

        f = omf.wrap(func).defaults(units='m')

        prob = om.Problem()
        prob.model.add_subsystem('indep', om.IndepVarComp('x', 100.0, units='cm'))
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(f))
        prob.model.connect('indep.x', 'C1.x')

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], 4.0, 0.00001)

    def test_units_varname_novalue(self):

        def func(x=2.0, units=2.0):
            y=x+units+1.
            return y

        f = (omf.wrap(func)
                .add_input('x', units='m')
                .add_output('y', units='m'))

        prob = om.Problem()
        prob.model.add_subsystem('indep', om.IndepVarComp('x', 100.0, units='cm'))
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(f))
        prob.model.connect('indep.x', 'C1.x')

        with self.assertRaises(Exception) as cm:
            prob.setup()

        self.assertEqual(str(cm.exception),
                         "'C1' <class ExplicitFuncComp>: cannot use variable name 'units' because it's a reserved keyword.")

    def test_bad_varname(self):

        def func(x=2.0):
            return x+1.

        f = (omf.wrap(func)
                .add_input('x', units='m')
                .add_output('foo:bar'))

        prob = om.Problem()
        prob.model.add_subsystem('indep', om.IndepVarComp('x', 100.0))
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(f))
        prob.model.connect('indep.x', 'C1.x')

        with self.assertRaises(Exception) as cm:
            prob.setup()

        self.assertEqual(str(cm.exception),
                         "'C1' <class ExplicitFuncComp>: 'foo:bar' is not a valid variable name.")

    def test_common_units(self):

        def func(x, z=2.0):
            y=x+z+1.
            return y

        f = (omf.wrap(func)
             .defaults(units='m')
             .add_input('x', val=2.0))

        prob = om.Problem()
        prob.model.add_subsystem('indep', om.IndepVarComp('x', 100.0, units='cm'))
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(f))
        prob.model.connect('indep.x', 'comp.x')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp.y'], 4.0, 0.00001)

    def test_common_units_no_var_meta(self):
        # make sure common units are assigned when no specific variable metadata is provided

        def func(x=2.0):
            y=x+1.
            return y

        f = omf.wrap(func).defaults(units='m')

        prob = om.Problem()
        prob.model.add_subsystem('indep', om.IndepVarComp('x', 2.0, units='km'))
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(f))

        prob.model.connect('indep.x', 'comp.x')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp.y'], 2001., 0.00001)

    def test_shape_def(self):
        def func(x):
            y = 3.0*x + 2.5
            return y

        f = omf.wrap(func).declare_partials(of='*', wrt='*', method='cs').defaults(shape=(5,))

        p = om.Problem()
        model = p.model
        model.add_subsystem('comp', om.ExplicitFuncComp(f))

        p.setup()
        p.run_model()

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        assert_almost_equal(J, np.eye(5)*3., decimal=6)

    def test_shaped_scalar_val(self):

        def func(x):
            y =3.0*x + 2.5
            return y

        f = (omf.wrap(func)
             .add_input('x', shape=(5,), val=5)
             .add_output('y', shape=(5,)))

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f))

        p.setup()
        p.run_model()

        self.assertEqual(p['comp.x'].shape, (5,))

    def test_common_shape(self):

        def func(x):
            y =3.0*x + 2.5
            return y

        f = omf.wrap(func).defaults(shape=(5,)).declare_partials(of='*', wrt='*', method='cs')

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f))

        p.setup()
        p.run_model()

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        assert_almost_equal(J, np.eye(5)*3., decimal=6)

    def test_defaults_shape(self):

        def func(x):
            y =3.0*x + 2.5
            return y

        f = omf.wrap(func).defaults(shape=(5,)).declare_partials(of='*', wrt='*', method='cs')

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f))

        p.setup()
        p.run_model()

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        assert_almost_equal(J, np.eye(5)*3., decimal=6)

    def test_common_shape_with_values(self):
        p = om.Problem()
        model = p.model

        def func(x):
            y =3.0*x + 2.5
            return y

        f = (omf.wrap(func)
             .defaults(shape=(5,))
             .declare_partials(of='*', wrt='*', method='cs')
             .add_input('x', val=np.zeros(5)))

        model.add_subsystem('comp', om.ExplicitFuncComp(f))

        p.setup()
        p.run_model()

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        assert_almost_equal(J, np.eye(5)*3., decimal=6)

    def test_array_lhs(self):
        def func(x=np.array([1., 2., 3.])):
            y=np.array([x[1], x[0]])
            return y

        f = omf.wrap(func).add_output('y', shape=2)

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(f))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], np.array([2., 1.]), 0.00001)

    def test_simple_array_model(self):
        def func(x):
            y = np.array([2.0*x[0]+7.0*x[1], 5.0*x[0]-3.0*x[1]])
            return y

        f = omf.wrap(func).declare_partials(of='*', wrt='*', method='cs').defaults(shape=2)

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(f))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        assert_check_partials(data, atol=1e-5, rtol=1e-5)

    def test_simple_array_model2(self):
        def func(x):
            y = np.array([[2., 7.], [5., -3.]]).dot(x)
            return y

        f = (omf.wrap(func)
             .declare_partials(of='*', wrt='*', method='cs')
             .add_input('x', shape=2)
             .add_output('y', shape=2))

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(f))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        assert_check_partials(data, atol=1e-5, rtol=1e-5)

    def test_complex_step_multivars(self):
        def func(a=np.arange(1,4,dtype=float), b=np.arange(3,6,dtype=float), c=np.arange(5,8,dtype=float)):
            x = a**2 + c * 3.
            y = b * -1.
            z = 1.5 * a + b * b - c
            return x, y, z

        f = (omf.wrap(func)
             .declare_partials(of='*', wrt='*', method='cs')
             .defaults(shape=3))

        prob = om.Problem(om.Group())
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(f))
        prob.set_solver_print(level=0)

        prob.setup(mode='fwd')
        prob.run_model()

        J = prob.compute_totals(of=['comp.x', 'comp.y', 'comp.z'], wrt=['comp.a', 'comp.b', 'comp.c'], return_format='flat_dict')
        Jcomp = prob.model.comp._jacobian._subjacs_info

        assert_near_equal(J['comp.x', 'comp.a'], np.diag(np.arange(1,4,dtype=float)*2.), 0.00001)
        assert_near_equal(J['comp.x', 'comp.b'], np.zeros((3,3)), 0.00001)
        assert_near_equal(J['comp.x', 'comp.c'], np.eye(3)*3., 0.00001)

        assert_near_equal(J['comp.y', 'comp.a'], np.zeros((3,3)), 0.00001)
        assert_near_equal(J['comp.y', 'comp.b'], -np.eye(3), 0.00001)
        assert_near_equal(J['comp.y', 'comp.c'], np.zeros((3,3)), 0.00001)

        assert_near_equal(J['comp.z', 'comp.a'], np.eye(3)*1.5, 0.00001)
        assert_near_equal(J['comp.z', 'comp.b'], np.diag(np.arange(3,6,dtype=float)*2.), 0.00001)
        assert_near_equal(J['comp.z', 'comp.c'], -np.eye(3), 0.00001)

        prob.setup(mode='rev')
        prob.run_model()

        J = prob.compute_totals(['comp.x', 'comp.y', 'comp.z'], wrt=['comp.a', 'comp.b', 'comp.c'], return_format='flat_dict')
        Jcomp = prob.model.comp._jacobian._subjacs_info

        assert_near_equal(J['comp.x', 'comp.a'], np.diag(np.arange(1,4,dtype=float)*2.), 0.00001)
        assert_near_equal(J['comp.x', 'comp.b'], np.zeros((3,3)), 0.00001)
        assert_near_equal(J['comp.x', 'comp.c'], np.eye(3)*3., 0.00001)

        assert_near_equal(J['comp.y', 'comp.a'], np.zeros((3,3)), 0.00001)
        assert_near_equal(J['comp.y', 'comp.b'], -np.eye(3), 0.00001)
        assert_near_equal(J['comp.y', 'comp.c'], np.zeros((3,3)), 0.00001)

        assert_near_equal(J['comp.z', 'comp.a'], np.eye(3)*1.5, 0.00001)
        assert_near_equal(J['comp.z', 'comp.b'], np.diag(np.arange(3,6,dtype=float)*2.), 0.00001)
        assert_near_equal(J['comp.z', 'comp.c'], -np.eye(3), 0.00001)

    def test_complex_step_multivars_coloring(self):
        def func(a=np.arange(1,4,dtype=float), b=np.arange(3,6,dtype=float), c=np.arange(5,8,dtype=float)):
            x = a**2 + c * 3.
            y = b * -1.
            z = 1.5 * a + b * b - c
            return x, y, z

        f = (omf.wrap(func)
             .declare_coloring(wrt='*')
             .defaults(shape=3))

        prob = om.Problem(om.Group())
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(f))
        prob.set_solver_print(level=0)

        prob.setup(mode='fwd')
        prob.run_model()

        J = prob.compute_totals(of=['comp.x', 'comp.y', 'comp.z'], wrt=['comp.a', 'comp.b', 'comp.c'], return_format='flat_dict')

        assert_near_equal(J['comp.x', 'comp.a'], np.diag(np.arange(1,4,dtype=float)*2.), 0.00001)
        assert_near_equal(J['comp.x', 'comp.b'], np.zeros((3,3)), 0.00001)
        assert_near_equal(J['comp.x', 'comp.c'], np.eye(3)*3., 0.00001)

        assert_near_equal(J['comp.y', 'comp.a'], np.zeros((3,3)), 0.00001)
        assert_near_equal(J['comp.y', 'comp.b'], -np.eye(3), 0.00001)
        assert_near_equal(J['comp.y', 'comp.c'], np.zeros((3,3)), 0.00001)

        assert_near_equal(J['comp.z', 'comp.a'], np.eye(3)*1.5, 0.00001)
        assert_near_equal(J['comp.z', 'comp.b'], np.diag(np.arange(3,6,dtype=float)*2.), 0.00001)
        assert_near_equal(J['comp.z', 'comp.c'], -np.eye(3), 0.00001)

        prob.setup(mode='rev')
        prob.run_model()

        J = prob.compute_totals(['comp.x', 'comp.y', 'comp.z'], wrt=['comp.a', 'comp.b', 'comp.c'], return_format='flat_dict')

        assert_near_equal(J['comp.x', 'comp.a'], np.diag(np.arange(1,4,dtype=float)*2.), 0.00001)
        assert_near_equal(J['comp.x', 'comp.b'], np.zeros((3,3)), 0.00001)
        assert_near_equal(J['comp.x', 'comp.c'], np.eye(3)*3., 0.00001)

        assert_near_equal(J['comp.y', 'comp.a'], np.zeros((3,3)), 0.00001)
        assert_near_equal(J['comp.y', 'comp.b'], -np.eye(3), 0.00001)
        assert_near_equal(J['comp.y', 'comp.c'], np.zeros((3,3)), 0.00001)

        assert_near_equal(J['comp.z', 'comp.a'], np.eye(3)*1.5, 0.00001)
        assert_near_equal(J['comp.z', 'comp.b'], np.diag(np.arange(3,6,dtype=float)*2.), 0.00001)
        assert_near_equal(J['comp.z', 'comp.c'], -np.eye(3), 0.00001)

    def test_abs_array_complex_step(self):
        def func(x=np.ones(3)*-2.0):
            y=2.0*abs(x)
            return y

        f = omf.wrap(func).add_output('y', shape=(3,)).declare_partials(of='*', wrt='*', method='cs')

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExplicitFuncComp(f))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(C1._outputs['y'], np.ones(3)*4.0, 0.00001)

        # any positive C1.x should give a 2.0 derivative for dy/dx
        C1._inputs['x'] = np.ones(3)*1.0e-10
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], np.eye(3)*2.0, 0.00001)

        C1._inputs['x'] = np.ones(3)*-3.0
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], np.eye(3)*-2.0, 0.00001)

        C1._inputs['x'] = np.zeros(3)
        C1._linearize()
        assert_near_equal(C1._jacobian['y', 'x'], np.eye(3)*2.0, 0.00001)

        C1._inputs['x'] = np.array([1.5, -0.6, 2.4])
        C1._linearize()
        expect = np.zeros((3, 3))
        expect[0, 0] = 2.0
        expect[1, 1] = -2.0
        expect[2, 2] = 2.0

        assert_near_equal(C1._jacobian['y', 'x'], expect, 0.00001)

    def test_exec_comp_deriv_sparsity(self):
        # Check to make sure that when an ExplicitFuncComp has more than one
        # expression that only the partials that are needed are declared and computed

        # with has_diag_partials set to the default of False and just scalars

        def func(x1=1.0, x2=2.0):
            y1=2.0*x1+1.
            y2=3.0*x2-1.
            return y1, y2

        f = (omf.wrap(func)
             .declare_partials(of='y1', wrt='x1', method='cs')
             .declare_partials(of='y2', wrt='x2', method='cs'))

        p = om.Problem()
        comp = p.model.add_subsystem('comp', om.ExplicitFuncComp(f))
        p.setup()
        p.final_setup()

        # make sure only the partials that are needed are declared
        declared_partials = comp._declared_partials_patterns
        self.assertListEqual( sorted([('y1', 'x1'), ('y2', 'x2') ]),
                              sorted(declared_partials.keys()))

        p.run_model()

        # make sure only what is needed was computed
        subjacs_info = comp._jacobian._subjacs_info
        self.assertListEqual(sorted([('comp.y1', 'comp.x1'), ('comp.y2', 'comp.x2'),
                                     ('comp.y1', 'comp.y1'),('comp.y2', 'comp.y2')]),
                             sorted(subjacs_info.keys()))

        # make sure the result of compute_partials is correct
        J = p.compute_totals(of=['comp.y1'], wrt=['comp.x1'], return_format='array')
        self.assertEqual(2.0, J)
        J = p.compute_totals(of=['comp.y2'], wrt=['comp.x2'], return_format='array')
        self.assertEqual(3.0, J)

        # make sure this works with arrays

        def func2(x1=np.ones(5), x2=np.ones(5)):
            y1=2.0*x1+1.
            y2=3.0*x2-1.
            return y1, y2

        f = (omf.wrap(func2)
             .defaults(shape=5)
             .declare_partials(of='y1', wrt='x1', method='cs')
             .declare_partials(of='y2', wrt='x2', method='cs'))

        p = om.Problem()
        comp = p.model.add_subsystem('comp', om.ExplicitFuncComp(f))
        p.setup()
        p.final_setup()

        declared_partials = comp._declared_partials_patterns
        self.assertListEqual( sorted([('y1', 'x1'), ('y2', 'x2') ]),
                              sorted(declared_partials.keys()))

        p.run_model()
        J = p.compute_totals(of=['comp.y1'], wrt=['comp.x1'], return_format='array')
        self.assertTrue(np.all(2.0*np.identity(5) == J))
        J = p.compute_totals(of=['comp.y2'], wrt=['comp.x2'], return_format='array')
        self.assertTrue(np.all(3.0*np.identity(5) == J))

    def test_feature_defaults(self):

        def func(x=0., y=0.):
            z = x + y
            return z

        f = omf.wrap(func).defaults(units='inch')

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExplicitFuncComp(f))

        prob.setup()

        prob.set_val('comp.x', 12.0, units='inch')
        prob.set_val('comp.y', 1.0, units='ft')

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('comp.z'), 24.0, 0.00001)

    def test_list_outputs_resids_tol(self):

        def func(a=2.0, b=5.0, c=3.0, x=np.ones(2)):
            y = a * x ** 2 + b * x + c
            return y

        f = omf.wrap(func).add_output('y', shape=2)

        prob = om.Problem()
        model = prob.model
        model.add_subsystem("quad_1", om.ExplicitFuncComp(f))

        balance = model.add_subsystem("balance", om.BalanceComp())
        balance.add_balance("x_1", val=np.array([1, -1]), rhs_val=np.array([0., 0.]))
        model.connect("balance.x_1", "quad_1.x")
        model.connect("quad_1.y", "balance.lhs:x_1")

        prob.model.linear_solver = om.ScipyKrylov()
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()
        prob.model.nonlinear_solver.options["maxiter"] = 0
        prob.run_model()

        stream = StringIO()
        outputs = prob.model.list_outputs(residuals=True, residuals_tol=1e-5, out_stream=stream)

        text = stream.getvalue()
        self.assertTrue("balance" in text)
        self.assertTrue("x_1" in text)


class ExplicitFuncCompCountRuns(om.ExplicitFuncComp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_count = 0

    def compute(self, inputs, outputs):
        super().compute(inputs, outputs)
        self.compute_count += 1


# user defined compute_partials tests #

class TestFuncCompUserPartials(unittest.TestCase):
    def test_user_partials(self):

        def func3(x1=np.ones(5), x2=np.ones(5)):
            y1=2.0*x1+1.
            y2=3.0*x2-1.
            return y1, y2

        f = (omf.wrap(func3)
                .defaults(shape=5)
                .declare_partials(of='*', wrt='*', method='cs'))

        p = om.Problem()
        p.model.add_subsystem('comp', ExplicitFuncCompCountRuns(f))

        p.setup(mode='fwd')
        p.run_model()

        assert_check_partials(p.check_partials(includes=['comp'], out_stream=None))

    def test_user_partials_sparse(self):
        def func3(x1=np.ones(5), x2=np.ones(5)):
            y1=2.0*x1+1.
            y2=3.0*x2-1.
            return y1, y2

        f = (omf.wrap(func3)
                .defaults(shape=5)
                .declare_partials(of=['y1'], wrt=['x1'], method='cs',
                                  rows=np.arange(5), cols=np.arange(5))
                .declare_partials(of=['y2'], wrt=['x2'], method='cs',
                                  rows=np.arange(5), cols=np.arange(5)))

        p = om.Problem()
        p.model.add_subsystem('comp', ExplicitFuncCompCountRuns(f))

        p.setup(mode='fwd')
        p.run_driver()

        assert_check_partials(p.check_partials(includes=['comp'], out_stream=None))

        coloring = compute_total_coloring(p, mode='fwd', of=['comp.y1', 'comp.y2'], wrt=['comp.x1', 'comp.x2'])

        self.assertEqual(coloring.total_solves(), 1)  # verify that sparsity had an effect


# coloring tests #

def mat_factory(ninputs, noutputs):
    inshapes = list(zip(np.random.randint(1, 4, ninputs), np.random.randint(1, 4, ninputs)))
    outshapes = list(zip(np.random.randint(1, 4, noutputs), np.random.randint(1, 4, noutputs)))

    nrows = np.sum([np.prod(shp) for shp in outshapes])
    ncols = np.sum([np.prod(shp) for shp in inshapes])

    # create a sparse matrix
    mat = np.random.random((nrows, ncols)) < 0.2

    # find any zero cols and insert a nonzero value in them
    for col in np.where(np.count_nonzero(mat, axis=0) == 0)[0]:
        mat[np.random.randint(nrows), col] = True

    # find any zero rows and insert a nonzero value in them
    for row in np.where(np.count_nonzero(mat, axis=1) == 0)[0]:
        mat[row, np.random.randint(ncols)] = True

    # convert sparse matrix to randomized floats
    fmat = np.random.random(mat.shape)
    fmat[mat == 0] = 0.
    return fmat, inshapes, outshapes


def ovec2outs(ovec, outsizes):
    start = end = 0
    for sz in outsizes:
        end += sz
        yield ovec[start:end]
        start = end


class TestFuncCompColoring(unittest.TestCase):
    def test_coloring1(self):
        mat, inshapes, outshapes = mat_factory(3, 2)
        outsizes = [np.prod(shp) for shp in outshapes]

        def func(a, b, c):
            ivec = np.hstack([a.flat, b.flat, c.flat])
            ovec = mat.dot(ivec)
            x, y = ovec2outs(ovec, outsizes)
            return x, y

        f = (omf.wrap(func)
             .add_inputs(a={'shape': inshapes[0]}, b={'shape': inshapes[1]}, c={'shape': inshapes[2]})
             .add_outputs(x={'shape': outshapes[0]}, y={'shape': outshapes[1]})
             .declare_coloring(wrt='*', method='cs', show_summary=False)
             )

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f))

        p.setup(mode='fwd')
        p.run_model()
        assert_check_totals(p.check_totals(of=['comp.x', 'comp.y'], wrt=['comp.a', 'comp.b', 'comp.c'], method='cs', out_stream=None))

        p.setup(mode='rev')
        p.run_model()
        assert_check_totals(p.check_totals(of=['comp.x', 'comp.y'], wrt=['comp.a', 'comp.b', 'comp.c'], method='cs', out_stream=None))

    def test_coloring2(self):
        # this test uses a very narrow matrix so the attempt at partial coloring will abort.
        # There used to be a bug where the total derivatives would be incorrect when this
        # happened.
        mat = np.array([[0.14898778],
                        [0.19860233],
                        [0.81899035],
                        [0.78498818],
                        [0.68436335],
                        [0.93677595],
                        [0.33964473],
                        [0.82057559],
                        [0.62672187],
                        [0.52089597],
                        [0.28524249],
                        [0.62003238]])

        inshapes = [1]
        outshapes = [2, 3, 7]
        outsizes = [np.prod(shp) for shp in outshapes]

        def func(a):
            ovec = mat.dot(a.flat)
            x, y, z = ovec2outs(ovec, outsizes)
            return x, y, z

        f = (omf.wrap(func)
             .add_inputs(a={'shape': inshapes[0]})
             .add_outputs(x={'shape': outshapes[0]}, y={'shape': outshapes[1]}, z={'shape': outshapes[2]})
             .declare_coloring(wrt='*', method='cs', show_summary=False)
             )

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f))

        p.setup(mode='fwd')
        p.run_model()
        assert_check_totals(p.check_totals(of=['comp.x', 'comp.y', 'comp.z'], wrt=['comp.a'], out_stream=None))

        p.setup(mode='rev')
        p.run_model()
        assert_check_totals(p.check_totals(of=['comp.x', 'comp.y', 'comp.z'], wrt=['comp.a'], out_stream=None))


class TestComputePartials(unittest.TestCase):
    def test_user_compute_partials_func(self):
        def J_func(x, y, z, J):

            # the following sub-jacs are 4x4 based on the sizes of foo, bar, x, and y, but the partials
            # were declared specifying rows and cols (in this case sub-jacs are diagonal), so we only
            # store the nonzero values of the sub-jacs, resulting in an actual size of 4 rather than 4x4.
            J['foo', 'x'] = -3*np.log(z)/(3*x+2*y)**2
            J['foo', 'y'] = -2*np.log(z)/(3*x+2*y)**2

            J['bar', 'x'] = 2.*np.ones(4)
            J['bar', 'y'] = np.ones(4)

            # z is a scalar so the true size of this sub-jac is 4x1
            J['foo', 'z'] = 1/(z*(3*x+2*y))

        def func(x=np.zeros(4), y=np.ones(4), z=3):
            foo = np.log(z)/(3*x+2*y)
            bar = 2.*x + y
            return foo, bar

        f = (omf.wrap(func)
                .defaults(units='m')
                .add_output('foo', units='1/m', shape=4)
                .add_output('bar', shape=4)
                .declare_partials(of='foo', wrt=('x', 'y'), rows=np.arange(4), cols=np.arange(4))
                .declare_partials(of='foo', wrt='z')
                .declare_partials(of='bar', wrt=('x', 'y'), rows=np.arange(4), cols=np.arange(4)))

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f, compute_partials=J_func))
        p.setup(force_alloc_complex=True)
        p.run_model()
        assert_check_totals(p.check_totals(of=['comp.foo', 'comp.bar'], wrt=['comp.x', 'comp.y', 'comp.z'], method='cs'))


@unittest.skipIf(jax is None, "jax is not installed")
class TestJax(unittest.TestCase):
    def check_derivs(self, mode, shape, use_jit):
        def func(a, b, c):
            x = 2. * a * b + 3. * c
            return x

        f = omf.wrap(func).defaults(shape=shape).declare_partials(of='*', wrt='*', method='jax')
        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f, use_jax=True, use_jit=use_jit))
        p.setup(mode=mode)
        p.run_model()
        J = p.compute_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'])

        I = np.eye(np.prod(shape, dtype=int))
        assert_near_equal(J['comp.x', 'comp.a'], I * 2.)
        assert_near_equal(J['comp.x', 'comp.b'], I * 2.)
        assert_near_equal(J['comp.x', 'comp.c'], I * 3.)

    def test_fwd3x2(self):
        self.check_derivs('fwd', (3,2), use_jit=False)

    def test_fwd_jit3x2(self):
        self.check_derivs('fwd', (3,2), use_jit=True)

    def test_rev3x2(self):
        self.check_derivs('rev', (3,2), use_jit=False)

    def test_rev_jit3x2(self):
        self.check_derivs('rev', (3,2), use_jit=True)

    def test_fwd(self):
        self.check_derivs('fwd', (), use_jit=False)

    def test_fwd_jit(self):
        self.check_derivs('fwd', (), use_jit=True)

    def test_rev(self):
        self.check_derivs('rev', (), use_jit=False)

    def test_rev_jit(self):
        self.check_derivs('rev', (), use_jit=True)


@unittest.skipIf(jax is None, "jax is not installed")
class TestJaxMixedShapes1output(unittest.TestCase):
    def check_derivs(self, mode, m, n, o):
        def func(a, b, c):
            x = 2. * a.dot(b) + c
            return x

        f = omf.wrap(func).declare_partials(of='*', wrt='*', method='jax')
        ishapes = {'a': (n,m), 'b': (m,o), 'c': (n,o)}
        oshapes = {'x': (n,o)}

        for name in ['a', 'b', 'c']:
            f.add_input(name, shape=ishapes[name])

        for name in ['x']:
            f.add_output(name, shape=oshapes[name])

        rand_inputs = {
            n: np.random.random(ishapes[n]) for n in ('a', 'b', 'c')
        }

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f, use_jax=True))
        p.setup(mode=mode)

        for n in ('a', 'b', 'c'):
            p[f"comp.{n}"] = rand_inputs[n]

        p.run_model()
        J = p.compute_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'])

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExecComp(['x=2.*a.dot(b)+c'],
                                                  x={'shape':oshapes['x']},
                                                  a={'shape':ishapes['a']},
                                                  b={'shape':ishapes['b']},
                                                  c={'shape':ishapes['c']},
                                                  ))
        p.setup(mode=mode)

        for n in ('a', 'b', 'c'):
            p[f"comp.{n}"] = rand_inputs[n]

        p.run_model()
        Jchk = p.compute_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'])

        for inp in ['comp.a', 'comp.b', 'comp.c']:
            assert_near_equal(J['comp.x', inp], Jchk['comp.x', inp])

    def test1fwd(self):
        self.check_derivs('fwd', 2, 3, 5)

    def test1rev(self):
        self.check_derivs('rev', 2, 3, 5)

    def test2fwd(self):
        self.check_derivs('fwd', 9, 3, 5)

    def test2rev(self):
        self.check_derivs('rev', 9, 3, 5)


@unittest.skipIf(jax is None, "jax is not installed")
class TestJaxMixedShapes2outputs(unittest.TestCase):
    def check_derivs(self, mode, m, n, o, p, q):
        def func(a, b, c):
            x = 2. * a.dot(b)
            y = 3. * c
            return x, y

        f = omf.wrap(func).declare_partials(of='*', wrt='*', method='jax')
        ishapes = {'a': (n,m), 'b': (m,o), 'c': (p,q)}
        oshapes = {'x': (n,o), 'y': (p,q)}

        for name in ['a', 'b', 'c']:
            f.add_input(name, shape=ishapes[name])

        for name in ['x', 'y']:
            f.add_output(name, shape=oshapes[name])

        rand_inputs = {
            n: np.random.random(ishapes[n]) for n in ('a', 'b', 'c')
        }

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f, use_jax=True))
        p.setup(mode=mode)

        for n in ('a', 'b', 'c'):
            p[f"comp.{n}"] = rand_inputs[n]

        p.run_model()
        J = p.compute_totals(of=['comp.x', 'comp.y'], wrt=['comp.a', 'comp.b', 'comp.c'])

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExecComp(['x=2.*a.dot(b)', 'y=3.*c'],
                                                  x={'shape':oshapes['x']},
                                                  y={'shape':oshapes['y']},
                                                  a={'shape':ishapes['a']},
                                                  b={'shape':ishapes['b']},
                                                  c={'shape':ishapes['c']},
                                                  ))
        p.setup(mode=mode)

        for n in ('a', 'b', 'c'):
            p[f"comp.{n}"] = rand_inputs[n]

        p.run_model()
        Jchk = p.compute_totals(of=['comp.x', 'comp.y'], wrt=['comp.a', 'comp.b', 'comp.c'])

        for out in ['comp.x', 'comp.y']:
            for inp in ['comp.a', 'comp.b', 'comp.c']:
                assert_near_equal(J[out, inp], Jchk[out, inp])

    def test1fwd(self):
        self.check_derivs('fwd', 2, 3, 5, 7, 1)

    def test1rev(self):
        self.check_derivs('rev', 2, 3, 5, 7, 1)

    def test2fwd(self):
        self.check_derivs('fwd', 9, 3, 5, 7, 2)

    def test2rev(self):
        self.check_derivs('rev', 9, 3, 5, 7, 2)


@unittest.skipIf(jax is None, "jax is not installed")
class TestJaxNumpy(unittest.TestCase):
    def check_derivs(self, mode, shape, use_jit):
        def func(a, b, c):
            x = np.sin(a) * b + 3. * c
            return x

        f = omf.wrap(func).defaults(shape=shape).declare_partials(of='*', wrt='*', method='jax')
        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f, use_jax=True, use_jit=use_jit))
        p.setup(mode=mode)
        p['comp.a'] = 1.0
        p['comp.b'] = 2.0
        p['comp.c'] = 3.0
        p.run_model()

        assert_check_partials(p.check_partials(includes=['comp'], method='fd', out_stream=None), atol=1e-5)

        J = p.compute_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'])

        I = np.eye(np.prod(shape)) if shape else np.eye(1)
        assert_near_equal(J['comp.x', 'comp.a'], I * p['comp.b'].ravel() * np.cos(p['comp.a']).ravel(), tolerance=1e-7)
        assert_near_equal(J['comp.x', 'comp.b'], I * np.sin(p['comp.a']).ravel(), tolerance=1e-7)
        assert_near_equal(J['comp.x', 'comp.c'], I * 3., tolerance=1e-7)

    def test_fwd3x2(self):
        self.check_derivs('fwd', (3,2), use_jit=False)

    def test_fwd_jit3x2(self):
        self.check_derivs('fwd', (3,2), use_jit=True)

    def test_rev3x2(self):
        self.check_derivs('rev', (3,2), use_jit=False)

    def test_rev_jit3x2(self):
        self.check_derivs('rev', (3,2), use_jit=True)

    def test_fwd(self):
        self.check_derivs('fwd', (), use_jit=False)

    def test_fwd_jit(self):
        self.check_derivs('fwd', (), use_jit=True)

    def test_rev(self):
        self.check_derivs('rev', (), use_jit=False)

    def test_rev_jit(self):
        self.check_derivs('rev', (), use_jit=True)


@unittest.skipIf(jax is None, "jax is not installed")
class TestJax2retvals(unittest.TestCase):
    def check_derivs(self, mode, shape, use_jit):
        def func(a, b, c):
            x = 2. * a * b + 3. * c
            y = 5. * a * c - 2.5 * b
            return x, y

        f = omf.wrap(func).defaults(shape=shape).declare_partials(of='*', wrt='*', method='jax')
        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f, use_jax=True, use_jit=use_jit))
        p.setup(mode=mode)
        p.run_model()
        J = p.compute_totals(of=['comp.x', 'comp.y'], wrt=['comp.a', 'comp.b', 'comp.c'])

        I = np.eye(np.prod(shape)) if shape else np.eye(1)
        assert_near_equal(J['comp.x', 'comp.a'], I * 2.)
        assert_near_equal(J['comp.x', 'comp.b'], I * 2.)
        assert_near_equal(J['comp.x', 'comp.c'], I * 3.)
        assert_near_equal(J['comp.y', 'comp.a'], I * 5.)
        assert_near_equal(J['comp.y', 'comp.b'], I * -2.5)
        assert_near_equal(J['comp.y', 'comp.c'], I * 5.)

    def test_fwd3x1(self):
        self.check_derivs('fwd', (3,1), use_jit=False)

    def test_fwd3x2(self):
        self.check_derivs('fwd', (3,2), use_jit=False)

    def test_fwd_jit3x2(self):
        self.check_derivs('fwd', (3,2), use_jit=True)

    def test_rev3x2(self):
        self.check_derivs('rev', (3,2), use_jit=False)

    def test_rev_jit3x2(self):
        self.check_derivs('rev', (3,2), use_jit=True)

    def test_fwd(self):
        self.check_derivs('fwd', (), use_jit=False)

    def test_fwd_jit(self):
        self.check_derivs('fwd', (), use_jit=True)

    def test_rev(self):
        self.check_derivs('rev', (), use_jit=False)

    def test_rev_jit(self):
        self.check_derivs('rev', (), use_jit=True)


@unittest.skipIf(jax is None, "jax is not installed")
class TestJaxNonDifferentiableArgs(unittest.TestCase):
    def check_derivs(self, mode, use_jit, method):
        def func(a, b, c, ex1, ex2):
            x = 2. * a * b + 3. * c
            y = 5. * a * c - 2.5 * b
            return x, y

        f = (omf.wrap(func)
                .defaults(shape=3)
                .declare_option('ex1', default='foo')
                .declare_option('ex2', default='bar')
                .declare_partials(of='*', wrt='*', method=method)
                )

        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f, use_jit=use_jit))
        p.setup(mode=mode)
        p.run_model()
        J = p.compute_totals(of=['comp.x', 'comp.y'], wrt=['comp.a', 'comp.b', 'comp.c'])

        I = np.eye(3)
        assert_near_equal(J['comp.x', 'comp.a'], I * 2.)
        assert_near_equal(J['comp.x', 'comp.b'], I * 2.)
        assert_near_equal(J['comp.x', 'comp.c'], I * 3.)
        assert_near_equal(J['comp.y', 'comp.a'], I * 5.)
        assert_near_equal(J['comp.y', 'comp.b'], I * -2.5)
        assert_near_equal(J['comp.y', 'comp.c'], I * 5.)

    def test_fwd(self):
        self.check_derivs('fwd', use_jit=False, method='jax')

    def test_fwd_jit(self):
        self.check_derivs('fwd', use_jit=True, method='jax')

    def test_rev(self):
        self.check_derivs('rev', use_jit=False, method='jax')

    def test_rev_jit(self):
        self.check_derivs('rev', use_jit=True, method='jax')

    def test_fwd_cs(self):
        self.check_derivs('fwd', use_jit=False, method='cs')

    def test_rev_cs(self):
        self.check_derivs('rev', use_jit=False, method='cs')


@unittest.skipIf(jax is None, "jax is not installed")
class TestJax2retvalsColoring(unittest.TestCase):
    def check_derivs(self, mode, shape, use_jit):
        def func(a, b, c):
            x = 2. * a * b + 3. * c
            y = 5. * a * c - 2.5 * b
            return x, y

        f = (omf.wrap(func)
                .defaults(shape=shape)
                .declare_partials(of='*', wrt='*', method='jax')
                .declare_coloring(wrt='*', method='jax')
        )
        p = om.Problem()
        p.model.add_subsystem('comp', om.ExplicitFuncComp(f, use_jax=True, use_jit=use_jit))
        p.setup(mode=mode)
        p.run_model()
        J = p.compute_totals(of=['comp.x', 'comp.y'], wrt=['comp.a', 'comp.b', 'comp.c'])

        I = np.eye(np.prod(shape)) if shape else np.eye(1)
        assert_near_equal(J['comp.x', 'comp.a'], I * 2.)
        assert_near_equal(J['comp.x', 'comp.b'], I * 2.)
        assert_near_equal(J['comp.x', 'comp.c'], I * 3.)
        assert_near_equal(J['comp.y', 'comp.a'], I * 5.)
        assert_near_equal(J['comp.y', 'comp.b'], I * -2.5)
        assert_near_equal(J['comp.y', 'comp.c'], I * 5.)

    def test_fwd3x2(self):
        self.check_derivs('fwd', (3,2), use_jit=False)

    def test_fwd_jit3x2(self):
        self.check_derivs('fwd', (3,2), use_jit=True)

    def test_rev3x2(self):
        self.check_derivs('rev', (3,2), use_jit=False)

    def test_rev_jit3x2(self):
        self.check_derivs('rev', (3,2), use_jit=True)

    def test_fwd(self):
        self.check_derivs('fwd', (), use_jit=False)

    def test_fwd_jit(self):
        self.check_derivs('fwd', (), use_jit=True)

    def test_rev(self):
        self.check_derivs('rev', (), use_jit=False)

    def test_rev_jit(self):
        self.check_derivs('rev', (), use_jit=True)

if __name__ == "__main__":
    unittest.main()
