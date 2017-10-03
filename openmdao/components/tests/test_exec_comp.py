import unittest
import math

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp
from openmdao.devtools.testutil import assert_rel_error


class TestExecComp(unittest.TestCase):

    def test_colon_vars(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('C1', ExecComp('y=foo:bar+1.'))
        with self.assertRaises(Exception) as context:
            prob.setup(check=False)
        self.assertEqual(str(context.exception),
                         "C1: failed to compile expression 'y=foo:bar+1.'.")

    def test_bad_kwargs(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('C1', ExecComp('y=x+1.', xx=2.0))
        with self.assertRaises(Exception) as context:
            prob.setup(check=False)
        self.assertEqual(str(context.exception),
                         "C1: arg 'xx' in call to ExecComp() does not refer to any variable in the expressions ['y=x+1.']")

    def test_bad_kwargs_meta(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('C1', ExecComp('y=x+1.', x={'val': 2.0, 'low': 0.0, 'high': 10.0, 'units': 'ft'}))
        with self.assertRaises(Exception) as context:
            prob.setup(check=False)
        self.assertEqual(str(context.exception),
                         "C1: the following metadata names were not recognized for variable 'x': ['high', 'low', 'val']")

    def test_name_collision_const(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('C1', ExecComp('e=x+1.'))
        with self.assertRaises(Exception) as context:
            prob.setup(check=False)
        self.assertEqual(str(context.exception),
                         "C1: cannot assign to variable 'e' because it's already defined as an internal function or constant.")

    def test_name_collision_func(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('C1', ExecComp('sin=x+1.'))
        with self.assertRaises(Exception) as context:
            prob.setup(check=False)
        self.assertEqual(str(context.exception),
                         "C1: cannot assign to variable 'sin' because it's already defined as an internal function or constant.")

    def test_func_as_rhs_var(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('C1', ExecComp('y=sin+1.'))
        with self.assertRaises(Exception) as context:
            prob.setup(check=False)
        self.assertEqual(str(context.exception),
                         "C1: cannot use 'sin' as a variable because it's already defined as an internal function.")

    def test_mixed_type(self):
        prob = Problem(model=Group())
        C1 = prob.model.add_subsystem('C1', ExecComp('y=numpy.sum(x)',
                                                     x=np.arange(10,dtype=float)))
        prob.setup(check=False)

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], 45.0, 0.00001)

    def test_simple(self):
        prob = Problem(model=Group())
        C1 = prob.model.add_subsystem('C1', ExecComp('y=x+1.', x=2.0))

        prob.setup(check=False)

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], 3.0, 0.00001)

    def test_for_spaces(self):
        prob = Problem(model=Group())
        C1 = prob.model.add_subsystem('C1', ExecComp('y = pi * x', x=2.0))

        prob.setup(check=False)

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)
        self.assertTrue('pi' not in C1._inputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], 2 * math.pi, 0.00001)

    def test_units(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('indep', IndepVarComp('x', 100.0, units='cm'))
        C1 = prob.model.add_subsystem('C1', ExecComp('y=x+z+1.',
                                                     x={'value': 2.0, 'units': 'm'},
                                                     y={'units': 'm'},
                                                     z=2.0))
        prob.model.connect('indep.x', 'C1.x')

        prob.setup(check=False)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], 4.0, 0.00001)

    def test_math(self):
        prob = Problem(model=Group())
        C1 = prob.model.add_subsystem('C1', ExecComp('y=sin(x)', x=2.0))

        prob.setup(check=False)

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], math.sin(2.0), 0.00001)

    def test_array(self):
        prob = Problem(model=Group())
        C1 = prob.model.add_subsystem('C1', ExecComp('y=x[1]',
                                                     x=np.array([1.,2.,3.]),
                                                     y=0.0))

        prob.setup(check=False)

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], 2.0, 0.00001)

    def test_array_lhs(self):
        prob = Problem(model=Group())
        C1 = prob.model.add_subsystem('C1', ExecComp(['y[0]=x[1]', 'y[1]=x[0]'],
                                                     x=np.array([1.,2.,3.]),
                                                     y=np.array([0.,0.])))

        prob.setup(check=False)

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], np.array([2.,1.]), 0.00001)

    def test_simple_array_model(self):
        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p1', IndepVarComp('x', np.ones([2])))
        prob.model.add_subsystem('comp', ExecComp(['y[0]=2.0*x[0]+7.0*x[1]',
                                                   'y[1]=5.0*x[0]-3.0*x[1]'],
                                                  x=np.zeros([2]), y=np.zeros([2])))

        prob.model.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.set_solver_print(level=0)
        prob.run_model()

        data = prob.check_partials()

        assert_rel_error(self, data['comp'][('y','x')]['abs error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][2], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][2], 0.0, 1e-5)

    def test_simple_array_model2(self):
        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p1', IndepVarComp('x', np.ones([2])))
        prob.model.add_subsystem('comp', ExecComp('y = mat.dot(x)',
                                                  x=np.zeros((2,)), y=np.zeros((2,)),
                                                  mat=np.array([[2.,7.],[5.,-3.]])))

        prob.model.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.set_solver_print(level=0)
        prob.run_model()

        data = prob.check_partials()

        assert_rel_error(self, data['comp'][('y','x')]['abs error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][2], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][2], 0.0, 1e-5)

    def test_complex_step(self):
        prob = Problem(model=Group())
        C1 = prob.model.add_subsystem('C1', ExecComp(['y=2.0*x+1.'], x=2.0))

        prob.setup(check=False)

        # Conclude setup but don't run model.
        prob.final_setup()

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], 5.0, 0.00001)

        C1._linearize()

        assert_rel_error(self, C1.jacobian[('y','x')], [[-2.0]], 0.00001)

    def test_complex_step2(self):
        prob = Problem(Group())
        prob.model.add_subsystem('p1', IndepVarComp('x', 2.0))
        prob.model.add_subsystem('comp', ExecComp('y=x*x + x*2.0'))
        prob.model.connect('p1.x', 'comp.x')
        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        J = prob.compute_totals(['comp.y'], ['p1.x'], return_format='flat_dict')
        assert_rel_error(self, J['comp.y', 'p1.x'], np.array([[6.0]]), 0.00001)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_totals(['comp.y'], ['p1.x'], return_format='flat_dict')
        assert_rel_error(self, J['comp.y', 'p1.x'], np.array([[6.0]]), 0.00001)

    def test_abs_complex_step(self):
        prob = Problem(model=Group())
        C1 = prob.model.add_subsystem('C1', ExecComp('y=2.0*abs(x)', x=-2.0))

        prob.setup(check=False)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], 4.0, 0.00001)

        # any negative C1.x should give a 2.0 derivative for dy/dx
        C1._inputs['x'] = -1.0e-10
        C1._linearize()
        assert_rel_error(self, C1.jacobian['y','x'], [[2.0]], 0.00001)

        C1._inputs['x'] = 3.0
        C1._linearize()
        assert_rel_error(self, C1.jacobian['y','x'], [[-2.0]], 0.00001)

        C1._inputs['x'] = 0.0
        C1._linearize()
        assert_rel_error(self, C1.jacobian['y','x'], [[-2.0]], 0.00001)

    def test_abs_array_complex_step(self):
        prob = Problem(model=Group())
        C1 = prob.model.add_subsystem('C1', ExecComp('y=2.0*abs(x)',
                                                     x=np.ones(3)*-2.0, y=np.zeros(3)))

        prob.setup(check=False)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, C1._outputs['y'], np.ones(3)*4.0, 0.00001)

        # any negative C1.x should give a 2.0 derivative for dy/dx
        C1._inputs['x'] = np.ones(3)*-1.0e-10
        C1._linearize()
        assert_rel_error(self, C1.jacobian['y','x'], np.eye(3)*2.0, 0.00001)

        C1._inputs['x'] = np.ones(3)*3.0
        C1._linearize()
        assert_rel_error(self, C1.jacobian['y','x'], np.eye(3)*-2.0, 0.00001)

        C1._inputs['x'] = np.zeros(3)
        C1._linearize()
        assert_rel_error(self, C1.jacobian['y','x'], np.eye(3)*-2.0, 0.00001)

        C1._inputs['x'] = np.array([1.5, -0.6, 2.4])
        C1._linearize()
        expect = np.zeros((3,3))
        expect[0,0] = -2.0
        expect[1,1] = 2.0
        expect[2,2] = -2.0

        assert_rel_error(self, C1.jacobian['y','x'], expect, 0.00001)

    def test_feature_simple(self):
        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('p', IndepVarComp('x', 2.0))
        model.add_subsystem('comp', ExecComp('y=x+1.'))

        model.connect('p.x', 'comp.x')

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['comp.y'], 3.0, 0.00001)

    def test_feature_array(self):
        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('p', IndepVarComp('x', np.array([1., 2., 3.])))
        model.add_subsystem('comp', ExecComp('y=x[1]',
                                             x=np.array([1.,2.,3.]),
                                             y=0.0))
        model.connect('p.x', 'comp.x')

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['comp.y'], 2.0, 0.00001)

    def test_feature_math(self):
        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('p1', IndepVarComp('x', np.pi/2.0))
        model.add_subsystem('p2', IndepVarComp('y', np.pi/2.0))
        model.add_subsystem('comp', ExecComp('z = sin(x)**2 + cos(y)**2'))

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 1.0, 0.00001)

    def test_feature_numpy(self):
        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('p', IndepVarComp('x', np.array([1., 2., 3.])))
        model.add_subsystem('comp', ExecComp('y=numpy.sum(x)', x=np.zeros((3, ))))
        model.connect('p.x', 'comp.x')

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['comp.y'], 6.0, 0.00001)

    def test_feature_metadata(self):
        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 12.0, units='inch'))
        model.add_subsystem('p2', IndepVarComp('y', 1.0, units='ft'))
        model.add_subsystem('comp', ExecComp('z=x+y',
                                             x={'value': 0.0, 'units':'inch'},
                                             y={'value': 0.0, 'units': 'inch'},
                                             z={'value': 0.0, 'units': 'inch'}))
        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 24.0, 0.00001)
if __name__ == "__main__":
    unittest.main()
