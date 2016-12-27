import unittest
import math

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp
from openmdao.devtools.testutil import assert_rel_error

class TestExecComp(unittest.TestCase):

    def test_bad_kwargs(self):
        prob = Problem(root=Group())
        try:
            C1 = prob.root.add_subsystem('C1', ExecComp('y=x+1.', xx=2.0))
        except Exception as err:
            self.assertEqual(str(err), "Arg 'xx' in call to ExecComp() does not refer to any variable in the expressions ['y=x+1.']")

    def test_mixed_type(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp('y=numpy.sum(x)',
                                          x=np.arange(10,dtype=float)))
        prob.setup(check=False)

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], 45.0, 0.00001)

    def test_simple(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp('y=x+1.', x=2.0))

        prob.setup(check=False)

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], 3.0, 0.00001)

    def test_for_spaces(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp('y = pi * x', x=2.0))

        prob.setup(check=False)

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)
        self.assertTrue('pi' not in C1._inputs)

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], 2 * math.pi, 0.00001)

    def test_units(self):
        raise unittest.SkipTest("no units support yet")
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp('y=x+z+1.', x=2.0, z=2.0, units={'x':'m','y':'m'}))

        prob.setup(check=False)

        self.assertTrue('units' in C1._inputs['x'])
        self.assertTrue(C1._inputs['x']['units'] == 'm')
        self.assertTrue('units' in C1._outputs['y'])
        self.assertTrue(C1._outputs['y']['units'] == 'm')
        self.assertFalse('units' in C1._inputs['z'])

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], 5.0, 0.00001)

    def test_math(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp('y=sin(x)', x=2.0))

        prob.setup(check=False)

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], math.sin(2.0), 0.00001)

    def test_array(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp('y=x[1]', x=np.array([1.,2.,3.]), y=0.0))

        prob.setup(check=False)

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], 2.0, 0.00001)

    def test_array_lhs(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp(['y[0]=x[1]', 'y[1]=x[0]'],
                                          x=np.array([1.,2.,3.]), y=np.array([0.,0.])))

        prob.setup(check=False)

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], np.array([2.,1.]), 0.00001)

    def test_simple_array_model(self):
        prob = Problem()
        prob.root = Group()
        prob.root.add_subsystem('comp', ExecComp(['y[0]=2.0*x[0]+7.0*x[1]',
                                        'y[1]=5.0*x[0]-3.0*x[1]'],
                                       x=np.zeros([2]), y=np.zeros([2])))

        prob.root.add_subsystem('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.root.suppress_solver_output = True
        prob.run()

        raise unittest.SkipTest("no check_partial_derivatives function")
        data = prob.check_partial_derivatives(out_stream=None)

        assert_rel_error(self, data['comp'][('y','x')]['abs error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][2], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][2], 0.0, 1e-5)

    def test_simple_array_model2(self):
        prob = Problem()
        prob.root = Group()
        comp = prob.root.add_subsystem('comp', ExecComp('y = mat.dot(x)',
                                              x=np.zeros((2,)), y=np.zeros((2,)),
                                              mat=np.array([[2.,7.],[5.,-3.]])))

        p1 = prob.root.add_subsystem('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.root.suppress_solver_output = True
        prob.run()

        raise unittest.SkipTest("no check_partial_derivatives function")
        data = prob.check_partial_derivatives(out_stream=None)

        assert_rel_error(self, data['comp'][('y','x')]['abs error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][2], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][2], 0.0, 1e-5)

    def test_complex_step(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp(['y=2.0*x+1.'], x=2.0))

        prob.setup(check=False)

        self.assertTrue('x' in C1._inputs)
        self.assertTrue('y' in C1._outputs)

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], 5.0, 0.00001)

        C1._linearize(True)

        assert_rel_error(self, C1._jacobian[('y','x')], -2.0, 0.00001)

    def test_complex_step2(self):
        prob = Problem(Group())
        comp = prob.root.add_subsystem('comp', ExecComp('y=x*x + x*2.0'))
        prob.root.add_subsystem('p1', IndepVarComp('x', 2.0))
        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.root.suppress_solver_output = True
        prob.run()

        raise unittest.SkipTest("problem missing calc_gradient")
        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp.y']['p1.x'], np.array([6.0]), 0.00001)

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['comp.y']['p1.x'], np.array([6.0]), 0.00001)

    def test_abs_complex_step(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp('y=2.0*abs(x)', x=-2.0))

        prob.setup(check=False)
        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], 4.0, 0.00001)

        # any negative C1.x should give a 2.0 derivative for dy/dx
        C1._inputs['x'] = -1.0e-10
        C1._linearize()
        assert_rel_error(self, C1._jacobian[('y','x')], 2.0, 0.00001)

        C1._inputs['x'] = 3.0
        C1._linearize()
        assert_rel_error(self, C1._jacobian[('y','x')], -2.0, 0.00001)

        C1._inputs['x'] = 0.0
        C1._linearize()
        assert_rel_error(self, C1._jacobian[('y','x')], -2.0, 0.00001)

    def test_abs_array_complex_step(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp('y=2.0*abs(x)',
                                          x=np.ones(3)*-2.0, y=np.zeros(3)))

        prob.setup(check=False)
        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['y'], np.ones(3)*4.0, 0.00001)

        # any negative C1.x should give a 2.0 derivative for dy/dx
        C1._inputs['x'] = np.ones(3)*-1.0e-10
        C1._linearize()
        assert_rel_error(self, C1._jacobian[('y','x')], np.eye(3)*2.0, 0.00001)

        C1._inputs['x'] = np.ones(3)*3.0
        C1._linearize()
        assert_rel_error(self, C1._jacobian[('y','x')], np.eye(3)*-2.0, 0.00001)

        C1._inputs['x'] = np.zeros(3)
        C1._linearize()
        assert_rel_error(self, C1._jacobian[('y','x')], np.eye(3)*-2.0, 0.00001)

        C1._inputs['x'] = np.array([1.5, -0.6, 2.4])
        C1._linearize()
        expect = np.zeros((3,3))
        expect[0,0] = -2.0
        expect[1,1] = 2.0
        expect[2,2] = -2.0

        assert_rel_error(self, C1._jacobian[('y','x')], expect, 0.00001)

    def test_colon_names(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp('a:y=a:x+1.+b',
                                     inits={'a:x':2.0}, b=0.5))
        prob.setup(check=False)

        self.assertTrue('a:x' in C1._inputs)
        self.assertTrue('a:y' in C1._outputs)

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['a:y'], 3.5, 0.00001)

    def test_complex_step_colons(self):
        prob = Problem(root=Group())
        C1 = prob.root.add_subsystem('C1', ExecComp(['foo:y=2.0*foo:bar:x+1.'], inits={'foo:bar:x':2.0}))

        prob.setup(check=False)

        self.assertTrue('foo:bar:x' in C1._inputs)
        self.assertTrue('foo:y' in C1._outputs)

        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, C1._outputs['foo:y'], 5.0, 0.00001)

        C1._linearize()

        assert_rel_error(self, C1._jacobian[('foo:y','foo:bar:x')], -2.0, 0.00001)

    def test_complex_step2_colons(self):
        prob = Problem(Group())
        comp = prob.root.add_subsystem('comp', ExecComp('foo:y=foo:x*foo:x + foo:x*2.0'))
        prob.root.add_subsystem('p1', IndepVarComp('x', 2.0))
        prob.root.connect('p1.x', 'comp.foo:x')

        prob.setup(check=False)
        prob.root.suppress_solver_output = True
        prob.run()

        raise unittest.SkipTest("problem missing calc_gradient")
        J = prob.calc_gradient(['p1.x'], ['comp.foo:y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp.foo:y']['p1.x'], np.array([6.0]), 0.00001)

        J = prob.calc_gradient(['p1.x'], ['comp.foo:y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['comp.foo:y']['p1.x'], np.array([6.0]), 0.00001)

    def test_simple_array_model2_colons(self):
        prob = Problem()
        prob.root = Group()
        comp = prob.root.add_subsystem('comp', ExecComp('foo:y = foo:mat.dot(x)',
                                              inits={'foo:y':np.zeros((2,)),
                                                     'foo:mat':np.array([[2.,7.],[5.,-3.]])},
                                              x=np.zeros((2,))))

        p1 = prob.root.add_subsystem('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.root.suppress_solver_output = True
        prob.run()

        raise unittest.SkipTest("no check_partial_derivatives function")
        data = prob.check_partial_derivatives(out_stream=None)

        assert_rel_error(self, data['comp'][('foo:y','x')]['abs error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['abs error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['abs error'][2], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['rel error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['rel error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['rel error'][2], 0.0, 1e-5)

if __name__ == "__main__":
    unittest.main()
