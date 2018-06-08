""" Testing for group finite differencing."""

import unittest
import itertools
from parameterized import parameterized

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ScipyKrylov, ExecComp, NewtonSolver, \
     ExplicitComponent, DefaultVector, NonlinearBlockGS, LinearRunOnce, ParallelGroup
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.mpi import MPI
from openmdao.test_suite.components.impl_comp_array import TestImplCompArray, TestImplCompArrayDense
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.test_suite.components.sellar_feature import SellarNoDerivativesCS
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.components.unit_conv import SrcComp, TgtCompC, TgtCompF, TgtCompK
from openmdao.test_suite.groups.parallel_groups import FanInSubbedIDVC

try:
    from openmdao.parallel_api import PETScVector
    vector_class = PETScVector
except ImportError:
    from openmdao.api import DefaultVector
    vector_class = DefaultVector
    PETScVector = None


class TestGroupFiniteDifference(unittest.TestCase):

    def test_paraboloid(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = ScipyKrylov()
        model.approx_totals()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

        # 1 output x 2 inputs
        self.assertEqual(len(model._approx_schemes['fd']._exec_list), 2)

    def test_fd_count(self):
        # Make sure we aren't doing extra FD steps.

        class ParaboloidA(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=0.0)
                self.add_input('y', val=0.0)

                self.add_output('f_xy', val=0.0)
                self.add_output('g_xy', val=0.0)

                # makes extra calls to the model with no actual steps
                self.declare_partials(of='*', wrt='*', method='fd', form='forward', step=1e-6)

                self.count = 0

            def compute(self, inputs, outputs):
                x = inputs['x']
                y = inputs['y']

                outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
                g_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
                outputs['g_xy'] = g_xy *3

                self.count += 1

        prob = Problem()
        model = prob.model
        model.add_subsystem('px', IndepVarComp('x', val=3.0))
        model.add_subsystem('py', IndepVarComp('y', val=5.0))
        model.add_subsystem('parab', ParaboloidA())

        model.connect('px.x', 'parab.x')
        model.connect('py.y', 'parab.y')

        model.add_design_var('px.x', lower=-50, upper=50)
        model.add_design_var('py.y', lower=-50, upper=50)
        model.add_objective('parab.f_xy')

        prob.setup(check=False)
        prob.run_model()
        print(prob.compute_totals(of=['parab.f_xy'], wrt=['px.x', 'py.y']))

        # 1. run_model; 2. step x; 3. step y
        self.assertEqual(model.parab.count, 3)

    def test_paraboloid_subbed(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        sub = model.add_subsystem('sub', Group(), promotes=['x', 'y', 'f_xy'])
        sub.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = ScipyKrylov()
        sub.approx_totals()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

        Jfd = sub._jacobian
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.comp.x'], [[-6.0]], 1e-6)
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.comp.y'], [[8.0]], 1e-6)

        # 1 output x 2 inputs
        self.assertEqual(len(sub._approx_schemes['fd']._exec_list), 2)

    def test_paraboloid_subbed_in_setup(self):
        class MyModel(Group):

            def setup(self):
                self.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

                self.approx_totals()

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        sub = model.add_subsystem('sub', MyModel(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = ScipyKrylov()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

        Jfd = sub._jacobian
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.comp.x'], [[-6.0]], 1e-6)
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.comp.y'], [[8.0]], 1e-6)

        # 1 output x 2 inputs
        self.assertEqual(len(sub._approx_schemes['fd']._exec_list), 2)

    def test_paraboloid_subbed_with_connections(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0))
        model.add_subsystem('p2', IndepVarComp('y', 0.0))
        sub = model.add_subsystem('sub', Group())
        sub.add_subsystem('bx', ExecComp('xout = xin'))
        sub.add_subsystem('by', ExecComp('yout = yin'))
        sub.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'sub.bx.xin')
        model.connect('sub.bx.xout', 'sub.comp.x')
        model.connect('p2.y', 'sub.by.yin')
        model.connect('sub.by.yout', 'sub.comp.y')

        model.linear_solver = ScipyKrylov()
        sub.approx_totals()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['sub.comp.f_xy']
        wrt = ['p1.x', 'p2.y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['sub.comp.f_xy', 'p1.x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['sub.comp.f_xy', 'p2.y'], [[8.0]], 1e-6)

        Jfd = sub._jacobian
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.bx.xin'], [[-6.0]], 1e-6)
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.by.yin'], [[8.0]], 1e-6)

        # 3 outputs x 2 inputs
        self.assertEqual(len(sub._approx_schemes['fd']._exec_list), 6)

    def test_arrray_comp(self):

        class DoubleArrayFD(DoubleArrayComp):

            def compute_partials(self, inputs, partials):
                """
                Override deriv calculation.
                """
                pass

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x1', val=np.ones(2)))
        model.add_subsystem('p2', IndepVarComp('x2', val=np.ones(2)))
        comp = model.add_subsystem('comp', DoubleArrayFD())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        model.linear_solver = ScipyKrylov()
        model.approx_totals()

        prob.setup(check=False)
        prob.run_model()
        model.run_linearize()

        Jfd = model._jacobian
        assert_rel_error(self, Jfd['comp.y1', 'p1.x1'], comp.JJ[0:2, 0:2], 1e-6)
        assert_rel_error(self, Jfd['comp.y1', 'p2.x2'], comp.JJ[0:2, 2:4], 1e-6)
        assert_rel_error(self, Jfd['comp.y2', 'p1.x1'], comp.JJ[2:4, 0:2], 1e-6)
        assert_rel_error(self, Jfd['comp.y2', 'p2.x2'], comp.JJ[2:4, 2:4], 1e-6)

    def test_implicit_component_fd(self):
        # Somehow this wasn't tested in the original fd tests (which are mostly feature tests.)

        class TestImplCompArrayDense(TestImplCompArray):

            def setup(self):
                super(TestImplCompArrayDense, self).setup()
                self.declare_partials('*', '*', method='fd')

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p_rhs', IndepVarComp('rhs', val=np.ones(2)))
        sub = model.add_subsystem('sub', Group())
        comp = sub.add_subsystem('comp', TestImplCompArrayDense())
        model.connect('p_rhs.rhs', 'sub.comp.rhs')

        model.linear_solver = ScipyKrylov()

        prob.setup(check=False)
        prob.run_model()
        model.run_linearize()

        Jfd = comp._jacobian
        assert_rel_error(self, Jfd['sub.comp.x', 'sub.comp.rhs'], -np.eye(2), 1e-6)
        assert_rel_error(self, Jfd['sub.comp.x', 'sub.comp.x'], comp.mtx, 1e-6)

    def test_around_newton(self):
        # For a group that is set to FD that has a Newton solver, make sure it doesn't
        # try to FD itself while solving.

        class TestImplCompArrayDenseNoSolve(TestImplCompArrayDense):
            def solve_nonlinear(self, inputs, outputs):
                """ Disable local solve."""
                pass


        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p_rhs', IndepVarComp('rhs', val=np.array([2, 4])))
        comp = model.add_subsystem('comp', TestImplCompArrayDenseNoSolve())
        model.connect('p_rhs.rhs', 'comp.rhs')

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyKrylov()
        model.approx_totals()

        prob.setup(check=False)
        prob.run_model()
        model.approx_totals()
        assert_rel_error(self, prob['comp.x'], [1.97959184, 4.02040816], 1e-5)

        model.run_linearize()

        of = ['comp.x']
        wrt = ['p_rhs.rhs']
        Jfd = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, Jfd['comp.x', 'p_rhs.rhs'], [[1.01020408, -0.01020408], [-0.01020408,  1.01020408]], 1e-5)

    def test_step_size(self):
        # Test makes sure option metadata propagates to the fd function
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = ScipyKrylov()

        # Worse step so that our answer will be off a wee bit.
        model.approx_totals(step=1e-2)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-5.99]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.01]], 1e-6)

    def test_unit_conv_group(self):

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        sub1 = prob.model.add_subsystem('sub1', Group())
        sub2 = prob.model.add_subsystem('sub2', Group())

        sub1.add_subsystem('src', SrcComp())
        sub2.add_subsystem('tgtF', TgtCompF())
        sub2.add_subsystem('tgtC', TgtCompC())
        sub2.add_subsystem('tgtK', TgtCompK())

        prob.model.connect('x1', 'sub1.src.x1')
        prob.model.connect('sub1.src.x2', 'sub2.tgtF.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtC.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtK.x2')

        sub2.approx_totals(method='fd')

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['sub1.src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtK.x3'], 373.15, 1e-6)

        wrt = ['x1']
        of = ['sub2.tgtF.x3', 'sub2.tgtC.x3', 'sub2.tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_sellar(self):
        # Basic sellar test.

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                               z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        nlbgs = prob.model.nonlinear_solver = NonlinearBlockGS()

        model.approx_totals(method='fd', step=1e-5)

        prob.setup(check=False)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_desvar_with_indices(self):
         # Just desvars on this one to cover code missed by desvar+response test.

        class ArrayComp2D(ExplicitComponent):
            """
            A fairly simple array component.
            """

            def setup(self):

                self.JJ = np.array([[1.0, 3.0, -2.0, 7.0],
                                    [6.0, 2.5, 2.0, 4.0],
                                    [-1.0, 0.0, 8.0, 1.0],
                                    [1.0, 4.0, -5.0, 6.0]])

                # Params
                self.add_input('x1', np.zeros([4]))

                # Unknowns
                self.add_output('y1', np.zeros([4]))

                # Derivatives
                self.declare_partials('*', '*')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.JJ.dot(inputs['x1'])

            def compute_partials(self, inputs, partials):
                """
                Analytical derivatives.
                """
                partials[('y1', 'x1')] = self.JJ

        prob = Problem()
        prob.model = model = Group()
        model.add_subsystem('x_param1', IndepVarComp('x1', np.ones((4))),
                            promotes=['x1'])
        mycomp = model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_constraint('y1')

        prob.set_solver_print(level=0)
        model.approx_totals(method='fd')

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = mycomp.JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['y1', 'x1'][0][0], Jbase[0, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][0][1], Jbase[0, 3], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][2][0], Jbase[2, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][2][1], Jbase[2, 3], 1e-8)

    def test_desvar_and_response_with_indices(self):

        class ArrayComp2D(ExplicitComponent):
            """
            A fairly simple array component.
            """

            def setup(self):

                self.JJ = np.array([[1.0, 3.0, -2.0, 7.0],
                                    [6.0, 2.5, 2.0, 4.0],
                                    [-1.0, 0.0, 8.0, 1.0],
                                    [1.0, 4.0, -5.0, 6.0]])

                # Params
                self.add_input('x1', np.zeros([4]))

                # Unknowns
                self.add_output('y1', np.zeros([4]))

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.JJ.dot(inputs['x1'])

            def compute_partials(self, inputs, partials):
                """
                Analytical derivatives.
                """
                partials[('y1', 'x1')] = self.JJ

        prob = Problem()
        prob.model = model = Group()
        model.add_subsystem('x_param1', IndepVarComp('x1', np.ones((4))),
                            promotes=['x1'])
        mycomp = model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_constraint('y1', indices=[0, 2])

        prob.set_solver_print(level=0)
        model.approx_totals(method='fd')

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = mycomp.JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['y1', 'x1'][0][0], Jbase[0, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][0][1], Jbase[0, 3], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][1][0], Jbase[2, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][1][1], Jbase[2, 3], 1e-8)

    def test_full_model_fd(self):

        class DontCall(LinearRunOnce):
            def solve(self, vec_names, mode, rel_systems=None):
                raise RuntimeError("This solver should be ignored!")


        class Simple(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=0.0)
                self.add_output('y', val=0.0)

                self.declare_partials('y', 'x')

            def compute(self, inputs, outputs):
                x = inputs['x']
                outputs['y'] = 4.0*x


        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('comp', Simple(), promotes=['x', 'y'])

        model.linear_solver = DontCall()
        model.approx_totals()

        model.add_design_var('x')
        model.add_objective('y')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['comp.y']
        wrt = ['p1.x']
        derivs = prob.driver._compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, derivs['comp.y']['p1.x'], [[4.0]], 1e-6)

    def test_newton_with_densejac_under_full_model_fd(self):
        # Basic sellar test.

        prob = Problem()
        model = prob.model = Group(assembled_jac_type='dense')
        sub = model.add_subsystem('sub', Group(), promotes=['*'])

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                               z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = NewtonSolver()
        sub.linear_solver = ScipyKrylov(assemble_jac=True)

        model.approx_totals(method='fd', step=1e-5)

        prob.setup(check=False)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_newton_with_cscjac_under_full_model_fd(self):
        # Basic sellar test.

        prob = Problem()
        model = prob.model = Group()
        sub = model.add_subsystem('sub', Group(), promotes=['*'])

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                               z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = NewtonSolver()
        sub.linear_solver = ScipyKrylov(assemble_jac=True)

        model.approx_totals(method='fd', step=1e-5)

        prob.setup(check=False)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)


@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class TestGroupFiniteDifferenceMPI(unittest.TestCase):

    N_PROCS = 2

    def test_indepvarcomp_under_par_sys(self):
        prob = Problem()
        prob.model = FanInSubbedIDVC()

        prob.setup(local_vector_class=vector_class, check=False, mode='rev')
        prob.model.approx_totals()
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.compute_totals(wrt=['sub.sub1.p1.x', 'sub.sub2.p2.x'], of=['sum.y'])
        assert_rel_error(self, J['sum.y', 'sub.sub1.p1.x'], [[2.0]], 1.0e-6)
        assert_rel_error(self, J['sum.y', 'sub.sub2.p2.x'], [[4.0]], 1.0e-6)


def title(txt):
    """ Provide nice title for parameterized testing."""
    return str(txt).split('.')[-1].replace("'", '').replace('>', '')


class TestGroupComplexStep(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()

    def tearDown(self):
        # Global stuff seems to not get cleaned up if test fails.
        try:
            self.prob.model._outputs._vector_info._under_complex_step = False
        except:
            pass

    @parameterized.expand(itertools.product(
        [DefaultVector, PETScVector],
        ), testcase_func_name=lambda f, n, p: 'test_paraboloid_'+'_'.join(title(a) for a in p.args)
    )
    def test_paraboloid(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = ScipyKrylov()
        model.approx_totals(method='cs')

        prob.setup(check=False, mode='fwd', local_vector_class=vec_class)
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

        # 1 output x 2 inputs
        self.assertEqual(len(model._approx_schemes['cs']._exec_list), 2)

    @parameterized.expand(itertools.product(
        [DefaultVector, PETScVector],
        ), testcase_func_name=lambda f, n, p: 'test_paraboloid_subbed_'+'_'.join(title(a) for a in p.args)
    )
    def test_paraboloid_subbed(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        sub = model.add_subsystem('sub', Group(), promotes=['x', 'y', 'f_xy'])
        sub.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = ScipyKrylov()
        sub.approx_totals(method='cs')

        prob.setup(check=False, mode='fwd', local_vector_class=vec_class)
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

        Jfd = sub._jacobian
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.comp.x'], [[-6.0]], 1e-6)
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.comp.y'], [[8.0]], 1e-6)

        # 1 output x 2 inputs
        self.assertEqual(len(sub._approx_schemes['cs']._exec_list), 2)

    @parameterized.expand(itertools.product(
        [DefaultVector, PETScVector],
        ), testcase_func_name=lambda f, n, p: 'test_paraboloid_subbed_with_connections_'+'_'.join(title(a) for a in p.args)
    )
    def test_paraboloid_subbed_with_connections(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0))
        model.add_subsystem('p2', IndepVarComp('y', 0.0))
        sub = model.add_subsystem('sub', Group())
        sub.add_subsystem('bx', ExecComp('xout = xin'))
        sub.add_subsystem('by', ExecComp('yout = yin'))
        sub.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'sub.bx.xin')
        model.connect('sub.bx.xout', 'sub.comp.x')
        model.connect('p2.y', 'sub.by.yin')
        model.connect('sub.by.yout', 'sub.comp.y')

        model.linear_solver = ScipyKrylov()
        sub.approx_totals(method='cs')

        prob.setup(check=False, mode='fwd', local_vector_class=vec_class)
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['sub.comp.f_xy']
        wrt = ['p1.x', 'p2.y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['sub.comp.f_xy', 'p1.x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['sub.comp.f_xy', 'p2.y'], [[8.0]], 1e-6)

        Jfd = sub._jacobian
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.bx.xin'], [[-6.0]], 1e-6)
        assert_rel_error(self, Jfd['sub.comp.f_xy', 'sub.by.yin'], [[8.0]], 1e-6)

        # 3 outputs x 2 inputs
        self.assertEqual(len(sub._approx_schemes['cs']._exec_list), 6)

    @parameterized.expand(itertools.product(
        [DefaultVector, PETScVector],
        ), testcase_func_name=lambda f, n, p: 'test_arrray_comp_'+'_'.join(title(a) for a in p.args)
    )
    def test_arrray_comp(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        class DoubleArrayFD(DoubleArrayComp):

            def compute_partials(self, inputs, partials):
                """
                Override deriv calculation.
                """
                pass

        prob = self.prob
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x1', val=np.ones(2)))
        model.add_subsystem('p2', IndepVarComp('x2', val=np.ones(2)))
        comp = model.add_subsystem('comp', DoubleArrayFD())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        model.linear_solver = ScipyKrylov()
        model.approx_totals(method='cs')

        prob.setup(check=False, local_vector_class=vec_class)
        prob.run_model()
        model.run_linearize()

        Jfd = model._jacobian
        assert_rel_error(self, Jfd['comp.y1', 'p1.x1'], comp.JJ[0:2, 0:2], 1e-6)
        assert_rel_error(self, Jfd['comp.y1', 'p2.x2'], comp.JJ[0:2, 2:4], 1e-6)
        assert_rel_error(self, Jfd['comp.y2', 'p1.x1'], comp.JJ[2:4, 0:2], 1e-6)
        assert_rel_error(self, Jfd['comp.y2', 'p2.x2'], comp.JJ[2:4, 2:4], 1e-6)

    @parameterized.expand(itertools.product(
        [DefaultVector, PETScVector],
        ), testcase_func_name=lambda f, n, p: 'test_unit_conv_group_'+'_'.join(title(a) for a in p.args)
    )
    def test_unit_conv_group(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        prob.model = Group()
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        sub1 = prob.model.add_subsystem('sub1', Group())
        sub2 = prob.model.add_subsystem('sub2', Group())

        sub1.add_subsystem('src', SrcComp())
        sub2.add_subsystem('tgtF', TgtCompF())
        sub2.add_subsystem('tgtC', TgtCompC())
        sub2.add_subsystem('tgtK', TgtCompK())

        prob.model.connect('x1', 'sub1.src.x1')
        prob.model.connect('sub1.src.x2', 'sub2.tgtF.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtC.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtK.x2')

        sub2.approx_totals(method='cs')

        prob.setup(check=False, local_vector_class=vec_class)
        prob.run_model()

        assert_rel_error(self, prob['sub1.src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtK.x3'], 373.15, 1e-6)

        wrt = ['x1']
        of = ['sub2.tgtF.x3', 'sub2.tgtC.x3', 'sub2.tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev', local_vector_class=vec_class)
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    @parameterized.expand(itertools.product(
        [DefaultVector, PETScVector],
        ), testcase_func_name=lambda f, n, p: 'test_sellar_'+'_'.join(title(a) for a in p.args)
    )
    def test_sellar(self, vec_class):
        # Basic sellar test.

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                               z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        nlbgs = prob.model.nonlinear_solver = NonlinearBlockGS()

        # Had to make this step larger so that solver would reconverge adequately.
        model.approx_totals(method='cs', step=1.0e-1)

        prob.setup(check=False, local_vector_class=vec_class)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_desvar_and_response_with_indices(self):

        class ArrayComp2D(ExplicitComponent):
            """
            A fairly simple array component.
            """

            def setup(self):

                self.JJ = np.array([[1.0, 3.0, -2.0, 7.0],
                                    [6.0, 2.5, 2.0, 4.0],
                                    [-1.0, 0.0, 8.0, 1.0],
                                    [1.0, 4.0, -5.0, 6.0]])

                # Params
                self.add_input('x1', np.zeros([4]))

                # Unknowns
                self.add_output('y1', np.zeros([4]))

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.JJ.dot(inputs['x1'])

            def compute_partials(self, inputs, partials):
                """
                Analytical derivatives.
                """
                partials[('y1', 'x1')] = self.JJ

        prob = Problem()
        prob.model = model = Group()
        model.add_subsystem('x_param1', IndepVarComp('x1', np.ones((4))),
                            promotes=['x1'])
        mycomp = model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_constraint('y1', indices=[0, 2])

        prob.set_solver_print(level=0)
        model.approx_totals(method='cs')

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = mycomp.JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['y1', 'x1'][0][0], Jbase[0, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][0][1], Jbase[0, 3], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][1][0], Jbase[2, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][1][1], Jbase[2, 3], 1e-8)

    def test_desvar_with_indices(self):
        # Just desvars on this one to cover code missed by desvar+response test.

        class ArrayComp2D(ExplicitComponent):
            """
            A fairly simple array component.
            """

            def setup(self):

                self.JJ = np.array([[1.0, 3.0, -2.0, 7.0],
                                    [6.0, 2.5, 2.0, 4.0],
                                    [-1.0, 0.0, 8.0, 1.0],
                                    [1.0, 4.0, -5.0, 6.0]])

                # Params
                self.add_input('x1', np.zeros([4]))

                # Unknowns
                self.add_output('y1', np.zeros([4]))

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.JJ.dot(inputs['x1'])

            def compute_partials(self, inputs, partials):
                """
                Analytical derivatives.
                """
                partials[('y1', 'x1')] = self.JJ

        prob = Problem()
        prob.model = model = Group()
        model.add_subsystem('x_param1', IndepVarComp('x1', np.ones((4))),
                            promotes=['x1'])
        mycomp = model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_constraint('y1')

        prob.set_solver_print(level=0)
        model.approx_totals(method='cs')

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = mycomp.JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['y1', 'x1'][0][0], Jbase[0, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][0][1], Jbase[0, 3], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][2][0], Jbase[2, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][2][1], Jbase[2, 3], 1e-8)


class TestComponentComplexStep(unittest.TestCase):

    def tearDown(self):
        # Global stuff seems to not get cleaned up if test fails.
        self.prob.model._outputs._vector_info._under_complex_step = False

    def test_implicit_component(self):

        class TestImplCompArrayDense(TestImplCompArray):

            def setup(self):
                super(TestImplCompArrayDense, self).setup()
                self.declare_partials('*', '*', method='cs')

        prob = self.prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p_rhs', IndepVarComp('rhs', val=np.ones(2)))
        sub = model.add_subsystem('sub', Group())
        comp = sub.add_subsystem('comp', TestImplCompArrayDense())
        model.connect('p_rhs.rhs', 'sub.comp.rhs')

        model.linear_solver = ScipyKrylov()

        prob.setup(check=False)
        prob.run_model()
        model.run_linearize()

        Jfd = comp._jacobian
        assert_rel_error(self, Jfd['sub.comp.x', 'sub.comp.rhs'], -np.eye(2), 1e-6)
        assert_rel_error(self, Jfd['sub.comp.x', 'sub.comp.x'], comp.mtx, 1e-6)

    def test_reconfigure(self):
        # In this test, we switch to 'cs' when we reconfigure.

        class TestImplCompArrayDense(TestImplCompArray):

            def initialize(self):
                self.mtx = np.array([
                    [0.99, 0.01],
                    [0.01, 0.99],
                ])
                self.count = 0

            def setup(self):
                super(TestImplCompArrayDense, self).setup()
                if self.count > 0:
                    self.declare_partials('*', '*', method='cs')
                else:
                    self.declare_partials('*', '*', method='fd')
                self.count += 1

        prob = self.prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p_rhs', IndepVarComp('rhs', val=np.ones(2)))
        sub = model.add_subsystem('sub', Group())
        comp = sub.add_subsystem('comp', TestImplCompArrayDense())
        model.connect('p_rhs.rhs', 'sub.comp.rhs')

        model.linear_solver = ScipyKrylov()

        prob.setup(check=False)
        prob.run_model()

        with self.assertRaises(RuntimeError) as context:
            model.resetup(setup_mode='reconf')

        msg = 'In order to activate complex step during reconfiguration, you need to set ' + \
            '"force_alloc_complex" to True during setup.'
        self.assertEqual(str(context.exception), msg)

        # This time, allocate complex in setup.
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()
        model.resetup(setup_mode='reconf')
        prob.run_model()

        model.run_linearize()
        Jfd = comp._jacobian
        assert_rel_error(self, Jfd['sub.comp.x', 'sub.comp.rhs'], -np.eye(2), 1e-6)
        assert_rel_error(self, Jfd['sub.comp.x', 'sub.comp.x'], comp.mtx, 1e-6)

    def test_vector_methods(self):

        class KenComp(ExplicitComponent):

            def setup(self):

                self.add_input('x1', np.array([[7.0, 3.0], [2.4, 3.33]]))
                self.add_output('y1', np.zeros((2, 2)))

                self.declare_partials('*', '*', method='cs')

            def compute(self, inputs, outputs):

                x1 = inputs['x1']

                outputs['y1'] = x1

                outputs['y1'][0][0] += 14.0
                outputs['y1'][0][1] *= 3.0
                outputs['y1'][1][0] -= 6.67
                outputs['y1'][1][1] /= 2.34

                pass #outputs['y1'] *= 1.0


        prob = self.prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', val=np.array([[7.0, 3.0], [2.4, 3.33]])))
        model.add_subsystem('comp', KenComp())
        model.connect('px.x', 'comp.x1')

        prob.setup(check=False)
        prob.run_model()

        of = ['comp.y1']
        wrt = ['px.x']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['comp.y1', 'px.x'][0][0], 1.0, 1e-6)
        assert_rel_error(self, derivs['comp.y1', 'px.x'][1][1], 3.0, 1e-6)
        assert_rel_error(self, derivs['comp.y1', 'px.x'][2][2], 1.0, 1e-6)
        assert_rel_error(self, derivs['comp.y1', 'px.x'][3][3], 1.0/2.34, 1e-6)


class ApproxTotalsFeature(unittest.TestCase):

    def test_basic(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ScipyKrylov, ExplicitComponent

        class CompOne(ExplicitComponent):

                def setup(self):
                    self.add_input('x', val=0.0)
                    self.add_output('y', val=np.zeros(25))
                    self._exec_count = 0

                def compute(self, inputs, outputs):
                    x = inputs['x']
                    outputs['y'] = np.arange(25) * x
                    self._exec_count += 1


        class CompTwo(ExplicitComponent):

                def setup(self):
                    self.add_input('y', val=np.zeros(25))
                    self.add_output('z', val=0.0)
                    self._exec_count = 0

                def compute(self, inputs, outputs):
                    y = inputs['y']
                    outputs['z'] = np.sum(y)
                    self._exec_count += 1


        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('comp1', CompOne(), promotes=['x', 'y'])
        comp2 = model.add_subsystem('comp2', CompTwo(), promotes=['y', 'z'])

        model.linear_solver = ScipyKrylov()
        model.approx_totals()

        prob.setup()
        prob.run_model()

        of = ['z']
        wrt = ['x']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['z', 'x'], [[300.0]], 1e-6)
        self.assertEqual(comp2._exec_count, 2)

    def test_basic_cs(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ScipyKrylov, ExplicitComponent

        class CompOne(ExplicitComponent):

                def setup(self):
                    self.add_input('x', val=0.0)
                    self.add_output('y', val=np.zeros(25))
                    self._exec_count = 0

                def compute(self, inputs, outputs):
                    x = inputs['x']
                    outputs['y'] = np.arange(25) * x
                    self._exec_count += 1


        class CompTwo(ExplicitComponent):

                def setup(self):
                    self.add_input('y', val=np.zeros(25))
                    self.add_output('z', val=0.0)
                    self._exec_count = 0

                def compute(self, inputs, outputs):
                    y = inputs['y']
                    outputs['z'] = np.sum(y)
                    self._exec_count += 1


        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('comp1', CompOne(), promotes=['x', 'y'])
        comp2 = model.add_subsystem('comp2', CompTwo(), promotes=['y', 'z'])

        model.linear_solver = ScipyKrylov()
        model.approx_totals(method='cs')

        prob.setup()
        prob.run_model()

        of = ['z']
        wrt = ['x']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['z', 'x'], [[300.0]], 1e-6)

    def test_arguments(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, ScipyKrylov, ExplicitComponent

        class CompOne(ExplicitComponent):

                def setup(self):
                    self.add_input('x', val=0.0)
                    self.add_output('y', val=np.zeros(25))
                    self._exec_count = 0

                def compute(self, inputs, outputs):
                    x = inputs['x']
                    outputs['y'] = np.arange(25) * x
                    self._exec_count += 1


        class CompTwo(ExplicitComponent):

                def setup(self):
                    self.add_input('y', val=np.zeros(25))
                    self.add_output('z', val=0.0)
                    self._exec_count = 0

                def compute(self, inputs, outputs):
                    y = inputs['y']
                    outputs['z'] = np.sum(y)
                    self._exec_count += 1


        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('comp1', CompOne(), promotes=['x', 'y'])
        comp2 = model.add_subsystem('comp2', CompTwo(), promotes=['y', 'z'])

        model.linear_solver = ScipyKrylov()
        model.approx_totals(method='fd', step=1e-7, form='central', step_calc='rel')

        prob.setup()
        prob.run_model()

        of = ['z']
        wrt = ['x']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['z', 'x'], [[300.0]], 1e-6)

    def test_sellarCS(self):
        # Just tests Newton on Sellar with FD derivs.
        import numpy as np

        from openmdao.api import Problem
        from openmdao.test_suite.components.sellar_feature import SellarNoDerivativesCS

        prob = Problem()
        prob.model = SellarNoDerivativesCS()

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)

if __name__ == "__main__":
    unittest.main()
