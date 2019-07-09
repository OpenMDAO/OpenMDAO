""" Testing for group finite differencing."""
import itertools
import unittest
from six import iterkeys
from six.moves import range

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.impl_comp_array import TestImplCompArray, TestImplCompArrayDense
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
    SellarDis2withDerivatives, SellarDis1CS, SellarDis2CS
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.components.unit_conv import SrcComp, TgtCompC, TgtCompF, TgtCompK
from openmdao.test_suite.groups.parallel_groups import FanInSubbedIDVC
from openmdao.test_suite.parametric_suite import parametric_suite
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.mpi import MPI

try:
    from openmdao.parallel_api import PETScVector
    vector_class = PETScVector
except ImportError:
    vector_class = om.DefaultVector
    PETScVector = None

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class TestGroupFiniteDifference(unittest.TestCase):

    def test_paraboloid(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = om.ScipyKrylov()
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
        self.assertEqual(len(model._approx_schemes['fd']._exec_dict), 2)

    def test_fd_count(self):
        # Make sure we aren't doing extra FD steps.

        class ParaboloidA(om.ExplicitComponent):
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
                outputs['g_xy'] = g_xy * 3

                self.count += 1

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('px', om.IndepVarComp('x', val=3.0))
        model.add_subsystem('py', om.IndepVarComp('y', val=5.0))
        model.add_subsystem('parab', ParaboloidA())

        model.connect('px.x', 'parab.x')
        model.connect('py.y', 'parab.y')

        model.add_design_var('px.x', lower=-50, upper=50)
        model.add_design_var('py.y', lower=-50, upper=50)
        model.add_objective('parab.f_xy')

        prob.setup()
        prob.run_model()
        J = prob.compute_totals(of=['parab.f_xy'], wrt=['px.x', 'py.y'])
        # print(J)

        # 1. run_model; 2. step x; 3. step y
        self.assertEqual(model.parab.count, 3)

    def test_fd_count_driver(self):
        # Make sure we aren't doing FD wrt any var that isn't in the driver desvar set.

        class ParaboloidA(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=0.0)
                self.add_input('y', val=0.0)

                self.add_output('f_xy', val=0.0)
                self.add_output('g_xy', val=0.0)

                self.count = 0

            def compute(self, inputs, outputs):
                x = inputs['x']
                y = inputs['y']

                outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
                g_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
                outputs['g_xy'] = g_xy * 3

                self.count += 1

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('px', om.IndepVarComp('x', val=3.0))
        model.add_subsystem('py', om.IndepVarComp('y', val=5.0))
        model.add_subsystem('parab', ParaboloidA())

        model.connect('px.x', 'parab.x')
        model.connect('py.y', 'parab.y')

        model.add_design_var('px.x', lower=-50, upper=50)
        model.add_objective('parab.f_xy')

        model.approx_totals(method='fd')

        prob.setup()
        prob.run_model()

        prob.driver._compute_totals(of=['parab.f_xy'], wrt=['px.x'], global_names=True)

        # 1. run_model; 2. step x
        self.assertEqual(model.parab.count, 2)

    def test_paraboloid_subbed(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        sub = model.add_subsystem('sub', om.Group(), promotes=['x', 'y', 'f_xy'])
        sub.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = om.ScipyKrylov()
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
        self.assertEqual(len(sub._approx_schemes['fd']._exec_dict), 2)

    def test_paraboloid_subbed_in_setup(self):
        class MyModel(om.Group):

            def setup(self):
                self.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

                self.approx_totals()

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        sub = model.add_subsystem('sub', MyModel(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = om.ScipyKrylov()

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
        self.assertEqual(len(sub._approx_schemes['fd']._exec_dict), 2)

    def test_paraboloid_subbed_with_connections(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0))
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0))
        sub = model.add_subsystem('sub', om.Group())
        sub.add_subsystem('bx', om.ExecComp('xout = xin'))
        sub.add_subsystem('by', om.ExecComp('yout = yin'))
        sub.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'sub.bx.xin')
        model.connect('sub.bx.xout', 'sub.comp.x')
        model.connect('p2.y', 'sub.by.yin')
        model.connect('sub.by.yout', 'sub.comp.y')

        model.linear_solver = om.ScipyKrylov()
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
        n_entries = 0
        for k, v in sub._approx_schemes['fd']._exec_dict.items():
            n_entries += len(v)
        self.assertEqual(n_entries, 6)

    def test_array_comp(self):

        class DoubleArrayFD(DoubleArrayComp):

            def compute_partials(self, inputs, partials):
                """
                Override deriv calculation.
                """
                pass

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x1', val=np.ones(2)))
        model.add_subsystem('p2', om.IndepVarComp('x2', val=np.ones(2)))
        comp = model.add_subsystem('comp', DoubleArrayFD())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        model.linear_solver = om.ScipyKrylov()
        model.approx_totals()

        prob.setup()
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

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p_rhs', om.IndepVarComp('rhs', val=np.ones(2)))
        sub = model.add_subsystem('sub', om.Group())
        comp = sub.add_subsystem('comp', TestImplCompArrayDense())
        model.connect('p_rhs.rhs', 'sub.comp.rhs')

        model.linear_solver = om.ScipyKrylov()

        prob.setup()
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

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p_rhs', om.IndepVarComp('rhs', val=np.array([2, 4])))
        model.add_subsystem('comp', TestImplCompArrayDenseNoSolve())
        model.connect('p_rhs.rhs', 'comp.rhs')

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()
        model.approx_totals()

        prob.setup()
        prob.run_model()
        model.approx_totals()
        assert_rel_error(self, prob['comp.x'], [1.97959184, 4.02040816], 1e-5)

        model.run_linearize()

        of = ['comp.x']
        wrt = ['p_rhs.rhs']
        Jfd = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, Jfd['comp.x', 'p_rhs.rhs'],
                         [[1.01020408, -0.01020408], [-0.01020408, 1.01020408]], 1e-5)

    def test_step_size(self):
        # Test makes sure option metadata propagates to the fd function
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = om.ScipyKrylov()

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

        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0), promotes=['x1'])
        sub1 = prob.model.add_subsystem('sub1', om.Group())
        sub2 = prob.model.add_subsystem('sub2', om.Group())

        sub1.add_subsystem('src', SrcComp())
        sub2.add_subsystem('tgtF', TgtCompF())
        sub2.add_subsystem('tgtC', TgtCompC())
        sub2.add_subsystem('tgtK', TgtCompK())

        prob.model.connect('x1', 'sub1.src.x1')
        prob.model.connect('sub1.src.x2', 'sub2.tgtF.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtC.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtK.x2')

        sub2.approx_totals(method='fd')

        prob.setup()
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

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        model.approx_totals(method='fd', step=1e-5)

        prob.setup()
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

        class ArrayComp2D(om.ExplicitComponent):
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

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param1', om.IndepVarComp('x1', np.ones((4))),
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

        class ArrayComp2D(om.ExplicitComponent):
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

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param1', om.IndepVarComp('x1', np.ones((4))),
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

        class DontCall(om.LinearRunOnce):
            def solve(self, vec_names, mode, rel_systems=None):
                raise RuntimeError("This solver should be ignored!")

        class Simple(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=0.0)
                self.add_output('y', val=0.0)

                self.declare_partials('y', 'x')

            def compute(self, inputs, outputs):
                x = inputs['x']
                outputs['y'] = 4.0*x

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
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

        prob = om.Problem()
        model = prob.model = om.Group(assembled_jac_type='dense')
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver()
        sub.linear_solver = om.ScipyKrylov(assemble_jac=True)

        model.approx_totals(method='fd', step=1e-5)

        prob.setup()
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

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver()
        sub.linear_solver = om.ScipyKrylov(assemble_jac=True)

        model.approx_totals(method='fd', step=1e-5)

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_approx_totals_multi_input_constrained_desvar(self):
        p = om.Problem()

        indeps = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['*'])

        indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                          0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                         -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        indeps.add_output('r', .7)

        arctan_yox = om.ExecComp('g=arctan(y/x)', vectorize=True,
                                 g=np.ones(10), x=np.ones(10), y=np.ones(10))

        p.model.add_subsystem('arctan_yox', arctan_yox)

        p.model.add_subsystem('circle', om.ExecComp('area=pi*r**2'))

        p.model.add_subsystem('r_con', om.ExecComp('g=x**2 + y**2 - r', vectorize=True,
                                                   g=np.ones(10), x=np.ones(10), y=np.ones(10)))

        p.model.connect('r', ('circle.r', 'r_con.r'))
        p.model.connect('x', ['r_con.x', 'arctan_yox.x'])
        p.model.connect('y', ['r_con.y', 'arctan_yox.y'])

        p.model.approx_totals(method='cs')

        p.model.add_design_var('x')
        p.model.add_design_var('y')
        p.model.add_design_var('r', lower=.5, upper=10)
        p.model.add_constraint('y', equals=0, indices=[0,])
        p.model.add_objective('circle.area', ref=-1)

        p.setup(derivatives=True)

        p.run_model()
        # Formerly a KeyError
        derivs = p.check_totals(compact_print=True, out_stream=None)
        assert_rel_error(self, 0.0, derivs['indeps.y', 'indeps.x']['abs error'][0])

        # Coverage
        derivs = p.driver._compute_totals(return_format='dict')
        assert_rel_error(self, np.zeros((1, 10)), derivs['indeps.y']['indeps.x'])

    def test_opt_with_linear_constraint(self):
        # Test for a bug where we weren't re-initializing things in-between computing totals on
        # linear constraints, and the nonlinear ones.
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        p = om.Problem()

        indeps = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['*'])

        indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                          0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                         -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        indeps.add_output('r', .7)

        arctan_yox = om.ExecComp('g=arctan(y/x)', vectorize=True,
                                 g=np.ones(10), x=np.ones(10), y=np.ones(10))

        p.model.add_subsystem('arctan_yox', arctan_yox)

        p.model.add_subsystem('circle', om.ExecComp('area=pi*r**2'))

        p.model.add_subsystem('r_con', om.ExecComp('g=x**2 + y**2 - r', vectorize=True,
                                                   g=np.ones(10), x=np.ones(10), y=np.ones(10)))

        thetas = np.linspace(0, np.pi/4, 10)
        p.model.add_subsystem('theta_con', om.ExecComp('g = x - theta', vectorize=True,
                                                       g=np.ones(10), x=np.ones(10),
                                                       theta=thetas))
        p.model.add_subsystem('delta_theta_con', om.ExecComp('g = even - odd', vectorize=True,
                                                             g=np.ones(10//2), even=np.ones(10//2),
                                                             odd=np.ones(10//2)))

        p.model.add_subsystem('l_conx', om.ExecComp('g=x-1', vectorize=True, g=np.ones(10), x=np.ones(10)))

        IND = np.arange(10, dtype=int)
        ODD_IND = IND[1::2]  # all odd indices
        EVEN_IND = IND[0::2]  # all even indices

        p.model.connect('r', ('circle.r', 'r_con.r'))
        p.model.connect('x', ['r_con.x', 'arctan_yox.x', 'l_conx.x'])
        p.model.connect('y', ['r_con.y', 'arctan_yox.y'])
        p.model.connect('arctan_yox.g', 'theta_con.x')
        p.model.connect('arctan_yox.g', 'delta_theta_con.even', src_indices=EVEN_IND)
        p.model.connect('arctan_yox.g', 'delta_theta_con.odd', src_indices=ODD_IND)

        p.driver = pyOptSparseDriver()
        p.driver.options['print_results'] = False
        p.model.approx_totals(method='fd')

        p.model.add_design_var('x')
        p.model.add_design_var('y')
        p.model.add_design_var('r', lower=.5, upper=10)

        # nonlinear constraints
        p.model.add_constraint('r_con.g', equals=0)

        p.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND)
        p.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)
        p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])
        p.model.add_constraint('y', equals=0, indices=[0,], linear=True)

        p.model.add_objective('circle.area', ref=-1)

        p.setup(mode='fwd', derivatives=True)

        p.run_driver()

        assert_rel_error(self, p['circle.area'], np.pi, 1e-6)


@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class TestGroupFiniteDifferenceMPI(unittest.TestCase):

    N_PROCS = 2

    def test_indepvarcomp_under_par_sys(self):
        prob = om.Problem()
        prob.model = FanInSubbedIDVC()

        prob.setup(local_vector_class=vector_class, check=False, mode='rev')
        prob.model.approx_totals()
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.compute_totals(wrt=['sub.sub1.p1.x', 'sub.sub2.p2.x'], of=['sum.y'])
        assert_rel_error(self, J['sum.y', 'sub.sub1.p1.x'], [[2.0]], 1.0e-6)
        assert_rel_error(self, J['sum.y', 'sub.sub2.p2.x'], [[4.0]], 1.0e-6)


@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class TestGroupCSMPI(unittest.TestCase):

    N_PROCS = 2

    def test_indepvarcomp_under_par_sys_par_cs(self):
        prob = om.Problem()
        prob.model = FanInSubbedIDVC(num_par_fd=2)
        prob.model.approx_totals(method='cs')

        prob.setup(local_vector_class=vector_class, check=False, mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.compute_totals(wrt=['sub.sub1.p1.x', 'sub.sub2.p2.x'], of=['sum.y'])
        assert_rel_error(self, J['sum.y', 'sub.sub1.p1.x'], [[2.0]], 1.0e-6)
        assert_rel_error(self, J['sum.y', 'sub.sub2.p2.x'], [[4.0]], 1.0e-6)

    def test_newton_with_direct_solver(self):
        # Make sure this works under mpi with bug fixed in norm calculation.

        if not PETScVector:
            raise unittest.SkipTest("PETSc is not installed")

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver()
        sub.linear_solver = om.DirectSolver(assemble_jac=False)
        sub.nonlinear_solver.options['atol'] = 1e-10
        sub.nonlinear_solver.options['rtol'] = 1e-10

        model.approx_totals(method='cs')

        prob.setup(check=False, local_vector_class=PETScVector)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, 1.0e-6)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, 1.0e-6)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)


@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class TestGroupFDMPI(unittest.TestCase):

    N_PROCS = 2

    def test_indepvarcomp_under_par_sys_par_fd(self):
        prob = om.Problem()
        prob.model = FanInSubbedIDVC(num_par_fd=2)

        prob.model.approx_totals(method='fd')
        prob.setup(local_vector_class=vector_class, check=False, mode='rev')
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

        self.prob = om.Problem()

    def tearDown(self):
        # Global stuff seems to not get cleaned up if test fails.
        try:
            self.prob.model._outputs._under_complex_step = False
        except Exception:
            pass

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_paraboloid_'+'_'.join(title(a) for a in p.args))
    def test_paraboloid(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = om.ScipyKrylov()
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
        self.assertEqual(len(model._approx_schemes['cs']._exec_dict), 2)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_paraboloid_subbed_'+'_'.join(title(a) for a in p.args))
    def test_paraboloid_subbed(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        sub = model.add_subsystem('sub', om.Group(), promotes=['x', 'y', 'f_xy'])
        sub.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = om.ScipyKrylov()
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
        self.assertEqual(len(sub._approx_schemes['cs']._exec_dict), 2)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_parab_subbed_with_connections_'+'_'.join(title(a) for a in p.args))
    def test_paraboloid_subbed_with_connections(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0))
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0))
        sub = model.add_subsystem('sub', om.Group())
        sub.add_subsystem('bx', om.ExecComp('xout = xin'))
        sub.add_subsystem('by', om.ExecComp('yout = yin'))
        sub.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'sub.bx.xin')
        model.connect('sub.bx.xout', 'sub.comp.x')
        model.connect('p2.y', 'sub.by.yin')
        model.connect('sub.by.yout', 'sub.comp.y')

        model.linear_solver = om.ScipyKrylov()
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
        n_entries = 0
        for k, v in sub._approx_schemes['cs']._exec_dict.items():
            n_entries += len(v)
        self.assertEqual(n_entries, 6)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_array_comp_'+'_'.join(title(a) for a in p.args))
    def test_array_comp(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        class DoubleArrayFD(DoubleArrayComp):

            def compute_partials(self, inputs, partials):
                """
                Override deriv calculation.
                """
                pass

        prob = self.prob
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x1', val=np.ones(2)))
        model.add_subsystem('p2', om.IndepVarComp('x2', val=np.ones(2)))
        comp = model.add_subsystem('comp', DoubleArrayFD())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        model.linear_solver = om.ScipyKrylov()
        model.approx_totals(method='cs')

        prob.setup(check=False, local_vector_class=vec_class)
        prob.run_model()
        model.run_linearize()

        Jfd = model._jacobian
        assert_rel_error(self, Jfd['comp.y1', 'p1.x1'], comp.JJ[0:2, 0:2], 1e-6)
        assert_rel_error(self, Jfd['comp.y1', 'p2.x2'], comp.JJ[0:2, 2:4], 1e-6)
        assert_rel_error(self, Jfd['comp.y2', 'p1.x1'], comp.JJ[2:4, 0:2], 1e-6)
        assert_rel_error(self, Jfd['comp.y2', 'p2.x2'], comp.JJ[2:4, 2:4], 1e-6)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_unit_conv_group_'+'_'.join(title(a) for a in p.args))
    def test_unit_conv_group(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0), promotes=['x1'])
        sub1 = prob.model.add_subsystem('sub1', om.Group())
        sub2 = prob.model.add_subsystem('sub2', om.Group())

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

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_sellar_'+'_'.join(title(a) for a in p.args))
    def test_sellar(self, vec_class):
        # Basic sellar test.

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = self.prob
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.nonlinear_solver.options['atol'] = 1e-50
        prob.model.nonlinear_solver.options['rtol'] = 1e-50

        model.approx_totals(method='cs')

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

        self.assertFalse(model._vectors['output']['linear']._alloc_complex,
                         msg="Linear vector should not be allocated as complex.")

    def test_desvar_and_response_with_indices(self):

        class ArrayComp2D(om.ExplicitComponent):
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

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param1', om.IndepVarComp('x1', np.ones((4))),
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

        class ArrayComp2D(om.ExplicitComponent):
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

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param1', om.IndepVarComp('x1', np.ones((4))),
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

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_newton_with_direct_solver_'+'_'.join(title(a) for a in p.args))
    def test_newton_with_direct_solver(self, vec_class):
        # Basic sellar test.

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver()
        sub.linear_solver = om.DirectSolver(assemble_jac=False)
        sub.nonlinear_solver.options['atol'] = 1e-10
        sub.nonlinear_solver.options['rtol'] = 1e-10

        model.approx_totals(method='cs')

        prob.setup(check=False, local_vector_class=vec_class)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, 1.0e-6)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, 1.0e-6)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_newton_with_direct_solver_dense_'+'_'.join(title(a) for a in p.args))
    def test_newton_with_direct_solver_dense(self, vec_class):
        # Basic sellar test.

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver()
        sub.linear_solver = om.DirectSolver()
        sub.options['assembled_jac_type'] = 'dense'

        sub.nonlinear_solver.options['atol'] = 1e-10
        sub.nonlinear_solver.options['rtol'] = 1e-10

        model.approx_totals(method='cs')

        prob.setup(check=False, local_vector_class=vec_class)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, 1.0e-6)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, 1.0e-6)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_newton_with_direct_solver_csc_'+'_'.join(title(a) for a in p.args))
    def test_newton_with_direct_solver_csc(self, vec_class):
        # Basic sellar test.

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver()
        sub.linear_solver = om.DirectSolver()
        sub.options['assembled_jac_type'] = 'csc'

        sub.nonlinear_solver.options['atol'] = 1e-10
        sub.nonlinear_solver.options['rtol'] = 1e-10

        model.approx_totals(method='cs')

        prob.setup(check=False, local_vector_class=vec_class)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, 1.0e-6)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, 1.0e-6)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_subbed_newton_gs_'+'_'.join(title(a) for a in p.args))
    def test_subbed_newton_gs(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives
        class SellarDerivatives(om.Group):

            def setup(self):
                self.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
                self.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

                self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
                self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
                sub = self.add_subsystem('sub', om.Group(), promotes=['*'])

                sub.linear_solver = om.DirectSolver(assemble_jac=True)
                sub.options['assembled_jac_type'] = 'csc'

                sub.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)', obj=0.0,
                                                         x=0.0, z=np.array([0.0, 0.0]), y1=0.0, y2=0.0),
                                  promotes=['obj', 'x', 'z', 'y1', 'y2'])

                sub.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1', con1=0.0, y1=0.0),
                                  promotes=['con1', 'y1'])
                sub.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0', con2=0.0, y2=0.0),
                                  promotes=['con2', 'y2'])

                self.nonlinear_solver = om.NewtonSolver()
                self.linear_solver = om.LinearBlockGS()
                self.linear_solver.options['maxiter'] = 25
                self.linear_solver.options['atol'] = 1e-16

        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        prob.model.approx_totals(method='cs')

        prob.run_model()

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, 1.0e-6)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, 1.0e-6)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_subbed_newton_gs_csc_external_mtx_'+'_'.join(title(a) for a in p.args))
    def test_subbed_newton_gs_csc_external_mtx(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives
        class SellarDerivatives(om.Group):

            def setup(self):
                self.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
                self.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

                self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
                self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
                sub = self.add_subsystem('sub', om.Group(), promotes=['*'])

                sub.linear_solver = om.DirectSolver(assemble_jac=True)
                sub.options['assembled_jac_type'] = 'csc'

                sub.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)', obj=0.0,
                                                         x=0.0, z=np.array([0.0, 0.0]), y1=0.0, y2=0.0),
                                  promotes=['obj', 'x', 'z', 'y1', 'y2'])

                sub.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1', con1=0.0, y1=0.0),
                                  promotes=['con1', 'y1'])
                sub.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0', con2=0.0, y2=0.0),
                                  promotes=['con2', 'y2'])

                self.nonlinear_solver = om.NewtonSolver()
                self.linear_solver = om.LinearBlockGS()
                self.linear_solver.options['maxiter'] = 25
                self.linear_solver.options['atol'] = 1e-16

        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        prob.model.approx_totals(method='cs')

        prob.run_model()

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, 1.0e-6)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, 1.0e-6)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_subbed_newton_gs_dense_external_mtx_'+'_'.join(title(a) for a in p.args))
    def test_subbed_newton_gs_dense_external_mtx(self, vec_class):

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives
        class SellarDerivatives(om.Group):

            def setup(self):
                self.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
                self.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

                self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
                self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
                sub = self.add_subsystem('sub', om.Group(), promotes=['*'])

                sub.linear_solver = om.DirectSolver(assemble_jac=True)
                sub.options['assembled_jac_type'] = 'dense'

                sub.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)', obj=0.0,
                                                         x=0.0, z=np.array([0.0, 0.0]), y1=0.0, y2=0.0),
                                  promotes=['obj', 'x', 'z', 'y1', 'y2'])

                sub.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1', con1=0.0, y1=0.0),
                                  promotes=['con1', 'y1'])
                sub.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0', con2=0.0, y2=0.0),
                                  promotes=['con2', 'y2'])

                self.nonlinear_solver = om.NewtonSolver()
                self.linear_solver = om.LinearBlockGS()
                self.linear_solver.options['maxiter'] = 25
                self.linear_solver.options['atol'] = 1e-16

        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        prob.model.approx_totals(method='cs')

        prob.run_model()

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, 1.0e-6)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, 1.0e-6)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)

    @parameterized.expand(itertools.product([om.DefaultVector, PETScVector]),
                          name_func=lambda f, n, p:
                          'test_newton_with_krylov_solver_'+'_'.join(title(a) for a in p.args))
    def test_newton_with_krylov_solver(self, vec_class):
        # Basic sellar test.

        if not vec_class:
            raise unittest.SkipTest("PETSc is not installed")

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver()
        sub.linear_solver = om.ScipyKrylov()
        sub.nonlinear_solver.options['atol'] = 1e-10
        sub.nonlinear_solver.options['rtol'] = 1e-10
        sub.linear_solver.options['atol'] = 1e-15

        model.approx_totals(method='cs', step=1e-14)

        prob.setup(check=False, local_vector_class=vec_class)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, 1.0e-6)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, 1.0e-6)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)

    def test_newton_with_cscjac_under_cs(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver()
        sub.linear_solver = om.ScipyKrylov(assemble_jac=True)
        sub.nonlinear_solver.options['atol'] = 1e-20
        sub.nonlinear_solver.options['rtol'] = 1e-20

        model.approx_totals(method='cs', step=1e-12)

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z', 'x']
        of = ['obj', 'con1']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)

    def test_newton_with_fd_group(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])
        subfd = sub.add_subsystem('subfd', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        subfd.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        subfd.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        # Finite difference for the Newton linear solve only
        subfd.approx_totals(method='fd')

        sub.nonlinear_solver = om.NewtonSolver()
        sub.nonlinear_solver.options['maxiter'] = 12
        sub.linear_solver = om.DirectSolver(assemble_jac=False)
        sub.nonlinear_solver.options['atol'] = 1e-20
        sub.nonlinear_solver.options['rtol'] = 1e-20

        # Complex Step for top derivatives
        model.approx_totals(method='cs', step=1e-14)

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, 1.0e-6)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, 1.0e-6)
        assert_rel_error(self, J['obj', 'x'][0][0], 2.98061391, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][0], -9.61002186, 1.0e-6)
        assert_rel_error(self, J['con1', 'z'][0][1], -0.78449158, 1.0e-6)
        assert_rel_error(self, J['con1', 'x'][0][0], -0.98061448, 1.0e-6)

    def test_nested_complex_step_unsupported(self):
        # Basic sellar test.

        prob = self.prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1CS(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2CS(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        prob.model.nonlinear_solver = om.NewtonSolver()
        prob.model.linear_solver = om.DirectSolver(assemble_jac=False)

        prob.model.approx_totals(method='cs')
        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

        outs = prob.model.list_outputs(residuals=True, out_stream=None)
        for j in range(len(outs)):
            val = np.linalg.norm(outs[j][1]['resids'])
            self.assertLess(val, 1e-8, msg="Check if CS cleans up after itself.")


class TestComponentComplexStep(unittest.TestCase):

    def test_implicit_component(self):

        class TestImplCompArrayDense(TestImplCompArray):

            def setup(self):
                super(TestImplCompArrayDense, self).setup()
                self.declare_partials('*', '*', method='cs')

        prob = self.prob = om.Problem()
        model = prob.model

        model.add_subsystem('p_rhs', om.IndepVarComp('rhs', val=np.ones(2)))
        sub = model.add_subsystem('sub', om.Group())
        comp = sub.add_subsystem('comp', TestImplCompArrayDense())
        model.connect('p_rhs.rhs', 'sub.comp.rhs')

        model.linear_solver = om.ScipyKrylov()

        prob.setup()
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

        prob = self.prob = om.Problem()
        model = prob.model

        model.add_subsystem('p_rhs', om.IndepVarComp('rhs', val=np.ones(2)))
        sub = model.add_subsystem('sub', om.Group())
        comp = sub.add_subsystem('comp', TestImplCompArrayDense())
        model.connect('p_rhs.rhs', 'sub.comp.rhs')

        model.linear_solver = om.ScipyKrylov()

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as context:
            model.resetup(setup_mode='reconf')

        msg = "TestImplCompArrayDense (sub.comp): In order to activate complex step during reconfiguration, " \
              "you need to set 'force_alloc_complex' to True during setup. " \
              "e.g. 'problem.setup(force_alloc_complex=True)'"
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

        class KenComp(om.ExplicitComponent):

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

                outputs['y1'] *= 1.0

        prob = self.prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', val=np.array([[7.0, 3.0], [2.4, 3.33]])))
        model.add_subsystem('comp', KenComp())
        model.connect('px.x', 'comp.x1')

        prob.setup()
        prob.run_model()

        of = ['comp.y1']
        wrt = ['px.x']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['comp.y1', 'px.x'][0][0], 1.0, 1e-6)
        assert_rel_error(self, derivs['comp.y1', 'px.x'][1][1], 3.0, 1e-6)
        assert_rel_error(self, derivs['comp.y1', 'px.x'][2][2], 1.0, 1e-6)
        assert_rel_error(self, derivs['comp.y1', 'px.x'][3][3], 1.0/2.34, 1e-6)

    def test_sellar_comp_cs(self):
        # Basic sellar test.

        prob = self.prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1CS(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2CS(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.linear_solver = om.DirectSolver(assemble_jac=False)

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

        outs = prob.model.list_outputs(residuals=True, out_stream=None)
        for j in range(len(outs)):
            val = np.linalg.norm(outs[j][1]['resids'])
            self.assertLess(val, 1e-8, msg="Check if CS cleans up after itself.")

    def test_stepsizes_under_complex_step(self):
        import openmdao.api as om

        class SimpleComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0)

                self.declare_partials(of='y', wrt='x', method='cs')
                self.count = 0

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0*inputs['x']

                if self.under_complex_step:

                    # Local cs
                    if self.count == 0 and inputs['x'].imag != 1.0e-40:
                        msg = "Wrong stepsize for local CS"
                        raise RuntimeError(msg)

                    # Global cs with default setting.
                    if self.count == 1 and inputs['x'].imag != 1.0e-40:
                        msg = "Wrong stepsize for default global CS"
                        raise RuntimeError(msg)

                    # Global cs with user setting.
                    if self.count == 3 and inputs['x'].imag != 1.0e-12:
                        msg = "Wrong stepsize for user global CS"
                        raise RuntimeError(msg)

                    # Check partials cs with default setting forward.
                    if self.count == 4 and inputs['x'].imag != 1.0e-40:
                        msg = "Wrong stepsize for check partial default CS forward"
                        raise RuntimeError(msg)

                    # Check partials cs with default setting.
                    if self.count == 5 and inputs['x'].imag != 1.0e-40:
                        msg = "Wrong stepsize for check partial default CS"
                        raise RuntimeError(msg)

                    # Check partials cs with user setting forward.
                    if self.count == 6 and inputs['x'].imag != 1.0e-40:
                        msg = "Wrong stepsize for check partial user CS forward"
                        raise RuntimeError(msg)

                    # Check partials cs with user setting.
                    if self.count == 7 and inputs['x'].imag != 1.0e-14:
                        msg = "Wrong stepsize for check partial user CS"
                        raise RuntimeError(msg)

                    self.count += 1

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = 3.

        prob = om.Problem()
        prob.model.add_subsystem('px', om.IndepVarComp('x', val=1.0))
        prob.model.add_subsystem('comp', SimpleComp())
        prob.model.connect('px.x', 'comp.x')

        prob.model.add_design_var('px.x', lower=-100, upper=100)
        prob.model.add_objective('comp.y')

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        prob.check_totals(method='cs', out_stream=None)

        prob.check_totals(method='cs', step=1e-12, out_stream=None)

        prob.check_partials(method='cs', out_stream=None)

        prob.check_partials(method='cs', step=1e-14, out_stream=None)

    def test_feature_under_complex_step(self):
        import openmdao.api as om

        class SimpleComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0)

                self.declare_partials(of='y', wrt='x', method='cs')

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0*inputs['x']

                if self.under_complex_step:
                    print("Under complex step")
                    print("x", inputs['x'])
                    print("y", outputs['y'])

        prob = om.Problem()
        prob.model.add_subsystem('px', om.IndepVarComp('x', val=1.0))
        prob.model.add_subsystem('comp', SimpleComp())
        prob.model.connect('px.x', 'comp.x')

        prob.model.add_design_var('px.x', lower=-100, upper=100)
        prob.model.add_objective('comp.y')

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        prob.compute_totals(of=['comp.y'], wrt=['px.x'])


class ApproxTotalsFeature(unittest.TestCase):

    def test_basic(self):
        import numpy as np

        import openmdao.api as om

        class CompOne(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_output('y', val=np.zeros(25))
                self._exec_count = 0

            def compute(self, inputs, outputs):
                x = inputs['x']
                outputs['y'] = np.arange(25) * x
                self._exec_count += 1

        class CompTwo(om.ExplicitComponent):

            def setup(self):
                self.add_input('y', val=np.zeros(25))
                self.add_output('z', val=0.0)
                self._exec_count = 0

            def compute(self, inputs, outputs):
                y = inputs['y']
                outputs['z'] = np.sum(y)
                self._exec_count += 1

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('comp1', CompOne(), promotes=['x', 'y'])
        comp2 = model.add_subsystem('comp2', CompTwo(), promotes=['y', 'z'])

        model.linear_solver = om.ScipyKrylov()
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

        import openmdao.api as om

        class CompOne(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_output('y', val=np.zeros(25))
                self._exec_count = 0

            def compute(self, inputs, outputs):
                x = inputs['x']
                outputs['y'] = np.arange(25) * x
                self._exec_count += 1

        class CompTwo(om.ExplicitComponent):

            def setup(self):
                self.add_input('y', val=np.zeros(25))
                self.add_output('z', val=0.0)
                self._exec_count = 0

            def compute(self, inputs, outputs):
                y = inputs['y']
                outputs['z'] = np.sum(y)
                self._exec_count += 1

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('comp1', CompOne(), promotes=['x', 'y'])
        model.add_subsystem('comp2', CompTwo(), promotes=['y', 'z'])

        model.linear_solver = om.ScipyKrylov()
        model.approx_totals(method='cs')

        prob.setup()
        prob.run_model()

        of = ['z']
        wrt = ['x']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['z', 'x'], [[300.0]], 1e-6)

    def test_arguments(self):
        import numpy as np

        import openmdao.api as om

        class CompOne(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_output('y', val=np.zeros(25))
                self._exec_count = 0

            def compute(self, inputs, outputs):
                x = inputs['x']
                outputs['y'] = np.arange(25) * x
                self._exec_count += 1

        class CompTwo(om.ExplicitComponent):

            def setup(self):
                self.add_input('y', val=np.zeros(25))
                self.add_output('z', val=0.0)
                self._exec_count = 0

            def compute(self, inputs, outputs):
                y = inputs['y']
                outputs['z'] = np.sum(y)
                self._exec_count += 1

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('comp1', CompOne(), promotes=['x', 'y'])
        model.add_subsystem('comp2', CompTwo(), promotes=['y', 'z'])

        model.linear_solver = om.ScipyKrylov()
        model.approx_totals(method='fd', step=1e-7, form='central', step_calc='rel')

        prob.setup()
        prob.run_model()

        of = ['z']
        wrt = ['x']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['z', 'x'], [[300.0]], 1e-6)

    def test_sellarCS(self):
        # Just tests Newton on Sellar with FD derivs.
        import openmdao.api as om
        from openmdao.test_suite.components.sellar_feature import SellarNoDerivativesCS

        prob = om.Problem()
        prob.model = SellarNoDerivativesCS()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)


class ParallelFDParametricTestCase(unittest.TestCase):

    @parametric_suite(
        assembled_jac=[False],
        jacobian_type=['dense'],
        partial_type=['array'],
        partial_method=['fd', 'cs'],
        num_var=[3],
        var_shape=[(2, 3), (2,)],
        connection_type=['explicit'],
        run_by_default=True,
    )
    def test_subset(self, param_instance):
        param_instance.linear_solver_class = om.DirectSolver
        param_instance.linear_solver_options = {}  # defaults not valid for DirectSolver

        param_instance.setup()
        problem = param_instance.problem
        model = problem.model

        expected_values = model.expected_values
        if expected_values:
            actual = {key: problem[key] for key in iterkeys(expected_values)}
            assert_rel_error(self, actual, expected_values, 1e-4)

        expected_totals = model.expected_totals
        if expected_totals:
            # Forward Derivatives Check
            totals = param_instance.compute_totals('fwd')
            assert_rel_error(self, totals, expected_totals, 1e-4)

            # Reverse Derivatives Check
            totals = param_instance.compute_totals('rev')
            assert_rel_error(self, totals, expected_totals, 1e-4)


if __name__ == "__main__":
    unittest.main()
