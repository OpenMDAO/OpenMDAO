"""Test the DirectSolver linear solver class."""

from __future__ import division, print_function

import unittest
from six import assertRaisesRegex, iteritems

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, NewtonSolver, ExecComp, \
     NewtonSolver, BalanceComp, ExplicitComponent, ImplicitComponent
from openmdao.solvers.linear.tests.linear_test_base import LinearSolverTests
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleJacVec
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.utils.assert_utils import assert_rel_error


class NanComp(ExplicitComponent):
    def setup(self):
        self.add_input('x', 1.0)
        self.add_output('y', 1.0)

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = 3.0*inputs['x']

    def compute_partials(self, inputs, partials):
        """Intentionally incorrect derivative."""
        J = partials
        J['y', 'x'] = np.NaN


class SingularComp(ImplicitComponent):
    def setup(self):
        self.add_input('x', 1.0)
        self.add_output('y', 1.0)

        self.declare_partials(of='*', wrt='*')

    def compute_partials(self, inputs, partials):
        """Intentionally incorrect derivative."""
        J = partials
        J['y', 'x'] = 0.0
        J['y', 'y'] = 0.0


class NanComp2(ExplicitComponent):
    def setup(self):
        self.add_input('x', 1.0)
        self.add_output('y', 1.0)
        self.add_output('y2', 1.0)

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = 3.0*inputs['x']
        outputs['y2'] = 2.0*inputs['x']

    def compute_partials(self, inputs, partials):
        """Intentionally incorrect derivative."""
        J = partials
        J['y', 'x'] = np.NaN
        J['y2', 'x'] = 2.0

class DupPartialsComp(ExplicitComponent):
    def setup(self):
        self.add_input('c', np.zeros(19))
        self.add_output('x', np.zeros(11))

        rows = [0,  1,  4, 10, 7, 9, 10, 4]
        cols = [0, 18, 11,  2, 5, 9,  2, 11]
        self.declare_partials(of='x', wrt='c', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        pass

    def compute_partials(self, inputs, partials):
        pass


class TestDirectSolver(LinearSolverTests.LinearSolverTestCase):

    linear_solver_class = DirectSolver

    # DirectSolver doesn't iterate.
    def test_solve_linear_maxiter(self):
        # Test that using options that should not exist in class cause an error
        solver = DirectSolver()

        msg = "\"Option '%s' cannot be set because it has not been declared.\""

        for option in ['atol', 'rtol', 'maxiter', 'err_on_maxiter']:
            with self.assertRaises(KeyError) as context:
                solver.options[option] = 1

            self.assertEqual(str(context.exception), msg % option)

    def test_solve_on_subsystem(self):
        """solve an implicit system with DirectSolver attached to a subsystem"""

        p = Problem()
        model = p.model
        dv = model.add_subsystem('des_vars', IndepVarComp())
        # just need a dummy variable so the sizes don't match between root and g1
        dv.add_output('dummy', val=1.0, shape=10)

        g1 = model.add_subsystem('g1', TestImplicitGroup(lnSolverClass=DirectSolver))

        p.setup()

        g1.linear_solver.options['assemble_jac'] = False

        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        # forward
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        g1._linearize(g1._assembled_jac)
        g1.linear_solver._linearize()
        g1.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output, g1.expected_solution, 1e-15)

        # reverse
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        g1.linear_solver._linearize()
        g1.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output, g1.expected_solution, 3e-15)

    def test_rev_mode_bug(self):

        prob = Problem()
        prob.model = SellarDerivatives(nonlinear_solver=NewtonSolver(), linear_solver=DirectSolver())

        prob.setup(check=False, mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['x', 'z']
        of = ['obj', 'con1', 'con2']

        Jbase = {}
        Jbase['con1', 'x'] = [[-0.98061433]]
        Jbase['con1', 'z'] = np.array([[-9.61002285, -0.78449158]])
        Jbase['con2', 'x'] = [[0.09692762]]
        Jbase['con2', 'z'] = np.array([[1.94989079, 1.0775421]])
        Jbase['obj', 'x'] = [[2.98061392]]
        Jbase['obj', 'z'] = np.array([[9.61001155, 1.78448534]])

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        for key, val in iteritems(Jbase):
            assert_rel_error(self, J[key], val, .00001)

        # In the bug, the solver mode got switched from fwd to rev when it shouldn't
        # have been, causing a singular matrix and NaNs in the output.
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_multi_dim_src_indices(self):
        prob = Problem()
        model = prob.model
        size = 5

        model.add_subsystem('indeps', IndepVarComp('x', np.arange(5).reshape((1,size,1))))
        model.add_subsystem('comp', ExecComp('y = x * 2.', x=np.zeros((size,)), y=np.zeros((size,))))
        src_indices = [[0, i, 0] for i in range(size)]
        model.connect('indeps.x', 'comp.x', src_indices=src_indices)

        model.linear_solver = DirectSolver()
        prob.setup()
        prob.run_model()

        J = prob.compute_totals(wrt=['indeps.x'], of=['comp.y'], return_format='array')
        np.testing.assert_almost_equal(J, np.eye(size) * 2.)

    def test_raise_error_on_singular(self):
        prob = Problem()
        model = prob.model

        comp = IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=Group())
        teg.add_subsystem('dynamics', ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = DirectSolver(assemble_jac=False)

        teg.nonlinear_solver = NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4

        prob.setup()
        prob.set_solver_print(level=0)

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected_msg = "Singular entry found in 'thrust_equilibrium_group' for column associated with state/residual 'thrust'."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_dup_partials(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('des_vars', IndepVarComp('x', 1.0), promotes=['*'])
        model.add_subsystem('dupcomp', DupPartialsComp())

        model.linear_solver = DirectSolver(assemble_jac=True)

        with self.assertRaises(Exception) as cm:
            prob.setup()

        expected_msg = "dupcomp: declare_partials has been called with rows and cols that specify the following duplicate subjacobian entries: [(4, 11), (10, 2)]."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_singular_with_densejac(self):
        prob = Problem()
        model = prob.model

        comp = IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=Group())
        teg.add_subsystem('dynamics', ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = DirectSolver(assemble_jac=True)
        teg.options['assembled_jac_type'] = 'dense'

        teg.nonlinear_solver = NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4

        prob.setup()
        prob.set_solver_print(level=0)

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected_msg = "Singular entry found in 'thrust_equilibrium_group' for column associated with state/residual 'thrust'."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_singular_with_sparsejac(self):
        prob = Problem()
        model = prob.model

        comp = IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=Group())
        teg.add_subsystem('dynamics', ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = DirectSolver(assemble_jac=True)

        teg.nonlinear_solver = NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4

        prob.setup()
        prob.set_solver_print(level=0)

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected_msg = "Singular entry found in 'thrust_equilibrium_group' for row associated with state/residual 'thrust'."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_no_error_on_singular(self):
        prob = Problem()
        model = prob.model

        comp = IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=Group())
        teg.add_subsystem('dynamics', ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = DirectSolver(assemble_jac=False)

        teg.nonlinear_solver = NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4

        prob.setup()
        prob.set_solver_print(level=0)

        teg.linear_solver.options['err_on_singular'] = False
        prob.run_model()

    def test_raise_error_on_nan(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('p', IndepVarComp('x', 2.0))
        model.add_subsystem('c1', ExecComp('y = 4.0*x'))
        sub = model.add_subsystem('sub', Group())
        sub.add_subsystem('c2', NanComp())
        model.add_subsystem('c3', ExecComp('y = 4.0*x'))
        model.add_subsystem('c4', NanComp2())
        model.add_subsystem('c5', ExecComp('y = 3.0*x'))
        model.add_subsystem('c6', ExecComp('y = 2.0*x'))

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'sub.c2.x')
        model.connect('sub.c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x')
        model.connect('c4.y2', 'c6.x')

        model.linear_solver = DirectSolver(assemble_jac=False)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "NaN entries found in '' for rows associated with states/residuals ['sub.c2.y', 'c4.y']."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_nan_sparse(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('p', IndepVarComp('x', 2.0))
        model.add_subsystem('c1', ExecComp('y = 4.0*x'))
        sub = model.add_subsystem('sub', Group())
        sub.add_subsystem('c2', NanComp())
        model.add_subsystem('c3', ExecComp('y = 4.0*x'))
        model.add_subsystem('c4', NanComp2())
        model.add_subsystem('c5', ExecComp('y = 3.0*x'))
        model.add_subsystem('c6', ExecComp('y = 2.0*x'))

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'sub.c2.x')
        model.connect('sub.c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x')
        model.connect('c4.y2', 'c6.x')

        model.linear_solver = DirectSolver(assemble_jac=True)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "NaN entries found in '' for rows associated with states/residuals ['sub.c2.y', 'c4.y']."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_nan_dense(self):

        prob = Problem(model=Group(assembled_jac_type='dense'))
        model = prob.model

        model.add_subsystem('p', IndepVarComp('x', 2.0))
        model.add_subsystem('c1', ExecComp('y = 4.0*x'))
        sub = model.add_subsystem('sub', Group())
        sub.add_subsystem('c2', NanComp())
        model.add_subsystem('c3', ExecComp('y = 4.0*x'))
        model.add_subsystem('c4', NanComp2())
        model.add_subsystem('c5', ExecComp('y = 3.0*x'))
        model.add_subsystem('c6', ExecComp('y = 2.0*x'))

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'sub.c2.x')
        model.connect('sub.c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x')
        model.connect('c4.y2', 'c6.x')

        model.linear_solver = DirectSolver(assemble_jac=True)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "NaN entries found in '' for rows associated with states/residuals ['sub.c2.y', 'c4.y']."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_error_on_NaN_bug(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p', IndepVarComp('x', 2.0*np.ones((2, 2))))
        model.add_subsystem('c1', ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c2', ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c3', ExecComp('y = 3.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c4', ExecComp('y = 2.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c5', NanComp())

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'c2.x')
        model.connect('c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x', src_indices=([0]))

        model.linear_solver = DirectSolver(assemble_jac=True)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "NaN entries found in '' for rows associated with states/residuals ['c5.y']."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_error_on_singular_with_sparsejac_bug(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p', IndepVarComp('x', 2.0*np.ones((2, 2))))
        model.add_subsystem('c1', ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c2', ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c3', ExecComp('y = 3.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c4', ExecComp('y = 2.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c5', SingularComp())

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'c2.x')
        model.connect('c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x', src_indices=([0]))

        model.linear_solver = DirectSolver(assemble_jac=True)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "Singular entry found in '' for row associated with state/residual 'c5.y'."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_error_on_singular_with_densejac_bug(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p', IndepVarComp('x', 2.0*np.ones((2, 2))))
        model.add_subsystem('c1', ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c2', ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c3', ExecComp('y = 3.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c4', ExecComp('y = 2.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c5', SingularComp())

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'c2.x')
        model.connect('c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x', src_indices=([0]))

        model.linear_solver = DirectSolver(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "Singular entry found in '' for column associated with state/residual 'c5.y'."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_underdetermined_csc(self):

        class DCgenerator(ImplicitComponent):

            def setup(self):
                self.add_input('V_bus', val=1.0)
                self.add_input('V_out', val=1.0)

                self.add_output('I_out', val=-2.0)
                self.add_output('P_out', val=-2.0)

                self.declare_partials('I_out', 'V_bus', val=1.0)
                self.declare_partials('I_out', 'V_out', val=-1.0)
                self.declare_partials('P_out', ['V_out', 'I_out'])
                self.declare_partials('P_out', 'P_out', val=-1.0)

            def apply_nonlinear(self, inputs, outputs, resids):
                resids['I_out'] = inputs['V_bus'] - inputs['V_out']
                resids['P_out'] = inputs['V_out'] * outputs['I_out'] - outputs['P_out']

            def linearize(self, inputs, outputs, J):
                J['P_out', 'V_out'] = outputs['I_out']
                J['P_out', 'I_out'] = inputs['V_out']

        class RectifierCalcs(ImplicitComponent):

            def setup(self):
                self.add_input('P_out', val=1.0)

                self.add_output('P_in', val=1.0)
                self.add_output('V_out', val=1.0)
                self.add_output('Q_in', val=1.0)

                self.declare_partials('P_in', 'P_out', val=1.0)
                self.declare_partials('P_in', 'P_in', val=-1.0)
                self.declare_partials('V_out', 'V_out', val=-1.0)
                self.declare_partials('Q_in', 'P_in', val=1.0)
                self.declare_partials('Q_in', 'Q_in', val=-1.0)

            def apply_nonlinear(self, inputs, outputs, resids):
                resids['P_in'] = inputs['P_out'] - outputs['P_in']
                resids['V_out'] = 1.0 - outputs['V_out']
                resids['Q_in'] = outputs['P_in'] - outputs['Q_in']

        class Rectifier(Group):

            def setup(self):
                self.add_subsystem('gen', DCgenerator(), promotes=[('V_bus', 'Vm_dc'), 'P_out'])

                self.add_subsystem('calcs', RectifierCalcs(), promotes=['P_out', ('V_out', 'Vm_dc')])

                self.nonlinear_solver = NewtonSolver()
                self.linear_solver = DirectSolver()

        prob = Problem()
        prob.model.add_subsystem('sub', Rectifier())

        prob.setup()
        prob.set_solver_print(level=0)

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected_msg = "Identical rows or columns found in jacobian in 'sub'. Problem is underdetermined."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_matvec_error_raised(self):
        prob = Problem()
        model = prob.model
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleJacVec(),
                            promotes=['length', 'width', 'area'])

        model.linear_solver = self.linear_solver_class()
        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')

        prob['width'] = 2.0

        msg = "AssembledJacobian not supported for matrix-free subcomponent."
        with assertRaisesRegex(self, Exception, msg):
            prob.run_model()


class TestDirectSolverFeature(unittest.TestCase):

    def test_specify_solver(self):

        from openmdao.api import Problem, DirectSolver
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.linear_solver = DirectSolver()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)


if __name__ == "__main__":
    unittest.main()
