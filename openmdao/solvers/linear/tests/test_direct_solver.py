"""Test the DirectSolver linear solver class."""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.core.tests.test_distrib_derivs import DistribExecComp
from openmdao.solvers.linear.tests.linear_test_base import LinearSolverTests
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleJacVec
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI
try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class NanComp(om.ExplicitComponent):
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


class SingularComp(om.ImplicitComponent):
    def setup(self):
        self.add_input('x', 1.0)
        self.add_output('y', 1.0)

        self.declare_partials(of='*', wrt='*')

    def compute_partials(self, inputs, partials):
        """Intentionally incorrect derivative."""
        J = partials
        J['y', 'x'] = 0.0
        J['y', 'y'] = 0.0


class NanComp2(om.ExplicitComponent):
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

class DupPartialsComp(om.ExplicitComponent):
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

    linear_solver_class = om.DirectSolver

    # DirectSolver doesn't iterate.
    def test_solve_linear_maxiter(self):
        # Test that using options that should not exist in class cause an error
        solver = om.DirectSolver()

        msg = "\"DirectSolver: Option '%s' cannot be set because it has not been declared.\""

        for option in ['atol', 'rtol', 'maxiter', 'err_on_non_converge']:
            with self.assertRaises(KeyError) as context:
                solver.options[option] = 1

            self.assertEqual(str(context.exception), msg % option)

    def test_solve_on_subsystem(self):
        """solve an implicit system with DirectSolver attached to a subsystem"""

        p = om.Problem()
        model = p.model
        dv = model.add_subsystem('des_vars', om.IndepVarComp())
        # just need a dummy variable so the sizes don't match between root and g1
        dv.add_output('dummy', val=1.0, shape=10)

        g1 = model.add_subsystem('g1', TestImplicitGroup(lnSolverClass=om.DirectSolver))

        p.setup()

        g1.linear_solver.options['assemble_jac'] = False

        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        # forward
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        g1._linearize(g1._assembled_jac)
        g1.linear_solver._linearize()
        g1.run_solve_linear(['linear'], 'fwd')

        output = d_outputs.asarray()
        assert_near_equal(output, g1.expected_solution, 1e-15)

        # reverse
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        g1.linear_solver._linearize()
        g1.run_solve_linear(['linear'], 'rev')

        output = d_residuals.asarray()
        assert_near_equal(output, g1.expected_solution, 3e-15)

    def test_rev_mode_bug(self):

        prob = om.Problem()
        prob.model = SellarDerivatives(nonlinear_solver=om.NewtonSolver(solve_subsystems=False),
                                       linear_solver=om.DirectSolver())

        prob.setup(check=False, mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['y2'], 12.05848819, .00001)

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
        for key, val in Jbase.items():
            assert_near_equal(J[key], val, .00001)

        # In the bug, the solver mode got switched from fwd to rev when it shouldn't
        # have been, causing a singular matrix and NaNs in the output.
        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['y2'], 12.05848819, .00001)

    def test_multi_dim_src_indices(self):
        prob = om.Problem()
        model = prob.model
        size = 5

        model.add_subsystem('indeps', om.IndepVarComp('x', np.arange(5).reshape((1,size,1))))
        model.add_subsystem('comp', om.ExecComp('y = x * 2.', x=np.zeros((size,)), y=np.zeros((size,))))
        src_indices = [[0, i, 0] for i in range(size)]
        model.connect('indeps.x', 'comp.x', src_indices=src_indices)

        model.linear_solver = om.DirectSolver()
        prob.setup()
        prob.run_model()

        J = prob.compute_totals(wrt=['indeps.x'], of=['comp.y'], return_format='array')
        np.testing.assert_almost_equal(J, np.eye(size) * 2.)

    def test_raise_error_on_singular(self):
        prob = om.Problem()
        model = prob.model

        comp = om.IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=om.Group())
        teg.add_subsystem('dynamics', om.ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = om.BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = om.DirectSolver(assemble_jac=False)

        teg.nonlinear_solver = om.NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4

        prob.setup()
        prob.set_solver_print(level=0)
        prob.final_setup()

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected_msg = "Singular entry found in 'thrust_equilibrium_group' <class Group> for row associated with state/residual 'thrust' ('thrust_equilibrium_group.thrust_bal.thrust') index 0."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_dup_partials(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('des_vars', om.IndepVarComp('x', 1.0), promotes=['*'])
        model.add_subsystem('dupcomp', DupPartialsComp())

        model.linear_solver = om.DirectSolver(assemble_jac=True)

        with self.assertRaises(Exception) as cm:
            prob.setup()

        expected_msg = "'dupcomp' <class DupPartialsComp>: d(x)/d(c): declare_partials has been called with rows and cols that specify the following duplicate subjacobian entries: [(4, 11), (10, 2)]."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_singular_with_densejac(self):
        prob = om.Problem()
        model = prob.model

        comp = om.IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=om.Group())
        teg.add_subsystem('dynamics', om.ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = om.BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = om.DirectSolver(assemble_jac=True)
        teg.options['assembled_jac_type'] = 'dense'

        teg.nonlinear_solver = om.NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4

        prob.setup()
        prob.set_solver_print(level=0)

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected_msg = "Singular entry found in 'thrust_equilibrium_group' <class Group> for row associated with state/residual 'thrust' ('thrust_equilibrium_group.thrust_bal.thrust') index 0."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_singular_with_sparsejac(self):
        prob = om.Problem()
        model = prob.model

        comp = om.IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=om.Group())
        teg.add_subsystem('dynamics', om.ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = om.BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = om.DirectSolver(assemble_jac=True)

        teg.nonlinear_solver = om.NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4

        prob.setup()
        prob.set_solver_print(level=0)

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected_msg = "Singular entry found in 'thrust_equilibrium_group' <class Group> for row associated with state/residual 'thrust' ('thrust_equilibrium_group.thrust_bal.thrust') index 0."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_no_error_on_singular(self):
        prob = om.Problem()
        model = prob.model

        comp = om.IndepVarComp()
        comp.add_output('dXdt:TAS', val=1.0)
        comp.add_output('accel_target', val=2.0)
        model.add_subsystem('des_vars', comp, promotes=['*'])

        teg = model.add_subsystem('thrust_equilibrium_group', subsys=om.Group())
        teg.add_subsystem('dynamics', om.ExecComp('z = 2.0*thrust'), promotes=['*'])

        thrust_bal = om.BalanceComp()
        thrust_bal.add_balance(name='thrust', val=1207.1, lhs_name='dXdt:TAS',
                               rhs_name='accel_target', eq_units='m/s**2', lower=-10.0, upper=10000.0)

        teg.add_subsystem(name='thrust_bal', subsys=thrust_bal,
                          promotes_inputs=['dXdt:TAS', 'accel_target'],
                          promotes_outputs=['thrust'])

        teg.linear_solver = om.DirectSolver(assemble_jac=False)

        teg.nonlinear_solver = om.NewtonSolver()
        teg.nonlinear_solver.options['solve_subsystems'] = True
        teg.nonlinear_solver.options['max_sub_solves'] = 1
        teg.nonlinear_solver.options['atol'] = 1e-4

        prob.setup()
        prob.set_solver_print(level=0)

        teg.linear_solver.options['err_on_singular'] = False
        prob.run_model()

    def test_raise_error_on_nan(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 2.0))
        model.add_subsystem('c1', om.ExecComp('y = 4.0*x'))
        sub = model.add_subsystem('sub', om.Group())
        sub.add_subsystem('c2', NanComp())
        model.add_subsystem('c3', om.ExecComp('y = 4.0*x'))
        model.add_subsystem('c4', NanComp2())
        model.add_subsystem('c5', om.ExecComp('y = 3.0*x'))
        model.add_subsystem('c6', om.ExecComp('y = 2.0*x'))

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'sub.c2.x')
        model.connect('sub.c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x')
        model.connect('c4.y2', 'c6.x')

        model.linear_solver = om.DirectSolver(assemble_jac=False)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "NaN entries found in <model> <class Group> for rows associated with states/residuals ['sub.c2.y', 'c4.y']."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_nan_sparse(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 2.0))
        model.add_subsystem('c1', om.ExecComp('y = 4.0*x'))
        sub = model.add_subsystem('sub', om.Group())
        sub.add_subsystem('c2', NanComp())
        model.add_subsystem('c3', om.ExecComp('y = 4.0*x'))
        model.add_subsystem('c4', NanComp2())
        model.add_subsystem('c5', om.ExecComp('y = 3.0*x'))
        model.add_subsystem('c6', om.ExecComp('y = 2.0*x'))

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'sub.c2.x')
        model.connect('sub.c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x')
        model.connect('c4.y2', 'c6.x')

        model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "NaN entries found in <model> <class Group> for rows associated with states/residuals ['sub.c2.y', 'c4.y']."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_raise_error_on_nan_dense(self):

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 2.0))
        model.add_subsystem('c1', om.ExecComp('y = 4.0*x'))
        sub = model.add_subsystem('sub', om.Group())
        sub.add_subsystem('c2', NanComp())
        model.add_subsystem('c3', om.ExecComp('y = 4.0*x'))
        model.add_subsystem('c4', NanComp2())
        model.add_subsystem('c5', om.ExecComp('y = 3.0*x'))
        model.add_subsystem('c6', om.ExecComp('y = 2.0*x'))

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'sub.c2.x')
        model.connect('sub.c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x')
        model.connect('c4.y2', 'c6.x')

        model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "NaN entries found in <model> <class Group> for rows associated with states/residuals ['sub.c2.y', 'c4.y']."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_error_on_NaN_bug(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 2.0*np.ones((2, 2))))
        model.add_subsystem('c1', om.ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c2', om.ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c3', om.ExecComp('y = 3.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c4', om.ExecComp('y = 2.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c5', NanComp())

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'c2.x')
        model.connect('c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x', src_indices=([0]))

        model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "NaN entries found in <model> <class Group> for rows associated with states/residuals ['c5.y']."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_error_on_singular_with_sparsejac_bug(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 2.0*np.ones((2, 2))))
        model.add_subsystem('c1', om.ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c2', om.ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c3', om.ExecComp('y = 3.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c4', om.ExecComp('y = 2.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c5', SingularComp())

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'c2.x')
        model.connect('c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x', src_indices=([0]))

        model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "Singular entry found in <model> <class Group> for row associated with state/residual 'c5.y' index 0."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_error_on_singular_with_densejac_bug(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 2.0*np.ones((2, 2))))
        model.add_subsystem('c1', om.ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c2', om.ExecComp('y = 4.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c3', om.ExecComp('y = 3.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c4', om.ExecComp('y = 2.0*x', x=np.zeros((2, 2)), y=np.zeros((2, 2))))
        model.add_subsystem('c5', SingularComp())

        model.connect('p.x', 'c1.x')
        model.connect('c1.y', 'c2.x')
        model.connect('c2.y', 'c3.x')
        model.connect('c3.y', 'c4.x')
        model.connect('c4.y', 'c5.x', src_indices=([0]))

        model.linear_solver = om.DirectSolver(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.compute_totals(of=['c5.y'], wrt=['p.x'])

        expected_msg = "Singular entry found in <model> <class Group> for row associated with state/residual 'c5.y' index 0."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_error_msg_underdetermined_1(self):

        class DCgenerator(om.ImplicitComponent):

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

        class RectifierCalcs(om.ImplicitComponent):

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

        class Rectifier(om.Group):

            def setup(self):
                self.add_subsystem('gen', DCgenerator(), promotes=[('V_bus', 'Vm_dc'), 'P_out'])

                self.add_subsystem('calcs', RectifierCalcs(), promotes=['P_out', ('V_out', 'Vm_dc')])

                self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.linear_solver = om.DirectSolver()

        prob = om.Problem()
        prob.model.add_subsystem('sub', Rectifier())

        prob.setup()
        prob.set_solver_print(level=0)

        with self.assertRaises(RuntimeError) as cm:
            prob.run_model()

        expected = "Jacobian in 'sub' is not full rank. The following set of states/residuals contains one or more equations that is a linear combination of the others: \n"
        expected += " 'gen.I_out' ('sub.gen.I_out') index 0.\n"
        expected += " 'Vm_dc' ('sub.calcs.V_out') index 0.\n"

        self.assertEqual(expected, str(cm.exception))

    def test_error_msg_underdetermined_2(self):

        class E1(om.ImplicitComponent):

            def setup(self):
                self.add_input('a', 1.0)
                self.add_input('aa', 1.0)
                self.add_input('y', 1.0)
                self.add_input('z', 1.0)
                self.add_output('x', 1.0)

                self.declare_partials('x', 'x', val=1.0)
                self.declare_partials('x', 'y', val=1.0)
                self.declare_partials('x', 'z', val=1.0)
                self.declare_partials('x', 'a', val=-1.0)
                self.declare_partials('x', 'aa', val=-1.0)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = outputs['x'] + inputs['y'] + inputs['z'] - inputs['a'] - inputs['aa']


        class E2(om.ImplicitComponent):

            def setup(self):
                self.add_input('x', 1.0)
                self.add_output('y', 1.0)

                self.declare_partials('y', 'x', val=2.063e-4)
                self.declare_partials('y', 'y')

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['y'] = 2.063e-4 * inputs['x'] - outputs['y'] ** 2

            def linearize(self, inputs, outputs, jacobian):
                jacobian['y', 'y'] = -2.0 * outputs['y']


        class E3(om.ImplicitComponent):

            def setup(self):
                self.add_input('a', 1.0)
                self.add_input('aa', 1.0)
                self.add_input('x', 1.0)
                self.add_input('y', 1.0)
                self.add_output('z', 1.0)

                self.declare_partials('z', 'x', val=2.0)
                self.declare_partials('z', 'y', val=1.0)
                self.declare_partials('z', 'z', val=-4.0)
                self.declare_partials('z', 'a', val=-1.0)
                self.declare_partials('z', 'aa', val=-1.0)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['z'] = 2.0 * inputs['x'] + inputs['y'] - 4.0 * outputs['z'] - inputs['a'] - inputs['aa']


        class E3bad(om.ImplicitComponent):

            def setup(self):
                self.add_input('a', 1.0)
                self.add_input('aa', 1.0)
                self.add_input('x', 1.0)
                self.add_input('y', 1.0)
                self.add_output('z', 1.0)

                self.declare_partials('z', 'x', val=1.0)
                self.declare_partials('z', 'y', val=1.0)
                self.declare_partials('z', 'z', val=1.0)
                self.declare_partials('z', 'a', val=-1.0)
                self.declare_partials('z', 'aa', val=-1.0)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['z'] = 2.0 * inputs['x'] + inputs['y'] - 4.0 * outputs['z'] - inputs['a'] - inputs['aa']

        # Configuration 1
        p = om.Problem()
        model = p.model

        ivc = om.IndepVarComp()
        ivc.add_output('aa', 1.0)
        model.add_subsystem('p', ivc, promotes=['aa'])

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = model.add_subsystem('sub2', om.Group())
        sub3 = model.add_subsystem('sub3', om.Group())

        sub1.add_subsystem('e1', E1(), promotes=['*'])
        sub1.add_subsystem('e2', E2(), promotes=['*'])
        sub1.add_subsystem('e3', E3(), promotes=['*'])

        sub2.add_subsystem('e1', E1(), promotes=['*'])
        sub2.add_subsystem('e2', E2(), promotes=['*'])
        sub2.add_subsystem('e3', E3bad(), promotes=['*'])

        sub3.add_subsystem('e1', E1(), promotes=['*'])
        sub3.add_subsystem('e2', E2(), promotes=['*'])
        sub3.add_subsystem('e3', E3(), promotes=['*'])

        model.connect('sub1.z', 'sub2.a')
        model.connect('sub2.z', 'sub3.a')
        model.connect('sub3.z', 'sub1.a')
        model.linear_solver = om.DirectSolver()
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        p.setup()
        with self.assertRaises(RuntimeError) as cm:
            p.run_model()

        expected = "Jacobian in '' is not full rank. The following set of states/residuals contains one or more equations that is a linear combination of the others: \n"
        expected += " 'sub2.x' ('sub2.e1.x') index 0.\n"
        expected += " 'sub2.z' ('sub2.e3.z') index 0.\n"

        self.assertEqual(expected, str(cm.exception))

        # Configuration 1 Dense
        p = om.Problem()
        model = p.model
        model.options['assembled_jac_type'] = 'dense'

        ivc = om.IndepVarComp()
        ivc.add_output('aa', 1.0)
        model.add_subsystem('p', ivc, promotes=['aa'])

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = model.add_subsystem('sub2', om.Group())
        sub3 = model.add_subsystem('sub3', om.Group())

        sub1.add_subsystem('e1', E1(), promotes=['*'])
        sub1.add_subsystem('e2', E2(), promotes=['*'])
        sub1.add_subsystem('e3', E3(), promotes=['*'])

        sub2.add_subsystem('e1', E1(), promotes=['*'])
        sub2.add_subsystem('e2', E2(), promotes=['*'])
        sub2.add_subsystem('e3', E3bad(), promotes=['*'])

        sub3.add_subsystem('e1', E1(), promotes=['*'])
        sub3.add_subsystem('e2', E2(), promotes=['*'])
        sub3.add_subsystem('e3', E3(), promotes=['*'])

        model.connect('sub1.z', 'sub2.a')
        model.connect('sub2.z', 'sub3.a')
        model.connect('sub3.z', 'sub1.a')
        model.linear_solver = om.DirectSolver()
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        p.setup()
        with self.assertRaises(RuntimeError) as cm:
            p.run_model()

        expected = "Jacobian in '' is not full rank. The following set of states/residuals contains one or more equations that is a linear combination of the others: \n"
        expected += " 'sub2.x' ('sub2.e1.x') index 0.\n"
        expected += " 'sub2.z' ('sub2.e3.z') index 0.\n"

        self.assertEqual(expected, str(cm.exception))

        # Configuration 2
        p = om.Problem()
        model = p.model

        ivc = om.IndepVarComp()
        ivc.add_output('aa', 1.0)
        model.add_subsystem('p', ivc, promotes=['aa'])

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = model.add_subsystem('sub2', om.Group())
        sub3 = model.add_subsystem('sub3', om.Group())

        sub1.add_subsystem('e1', E1(), promotes=['*'])
        sub1.add_subsystem('e2', E2(), promotes=['*'])
        sub1.add_subsystem('e3', E3(), promotes=['aa', 'x', 'y', 'z'])

        sub2.add_subsystem('e1', E1(), promotes=['*'])
        sub2.add_subsystem('e2', E2(), promotes=['*'])
        sub2.add_subsystem('e3', E3bad(), promotes=['aa', 'x', 'y', 'z'])

        sub3.add_subsystem('e1', E1(), promotes=['*'])
        sub3.add_subsystem('e2', E2(), promotes=['*'])
        sub3.add_subsystem('e3', E3(), promotes=['aa', 'x', 'y', 'z'])

        model.connect('sub1.z', 'sub2.a')
        model.connect('sub2.z', 'sub3.a')
        model.linear_solver = om.DirectSolver()
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        p.setup()
        with self.assertRaises(RuntimeError) as cm:
            p.run_model()

        expected = "Jacobian in '' is not full rank. The following set of states/residuals contains one or more equations that is a linear combination of the others: \n"
        expected += " '_auto_ivc.v0' index 0.\n"
        expected += " '_auto_ivc.v1' index 0.\n"
        expected += " '_auto_ivc.v2' index 0.\n"
        expected += " '_auto_ivc.v4' index 0.\n"
        expected += " 'sub1.x' ('sub1.e1.x') index 0.\n"
        expected += " 'sub1.y' ('sub1.e2.y') index 0.\n"
        expected += " 'sub1.z' ('sub1.e3.z') index 0.\n"
        expected += " 'sub2.x' ('sub2.e1.x') index 0.\n"
        expected += " 'sub2.z' ('sub2.e3.z') index 0.\n"
        expected += "Note that the problem may be in a single Component."

        self.assertEqual(expected, str(cm.exception))

    def test_matvec_error_raised(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param', om.IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleJacVec(),
                            promotes=['length', 'width', 'area'])

        model.linear_solver = self.linear_solver_class()
        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')

        prob['width'] = 2.0

        msg = "AssembledJacobian not supported for matrix-free subcomponent."
        with self.assertRaisesRegex(Exception, msg):
            prob.run_model()


@unittest.skipUnless(MPI and PETScVector, "only run with MPI and PETSc.")
class TestDirectSolverRemoteErrors(unittest.TestCase):

    N_PROCS = 2

    def test_distrib_direct(self):
        size = 3
        group = om.Group()

        group.add_subsystem('P', om.IndepVarComp('x', np.arange(size)))
        group.add_subsystem('C1', DistribExecComp(['y=2.0*x', 'y=3.0*x'], arr_size=size,
                                                  x=np.zeros(size),
                                                  y=np.zeros(size)))
        group.add_subsystem('C2', om.ExecComp(['z=3.0*y'],
                                           y=np.zeros(size),
                                           z=np.zeros(size)))

        prob = om.Problem()
        prob.model = group
        prob.model.linear_solver = om.DirectSolver()
        prob.model.connect('P.x', 'C1.x')
        prob.model.connect('C1.y', 'C2.y')


        prob.setup(check=False, mode='fwd')
        with self.assertRaises(Exception) as cm:
            prob.run_model()

        msg = "DirectSolver linear solver in <model> <class Group> cannot be used in or above a ParallelGroup or a " + \
            "distributed component."
        self.assertEqual(str(cm.exception), msg)

    def test_distrib_direct_subbed(self):
        size = 3
        prob = om.Problem()
        group = prob.model = om.Group()

        group.add_subsystem('P', om.IndepVarComp('x', np.arange(size)))
        sub = group.add_subsystem('sub', om.Group())

        sub.add_subsystem('C1', DistribExecComp(['y=2.0*x', 'y=3.0*x'], arr_size=size,
                                                x=np.zeros(size),
                                                y=np.zeros(size)))
        sub.add_subsystem('C2', om.ExecComp(['z=3.0*y'],
                                            y=np.zeros(size),
                                            z=np.zeros(size)))

        prob.model.linear_solver = om.DirectSolver()
        group.connect('P.x', 'sub.C1.x')
        group.connect('sub.C1.y', 'sub.C2.y')

        prob.setup(check=False, mode='fwd')
        with self.assertRaises(Exception) as cm:
            prob.run_model()

        msg = "DirectSolver linear solver in <model> <class Group> cannot be used in or above a ParallelGroup or a " + \
            "distributed component."
        self.assertEqual(str(cm.exception), msg)

    def test_par_direct_subbed(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 1.0))
        model.add_subsystem('p2', om.IndepVarComp('x', 1.0))

        parallel = model.add_subsystem('parallel', om.ParallelGroup())
        parallel.add_subsystem('c1', om.ExecComp(['y=-2.0*x']))
        parallel.add_subsystem('c2', om.ExecComp(['y=5.0*x']))

        model.add_subsystem('c3', om.ExecComp(['y=3.0*x1+7.0*x2']))

        model.connect("parallel.c1.y", "c3.x1")
        model.connect("parallel.c2.y", "c3.x2")

        model.connect("p1.x", "parallel.c1.x")
        model.connect("p2.x", "parallel.c2.x")

        model.linear_solver = om.DirectSolver()

        prob.setup(check=False, mode='fwd')
        with self.assertRaises(Exception) as cm:
            prob.run_model()

        msg = "DirectSolver linear solver in <model> <class Group> cannot be used in or above a ParallelGroup or a " + \
            "distributed component."
        self.assertEqual(str(cm.exception), msg)

    def test_par_direct(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('P', om.IndepVarComp('x', 1.0))
        par = model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('C1', om.ExecComp(['y=2.0*x']))
        par.add_subsystem('C2', om.ExecComp(['z=3.0*y']))

        model.linear_solver = om.DirectSolver()
        model.connect('P.x', 'par.C1.x')
        model.connect('P.x', 'par.C2.y')

        prob.setup()
        with self.assertRaises(Exception) as cm:
            prob.run_model()

        msg = "DirectSolver linear solver in <model> <class Group> cannot be used in or above a ParallelGroup or a " + \
            "distributed component."
        self.assertEqual(str(cm.exception), msg)



class TestDirectSolverFeature(unittest.TestCase):

    def test_specify_solver(self):

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem()
        model = prob.model = SellarDerivatives()

        model.linear_solver = om.DirectSolver()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_near_equal(J['obj', 'z'][0][1], 1.78448534, .00001)


class TestDirectSolverMPI(unittest.TestCase):

    N_PROCS = 2

    def test_serial_in_mpi(self):
        # Tests that we can take an MPI model with a DirectSolver and run it in mpi with more
        # procs. This verifies fix of a bug.

        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = om.DirectSolver(assemble_jac=True)
        g1.options['assembled_jac_type'] = 'dense'

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = om.DirectSolver(assemble_jac=True)
        g2.options['assembled_jac_type'] = 'dense'

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        model.nonlinear_solver.options['solve_subsystems'] = True

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, 1.0e-5)
        assert_near_equal(prob['g1.y2'], 0.80, 1.0e-5)
        assert_near_equal(prob['g2.y1'], 0.64, 1.0e-5)
        assert_near_equal(prob['g2.y2'], 0.80, 1.0e-5)


if __name__ == "__main__":
    unittest.main()
