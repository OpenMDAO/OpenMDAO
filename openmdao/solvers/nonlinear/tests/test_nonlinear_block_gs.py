"""Test the Nonlinear Block Gauss Seidel solver. """

import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.sellar import SellarDerivatives, \
    SellarDis1withDerivatives, SellarDis2withDerivatives, \
    SellarDis1, SellarDis2
from openmdao.utils.assert_utils import assert_near_equal

from openmdao.utils.mpi import MPI
try:
    from openmdao.api import PETScVector
except Exception:
    PETScVector = None


class TestNLBGaussSeidel(unittest.TestCase):

    def test_reraise_error(self):

        prob = om.Problem(model=DoubleSellar())
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NonlinearBlockGS()
        g1.nonlinear_solver.options['maxiter'] = 1
        g1.nonlinear_solver.options['err_on_non_converge'] = True
        g1.linear_solver = om.DirectSolver(assemble_jac=True)

        g2 = model.g2
        g2.nonlinear_solver = om.NonlinearBlockGS()
        g2.nonlinear_solver.options['maxiter'] = 1
        g2.nonlinear_solver.options['err_on_non_converge'] = True
        g2.linear_solver = om.DirectSolver(assemble_jac=True)

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.linear_solver = om.DirectSolver(assemble_jac=True)
        model.nonlinear_solver.options['err_on_non_converge'] = True
        model.nonlinear_solver.options['reraise_child_analysiserror'] = True

        prob.setup()

        with self.assertRaises(om.AnalysisError) as context:
            prob.run_model()

        msg = "Solver 'NL: NLBGS' on system 'g1' failed to converge in 1 iterations."
        self.assertEqual(str(context.exception), msg)

    def test_feature_set_options(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        nlbgs = model.nonlinear_solver = om.NonlinearBlockGS()

        nlbgs.options['maxiter'] = 20
        nlbgs.options['atol'] = 1e-6
        nlbgs.options['rtol'] = 1e-6

        prob.setup()
        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_feature_maxiter(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        prob.setup()
        nlbgs = model.nonlinear_solver = om.NonlinearBlockGS()

        #basic test of number of iterations
        nlbgs.options['maxiter'] = 1
        prob.run_model()
        self.assertEqual(model.nonlinear_solver._iter_count, 1)

        nlbgs.options['maxiter'] = 5
        prob.run_model()
        self.assertEqual(model.nonlinear_solver._iter_count, 5)

        #test of number of iterations AND solution after exit at maxiter
        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        nlbgs.options['maxiter'] = 3
        prob.set_solver_print()
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58914915, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05857185, .00001)
        self.assertEqual(model.nonlinear_solver._iter_count, 3)

    def test_feature_rtol(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        nlbgs = model.nonlinear_solver = om.NonlinearBlockGS()
        nlbgs.options['rtol'] = 1e-3

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.5883027, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_feature_atol(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        nlbgs = model.nonlinear_solver = om.NonlinearBlockGS()
        nlbgs.options['atol'] = 1e-4

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.5882856302, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_sellar(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        nlbgs = model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertEqual(model.nonlinear_solver._iter_count, 8)

        # Only one extra execution
        self.assertEqual(model.d1.execution_count, 8)

        # With run_apply_linear, we execute the components more times.

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

        nlbgs = model.nonlinear_solver = om.NonlinearBlockGS()
        nlbgs.options['use_apply_nonlinear'] = True

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertEqual(model.nonlinear_solver._iter_count, 7)

        # Nearly double the executions.
        self.assertEqual(model.d1.execution_count, 15)

    def test_sellar_analysis_error(self):
        # Tests Sellar behavior when AnalysisError is raised.

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

        nlbgs = model.nonlinear_solver = om.NonlinearBlockGS()
        nlbgs.options['maxiter'] = 2
        nlbgs.options['err_on_non_converge'] = True

        prob.setup()
        prob.set_solver_print(level=0)

        try:
            prob.run_model()
        except om.AnalysisError as err:
            self.assertEqual(str(err), "Solver 'NL: NLBGS' on system '' failed to converge in 2 iterations.")
        else:
            self.fail("expected AnalysisError")

    def test_sellar_group_nested(self):
        # This tests true nested gs. Subsolvers solve each Sellar system. Top
        # solver couples them together through variable x.

        # This version has the indepvarcomps removed so we can connect them together.
        class SellarModified(om.Group):
            """ Group containing the Sellar MDA. This version uses the disciplines
            with derivatives."""

            def __init__(self):
                super().__init__()

                self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
                self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

                self.nonlinear_solver = om.NonlinearBlockGS()
                self.linear_solver = om.ScipyKrylov()

        prob = om.Problem()
        root = prob.model
        root.nonlinear_solver = om.NonlinearBlockGS()
        root.nonlinear_solver.options['maxiter'] = 20
        root.add_subsystem('g1', SellarModified())
        root.add_subsystem('g2', SellarModified())

        root.connect('g1.y2', 'g2.x')
        root.connect('g2.y2', 'g1.x')

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

    def test_NLBGS_Aitken(self):

        prob = om.Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()
        model.nonlinear_solver.options['use_aitken'] = True
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)
        self.assertTrue(model.nonlinear_solver._iter_count == 6)

        #check that the relaxation factor is updated correctly
        assert_near_equal(model.nonlinear_solver._theta_n_1, 1.00, 0.001)

    def test_NLBGS_Aitken_initial_factor(self):

        prob = om.Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()

        model.nonlinear_solver.options['use_aitken'] = True
        model.nonlinear_solver.options['aitken_initial_factor'] = 0.33
        model.nonlinear_solver.options['maxiter'] = 1
        prob.run_model()
        self.assertTrue(model.nonlinear_solver._theta_n_1 == 0.33)


        model.nonlinear_solver.options['maxiter'] = 14
        prob.run_model()

        # should converge to the same solution
        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        # in more iterations
        self.assertTrue(model.nonlinear_solver._iter_count == 12)

        #check that the relaxation factor is updated correctly (should tend towards 1)
        self.assertGreater(model.nonlinear_solver._theta_n_1, 0.99)

    def test_NLBGS_Aitken_min_max_factor(self):

        prob = om.Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()

        model.nonlinear_solver.options['use_aitken'] = True
        model.nonlinear_solver.options['aitken_min_factor'] = 1.2
        model.nonlinear_solver.options['maxiter'] = 1
        prob.run_model()
        self.assertTrue(model.nonlinear_solver._theta_n_1 == 1.2)

        model.nonlinear_solver.options['aitken_max_factor'] = 0.7
        model.nonlinear_solver.options['aitken_min_factor'] = 0.1

        model.nonlinear_solver.options['maxiter'] = 1
        prob.run_model()
        self.assertTrue(model.nonlinear_solver._theta_n_1 == 0.7)

    def test_NLBGS_Aitken_cs(self):

        prob = om.Problem(model=SellarDerivatives(nonlinear_solver=om.NonlinearBlockGS))

        model = prob.model
        model.approx_totals(method='cs', step=1e-10)

        prob.setup()
        prob.set_solver_print(level=2)
        model.nonlinear_solver.options['use_aitken'] = True
        model.nonlinear_solver.options['atol'] = 1e-15
        model.nonlinear_solver.options['rtol'] = 1e-15

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        J = prob.compute_totals(of=['y1'], wrt=['x'])
        assert_near_equal(J['y1', 'x'][0][0], 0.98061448, 1e-6)

    def test_aitken_bug(self):
        class Spring(om.ExplicitComponent):
            def setup(self):
                self.add_input('dz')
                self.add_output('f')
                self.z0 = 0.0

            def compute(self, inputs, outputs):
                outputs['f'] = inputs['dz'] - self.z0

        class Forcer(om.ExplicitComponent):
            def setup(self):
                self.add_input('f')
                self.add_output('dz', val=0.0)

            def compute(self, inputs, outputs):
                f = inputs['f']
                if f > 1.0:
                    raise RuntimeError("Aitken should have prevented this.")
                outputs['dz'] = -10.0 * f + 3.5
                print('dz', outputs['dz'])

        class Coupled(om.Group):
            def setup(self):
                self.add_subsystem('spring', Spring())
                self.add_subsystem('forcer', Forcer())
                self.connect('spring.f', 'forcer.f')
                self.connect('forcer.dz', 'spring.dz')

        def create_model():
            model = om.Group()

            nonlinear_solver = om.NonlinearBlockGS(
                maxiter=250,
                iprint=2,
                use_aitken=True,
                aitken_min_factor=0.1,
                aitken_max_factor=1.0,
                aitken_initial_factor=0.1,
                rtol=1e-7,
                atol=1e-8,
            )

            coupling = model.add_subsystem('coupled_springs', Coupled())
            coupling.nonlinear_solver = nonlinear_solver
            return model

        prob = om.Problem()
        prob.model = create_model()
        prob.setup()
        prob.set_solver_print(0)

        # Will raise an exception if the bug is present.
        prob.run_model()

    def test_NLBGS_cs(self):

        prob = om.Problem(model=SellarDerivatives(nonlinear_solver=om.NonlinearBlockGS))

        model = prob.model
        model.approx_totals(method='cs')

        prob.setup()
        prob.set_solver_print(level=0)
        model.nonlinear_solver.options['atol'] = 1e-15
        model.nonlinear_solver.options['rtol'] = 1e-15

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        J = prob.compute_totals(of=['y1'], wrt=['x'])
        assert_near_equal(J['y1', 'x'][0][0], 0.98061448, 1e-6)

    def test_res_ref(self):

        class ContrivedSellarDis1(SellarDis1):

            def setup(self):
                super().setup()
                self.add_output('highly_nonlinear', val=1.0, res_ref=1e-4)
            def compute(self, inputs, outputs):
                super().compute(inputs, outputs)
                outputs['highly_nonlinear'] = 10*np.sin(10*inputs['y2'])

        p = om.Problem()
        model = p.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', ContrivedSellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

        nlbgs = model.nonlinear_solver = om.NonlinearBlockGS()

        nlbgs.options['maxiter'] = 20
        nlbgs.options['atol'] = 1e-6
        nlbgs.options['rtol'] = 1e-100

        p.setup()
        p.run_model()

        self.assertEqual(nlbgs._iter_count, 10, 'res_ref should make this take more iters.')

    def test_guess_nonlinear(self):
        class SmartGroup(om.Group):

            def setup(self):
                self.add_subsystem('c1', om.ExecComp('y = 2.7951 + 10.56*x**2 - 5.4*x**3 + 0.5*x**4'), promotes=['*'])
                self.add_subsystem('c2', om.ExecComp('x = y/8.954'), promotes=['*'])

                self.nonlinear_solver = om.NonlinearBlockGS()
                self.nonlinear_solver.options['maxiter'] = 100
                self.nonlinear_solver.options['atol'] = 1e-6

            def guess_nonlinear(self, inputs, outputs, residuals):
                x = outputs['x']
                y = outputs['y']

                if np.abs(x) > 1.0 or np.abs(y) > 10.0:
                    # Pull out of divergence zone.
                    x = outputs['x'] = 0.5
                    outputs['y'] = 2.7951 + 10.56*x**2 - 5.4*x**3 + 0.5*x**4

        prob = om.Problem(model=SmartGroup())

        prob.setup()
        prob.set_solver_print(level=0)

        # This will mess things up. Only guess_nonlinear can save us.
        prob['y'] = 1000.0
        prob['x'] = 1000.0

        prob.run_model()

        assert_near_equal(prob['x'], 0.67883021, 1e-5)


class SquareComp(om.ExplicitComponent):
    """Computes y = x**2 with analytic partials."""

    def setup(self):
        self.add_input('x', 22.0)
        self.add_output('y', 1.0)
        self.declare_partials('y', 'x')

    def compute(self, inputs, outputs):
        outputs['y'] = inputs['x'] ** 2

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = 2.0 * inputs['x']


class CoupledComp1(om.ExplicitComponent):
    """Computes y1 = x + 0.3*y2, half of a coupled pair with contraction ~0.6 per sweep."""

    def setup(self):
        self.add_input('x', 22.0)
        self.add_input('y2', 1.0)
        self.add_output('y1', 1.0)
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['y1'] = inputs['x'] + 0.3 * inputs['y2']

    def compute_partials(self, inputs, partials):
        partials['y1', 'x'] = 1.0
        partials['y1', 'y2'] = 0.3


class CoupledComp2(om.ExplicitComponent):
    """Computes y2 = 2*y1 + 0.1*x, the other half of the coupled pair."""

    def setup(self):
        self.add_input('x', 22.0)
        self.add_input('y1', 1.0)
        self.add_output('y2', 1.0)
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['y2'] = 2.0 * inputs['y1'] + 0.1 * inputs['x']

    def compute_partials(self, inputs, partials):
        partials['y2', 'y1'] = 2.0
        partials['y2', 'x'] = 0.1


class VecSquareComp(om.ExplicitComponent):
    """Computes y = x**2 elementwise for a length-3 vector, with analytic partials."""

    def setup(self):
        self.add_input('x', np.ones(3))
        self.add_output('y', np.ones(3))
        self.declare_partials('y', 'x', rows=np.arange(3), cols=np.arange(3))

    def compute(self, inputs, outputs):
        outputs['y'] = inputs['x'] ** 2

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = 2.0 * inputs['x']


class SquareZComp(om.ExplicitComponent):
    """Computes lhs = z**2 with analytic partials, used as the balance left-hand side."""

    def setup(self):
        self.add_input('z', 4.0)
        self.add_output('lhs', 16.0)
        self.declare_partials('lhs', 'z')

    def compute(self, inputs, outputs):
        outputs['lhs'] = inputs['z'] ** 2

    def compute_partials(self, inputs, partials):
        partials['lhs', 'z'] = 2.0 * inputs['z']


class TestNLBGSComplexStepReconverge(unittest.TestCase):
    """
    Regression tests for issue #3800.

    The cs_reconverge nudge must not translate the real part of IndepVarComp or auto_ivc
    outputs, because no Gauss-Seidel iteration recomputes them and the complex step total
    would then be evaluated at the wrong real point.
    """

    def _build_feedforward(self, irrelevant, use_aitken=False):
        prob = om.Problem()
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('x', 22.0)
        ivc.add_output('irrelevant', irrelevant)
        prob.model.add_subsystem('sq', SquareComp(), promotes=['*'])
        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.nonlinear_solver.options['use_aitken'] = use_aitken
        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=0)
        prob.run_model()
        return prob

    def _cs_total(self, prob, of='y', wrt='x'):
        data = prob.check_totals(of=[of], wrt=[wrt], method='cs', step=1e-40,
                                 out_stream=None)
        return data[of, wrt]['J_fd'].item()

    def test_cs_totals_at_declared_point(self):
        # dy/dx of y = x**2 at x = 22 is exactly 44.  On the unfixed code the nudge
        # shifts x by norm(outputs)*1e-10 and the error is ~9.7e-8.
        prob = self._build_feedforward(0.0)
        J = self._cs_total(prob)
        self.assertLess(abs(J - 44.0), 1e-10)

    def test_cs_totals_disconnected_output_invariance(self):
        # A disconnected IndepVarComp output must not influence dy/dx.  On the unfixed
        # code it inflates the nudge norm, giving errors up to 2.0 for irrelevant=1e10.
        vals = []
        for irrelevant in (0.0, 1.0, 1e6, 1e8, 1e10):
            prob = self._build_feedforward(irrelevant)
            J = self._cs_total(prob)
            self.assertLess(abs(J - 44.0), 1e-10,
                            msg=f'irrelevant={irrelevant}: J={J}')
            vals.append(J)
        self.assertEqual(len(set(vals)), 1,
                         msg=f'derivative varies with disconnected output: {vals}')

    def test_cs_totals_auto_ivc(self):
        # Same contract when the independent variable lives on the auto_ivc.
        prob = om.Problem()
        prob.model.add_subsystem('sq', SquareComp(), promotes=['*'])
        prob.model.add_subsystem('oth', om.ExecComp('z = 2.0*w'), promotes=['*'])
        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=0)
        prob.set_val('x', 22.0)
        prob.set_val('w', 1e8)
        prob.run_model()
        J = self._cs_total(prob)
        self.assertLess(abs(J - 44.0), 1e-10)

    def test_cs_totals_vector_indep_var(self):
        # Every element of a vector-valued independent output must be protected, not
        # just the first: diag(dy/dx) for y = x**2 at x = [3, 22, 5] is [6, 44, 10].
        # A mask that covered only the first element of each independent variable
        # would leave the other two translated and fail here.
        x = np.array([3.0, 22.0, 5.0])
        prob = om.Problem()
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('x', x)
        ivc.add_output('irrelevant', 1e8)
        prob.model.add_subsystem('sq', VecSquareComp(), promotes=['*'])
        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=0)
        prob.run_model()
        data = prob.check_totals(of=['y'], wrt=['x'], method='cs', step=1e-40,
                                 out_stream=None)
        J = data['y', 'x']['J_fd']
        assert_near_equal(J, np.diag(2.0 * x), 1e-10)

    def test_cs_totals_aitken(self):
        # The Aitken path shares the nudge and must satisfy the same contract.
        prob = self._build_feedforward(1e8, use_aitken=True)
        J = self._cs_total(prob)
        self.assertLess(abs(J - 44.0), 1e-10)

    def test_cs_totals_coupled_still_reconverges(self):
        # The nudge exists so that the imaginary fixed-point iteration actually runs in
        # coupled models.  With the solver-state nudge intact and tight tolerances the
        # complex step total matches the analytic 2.575; if the nudge were removed
        # entirely the solver would stop after two sweeps and the error would be ~0.3.
        prob = om.Problem()
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('x', 22.0)
        ivc.add_output('irrelevant', 1e8)
        prob.model.add_subsystem('c1', CoupledComp1(), promotes=['*'])
        prob.model.add_subsystem('c2', CoupledComp2(), promotes=['*'])
        prob.model.nonlinear_solver = om.NonlinearBlockGS(maxiter=500, atol=1e-14,
                                                          rtol=1e-14)
        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=0)
        prob.run_model()
        # y1 = x + 0.3*(2*y1 + 0.1*x)  =>  y1 = 2.575*x
        J = self._cs_total(prob, of='y1')
        self.assertLess(abs(J - 2.575), 1e-4)

    def test_cs_totals_coupled_disconnected_invariance(self):
        # In a coupled model the disconnected output must not influence the derivative
        # even indirectly: if it entered the nudge sizing norm it would change the
        # iteration count and therefore the converged imaginary part.
        vals = []
        for irrelevant in (0.0, 1e8):
            prob = om.Problem()
            ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
            ivc.add_output('x', 22.0)
            ivc.add_output('irrelevant', irrelevant)
            prob.model.add_subsystem('c1', CoupledComp1(), promotes=['*'])
            prob.model.add_subsystem('c2', CoupledComp2(), promotes=['*'])
            prob.model.nonlinear_solver = om.NonlinearBlockGS(maxiter=500)
            prob.setup(force_alloc_complex=True)
            prob.set_solver_print(level=0)
            prob.run_model()
            vals.append(self._cs_total(prob, of='y1'))
        self.assertEqual(vals[0], vals[1],
                         msg=f'coupled derivative varies with disconnected output: {vals}')

    def test_cs_totals_implicit_state_reconverges(self):
        # A genuine implicit state below the NLBGS must still be nudged and reconverge:
        # z solves z**2 = x, so dz/dx = 1/(2*sqrt(x)).  On the unfixed code a
        # disconnected 1e8 output shifts x and the error is ~2.4e-5.
        prob = om.Problem()
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('x', 22.0)
        ivc.add_output('irrelevant', 1e8)
        sub = prob.model.add_subsystem('sub', om.Group(), promotes=['*'])
        bal = om.BalanceComp()
        bal.add_balance('z', lhs_name='lhs', rhs_name='x', val=4.0)
        sub.add_subsystem('bal', bal, promotes=['*'])
        sub.add_subsystem('sqc', SquareZComp(), promotes=['*'])
        sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=30)
        sub.linear_solver = om.DirectSolver()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=0)
        prob.run_model()
        J = self._cs_total(prob, of='z')
        self.assertLess(abs(J - 1.0 / (2.0 * np.sqrt(22.0))), 1e-8)

    def test_cs_reconverge_nudge_semantics(self):
        # White-box check of the nudge itself: under complex step it must leave every
        # imaginary part untouched, leave outputs tagged 'openmdao:indep_var' untouched,
        # and shift the remaining outputs by exactly norm(those outputs) * 1e-10.
        prob = self._build_feedforward(1e8)
        model = prob.model
        solver = model.nonlinear_solver
        model._set_complex_step_mode(True)
        try:
            arr = model._outputs.asarray()
            idx = {name: model._outputs.get_range(name)[0]
                   for name in model._outputs._abs_iter()}
            arr[idx['ivc.x']] += 1e-40j       # the perturbation a wrt would carry
            arr[idx['sq.y']] += 3e-41j        # in-flight imaginary data on a state
            before = arr.copy()
            expected_nudge = np.linalg.norm(before[[idx['sq.y']]]) * 1e-10

            solver._iter_initialize()

            after = model._outputs.asarray()
            self.assertTrue(np.array_equal(after.imag, before.imag),
                            msg='nudge must not modify imaginary parts')
            self.assertEqual((after[idx['ivc.x']] - before[idx['ivc.x']]).real, 0.0,
                             msg='indep var was translated by the nudge')
            self.assertEqual((after[idx['ivc.irrelevant']] -
                              before[idx['ivc.irrelevant']]).real, 0.0,
                             msg='indep var was translated by the nudge')
            # exact up to one rounding of (y + nudge) in float64
            delta = (after[idx['sq.y']] - before[idx['sq.y']]).real
            self.assertLess(abs(delta - expected_nudge), 1e-6 * expected_nudge,
                            msg='solver-state nudge missing or mis-sized')
        finally:
            model._set_complex_step_mode(False)

    def test_cs_totals_repeated_calls_deterministic(self):
        # Repeated and interleaved cs/fd evaluations must not contaminate each other.
        prob = self._build_feedforward(1e8)
        first = self._cs_total(prob)
        prob.check_totals(of=['y'], wrt=['x'], method='fd', out_stream=None)
        for _ in range(3):
            self.assertEqual(self._cs_total(prob), first)
        self.assertLess(abs(first - 44.0), 1e-10)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProcTestCase1(unittest.TestCase):

    N_PROCS = 2

    def test_aitken(self):

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        p1 = model.add_subsystem('p1', om.ParallelGroup(), promotes=['*'])
        p1.add_subsystem('d1a', SellarDis1withDerivatives(), promotes=['x', 'z'])
        p1.add_subsystem('d1b', SellarDis1withDerivatives(), promotes=['x', 'z'])

        p2 = model.add_subsystem('p2', om.ParallelGroup(), promotes=['*'])
        p2.add_subsystem('d2a', SellarDis2withDerivatives(), promotes=['z'])
        p2.add_subsystem('d2b', SellarDis2withDerivatives(), promotes=['z'])

        model.connect('d1a.y1', 'd2a.y1')
        model.connect('d1b.y1', 'd2b.y1')
        model.connect('d2a.y2', 'd1a.y2')
        model.connect('d2b.y2', 'd1b.y2')

        model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()
        prob.set_solver_print(level=2)
        model.nonlinear_solver.options['use_aitken'] = True

        # Set one branch of Sellar close to the solution.
        prob.set_val('d2b.y2', 12.05848815)
        prob.set_val('d1b.y1', 25.58830237)

        prob.run_model()

        print(prob.get_val('d1a.y1', get_remote=True))
        print(prob.get_val('d2a.y1', get_remote=True))
        print(prob.get_val('d1b.y2', get_remote=True))
        print(prob.get_val('d2b.y2', get_remote=True))

        assert_near_equal(prob.get_val('d1a.y1', get_remote=True), 25.58830273, .00001)
        assert_near_equal(prob.get_val('d1b.y1', get_remote=True), 25.58830273, .00001)
        assert_near_equal(prob.get_val('d2a.y2', get_remote=True), 12.05848819, .00001)
        assert_near_equal(prob.get_val('d2b.y2', get_remote=True), 12.05848819, .00001)

        # Test that Aitken accelerated the convergence, normally takes 7.
        self.assertTrue(model.nonlinear_solver._iter_count == 6)

    def test_nonlinear_analysis_error(self):

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        p1 = model.add_subsystem('p1', om.ParallelGroup(), promotes=['*'])
        p1.add_subsystem('d1a', SellarDis1withDerivatives(), promotes=['x', 'z'])
        p1.add_subsystem('d1b', SellarDis1withDerivatives(), promotes=['x', 'z'])

        p2 = model.add_subsystem('p2', om.ParallelGroup(), promotes=['*'])
        p2.add_subsystem('d2a', SellarDis2withDerivatives(), promotes=['z'])
        p2.add_subsystem('d2b', SellarDis2withDerivatives(), promotes=['z'])

        model.connect('d1a.y1', 'd2a.y1')
        model.connect('d1b.y1', 'd2b.y1')
        model.connect('d2a.y2', 'd1a.y2')
        model.connect('d2b.y2', 'd1b.y2')

        model.nonlinear_solver = om.NonlinearBlockGS(maxiter=2, err_on_non_converge=True)

        prob.setup()
        prob.set_solver_print(level=2)

        # Set one branch of Sellar close to the solution.
        prob.set_val('d2b.y2', 12.05848815)
        prob.set_val('d1b.y1', 25.58830237)

        # test if the analysis error is raised properly on all procs
        try:
            prob.run_model()
        except om.AnalysisError as err:
            self.assertEqual(str(err), "Solver 'NL: NLBGS' on system '' failed to converge in 2 iterations.")
        else:
            self.fail("expected AnalysisError")


if __name__ == "__main__":
    unittest.main()
