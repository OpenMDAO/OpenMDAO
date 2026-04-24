"""Unit tests for the pymooDriver."""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.branin import Branin, BraninDiscrete
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

try:
    import cma  # noqa: F401
    # Verify cma is compatible with the installed NumPy version.
    # cma < 3.4 uses np.array(..., copy=False) which raises ValueError on NumPy >= 2.0.
    from cma.transformations import BoxConstraintsLinQuadTransformation
    BoxConstraintsLinQuadTransformation([[0.0, 1.0]])
    CMA_INSTALLED = True
except Exception:
    CMA_INSTALLED = False


# Check whether pymoo is installed.
try:
    import pymoo  # noqa: F401
    from openmdao.drivers.pymoo_driver import pymooDriver
    PYMOO_INSTALLED = True
except ImportError:
    PYMOO_INSTALLED = False


@unittest.skipUnless(not PYMOO_INSTALLED, 'only run if pymoo is NOT installed.')
@use_tempdirs
class TestNotInstalled(unittest.TestCase):
    """Test behavior when pymoo is not available."""

    def test_pymoo_not_installed(self):
        """
        Test that pymooDriver raises RuntimeError when pymoo is not installed.

        When pymoo is not available, attempting to instantiate pymooDriver
        should raise a RuntimeError with a message that includes 'pymoo'.
        """
        from openmdao.drivers.pymoo_driver import pymooDriver

        with self.assertRaises(RuntimeError) as ctx:
            pymooDriver()

        self.assertIn('pymoo', str(ctx.exception))


@unittest.skipUnless(MPI, 'MPI is required.')
@unittest.skipUnless(PYMOO_INSTALLED, 'pymoo is not installed.')
@use_tempdirs
class TestMPIIsolatedParallelism(unittest.TestCase):
    """
    Test population parallelism and model parallelism independently, one at a time.

    With two ranks only one mode can be active: either n_groups=2/procs_per_model=1
    (population parallelism) or n_groups=1/procs_per_model=2 (model parallelism).
    Tests exercising both simultaneously require four ranks and are in
    TestMPICombinedParallelism.
    """

    N_PROCS = 2

    def test_run_parallel_branin(self):
        """
        Test population parallelism with procs_per_model=1 (n_groups=2).

        Each rank runs its own model independently. Odd pop_size gives each group
        a different individual count per generation, so iter_counts must differ.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['run_parallel'] = True
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 25  # odd: unequal split across 2 groups
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 200)

        prob.setup()
        prob.run_driver()

        self.assertEqual(prob.comm.size, 2)
        self.assertEqual(prob.driver._problem_comm.size, 2)
        self.assertEqual(prob.model.comm.size, 1)

        # iter_counts differ because each rank evaluated a different subset.
        all_iters = prob.comm.allgather(prob.driver.iter_count)
        self.assertGreater(max(all_iters), min(all_iters),
                           f'expected unequal iter_counts, got {all_iters}')

        all_f = prob.comm.allgather(prob.get_val('comp.f').item())
        for f_val in all_f:
            assert_near_equal(f_val, all_f[0], tolerance=1e-10)
        assert_near_equal(all_f[0], 0.397887, tolerance=0.1)

    @unittest.skipUnless(PETScVector, 'PETSc is required.')
    def test_procs_per_model_parallel_group(self):
        """
        Test model parallelism with procs_per_model=2 (n_groups=1).

        Both ranks share a sub-communicator and cooperate on every evaluation
        via a ParallelGroup. Equal iter_counts confirm no population splitting.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', om.IndepVarComp('xI', 0.0))
        par = model.add_subsystem('par', om.ParallelGroup())

        par.add_subsystem('comp1', Branin())
        par.add_subsystem('comp2', Branin())

        model.connect('p2.xI', 'par.comp1.x0')
        model.connect('p1.xC', 'par.comp1.x1')
        model.connect('p2.xI', 'par.comp2.x0')
        model.connect('p1.xC', 'par.comp2.x1')

        model.add_subsystem('comp', om.ExecComp('f = f1 + f2'))
        model.connect('par.comp1.f', 'comp.f1')
        model.connect('par.comp2.f', 'comp.f2')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 25
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 200)

        prob.setup()
        prob.run_driver()

        self.assertEqual(prob.comm.size, 2)
        self.assertEqual(prob.driver._problem_comm.size, 2)
        self.assertEqual(prob.model.comm.size, 2)

        all_iters = prob.comm.allgather(prob.driver.iter_count)
        self.assertEqual(all_iters[0], all_iters[1],
                         f'expected equal iter_counts, got {all_iters}')

        assert_near_equal(prob.get_val('comp.f'), 2 * 0.397887, tolerance=0.1)


@unittest.skipUnless(MPI and PETScVector, 'MPI and PETSc are required.')
@unittest.skipUnless(PYMOO_INSTALLED, 'pymoo is not installed.')
@use_tempdirs
class TestMPICombinedParallelism(unittest.TestCase):
    """
    Test population parallelism and model parallelism active simultaneously.

    Four ranks (2 groups x 2 procs/model) is the minimum needed for both modes
    at once. Tests that isolate one mode at a time are in TestMPIIsolatedParallelism.
    """

    N_PROCS = 4

    def test_run_parallel_with_procs_per_model(self):
        """
        Test population parallelism and model parallelism simultaneously.

        Two evaluation groups (color = rank % 2) each use a 2-rank sub-communicator
        for a ParallelGroup model. Odd pop_size gives groups unequal work so
        within-group iter_counts match and across-group iter_counts differ.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', om.IndepVarComp('xI', 0.0))
        par = model.add_subsystem('par', om.ParallelGroup())

        par.add_subsystem('comp1', Branin())
        par.add_subsystem('comp2', Branin())

        model.connect('p2.xI', 'par.comp1.x0')
        model.connect('p1.xC', 'par.comp1.x1')
        model.connect('p2.xI', 'par.comp2.x0')
        model.connect('p1.xC', 'par.comp2.x1')

        model.add_subsystem('comp', om.ExecComp('f = f1 + f2'))
        model.connect('par.comp1.f', 'comp.f1')
        model.connect('par.comp2.f', 'comp.f2')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 25  # odd: unequal split across 2 groups
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 100)

        prob.setup()
        prob.run_driver()

        self.assertEqual(prob.comm.size, 4)
        self.assertEqual(prob.driver._problem_comm.size, 4)
        self.assertEqual(prob.model.comm.size, 2)

        # color = rank % 2: group 0 = ranks 0,2; group 1 = ranks 1,3
        all_iters = prob.comm.allgather(prob.driver.iter_count)
        self.assertEqual(all_iters[0], all_iters[2],
                         f'expected equal iter_counts within group 0, got {all_iters}')
        self.assertEqual(all_iters[1], all_iters[3],
                         f'expected equal iter_counts within group 1, got {all_iters}')
        self.assertNotEqual(all_iters[0], all_iters[1],
                            f'expected unequal iter_counts across groups, got {all_iters}')

        all_f = prob.comm.allgather(prob.get_val('comp.f').item())
        for f_val in all_f:
            assert_near_equal(f_val, all_f[0], tolerance=1e-10)
        assert_near_equal(all_f[0], 2 * 0.397887, tolerance=0.1)

    def test_indivisible_procs_raises(self):
        """Test that N_PROCS not evenly divisible by procs_per_model raises RuntimeError."""
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('par', om.ParallelGroup())

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 3
        prob.driver.options['disp'] = False

        with self.assertRaises(RuntimeError) as ctx:
            prob.setup()

        self.assertIn('procs_per_model', str(ctx.exception))
        self.assertIn('3', str(ctx.exception))


@unittest.skipUnless(PYMOO_INSTALLED, 'pymoo is not installed.')
@use_tempdirs
class TestPymooDriver(unittest.TestCase):
    """Tests for the pymooDriver."""

    def test_unconstrained_single_variable(self):
        """
        Test unconstrained single-objective optimization on a simple quadratic.

        Minimizes f = (x-3)^2, which has a known global minimum of 0 at x=3.
        Verifies that GA finds a solution near the true optimum.
        """
        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('f = (x-3.0)**2'),
                                 promotes=['*'])
        prob.model.add_design_var('x', lower=-10.0, upper=10.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 20
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 200)

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob.get_val('x'), 3.0, tolerance=0.05)
        assert_near_equal(prob.get_val('f'), 0.0, tolerance=0.01)

    def test_upper_bound_constraint(self):
        """
        Test optimization with an active upper bound inequality constraint.

        Minimizes the Paraboloid subject to c = -x + y <= -15 using GA.
        The known constrained optimum is x=7.167, y=-7.833. Verifies that
        the constraint is satisfied and the solution is near the known optimum.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = -x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 50
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 500)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()
        prob.run_driver()

        self.assertLessEqual(prob.get_val('c'), -15.0 + 0.5)
        assert_near_equal(prob.get_val('x'), 7.16667, tolerance=0.05)
        assert_near_equal(prob.get_val('y'), -7.833334, tolerance=0.05)

    def test_lower_bound_constraint(self):
        """
        Test optimization with an active lower bound inequality constraint.

        Minimizes the Paraboloid subject to c = x - y >= 15 using GA.
        The known constrained optimum is x=7.167, y=-7.833. Verifies that
        the lower bound constraint is satisfied at the solution.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 50
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 500)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0)

        prob.setup()
        prob.run_driver()

        self.assertGreaterEqual(prob.get_val('c'), 15.0 - 0.5)
        assert_near_equal(prob.get_val('x'), 7.16667, tolerance=0.05)
        assert_near_equal(prob.get_val('y'), -7.833334, tolerance=0.05)

    def test_two_sided_constraint(self):
        """
        Test optimization with a two-sided inequality constraint.

        Each side of a two-sided bound is converted into a separate pymoo
        inequality constraint. Verifies both bounds are enforced and the
        solution remains feasible.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp',
                            om.ExecComp('f = (x-3.0)**2 + (y-2.0)**2'),
                            promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 30
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 300)

        model.add_design_var('x', lower=0.0, upper=6.0)
        model.add_design_var('y', lower=0.0, upper=6.0)
        model.add_objective('f')
        model.add_constraint('c', lower=4.0, upper=6.0)

        prob.setup()
        prob.run_driver()

        c = prob.get_val('c')
        self.assertGreaterEqual(c, 4.0 - 0.5)
        self.assertLessEqual(c, 6.0 + 0.5)

    def test_equality_constraint(self):
        """
        Test optimization with an equality constraint.

        Minimizes f = (x-5)^2 + (y-5)^2 subject to x + y == 8.
        The equality constraint is converted to the pymoo form h == 0.
        The known constrained optimum is x=y=4 with f=2.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp',
                            om.ExecComp('f = (x-5.0)**2 + (y-5.0)**2'),
                            promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 50
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 400)

        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_design_var('y', lower=0.0, upper=10.0)
        model.add_objective('f')
        model.add_constraint('c', equals=8.0)

        prob.setup()
        prob.run_driver()

        # pymoo uses cv_eq tolerance (default 1e-4) when checking feasibility.
        assert_near_equal(prob.get_val('c'), 8.0, tolerance=0.1)

    def test_moo_problem_dimensions(self):
        """
        Test that n_obj, n_ieq_constr, and n_eq_constr on _moo_prob reflect the problem.

        Verifies the constraint counting logic in run() that builds pymooProblem.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp',
                            om.ExecComp(['f = x**2 + y**2', 'c1 = x', 'c2 = y']),
                            promotes=['*'])

        prob.set_solver_print(level=0)
        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 5
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 1)

        model.add_design_var('x', lower=-10.0, upper=10.0)
        model.add_design_var('y', lower=-10.0, upper=10.0)
        model.add_objective('f')
        model.add_constraint('c1', upper=3.0)
        model.add_constraint('c2', equals=1.0)

        prob.setup()
        prob.run_driver()

        self.assertEqual(prob.driver._moo_prob.n_obj, 1)
        self.assertEqual(prob.driver._moo_prob.n_ieq_constr, 1)
        self.assertEqual(prob.driver._moo_prob.n_eq_constr, 1)

    def test_two_sided_constraint_two_ieq_entries(self):
        """
        Test that a single two-sided constraint produces n_ieq_constr == 2.

        Each side of a two-sided bound is registered as a separate inequality
        constraint in pymooProblem, so lower=a/upper=b counts as two entries.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp',
                            om.ExecComp(['f = x**2', 'c = x']),
                            promotes=['*'])

        prob.set_solver_print(level=0)
        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 5
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 1)

        model.add_design_var('x', lower=-10.0, upper=10.0)
        model.add_objective('f')
        model.add_constraint('c', lower=1.0, upper=3.0)

        prob.setup()
        prob.run_driver()

        self.assertEqual(prob.driver._moo_prob.n_ieq_constr, 2)

    def test_constraint_g_convention(self):
        """
        Test that G values in pymoo_results are <= 0 at the optimal point.

        pymoo requires g <= 0 for satisfied constraints. The driver converts
        OpenMDAO upper bounds as g = value - bound and lower bounds as
        g = bound - value. Testing with an active upper bound constraint
        (unconstrained optimum outside the bound) catches a sign reversal:
        a swapped conversion would push the solution the wrong way and produce
        G > 0 at the reported optimum.
        """
        prob = om.Problem()
        model = prob.model

        # Unconstrained optimum at x=8, y=6; constraints force x<=3 and y>=-1.
        model.add_subsystem('comp',
                            om.ExecComp(['f = (x-8.0)**2 + (y-6.0)**2',
                                         'c1 = x', 'c2 = y']),
                            promotes=['*'])

        prob.set_solver_print(level=0)
        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 30
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 200)

        model.add_design_var('x', lower=-10.0, upper=10.0)
        model.add_design_var('y', lower=-10.0, upper=10.0)
        model.add_objective('f')
        model.add_constraint('c1', upper=3.0)   # active: pushes x from 8 to 3
        model.add_constraint('c2', lower=-1.0)  # inactive at optimum (y~6 >> -1)

        prob.setup()
        prob.run_driver()

        self.assertTrue(np.all(prob.driver.pymoo_results.G <= 0),
                        f'Expected all G <= 0 at optimum, got {prob.driver.pymoo_results.G}')

    def test_multi_objective_nsga2(self):
        """
        Test multi-objective optimization using NSGA2.

        Minimizes two competing objectives f1=(x+5)^2 and f2=(x-5)^2. Since
        the objectives are conflicting, no single point minimizes both; the
        result is a Pareto front. Verifies that driver.pareto is populated with
        non-empty arrays of the correct shape after the run.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp',
                            om.ExecComp(['f1 = (x+5.0)**2', 'f2 = (x-5.0)**2']),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'NSGA2'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 50
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 100)

        model.add_design_var('x', lower=-10.0, upper=10.0)
        model.add_objective('f1')
        model.add_objective('f2')

        prob.setup()
        prob.run_driver()

        self.assertIsNotNone(prob.driver.pareto['X'])
        self.assertIsNotNone(prob.driver.pareto['F'])

        # Each Pareto solution has one design variable and two objectives.
        self.assertEqual(prob.driver.pareto['X'].shape[1], 1)
        self.assertEqual(prob.driver.pareto['F'].shape[1], 2)
        self.assertGreater(len(prob.driver.pareto['X']), 0)

    def test_multi_objective_pareto_feasibility(self):
        """
        Test that NSGA2 Pareto solutions lie on the theoretical Pareto front.

        For f1=(x+5)^2 and f2=(x-5)^2, the true Pareto front is x in [-5, 5].
        All returned Pareto solutions should have x in this interval.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp',
                            om.ExecComp(['f1 = (x+5.0)**2', 'f2 = (x-5.0)**2']),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'NSGA2'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 50
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 100)

        model.add_design_var('x', lower=-10.0, upper=10.0)
        model.add_objective('f1')
        model.add_objective('f2')

        prob.setup()
        prob.run_driver()

        x_pareto = prob.driver.pareto['X'].flatten()
        self.assertTrue(np.all(x_pareto >= -5.1))
        self.assertTrue(np.all(x_pareto <= 5.1))

    def test_nsga2_with_constraint(self):
        """
        Test NSGA2 multi-objective optimization with an inequality constraint.

        Minimizes f1=(x+5)^2 and f2=(x-5)^2 subject to x <= 3. The constraint
        restricts the Pareto front to x in [-5, 3]. All returned Pareto solutions
        must satisfy the constraint.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            'comp',
            om.ExecComp(['f1 = (x+5.0)**2', 'f2 = (x-5.0)**2', 'c = x']),
            promotes=['*']
        )

        prob.set_solver_print(level=0)

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'NSGA2'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 50
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 100)

        model.add_design_var('x', lower=-10.0, upper=10.0)
        model.add_objective('f1')
        model.add_objective('f2')
        model.add_constraint('c', upper=3.0)

        prob.setup()
        prob.run_driver()

        self.assertIsNotNone(prob.driver.pareto['X'])
        x_pareto = prob.driver.pareto['X'].flatten()
        self.assertTrue(np.all(x_pareto <= 3.1))

    def test_mixed_integer_branin(self):
        """
        Test MixedVariableGA on the Branin function with continuous design variables.

        MixedVariableGA is required whenever discrete variables are present, but
        also works with continuous-only problems. With both variables treated as
        continuous Real types, the optimizer can reach the true global minimum
        of approximately 0.3979.
        """
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('xC', 7.5)
        model.set_input_defaults('xI', 0.0)

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'MixedVariableGA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 25
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 200)

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob.get_val('comp.f'), 0.397887, tolerance=0.05)

    def test_mixed_integer_branin_discrete(self):
        """
        Test MixedVariableGA with a native OpenMDAO discrete integer design variable.

        BraninDiscrete declares x0 as a discrete integer input. MixedVariableGA
        handles discrete variables through pymoo's Integer type. The expected
        optimal integer value for x0 is 3 or -3.
        """
        prob = om.Problem()
        model = prob.model

        indep = om.IndepVarComp()
        indep.add_discrete_output('xI', val=0)
        indep.add_output('xC', val=7.5)
        model.add_subsystem('p', indep)
        model.add_subsystem('comp', BraninDiscrete())
        model.connect('p.xI', 'comp.x0')
        model.connect('p.xC', 'comp.x1')

        model.add_design_var('p.xI', lower=-5, upper=10)
        model.add_design_var('p.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'MixedVariableGA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 25
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 200)

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob.get_val('comp.f'), 0.49399549, tolerance=0.1)
        self.assertIn(prob.get_val('p.xI'), [3, -3])
        self.assertIsInstance(prob.get_val('p.xI'), int)

    def test_missing_objective_raises(self):
        """
        Test that run_driver raises an exception when no objective is defined.

        An optimization problem without an objective is ill-posed. pymooDriver
        should raise before running the optimizer.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('f = x**2'), promotes=['*'])
        model.add_design_var('x', lower=-50.0, upper=50.0)

        prob.driver = om.pymooDriver()
        prob.driver.options['disp'] = False

        prob.setup()

        with self.assertRaises(Exception):
            prob.run_driver()

    def test_no_design_vars_raises(self):
        """
        Test that run_driver raises RuntimeError when no design variables are defined.

        Without design variables there is nothing for the optimizer to vary.
        pymooDriver should raise RuntimeError with a message mentioning design
        variables.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('f = x**2'), promotes=['*'])
        model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['disp'] = False

        prob.setup()

        with self.assertRaises(RuntimeError) as ctx:
            prob.run_driver()

        self.assertIn('design variable', str(ctx.exception).lower())

    def test_single_obj_algorithm_with_multiple_objectives_raises(self):
        """
        Test that using a single-objective algorithm with two objectives raises RuntimeError.

        GA supports only one objective. Using it with a multi-objective problem
        should raise RuntimeError during setup, referencing the algorithm name
        and 'multiple objectives'.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp',
                            om.ExecComp(['f1 = x**2', 'f2 = (x-1.0)**2']),
                            promotes=['*'])

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-10.0, upper=10.0)
        model.add_objective('f1')
        model.add_objective('f2')

        prob.setup()

        with self.assertRaises(RuntimeError) as ctx:
            prob.run_driver()

        self.assertIn('GA', str(ctx.exception))
        self.assertIn('multiple objectives', str(ctx.exception).lower())

    def test_discrete_vars_without_mixed_var_ga_raises(self):
        """
        Test that discrete design variables with a non-MixedVariableGA algorithm raises.

        Discrete (integer) design variables are only supported by MixedVariableGA.
        Using any other algorithm with discrete variables should raise RuntimeError
        that mentions MixedVariableGA.
        """
        prob = om.Problem()
        model = prob.model

        indep = om.IndepVarComp()
        indep.add_discrete_output('xI', val=0)
        indep.add_output('xC', val=7.5)
        model.add_subsystem('p', indep)
        model.add_subsystem('comp', BraninDiscrete())
        model.connect('p.xI', 'comp.x0')
        model.connect('p.xC', 'comp.x1')

        model.add_design_var('p.xI', lower=-5, upper=10)
        model.add_design_var('p.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False

        prob.setup()

        with self.assertRaises(RuntimeError) as ctx:
            prob.run_driver()

        self.assertIn('MixedVariableGA', str(ctx.exception))

    def test_analysis_error_propagation(self):
        """
        Test that exceptions raised inside the model propagate out of run_driver.

        When a component raises an exception during compute, the pymoo callback
        catches it, stores it, and re-raises it after the optimization loop exits
        so the user receives the original exception type.
        """
        class AlwaysFailComp(om.ExplicitComponent):
            """Component that always raises ValueError in compute."""

            def setup(self):
                self.add_input('x', 1.0)
                self.add_output('f', 1.0)

            def compute(self, inputs, outputs):
                raise ValueError('intentional failure in compute')

        prob = om.Problem()
        prob.model.add_subsystem('comp', AlwaysFailComp(), promotes=['*'])
        prob.model.add_design_var('x', lower=-5.0, upper=10.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 10
        prob.driver.run_settings['termination'] = ('n_gen', 5)

        prob.setup()

        with self.assertRaises(ValueError) as ctx:
            prob.run_driver()

        self.assertIn('intentional failure in compute', str(ctx.exception))

    def test_run_parallel_without_mpi_raises(self):
        """
        Test that run_parallel=True raises RuntimeError when MPI is not available.

        Population-level parallelism requires MPI. When MPI is not installed,
        _setup_comm should raise RuntimeError mentioning 'run_parallel'.
        """
        if MPI:
            raise unittest.SkipTest('MPI is available; this test requires no MPI.')

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('f = x**2'), promotes=['*'])
        prob.model.add_design_var('x', lower=-5.0, upper=5.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['run_parallel'] = True
        prob.driver.options['disp'] = False

        with self.assertRaises(RuntimeError) as ctx:
            prob.setup()

        self.assertIn('run_parallel', str(ctx.exception))

    def test_procs_per_model_without_mpi_raises(self):
        """
        Test that procs_per_model != 1 raises RuntimeError when MPI is not available.

        Sub-communicator splitting requires MPI. Setting procs_per_model > 1
        without MPI should raise RuntimeError mentioning 'procs_per_model'.
        """
        if MPI:
            raise unittest.SkipTest('MPI is available; this test requires no MPI.')

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('f = x**2'), promotes=['*'])
        prob.model.add_design_var('x', lower=-5.0, upper=5.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['procs_per_model'] = 2
        prob.driver.options['disp'] = False

        with self.assertRaises(RuntimeError) as ctx:
            prob.setup()

        self.assertIn('procs_per_model', str(ctx.exception))

    def test_pymoo_results_attribute_populated(self):
        """
        Test that the pymoo_results attribute is set after a successful run.

        After run_driver() completes, driver.pymoo_results should contain the
        pymoo Result object from minimize(). This allows users to inspect the
        raw pymoo output, including optimal design variables and convergence info.
        """
        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('f = (x-3.0)**2'),
                                 promotes=['*'])
        prob.model.add_design_var('x', lower=-10.0, upper=10.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 10
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 20)

        prob.setup()
        prob.run_driver()

        self.assertIsNotNone(prob.driver.pymoo_results)
        self.assertIsNotNone(prob.driver.pymoo_results.X)

    def test_driver_name(self):
        """
        Test that _get_name returns the correctly formatted name for GA.

        The driver name follows the format 'pymoo_<optimizer_name>' and is used
        for case recording identifiers and logging.
        """
        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('f = x**2'), promotes=['*'])
        prob.model.add_design_var('x', lower=-5.0, upper=5.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False

        prob.setup()

        self.assertEqual(prob.driver._get_name(), 'pymoo_GA')

    def test_support_flags_single_obj_algorithm(self):
        """
        Test that support flags are updated correctly for single-objective algorithms.

        After setup with GA, supports['multiple_objectives'] and
        supports['integer_design_vars'] should both be False, while
        supports['inequality_constraints'] should be True.
        """
        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('f = x**2'), promotes=['*'])
        prob.model.add_design_var('x', lower=-5.0, upper=5.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False

        prob.setup()
        prob.final_setup()

        self.assertFalse(prob.driver.supports['multiple_objectives'])
        self.assertFalse(prob.driver.supports['integer_design_vars'])
        self.assertTrue(prob.driver.supports['inequality_constraints'])

    def test_support_flags_multi_obj_algorithm(self):
        """
        Test that support flags are updated correctly for multi-objective algorithms.

        After setup with NSGA2, supports['multiple_objectives'] should be True
        and supports['integer_design_vars'] should be False.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp',
                            om.ExecComp(['f1 = x**2', 'f2 = (x-1.0)**2']),
                            promotes=['*'])

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'NSGA2'
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-10.0, upper=10.0)
        model.add_objective('f1')
        model.add_objective('f2')

        prob.setup()
        prob.final_setup()

        self.assertTrue(prob.driver.supports['multiple_objectives'])
        self.assertFalse(prob.driver.supports['integer_design_vars'])

    def test_support_flags_mixed_var_ga(self):
        """
        Test that support flags are updated correctly for MixedVariableGA.

        After setup with MixedVariableGA, supports['integer_design_vars'] should
        be True and supports['multiple_objectives'] should be False.
        """
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('xC', 7.5)
        model.set_input_defaults('xI', 0.0)
        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'MixedVariableGA'
        prob.driver.options['disp'] = False

        prob.setup()
        prob.final_setup()

        self.assertTrue(prob.driver.supports['integer_design_vars'])
        self.assertFalse(prob.driver.supports['multiple_objectives'])

    def test_get_algorithm_single(self):
        """Test that get_algorithm resolves the correct single-obj class (soo, default naming)."""
        driver = om.pymooDriver()
        alg_class = driver.get_algorithm('GA')
        self.assertEqual(alg_class.__name__, 'GA')

    def test_get_algorithm_multi(self):
        """Test that get_algorithm resolves the correct multi-obj class (moo, default naming)."""
        driver = om.pymooDriver()
        alg_class = driver.get_algorithm('NSGA2')
        self.assertEqual(alg_class.__name__, 'NSGA2')

    def test_get_algorithm_mixed_var(self):
        """Test that MixedVariableGA is resolved from pymoo.core.mixed, not soo or moo."""
        driver = om.pymooDriver()
        alg_class = driver.get_algorithm('MixedVariableGA')
        self.assertEqual(alg_class.__name__, 'MixedVariableGA')

    def test_get_algorithm_non_default_single(self):
        """Test non-default module name mapping for a single-obj algorithm (NelderMead -> nelder)."""
        driver = om.pymooDriver()
        alg_class = driver.get_algorithm('NelderMead')
        self.assertEqual(alg_class.__name__, 'NelderMead')

    def test_get_algorithm_non_default_multi(self):
        """Test non-default module name mapping for a multi-obj algorithm (SMSEMOA -> sms)."""
        driver = om.pymooDriver()
        alg_class = driver.get_algorithm('SMSEMOA')
        self.assertEqual(alg_class.__name__, 'SMSEMOA')

    def test_alg_settings_pop_size_respected(self):
        """
        Test that alg_settings['pop_size'] is actually used by the algorithm.

        The alg_settings dict is unpacked into the pymoo algorithm constructor.
        With pop_size=P and n_gen=N generations, the driver evaluates the model
        at least P*N times (one per individual per generation), plus an initial
        run before the optimizer starts and a final run at the optimal point.
        Running with a larger population therefore produces a strictly greater
        iter_count under the same number of generations.
        """
        n_gen = 3

        def run_with_pop_size(pop_size):
            prob = om.Problem()
            prob.model.add_subsystem('comp', om.ExecComp('f = (x-3.0)**2'),
                                     promotes=['*'])
            prob.model.add_design_var('x', lower=-10.0, upper=10.0)
            prob.model.add_objective('f')

            prob.driver = om.pymooDriver()
            prob.driver.options['optimizer'] = 'GA'
            prob.driver.options['disp'] = False
            prob.driver.alg_settings['pop_size'] = pop_size
            prob.driver.run_settings['seed'] = 1
            prob.driver.run_settings['termination'] = ('n_gen', n_gen)

            prob.setup()
            prob.run_driver()
            return prob.driver.iter_count

        iter_small = run_with_pop_size(5)
        iter_large = run_with_pop_size(20)

        # A larger population must produce more model evaluations.
        self.assertGreater(iter_large, iter_small)

        # At minimum, pop_size * n_gen evaluations occur (plus bookkeeping).
        self.assertGreaterEqual(iter_small, 5 * n_gen)
        self.assertGreaterEqual(iter_large, 20 * n_gen)

    def test_run_settings_termination_limits_generations(self):
        """
        Test that run_settings['termination'] stops the optimizer at the right generation.

        The termination condition is passed directly to pymoo.optimize.minimize.
        With ('n_gen', N) and a fixed pop_size, iter_count grows proportionally
        to N. Running for twice as many generations produces approximately twice
        as many evaluations.
        """
        pop_size = 5

        def run_with_n_gen(n_gen):
            prob = om.Problem()
            prob.model.add_subsystem('comp', om.ExecComp('f = (x-3.0)**2'),
                                     promotes=['*'])
            prob.model.add_design_var('x', lower=-10.0, upper=10.0)
            prob.model.add_objective('f')

            prob.driver = om.pymooDriver()
            prob.driver.options['optimizer'] = 'GA'
            prob.driver.options['disp'] = False
            prob.driver.alg_settings['pop_size'] = pop_size
            prob.driver.run_settings['seed'] = 1
            prob.driver.run_settings['termination'] = ('n_gen', n_gen)

            prob.setup()
            prob.run_driver()
            return prob.driver.iter_count

        iter_short = run_with_n_gen(3)
        iter_long = run_with_n_gen(6)

        # More generations must produce more evaluations.
        self.assertGreater(iter_long, iter_short)

        # Each run must have at least pop_size * n_gen evaluations.
        self.assertGreaterEqual(iter_short, pop_size * 3)
        self.assertGreaterEqual(iter_long, pop_size * 6)

    def test_run_settings_seed_reproducibility(self):
        """
        Test that a fixed seed in run_settings yields identical results across runs.

        Random evolutionary algorithms are nondeterministic by default. Setting
        run_settings['seed'] should produce bit-for-bit identical design variable
        values for two independent runs with the same seed.
        """
        def run_with_seed(seed):
            prob = om.Problem()
            prob.model.add_subsystem('comp', om.ExecComp('f = (x-3.0)**2'),
                                     promotes=['*'])
            prob.model.add_design_var('x', lower=-10.0, upper=10.0)
            prob.model.add_objective('f')

            prob.driver = om.pymooDriver()
            prob.driver.options['optimizer'] = 'GA'
            prob.driver.options['disp'] = False
            prob.driver.alg_settings['pop_size'] = 10
            prob.driver.run_settings['seed'] = seed
            prob.driver.run_settings['termination'] = ('n_gen', 30)

            prob.setup()
            prob.run_driver()
            return prob.get_val('x').item()

        result_a = run_with_seed(42)
        result_b = run_with_seed(42)
        self.assertEqual(result_a, result_b)

    def test_disp_false_suppresses_output(self):
        """Test that disp=False produces no stdout output from pymoo."""
        import io
        import contextlib

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('f = (x-3.0)**2'),
                                 promotes=['*'])
        prob.model.add_design_var('x', lower=-10.0, upper=10.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 10
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 10)

        prob.setup()

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            prob.run_driver()

        output = captured.getvalue()
        self.assertNotIn('n_gen', output)
        self.assertNotIn('n_eval', output)
        self.assertNotIn('Optimization Complete', output)

    def test_fail_flag_when_no_feasible_solution(self):
        """
        Test that driver.fail is True when no feasible solution is found.

        Contradictory constraints (c1 <= -99 and c2 >= 99 on the same variable)
        make the problem infeasible. pymoo returns X=None for infeasible problems,
        which should set driver.fail=True.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp',
                            om.ExecComp(['f = x**2', 'c1 = x', 'c2 = x']),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'GA'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['pop_size'] = 10
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 5)

        model.add_design_var('x', lower=-100.0, upper=100.0)
        model.add_objective('f')
        # Contradictory constraints: x <= -99 AND x >= 99.
        model.add_constraint('c1', upper=-99.0)
        model.add_constraint('c2', lower=99.0)

        prob.setup()
        prob.run_driver()

        self.assertTrue(prob.driver.fail)

    @unittest.skipUnless(CMA_INSTALLED, 'cma package is not installed.')
    def test_cmaes_output_redirected_to_outputs_dir(self):
        """
        Test that the driver sets verb_filenameprefix to the outputs directory for CMAES.

        The driver builds a local copy of alg_settings and injects verb_filenameprefix
        pointing inside the problem outputs directory. This ensures cma (if it writes
        files) targets that directory rather than the working directory.

        Verified properties:
        - driver.alg_settings is not modified by the run (the driver uses a copy).
        - The outputs directory is created by the driver during the run.
        """
        prob = om.Problem()
        prob.model.add_subsystem('comp',
                                 om.ExecComp('f = (x-3.0)**2 + (y+1.0)**2'),
                                 promotes=['*'])
        prob.model.add_design_var('x', lower=-10.0, upper=10.0)
        prob.model.add_design_var('y', lower=-10.0, upper=10.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'CMAES'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['popsize'] = 10
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 5)

        prob.setup()
        prob.run_driver()

        # The driver modifies a copy of alg_settings, not alg_settings itself.
        self.assertNotIn('verb_filenameprefix', prob.driver.alg_settings)
        # The outputs directory must have been created to receive cma output.
        self.assertTrue(prob.get_outputs_dir().exists())

    @unittest.skipUnless(CMA_INSTALLED, 'cma package is not installed.')
    def test_cmaes_output_prefix_not_overridden_by_user(self):
        """
        Test that a user-supplied verb_filenameprefix in alg_settings is not overridden.

        The driver only sets verb_filenameprefix when it is absent from alg_settings.
        If the user supplies one, the driver must leave it unchanged.
        """
        custom_prefix = 'my_custom_cmaes_'

        prob = om.Problem()
        prob.model.add_subsystem('comp',
                                 om.ExecComp('f = (x-3.0)**2 + (y+1.0)**2'),
                                 promotes=['*'])
        prob.model.add_design_var('x', lower=-10.0, upper=10.0)
        prob.model.add_design_var('y', lower=-10.0, upper=10.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()
        prob.driver.options['optimizer'] = 'CMAES'
        prob.driver.options['disp'] = False
        prob.driver.alg_settings['popsize'] = 10
        prob.driver.alg_settings['verb_filenameprefix'] = custom_prefix
        prob.driver.run_settings['seed'] = 1
        prob.driver.run_settings['termination'] = ('n_gen', 5)

        prob.setup()
        prob.run_driver()

        # The user-supplied prefix must be preserved unchanged after the run.
        self.assertEqual(prob.driver.alg_settings['verb_filenameprefix'], custom_prefix)

    def test_citation(self):
        """
        Test that the pymooDriver cite attribute references the pymoo paper.

        The citation is used to acknowledge the underlying library. It should
        include 'pymoo' and the author 'Blank'.
        """
        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('f = x**2'), promotes=['*'])
        prob.model.add_design_var('x', lower=-5.0, upper=5.0)
        prob.model.add_objective('f')

        prob.driver = om.pymooDriver()

        self.assertIn('pymoo', prob.driver.cite)
        self.assertIn('Blank', prob.driver.cite)


if __name__ == '__main__':
    unittest.main()
