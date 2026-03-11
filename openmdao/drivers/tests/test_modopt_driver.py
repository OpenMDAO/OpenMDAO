"""Unit tests for the modOpt Driver."""

import copy
import functools
import pathlib
import unittest
import os.path
from io import StringIO
import sys

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_problem import ParaboloidProblem
from openmdao.test_suite.components.paraboloid_distributed import DistParab
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_totals
from openmdao.utils.general_utils import run_driver
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.om_warnings import OMDeprecationWarning
from openmdao.utils.mpi import MPI


# Check that modOpt is installed
try:
    import modopt as mo
    from openmdao.drivers.modopt_driver import modOptDriver
    MODOPT_INSTALLED = True
    # Try to use SLSQP as default test optimizer
    OPTIMIZER = 'SLSQP'
except ImportError:
    MODOPT_INSTALLED = False
    OPTIMIZER = None


def require_modopt_optimizer(optimizer_name):
    """
    Decorator to skip test if optimizer not available in modOpt.

    Parameters
    ----------
    optimizer_name : str
        Name of the modOpt optimizer to check for.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self):
            if not MODOPT_INSTALLED:
                raise unittest.SkipTest("modOpt is not installed")

            # Build the full list of available optimizers the first time this
            # decorator is triggered
            if not require_modopt_optimizer._initialized:
                for external_optimizer in ['SNOPT', 'IPOPT']:
                    try:
                        optimizer = getattr(mo, external_optimizer)
                        optimizer(mo.ProblemLite(x0=np.array([0.0])),
                                    turn_off_outputs=True)
                        require_modopt_optimizer._available_optimizers.add(external_optimizer)
                    except ImportError:
                        pass

            if optimizer_name not in require_modopt_optimizer._available_optimizers:
                raise unittest.SkipTest(f"{optimizer_name} requires additional installation")

            return func(self)
        return wrapper
    return decorator

require_modopt_optimizer._initialized = False
require_modopt_optimizer._available_optimizers = {
    'SLSQP', 'PySLSQP', 'COBYLA', 'COBYQA', 'BFGS', 'LBFGSB', 'TrustConstr',
    'NelderMead', 'OpenSQP'
}


class ParaboloidAE(om.ExplicitComponent):
    """Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.

    This version raises an analysis error during compute/compute_partials
    for testing error handling.
    """

    def __init__(self):
        super().__init__()
        self.fail_hard = False

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.eval_iter_count = 0
        self.eval_fail_at = 3

        self.grad_iter_count = 0
        self.grad_fail_at = 100

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        if self.eval_iter_count == self.eval_fail_at:
            self.eval_iter_count = 0

            if self.fail_hard:
                raise RuntimeError('This should error.')
            else:
                raise om.AnalysisError('Try again.')

        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
        self.eval_iter_count += 1

    def compute_partials(self, inputs, partials):
        """Jacobian for our paraboloid."""

        if self.grad_iter_count == self.grad_fail_at:
            self.grad_iter_count = 0

            if self.fail_hard:
                raise RuntimeError('This should error.')
            else:
                raise om.AnalysisError('Try again.')

        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2.0*x - 6.0 + y
        partials['f_xy', 'y'] = 2.0*y + 8.0 + x
        self.grad_iter_count += 1


class DummyComp(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 across MPI ranks.
    """
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('c', val=0.0)

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        noise = 1e-10
        if self.comm.rank == 0:
            outputs['c'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
        if self.comm.rank == 1:
            outputs['c'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0 + noise

    def compute_partials(self, inputs, partials):
        """
        Jacobian for our paraboloid.
        """
        x = inputs['x']
        y = inputs['y']

        partials['c', 'x'] = 2.0*x - 6.0 + y
        partials['c', 'y'] = 2.0*y + 8.0 + x


class DataSave(om.ExplicitComponent):
    """Saves run points so that we can verify that initial point is run."""

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_output('y', val=0.0)

        self.visited_points = []
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        x = inputs['x']
        self.visited_points.append(copy.copy(x))
        outputs['y'] = (x-3.0)**2

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        partials['y', 'x'] = 2.0*x - 6.0


@unittest.skipUnless(not MODOPT_INSTALLED, "only run if modOpt is NOT installed.")
@use_tempdirs
class TestNotInstalled(unittest.TestCase):
    """Test behavior when modOpt is not available."""

    def test_modopt_not_installed(self):
        """
        Test that modOptDriver raises RuntimeError when modOpt is not installed.

        When modOpt is not available, attempting to instantiate modOptDriver
        should raise a RuntimeError with a clear message.
        """
        from openmdao.drivers.modopt_driver import modOptDriver

        with self.assertRaises(RuntimeError) as ctx:
            modOptDriver()

        self.assertEqual(str(ctx.exception),
                         'modOptDriver is not available, modOpt is not installed.')


@unittest.skipUnless(MPI, "MPI is required.")
@use_tempdirs
class TestMPIScatter(unittest.TestCase):
    """Test modOptDriver with distributed components."""

    N_PROCS = 2

    @unittest.skipUnless(MODOPT_INSTALLED, "modOpt is not installed")
    def test_design_vars_on_all_procs_modopt(self):
        """
        Test that design variables are correctly synchronized across MPI ranks.

        Verifies that when design variables are distributed across all processes,
        the optimization results are consistent across ranks. All processes should
        see the same optimal design variable and objective values.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', DummyComp(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-15.0)

        prob.setup()
        prob.run_driver()

        proc_vals = prob.comm.allgather([prob['x'], prob['y'], prob['c'], prob['f_xy']])
        # f_xy is a scalar, so compare it separately
        self.assertAlmostEqual(proc_vals[0][3], proc_vals[1][3])
        np.testing.assert_array_almost_equal(proc_vals[0][:3], proc_vals[1][:3])

    @unittest.skipUnless(MODOPT_INSTALLED, "modOpt is not installed")
    def test_opt_distcomp(self):
        """
        Test optimization with distributed array components.

        Verifies that modOptDriver can optimize a problem with distributed
        array components. The constraint values should be satisfied at optimum.
        """
        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', np.ones((size, )))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParab(arr_size=size, deriv_type='dense'), promotes=['*'])
        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones((size, )),
                                               f_xy=np.ones((size, ))),
                            promotes_outputs=['*'])
        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['turn_off_outputs'] = True

        prob.setup(force_alloc_complex=True)

        prob.run_driver()

        con = prob.driver.get_constraint_values()
        obj = prob.driver.get_objective_values()

        assert_near_equal(obj['f_sum'], 0.0, 2e-6)
        assert_near_equal(con['f_xy'], np.zeros(7), 1e-5)


@unittest.skipUnless(MODOPT_INSTALLED, "modOpt is not installed")
@use_tempdirs
class TestModOptDriver(unittest.TestCase):
    """Main test class for modOptDriver functionality."""

    def test_simple_paraboloid_unconstrained(self):
        """
        Test modOptDriver with simple unconstrained paraboloid optimization.

        Verifies that SLSQP correctly optimizes a paraboloid problem without
        constraints. The optimal solution should be at x=6.6667, y=-7.3333.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['x'], 6.66666667, 1e-6)
        assert_near_equal(prob['y'], -7.3333333, 1e-6)

    def test_simple_paraboloid_upper(self):
        """
        Test modOptDriver with a simple upper bound constraint.

        Verifies that SLSQP correctly optimizes a paraboloid problem with an
        inequality constraint (con <= upper_bound). The optimal solution should
        satisfy the constraint at the boundary.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=0.0)

        prob.setup()
        prob.run_driver()

        # Check constraint is satisfied
        self.assertLessEqual(prob['c'], 0.0 + 1e-4)

    def test_simple_paraboloid_lower(self):
        """
        Test modOptDriver with a simple lower bound constraint.

        Verifies that SLSQP correctly optimizes a paraboloid problem with a
        lower bound constraint (con >= lower_bound). The optimal solution
        should satisfy the constraint.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-15.0)

        prob.setup()
        prob.run_driver()

        # Check constraint is satisfied
        self.assertGreaterEqual(prob['c'], -15.0 - 1e-4)

    def test_simple_paraboloid_equality(self):
        """
        Test modOptDriver with an equality constraint.

        Verifies that SLSQP correctly handles equality constraints where
        the constraint value must equal a specific target value.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=0.0)

        prob.setup()
        prob.run_driver()

        # Check constraint is satisfied
        assert_near_equal(prob['c'], 0.0, 1e-6)

    def test_simple_paraboloid_double_sided_low(self):
        """
        Test modOptDriver with a two-sided constraint where lower bound is active.

        Verifies that SLSQP correctly handles two-sided inequality constraints
        (lower <= con <= upper) where the lower bound is active at optimum.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-15.0, upper=100.0)

        prob.setup()
        prob.run_driver()

        # Check constraint is satisfied
        self.assertGreaterEqual(prob['c'], -15.0 - 1e-4)
        self.assertLessEqual(prob['c'], 100.0 + 1e-4)

    def test_simple_paraboloid_double_sided_high(self):
        """
        Test modOptDriver with a two-sided constraint where upper bound is active.

        Verifies that SLSQP correctly handles two-sided inequality constraints
        where the upper bound is active at optimum.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-50.0, upper=0.0)

        prob.setup()
        prob.run_driver()

        # Check constraint is satisfied
        self.assertGreaterEqual(prob['c'], -50.0 - 1e-4)
        self.assertLessEqual(prob['c'], 0.0 + 1e-4)

    def test_simple_constraint_indices(self):
        """
        Test modOptDriver with a 2D array component and specific constraint indices.

        Verifies that SLSQP correctly handles constraints on specific indices
        of an array output, rather than the entire array.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = areas - 20.0', c=np.zeros((2, 2)), areas=np.zeros((2, 2))),
                            promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('widths', lower=-50.0*np.ones((2, 2)), upper=50.0*np.ones((2, 2)))
        model.add_objective('o')
        model.add_constraint('c', indices=1, equals=0.0, flat_indices=True)

        prob.setup()
        prob.run_driver()

        # Constraint should be satisfied
        self.assertLessEqual(prob['c'][0][1], 0.0 + 1e-4)

    def test_simple_paraboloid_lower_linear(self):
        """
        Test modOptDriver with a linear inequality constraint.

        Verifies that modOptDriver correctly pre-computes and uses linear
        constraint Jacobians for efficiency. Linear constraints should be
        handled separately from nonlinear constraints.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=0.0, linear=True)

        prob.setup()
        prob.run_driver()

        # Check constraint is satisfied
        constraint_val = prob['x'] + prob['y']
        self.assertGreaterEqual(constraint_val, 0.0 - 1e-4)

    def test_simple_paraboloid_equality_linear(self):
        """
        Test modOptDriver with a linear equality constraint.

        Verifies that linear equality constraints are pre-computed and
        remain fixed during optimization (not re-evaluated each iteration).
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=0.0, linear=True)

        prob.setup()
        prob.run_driver()

        # Check constraint is satisfied
        assert_near_equal(prob['x'] + prob['y'], 0.0, 1e-6)

    def test_mixed_linear_nonlinear_constraints(self):
        """
        Test modOptDriver with both linear and nonlinear constraints.

        Verifies that modOptDriver correctly handles a mix of linear and
        nonlinear constraints, pre-computing linear Jacobians while
        evaluating nonlinear ones during optimization.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con1', om.ExecComp('c1 = x + y'), promotes=['*'])
        model.add_subsystem('con2', om.ExecComp('c2 = -x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        # Add a linear constraint
        model.add_constraint('c1', lower=-50.0, linear=True)
        # Add a nonlinear constraint
        model.add_constraint('c2', lower=-15.0)

        prob.setup()
        prob.run_driver()

        # Check both constraints are satisfied
        self.assertGreaterEqual(prob['c1'], -50.0 - 1e-4)
        self.assertGreaterEqual(prob['c2'], -15.0 - 1e-4)

    def test_scaled_desvars_fwd(self):
        """
        Test modOptDriver with scaled design variables in forward mode.

        Verifies that design variable scaling is correctly applied when
        using forward mode derivatives. Scaled variables should be
        properly converted to/from the reference frame.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0, scaler=0.1)
        model.add_design_var('y', lower=-50.0, upper=50.0, scaler=0.1)
        model.add_objective('f_xy')

        prob.setup(mode='fwd')
        prob.run_driver()

        # Solution should still be correct despite scaling at point the scaled
        # and unscaled points
        assert_near_equal(prob['x'], 6.66666667, 1e-4)
        assert_near_equal(prob['y'], -7.3333333, 1e-4)
        assert_near_equal(
            prob.driver._mo_prob.x.get_data(),
            np.array([0.666666667, -0.73333333]),
            1e-4
        )

    def test_scaled_desvars_rev(self):
        """
        Test modOptDriver with scaled design variables in reverse mode.

        Verifies that design variable scaling is correctly applied when
        using reverse mode derivatives.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0, scaler=0.1)
        model.add_design_var('y', lower=-50.0, upper=50.0, scaler=0.1)
        model.add_objective('f_xy')

        prob.setup(mode='rev')
        prob.run_driver()

        # Solution should still be correct despite scaling at point the scaled
        # and unscaled points
        assert_near_equal(prob['x'], 6.66666667, 1e-4)
        assert_near_equal(prob['y'], -7.3333333, 1e-4)
        assert_near_equal(
            prob.driver._mo_prob.x.get_data(),
            np.array([0.666666667, -0.73333333]),
            1e-4
        )

    def test_scaled_constraint_fwd(self):
        """
        Test modOptDriver with scaled constraints in forward mode.

        Verifies that constraint scaling is correctly applied in forward
        mode derivatives. Constraints should be properly scaled in both
        the problem formulation and solution verification.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=1.0, scaler=0.1)

        prob.setup(mode='fwd')
        prob.run_driver()

        # Constraint should be satisfied at both the scaled and unscaled points
        assert_near_equal(prob['c'], 1.0, 1e-4)
        assert_near_equal(prob.driver._mo_prob.con.get_data(), 0.1, 1e-4)

    def test_scaled_constraint_rev(self):
        """
        Test modOptDriver with scaled constraints in reverse mode.

        Verifies that constraint scaling is correctly applied in reverse
        mode derivatives.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=1.0, scaler=0.1)

        prob.setup(mode='rev')
        prob.run_driver()

        # Constraint should be satisfied at both the scaled and unscaled points
        assert_near_equal(prob['c'], 1.0, 1e-4)
        assert_near_equal(prob.driver._mo_prob.con.get_data(), 0.1, 1e-4)

    def test_scaled_objective_fwd(self):
        """
        Test modOptDriver with scaled objective in forward mode.

        Verifies that objective scaling is correctly applied in forward
        mode derivatives.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', scaler=0.1)

        prob.setup(mode='fwd')
        prob.run_driver()

        # Solution should be correct at both the scaled and unscaled points
        assert_near_equal(prob['f_xy'], 2783.1236497, 1e-4)
        assert_near_equal(prob.driver._mo_prob.obj['f_xy'], 278.31236497, 1e-4)

    def test_scaled_objective_rev(self):
        """
        Test modOptDriver with scaled objective in reverse mode.

        Verifies that objective scaling is correctly applied in reverse
        mode derivatives.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', scaler=0.1)

        prob.setup(mode='rev')
        prob.run_driver()

        # Solution should be correct at both the scaled and unscaled points
        assert_near_equal(prob['f_xy'], 2783.1236497, 1e-4)
        assert_near_equal(prob.driver._mo_prob.obj['f_xy'], 278.31236497, 1e-4)

    def test_missing_objective(self):
        """
        Test that modOptDriver raises error when no objective is defined.

        Verifies that attempting to run optimization without defining an
        objective raises an appropriate error with a clear message.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        # Note: No objective defined

        prob.setup()

        with self.assertRaises(RuntimeError) as ctx:
            prob.run_driver()

        self.assertIn('objective', str(ctx.exception).lower())

    def test_multiple_objectives_error(self):
        """
        Test that modOptDriver raises error with multiple objectives.

        Verifies that attempting to define multiple objectives raises an
        error, as modOpt supports single-objective optimization only.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_objective('x')

        prob.setup()

        # Try to add another objective (this should fail during run)
        with self.assertRaises(RuntimeError) as ctx:
            prob.run_driver()

        self.assertIn('does not support multiple objectives', str(ctx.exception).lower())

    def test_invalid_desvar_values(self):
        """
        Test handling of NaN/Inf in design variables.

        Verifies that the optimizer handles invalid design variable values
        (NaN, Inf) gracefully without crashing.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()

        # Set invalid values
        prob.set_val('x', np.nan)
        prob.set_val('y', 50.0)

        # Should handle gracefully
        prob.run_driver()

    def test_raised_error_objfunc(self):
        """
        Test handling when objective function evaluation raises an error.

        Verifies that if the objective function raises an AnalysisError,
        modOptDriver handles it appropriately without crashing.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        comp = model.add_subsystem('comp', ParaboloidAE(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        # Force to fail
        comp.fail_hard = True

        prob.setup()

        with self.assertRaises(RuntimeError):
            prob.run_driver()

    def test_raised_error_gradfunc(self):
        """
        Test handling when gradient evaluation raises an error.

        Verifies that if gradient computation raises an AnalysisError,
        modOptDriver handles it appropriately.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        comp = model.add_subsystem('comp', ParaboloidAE(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        # Force to fail
        comp.fail_hard = True
        comp.grad_fail_at = 2
        comp.eval_fail_at = 100

        prob.setup()

        with self.assertRaises(RuntimeError):
            prob.run_driver()

    def test_singular_jac_error_response(self):
        """
        Test singular Jacobian detection with error behavior in responses.

        Verifies that when singular_jac_behavior='error', an error is raised
        when a singular constraint Jacobian is detected. Only need to check a
        response resulting in a singular jacobian because if a design variable
        doesn't impact any responses it won't be declared in modopt as part of
        the constraint jacobian.
        """
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                              'z = 12.0'],),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['singular_jac_behavior'] = 'error'
        prob.driver.options['turn_off_outputs'] = True

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        # This constraint produces a zero row.
        prob.model.add_constraint('parab.z', equals=12.)

        prob.setup()

        with self.assertRaises(Exception) as msg:
            prob.run_driver()

        self.assertEqual(
            str(msg.exception),
            'Constraints or objectives [parab.z] cannot be impacted by the '
            'design variables of the problem because no partials were defined '
            'for them in their parent component(s).'
        )

    def test_singular_jac_warn(self):
        """
        Test singular Jacobian detection with warn behavior.

        Verifies that when singular_jac_behavior='warn', a warning is issued
        when a singular constraint Jacobian is detected, but optimization
        continues.
        """
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                              'z = 12.0'],),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True
        # Default behavior is 'warn'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.z')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        prob.setup()

        msg = (
            'Constraints or objectives [parab.z] cannot be impacted by the '
            'design variables of the problem because no partials were defined '
            'for them in their parent component(s).'
        )
        with assert_warning(UserWarning, msg):
            prob.run_driver()

    def test_singular_jac_ignore(self):
        """
        Test singular Jacobian detection with ignore behavior.

        Verifies that when singular_jac_behavior='ignore', no check is
        performed and optimization proceeds normally.
        """
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                              'z = 12.0'],),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['singular_jac_behavior'] = 'ignore'
        prob.driver.options['turn_off_outputs'] = True

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.z')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        prob.setup()

        # Will not raise an exception.
        prob.run_driver()

    @require_modopt_optimizer('SLSQP')
    def test_slsqp_basic(self):
        """
        Test SLSQP optimizer with basic constrained optimization.

        SLSQP (Sequential Least Squares Programming) is the default optimizer
        and supports all constraint types and bounds.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=0.0)

        prob.setup()
        prob.run_driver()

        # Check expected responses are met
        assert_near_equal(prob['f_xy'], -27.0, 1e-4)
        assert_near_equal(prob['c'], 0.0, 1e-4)

    @require_modopt_optimizer('PySLSQP')
    def test_pyslsqp_basic(self):
        """
        Test PySLSQP optimizer with basic constrained optimization.

        PySLSQP is a pure Python implementation of SLSQP, useful when the
        Fortran version is not available or preferred.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='PySLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=0.0)

        prob.setup()
        prob.run_driver()

        # Check expected responses are met
        assert_near_equal(prob['f_xy'], -27.0, 1e-4)
        assert_near_equal(prob['c'], 0.0, 1e-4)

    @require_modopt_optimizer('BFGS')
    def test_bfgs_unconstrained(self):
        """
        Test BFGS optimizer for unconstrained optimization.

        BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a gradient-based optimizer
        for unconstrained problems and does not support constraints.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='BFGS')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        # BFGS does not support constraints

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['x'], 6.66666667, 1e-5)
        assert_near_equal(prob['y'], -7.3333333, 1e-5)

    @require_modopt_optimizer('LBFGSB')
    def test_lbfgsb_bounds(self):
        """
        Test L-BFGS-B optimizer with bound constraints.

        L-BFGS-B (Limited-memory BFGS with bounds) supports bound constraints
        on design variables but not general nonlinear constraints.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='LBFGSB')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['x'], 6.66666667, 1e-5)
        assert_near_equal(prob['y'], -7.3333333, 1e-5)

    # @require_modopt_optimizer('TrustConstr')
    # def test_trustconstr_constrained(self):
    #     """
    #     Test TrustConstr optimizer with general constraints.

    #     Trust-region constrained algorithm supports equality and inequality
    #     constraints along with bounds.
    #     """
    #     prob = om.Problem()
    #     model = prob.model

    #     model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
    #     model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
    #     model.add_subsystem('comp', Paraboloid(), promotes=['*'])
    #     model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

    #     prob.set_solver_print(level=0)

    #     prob.driver = modOptDriver(optimizer='TrustConstr')
    #     prob.driver.opt_settings['ignore_exact_hessian'] = True
    #     prob.driver.options['disp'] = False
    #     prob.driver.options['maxiter'] = 1000
    #     prob.driver.options['turn_off_outputs'] = True

    #     model.add_design_var('x', lower=-50.0, upper=50.0)
    #     model.add_design_var('y', lower=-50.0, upper=50.0)
    #     model.add_objective('f_xy')
    #     model.add_constraint('c', equals=0.0)

    #     prob.setup()
    #     prob.run_driver()

    #     # Check expected responses are met
    #     # assert_near_equal(prob['f_xy'], -27.0, 1e-4)
    #     # assert_near_equal(prob['c'], 0.0, 1e-4)

    #     print("\n\n")
    #     print(prob['x'])
    #     print(prob['y'])
    #     print(prob['f_xy'])
    #     print(prob['c'])
    #     print("\n\n")

    @require_modopt_optimizer('COBYLA')
    def test_cobyla_gradient_free(self):
        """
        Test COBYLA gradient-free optimizer.

        COBYLA (Constrained Optimization BY Linear Approximation) is a
        derivative-free algorithm suitable for problems without gradients.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='COBYLA')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=0.0)

        prob.setup()
        prob.run_driver()

        # Check expected responses are met
        assert_near_equal(prob['f_xy'], -27.0, 1e-4)
        assert_near_equal(prob['c'], 0.0, 1e-4)

    @require_modopt_optimizer('COBYQA')
    def test_cobyqa_gradient_free(self):
        """
        Test COBYQA gradient-free optimizer.

        COBYQA (Constrained Optimization BY Quadratic Approximation) is a
        modern derivative-free algorithm with better accuracy than COBYLA.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='COBYQA')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=0.0)

        prob.setup()
        prob.run_driver()

        # Check expected responses are met
        assert_near_equal(prob['f_xy'], -27.0, 1e-4)
        assert_near_equal(prob['c'], 0.0, 1e-4)

    @require_modopt_optimizer('NelderMead')
    def test_neldermead_unconstrained(self):
        """
        Test Nelder-Mead optimizer for unconstrained optimization.

        Nelder-Mead is a gradient-free simplex algorithm for unconstrained
        optimization. It does not support constraints.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='NelderMead')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['f_xy'], -27.3333333, 1e-4)
        assert_near_equal(prob['x'], 6.66666667, 1e-4)
        assert_near_equal(prob['y'], -7.3333333, 1e-4)

    @require_modopt_optimizer('OpenSQP')
    def test_opensqp_basic(self):
        """
        Test OpenSQP optimizer with general constraints.

        OpenSQP is a sequential quadratic programming optimizer built into
        modOpt and supports equality and inequality constraints.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='OpenSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=0.0)

        prob.setup()
        prob.run_driver()

        # Check expected responses are met
        assert_near_equal(prob['f_xy'], -27.0, 1e-4)
        assert_near_equal(prob['c'], 0.0, 1e-4)

    @require_modopt_optimizer('SNOPT')
    def test_snopt_basic(self):
        """
        Test SNOPT optimizer with general constraints.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SNOPT')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=0.0)

        prob.setup()
        prob.run_driver()

        # Check expected responses are met
        assert_near_equal(prob['f_xy'], -27.0, 1e-4)
        assert_near_equal(prob['c'], 0.0, 1e-4)

    # @require_modopt_optimizer('IPOPT')
    # def test_ipopt_basic(self):
    #     """
    #     Test OpenSQP optimizer with general constraints.
    #     """
    #     prob = om.Problem()
    #     model = prob.model

    #     model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
    #     model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
    #     model.add_subsystem('comp', Paraboloid(), promotes=['*'])
    #     model.add_subsystem('con', om.ExecComp('c = x + y'), promotes=['*'])

    #     prob.set_solver_print(level=0)

    #     prob.driver = modOptDriver(optimizer='IPOPT')
    #     prob.driver.options['disp'] = False
    #     prob.driver.options['turn_off_outputs'] = True
    #     prob.driver.opt_settings['tol'] = 1e-6

    #     model.add_design_var('x', lower=-50.0, upper=50.0)
    #     model.add_design_var('y', lower=-50.0, upper=50.0)
    #     model.add_objective('f_xy')
    #     model.add_constraint('c', equals=0.0)

    #     prob.setup()
    #     prob.run_driver()

    #     # Check expected responses are met
    #     assert_near_equal(prob['f_xy'], -27.0, 1e-4)
    #     assert_near_equal(prob['c'], 0.0, 1e-4)

    def test_driver_sparsity(self):
        prob = om.Problem()
        prob.set_solver_print(level=0)
        prob.driver = driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True
        driver.declare_coloring()
        model = prob.model

        shape = (1, 5)
        ivc = model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('x', np.ones(shape))
        ivc.add_output('y', np.ones(shape))

        model.add_subsystem(
            'obj',
            om.ExecComp(
                'obj = sum(x + y)',
                obj=1.,
                x=np.ones(shape),
                y=np.ones(shape),
            )
        )
        model.add_subsystem(
            'con',
            om.ExecComp(
                'w = x',
                x=np.ones(shape),
                w=np.ones(shape)
            )
        )

        model.connect('ivc.x', 'obj.x')
        model.connect('ivc.x', 'con.x')
        model.connect('ivc.y', 'obj.y')

        model.add_design_var('ivc.x', lower=0.0, upper=1.0)
        model.add_design_var('ivc.y', lower=0.0, upper=1.0)

        model.add_objective('obj.obj')
        model.add_constraint('con.w', lower=0.0)

        prob.setup()
        prob.run_model()
        prob.run_driver()

        sparsity_truth = {
            'con.w': {
                'ivc.x': {
                    'coo': [np.array([0, 1, 2, 3, 4]),
                            np.array([0, 1, 2, 3, 4]),
                            np.array([0., 0., 0., 0., 0.])],
                    'shape': (5, 5)
                },
                'ivc.y': {
                    'coo': [np.array([]), np.array([]), np.array([])],
                    'shape': (5, 5)
                }
            }
        }

        # Check _con_subjacs, but it's a nested dict so check it in parts
        subjacs = prob.driver._con_subjacs
        self.assertListEqual(list(sparsity_truth.keys()), ['con.w'])
        self.assertListEqual(list(sparsity_truth['con.w'].keys()), ['ivc.x', 'ivc.y'])
        np.testing.assert_equal(
            sparsity_truth['con.w']['ivc.x']['coo'],
            subjacs['con.w']['ivc.x']['coo']
        )
        np.testing.assert_equal(
            sparsity_truth['con.w']['ivc.y']['coo'],
            subjacs['con.w']['ivc.y']['coo']
        )

    def test_no_print_option(self):
        """
        Test debug print option for modOptDriver.

        Verifies that the disp option controls verbosity of optimizer output.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        prob.run_driver()

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Optimizer shouldn't print out anything, only the hardcoded modOpt text
        self.assertEqual(output, 'Setting objective name as "f_xy".\n')

    def test_sellar_mdf(self):
        """
        Test modOptDriver on Sellar MDF problem with coupling.

        Verifies that the driver can handle problems with implicit components
        and nonlinear coupling (like the Sellar problem with multidisciplinary
        feasible architecture).
        """
        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped(nonlinear_solver=om.NonlinearBlockGS,
                                                      linear_solver=om.ScipyKrylov)

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['disp'] = False
        prob.driver.options['turn_off_outputs'] = True

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        # Check that constraints are satisfied
        assert_near_equal(prob['z'][0], 1.9776, 1e-3)
        assert_near_equal(prob['z'][1], 0.0, 1e-3)
        assert_near_equal(prob['x'], 0.0, 4e-3)

    def test_driver_supports(self):
        """
        Test driver supports dictionary for modOptDriver.

        Verifies that the driver correctly reports its capabilities (what
        types of constraints it supports, etc.) through the supports dict.
        """
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = modOptDriver(optimizer='SLSQP')
        prob.driver.options['turn_off_outputs'] = True

        # Check that supports dict is read-only
        with self.assertRaises(KeyError):
            prob.driver.supports['equality_constraints'] = False


if __name__ == '__main__':
    unittest.main()
