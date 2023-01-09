""" Unit tests for the Pyoptsparse Driver."""

import copy
import unittest

from packaging.version import Version

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_distributed import DistParab
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_totals
from openmdao.utils.general_utils import set_pyoptsparse_opt, run_driver
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse, set_env_vars_context
from openmdao.utils.mpi import MPI


# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver, UserRequestedException


class ParaboloidAE(om.ExplicitComponent):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
    This version raises an analysis error 50% of the time.
    The AE in ParaboloidAE stands for AnalysisError."""

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
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
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
        """ Jacobian for our paraboloid."""

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
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
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
    """ Saves run points so that we can verify that initial point is run."""

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_output('y', val=0.0)

        self.visited_points=[]
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        x = inputs['x']
        self.visited_points.append(copy.copy(x))
        outputs['y'] = (x-3.0)**2

    def compute_partials(self, inputs, partials):
        x = inputs['x']

        partials['y', 'x'] = 2.0*x - 6.0


@unittest.skipUnless(OPT is None, "only run if pyoptsparse is NOT installed.")
class TestNotInstalled(unittest.TestCase):

    def test_pyoptsparse_not_installed(self):
        # the import should not fail
        from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

        # but we get a RuntimeError if we try to instantiate
        with self.assertRaises(RuntimeError) as ctx:
            pyOptSparseDriver()

        self.assertEqual(str(ctx.exception),
                         'pyOptSparseDriver is not available, pyOptsparse is not installed.')


@unittest.skipUnless(MPI, "MPI is required.")
class TestMPIScatter(unittest.TestCase):
    N_PROCS = 2

    @require_pyoptsparse(OPTIMIZER)
    def test_design_vars_on_all_procs_pyopt(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', DummyComp(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver(optimizer=OPTIMIZER, print_results=False)
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-6

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-15.0)

        prob.setup()
        prob.run_driver()

        proc_vals = MPI.COMM_WORLD.allgather([prob['x'], prob['y'], prob['c'], prob['f_xy']])
        np.testing.assert_array_almost_equal(proc_vals[0], proc_vals[1])

    @require_pyoptsparse(OPTIMIZER)
    def test_opt_distcomp(self):
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

        prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

        prob.setup(force_alloc_complex=True)

        prob.run_driver()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()
        obj = prob.driver.get_objective_values()

        assert_near_equal(obj['sum.f_sum'], 0.0, 2e-6)
        assert_near_equal(con['parab.f_xy'],
                          np.zeros(7),
                          1e-5)

    @require_pyoptsparse('ParOpt')
    def test_paropt_distcomp(self):
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

        prob.driver = om.pyOptSparseDriver(optimizer='ParOpt')

        prob.setup(force_alloc_complex=True)

        prob.run_driver()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()
        obj = prob.driver.get_objective_values()

        assert_near_equal(obj['sum.f_sum'], 0.0, 4e-6)
        assert_near_equal(con['parab.f_xy'],
                          np.zeros(7),
                          1e-5)


@require_pyoptsparse(OPTIMIZER)
@use_tempdirs
class TestPyoptSparse(unittest.TestCase):

    def test_simple_paraboloid_upper(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver(optimizer=OPTIMIZER, print_results=False)
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_upper_indices(self):

        prob = om.Problem()
        model = prob.model

        size = 3
        model.add_subsystem('p1', om.IndepVarComp('x', np.array([50.0]*size)))
        model.add_subsystem('p2', om.IndepVarComp('y', np.array([50.0]*size)))
        model.add_subsystem('comp', om.ExecComp('f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                                x=np.zeros(size), y=np.zeros(size),
                                                f_xy=np.zeros(size)))
        model.add_subsystem('con', om.ExecComp('c = - x + y',
                                               c=np.zeros(size), x=np.zeros(size),
                                               y=np.zeros(size)))

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')
        model.connect('p1.x', 'con.x')
        model.connect('p2.y', 'con.y')

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        model.add_design_var('p1.x', indices=[1], lower=-50.0, upper=50.0)
        model.add_design_var('p2.y', indices=[1], lower=-50.0, upper=50.0)
        model.add_objective('comp.f_xy', index=1)
        model.add_constraint('con.c', indices=[1], upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['p1.x'], np.array([50., 7.16667, 50.]), 1e-6)
        assert_near_equal(prob['p2.y'], np.array([50., -7.833334, 50.]), 1e-6)

    def test_simple_paraboloid_lower(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)

        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_lower_linear(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0, linear=True)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

        self.assertEqual(prob.driver._quantities, ['comp.f_xy'])

        # make sure multiple driver runs don't grow the list of _quantities
        quants = copy.copy(prob.driver._quantities)
        for i in range(5):
            prob.run_driver()
            self.assertEqual(quants, prob.driver._quantities)

    def test_simple_paraboloid_equality(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_equality_linear(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 50.0)
        model.set_input_defaults('y', 50.0)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=-15.0, linear=True)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_double_sided_low(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-11.0, upper=-10.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['y'] - prob['x'], -11.0, 1e-6)

    def test_simple_paraboloid_double_sided_high(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_linear_with_y_intercept_eq(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 50.0)
        model.set_input_defaults('y', 50.0)

        parab = om.ExecComp('f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0')

        model.add_subsystem('comp', parab, promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y - 25.0'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=0.0, linear=True)

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob['x'], 19.5, 1e-6)
        assert_near_equal(prob['y'], 5.5, 1e-6)

    def test_simple_paraboloid_linear_with_y_intercept_ineq(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 50.0)
        model.set_input_defaults('y', 50.0)

        parab = om.ExecComp('f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0')

        model.add_subsystem('comp', parab, promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x + y - 25.0'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=0.0, linear=True)

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob['x'], 19.5, 1e-6)
        assert_near_equal(prob['y'], 5.5, 1e-6)

    def test_simple_array_comp2D(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = areas - 20.0', c=np.zeros((2, 2)), areas=np.zeros((2, 2))),
                            promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('c', equals=0.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        obj = prob['o']
        assert_near_equal(obj, 20.0, 1e-6)

    def test_simple_array_comp2D_array_lo_hi(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = areas - 20.0', c=np.zeros((2, 2)), areas=np.zeros((2, 2))),
                            promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False

        model.add_design_var('widths', lower=-50.0*np.ones((2, 2)), upper=50.0*np.ones((2, 2)))
        model.add_objective('o')
        model.add_constraint('c', equals=0.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        obj = prob['o']
        assert_near_equal(obj, 20.0, 1e-6)

    def test_driver_supports(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])

        prob.driver = pyOptSparseDriver(optimizer=OPTIMIZER, print_results=False)

        with self.assertRaises(KeyError) as raises_msg:
            prob.driver.supports['equality_constraints'] = False

        exception = raises_msg.exception

        msg = "pyOptSparseDriver: Tried to set read-only option 'equality_constraints'."

        self.assertEqual(exception.args[0], msg)

    def test_fan_out(self):
        # This tests sparse-response specification.
        # This is a slightly modified FanOut

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp1', om.ExecComp('y = 3.0*x'))
        model.add_subsystem('comp2', om.ExecComp('y = 5.0*x'))

        model.add_subsystem('obj', om.ExecComp('o = i1 + i2'))
        model.add_subsystem('con1', om.ExecComp('c = 15.0 - x'))
        model.add_subsystem('con2', om.ExecComp('c = 15.0 - x'))

        # hook up explicitly
        model.connect('comp1.y', 'obj.i1')
        model.connect('comp2.y', 'obj.i2')
        model.connect('comp1.y', 'con1.x')
        model.connect('comp2.y', 'con2.x')

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False

        model.add_design_var('comp1.x', lower=-50.0, upper=50.0)
        model.add_design_var('comp2.x', lower=-50.0, upper=50.0)
        model.add_objective('obj.o')
        model.add_constraint('con1.c', equals=0.0)
        model.add_constraint('con2.c', equals=0.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        obj = prob['obj.o']
        assert_near_equal(obj, 30.0, 1e-6)

        # Verify that pyOpt has the correct wrt names
        con1 = prob.driver.pyopt_solution.constraints['con1.c']
        self.assertEqual(con1.wrt, ['comp1.x'])
        con2 = prob.driver.pyopt_solution.constraints['con2.c']
        self.assertEqual(con2.wrt, ['comp2.x'])

    def test_inf_as_desvar_bounds(self):

        # User may use np.inf as a bound. It is unneccessary, but the user
        # may do it anyway, so make sure SLSQP doesn't blow up with it (bug
        # reported by rfalck)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-np.inf, upper=np.inf)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_pyopt_fd_solution(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['gradient_method'] = 'pyopt_fd'
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-4)
        assert_near_equal(prob['y'], -7.833334, 1e-4)

    def test_pyopt_fd_is_called(self):

        class ParaboloidApplyLinear(Paraboloid):
            def apply_linear(inputs, outputs, resids):
                raise Exception("OpenMDAO's finite difference has been called."
                                " pyopt_fd option has failed.")

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])

        model.add_subsystem('comp', ParaboloidApplyLinear(), promotes=['*'])

        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['gradient_method'] = 'pyopt_fd'
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-4)
        assert_near_equal(prob['y'], -7.833334, 1e-4)

    def test_snopt_fd_option_error(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['gradient_method'] = 'snopt_fd'
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        with self.assertRaises(Exception) as raises_cm:
            prob.run_driver()

        exception = raises_cm.exception

        msg = "SNOPT's internal finite difference can only be used with SNOPT"

        self.assertEqual(exception.args[0], msg)

    def test_unsupported_multiple_obj(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('comp2', Paraboloid())

        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['gradient_method'] = 'snopt_fd'
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_objective('comp2.f_xy')
        model.add_constraint('c', upper=-15.0)

        expected = 'Multiple objectives have been added to pyOptSparseDriver' \
                   ' but the selected optimizer (SLSQP) does not support' \
                   ' multiple objectives.'

        prob.setup()

        with self.assertRaises(RuntimeError) as cm:
            prob.final_setup()

        self.assertEqual(str(cm.exception), expected)

    def test_simple_paraboloid_scaled_desvars_fwd(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0, ref=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='fwd')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_fd(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 50.0)
        model.set_input_defaults('y', 50.0)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0, ref=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        model.approx_totals(method='fd')

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_cs(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0, ref=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        model.approx_totals(method='cs')

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_rev(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0, ref=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_fwd(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0, ref=10.)

        prob.setup(check=False, mode='fwd')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_fd(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0, ref=10.)

        model.approx_totals(method='fd')

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_cs(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0, ref=10.)

        model.approx_totals(method='cs')

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_rev(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0, ref=10.)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_fwd(self):

        prob = om.Problem()
        model = prob.model

        prob.set_solver_print(level=0)

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', ref=10.)
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='fwd')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_rev(self):

        prob = om.Problem()
        model = prob.model

        prob.set_solver_print(level=0)

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', ref=10.)
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_sellar_mdf(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped(nonlinear_solver=om.NonlinearBlockGS,
                                                      linear_solver=om.ScipyKrylov)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        elif OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-3
        prob.driver.options['print_results'] = False

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        assert_near_equal(prob['z'][0], 1.9776, 1e-3)
        assert_near_equal(prob['z'][1], 0.0, 1e-3)
        assert_near_equal(prob['x'], 0.0, 4e-3)

    def test_sellar_mdf_linear_con_directsolver(self):
        # This test makes sure that we call solve_nonlinear first if we have any linear constraints
        # to cache.

        class SellarDis1withDerivatives(om.ExplicitComponent):

            def setup(self):
                self.add_input('z', val=np.zeros(2))
                self.add_input('x', val=0.)
                self.add_input('y2', val=0.0)
                self.add_output('y1', val=0.0)

            def setup_partials(self):
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                z1 = inputs['z'][0]
                z2 = inputs['z'][1]
                x1 = inputs['x']
                y2 = inputs['y2']

                outputs['y1'] = z1**2 + z2 + x1 - 0.2*y2

            def compute_partials(self, inputs, partials):
                partials['y1', 'y2'] = -0.2
                partials['y1', 'z'] = np.array([[2.0 * inputs['z'][0], 1.0]])
                partials['y1', 'x'] = 1.0


        class SellarDis2withDerivatives(om.ExplicitComponent):

            def setup(self):
                self.add_input('z', val=np.zeros(2))
                self.add_input('y1', val=0.0)
                self.add_output('y2', val=0.0)

            def setup_partials(self):
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                z1 = inputs['z'][0]
                z2 = inputs['z'][1]
                y1 = inputs['y1']

                if y1.real < 0.0:
                    y1 *= -1

                outputs['y2'] = y1**.5 + z1 + z2

            def compute_partials(self, inputs, J):
                y1 = inputs['y1']
                if y1.real < 0.0:
                    y1 *= -1

                J['y2', 'y1'] = .5*y1**-.5
                J['y2', 'z'] = np.array([[1.0, 1.0]])


        class MySellarGroup(om.Group):

            def setup(self):
                self.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
                self.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

                self.mda = mda = self.add_subsystem('mda', om.Group(), promotes=['x', 'z', 'y1', 'y2'])
                mda.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
                mda.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

                self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                          z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                                   promotes=['obj', 'x', 'z', 'y1', 'y2'])

                self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
                self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

                self.linear_solver = om.DirectSolver()
                self.nonlinear_solver = om.NonlinearBlockGS()


        prob = om.Problem()
        model = prob.model = MySellarGroup()

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        elif OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-3
        prob.driver.options['print_results'] = False

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)
        model.add_constraint('x', upper=11.0, linear=True)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        assert_near_equal(prob['z'][0], 1.9776, 1e-3)
        assert_near_equal(prob['z'][1], 0.0, 1e-3)
        assert_near_equal(prob['x'], 0.0, 4e-3)

        # Piggyback test: make sure we can run the driver again as a subdriver without a keyerror.
        prob.driver.run()

    def test_raised_error_objfunc(self):

        # Component fails hard this time during execution, so we expect
        # pyoptsparse to raise.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])

        comp = model.add_subsystem('comp', ParaboloidAE(), promotes=['*'])

        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()

        # SNOPT has a weird cleanup problem when this fails, so we use SLSQP. For the
        # regular failure, it doesn't matter which opt we choose since they all fail through.
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['ACC'] = 1e-9

        prob.driver.options['print_results'] = False
        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)

        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        comp.fail_hard = True

        prob.setup()

        with self.assertRaises(Exception):
            prob.run_driver()

        # pyopt's failure message differs by platform and is not informative anyway

    def test_raised_error_sensfunc(self):

        # Component fails hard this time during gradient eval, so we expect
        # pyoptsparse to raise.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])

        comp = model.add_subsystem('comp', ParaboloidAE(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()

        # SNOPT has a weird cleanup problem when this fails, so we use SLSQP. For the
        # regular failure, it doesn't matter which opt we choose since they all fail through.
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['ACC'] = 1e-9

        prob.driver.options['print_results'] = False
        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)

        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        comp.fail_hard = True
        comp.grad_fail_at = 2
        comp.eval_fail_at = 100

        prob.setup()

        with self.assertRaises(Exception):
            prob.run_driver()

        # pyopt's failure message differs by platform and is not informative anyway

    def test_debug_print_option_totals(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        prob.driver.options['debug_print'] = ['totals']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False, mode='rev')

        failed, output = run_driver(prob)

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        self.assertTrue('In mode: rev.' in output)
        self.assertTrue("('comp.f_xy', [0])" in output)
        self.assertTrue('Elapsed Time:' in output)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        prob.driver.options['debug_print'] = ['totals']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False, mode='fwd')

        failed, output = run_driver(prob)

        self.assertFalse(failed, "Optimization failed, info = " +
                             str(prob.driver.pyopt_solution.optInform))

        self.assertTrue('In mode: fwd.' in output)
        self.assertTrue("('p2.y', [1])" in output)
        self.assertTrue('Elapsed Time:' in output)

    def test_debug_print_option_totals_no_ivc(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        model.set_input_defaults('x', 50.0)
        model.set_input_defaults('y', 50.0)

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        prob.driver.options['debug_print'] = ['totals']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False, mode='rev')

        failed, output = run_driver(prob)

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        self.assertTrue('In mode: rev.' in output)
        self.assertTrue("('comp.f_xy', [0])" in output)
        self.assertTrue('Elapsed Time:' in output)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        prob.driver.options['debug_print'] = ['totals']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False, mode='fwd')

        failed, output = run_driver(prob)

        self.assertFalse(failed, "Optimization failed, info = " +
                             str(prob.driver.pyopt_solution.optInform))

        self.assertTrue('In mode: fwd.' in output)
        self.assertTrue("('p2.y', [1])" in output)
        self.assertTrue('Elapsed Time:' in output)

    def test_debug_print_option(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        prob.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed, output = run_driver(prob)

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        output = output.split('\n')

        self.assertTrue(output.count("Design Vars") > 1,
                        "Should be more than one design vars header printed")
        self.assertTrue(output.count("Nonlinear constraints") > 1,
                        "Should be more than one nonlinear constraint header printed")
        self.assertTrue(output.count("Linear constraints") > 1,
                        "Should be more than one linear constraint header printed")
        self.assertTrue(output.count("Objectives") > 1,
                        "Should be more than one objective header printed")

        self.assertTrue(len([s for s in output if s.startswith("{'p1.x")]) > 1,
                        "Should be more than one p1.x printed")
        self.assertTrue(len([s for s in output if "'p2.y'" in s]) > 1,
                        "Should be more than one p2.y printed")
        self.assertTrue(len([s for s in output if s.startswith("{'con.c")]) > 1,
                        "Should be more than one con.c printed")
        self.assertTrue(len([s for s in output if s.startswith("{'comp.f_xy")]) > 1,
                        "Should be more than one comp.f_xy printed")

    def test_show_exception_bad_opt(self):

        # First, check if we have the optimizer for this test. If they do, then just skip it.
        _, loc_opt = set_pyoptsparse_opt('NLPQLP')
        if loc_opt == 'NLPQLP':
            raise unittest.SkipTest("Skipping because user has this optimizer.")

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.driver = pyOptSparseDriver()

        # We generally don't have a working NLPQLP install.
        prob.driver.options['optimizer'] = 'NLPQLP'
        prob.setup()

        # Test that we get exception.
        with self.assertRaises(ImportError) as raises_cm:
            prob.run_driver()

        self.assertTrue("NLPQLP is not available" in str(raises_cm.exception))

    # Travis testing core dumps on many of the machines. Probabaly a build problem with the NSGA source.
    # Limiting this to the single travis 1.14 machine for now.
    @unittest.skipUnless(Version(np.__version__) >= Version("1.13"), "numpy >= 1.13 is required.")
    def test_initial_run_NSGA2(self):
        _, local_opt = set_pyoptsparse_opt('NSGA2')
        if local_opt != 'NSGA2':
            raise unittest.SkipTest("pyoptsparse is not providing NSGA2")

        # Make sure all our opts have run the initial point just once.
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', val=1.0))
        comp = model.add_subsystem('comp1', DataSave())
        model.connect('p1.x', 'comp1.x')

        model.add_design_var('p1.x', lower=-100.0, upper=100.0)
        model.add_objective('comp1.y')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['print_results'] = False
        prob.driver.options['optimizer'] = 'NSGA2'
        prob.driver.opt_settings['maxGen'] = 1
        prob.driver.opt_settings['PrintOut'] = 0

        prob.setup()
        prob.run_driver()

        self.assertEqual(comp.visited_points[0], 1.0)
        self.assertNotEqual(comp.visited_points[1], 1.0)

    def test_initial_run_SLSQP(self):
        _, local_opt = set_pyoptsparse_opt('SLSQP')
        if local_opt != 'SLSQP':
            raise unittest.SkipTest("pyoptsparse is not providing SLSQP")

        # Make sure all our opts have run the initial point just once.
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', val=1.0))
        comp = model.add_subsystem('comp1', DataSave())
        model.connect('p1.x', 'comp1.x')

        model.add_design_var('p1.x', lower=-100.0, upper=100.0)
        model.add_objective('comp1.y')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['print_results'] = False
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['IPRINT'] = -1

        prob.setup()
        prob.run_driver()

        self.assertEqual(comp.visited_points[0], 1.0)
        self.assertNotEqual(comp.visited_points[1], 1.0)

    def test_initial_run_SNOPT(self):
        _, local_opt = set_pyoptsparse_opt('SNOPT')
        if local_opt != 'SNOPT':
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT")

        # Make sure all our opts have run the initial point just once.
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', val=1.0))
        comp = model.add_subsystem('comp1', DataSave())
        model.connect('p1.x', 'comp1.x')

        model.add_design_var('p1.x', lower=-100.0, upper=100.0)
        model.add_objective('comp1.y')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['print_results'] = False
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['iSumm'] = 0
        prob.driver.opt_settings['iPrint'] = 0

        prob.setup()
        prob.run_driver()

        self.assertEqual(comp.visited_points[0], 1.0)
        self.assertNotEqual(comp.visited_points[1], 1.0)

    # Seems to be a bug in numpy 1.12, fixed in later versions.
    @unittest.skipUnless(Version(np.__version__) >= Version("1.13"), "numpy >= 1.13 is required.")
    def test_initial_run_ALPSO(self):
        _, local_opt = set_pyoptsparse_opt('ALPSO')
        if local_opt != 'ALPSO':
            raise unittest.SkipTest("pyoptsparse is not providing ALPSO")

        # Make sure all our opts have run the initial point just once.
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', val=1.0))
        comp = model.add_subsystem('comp1', DataSave())
        model.connect('p1.x', 'comp1.x')

        model.add_design_var('p1.x', lower=-100.0, upper=100.0)
        model.add_objective('comp1.y')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['print_results'] = False
        prob.driver.options['optimizer'] = 'ALPSO'
        prob.driver.opt_settings['fileout'] = 0

        prob.setup()
        prob.run_driver()

        self.assertEqual(comp.visited_points[0], 1.0)
        self.assertNotEqual(comp.visited_points[1], 1.0)

    def test_initial_run_PSQP(self):
        _, local_opt = set_pyoptsparse_opt('PSQP')
        if local_opt != 'PSQP':
            raise unittest.SkipTest("pyoptsparse is not providing PSQP")

        # Make sure all our opts have run the initial point just once.
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', val=1.0))
        comp = model.add_subsystem('comp1', DataSave())
        model.connect('p1.x', 'comp1.x')

        model.add_design_var('p1.x', lower=-100.0, upper=100.0)
        model.add_objective('comp1.y')
        model.add_constraint('p1.x', lower=-200.0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['print_results'] = False
        prob.driver.options['optimizer'] = 'PSQP'
        prob.driver.opt_settings['IPRINT'] = 0

        prob.setup()
        prob.run_driver()

        self.assertEqual(comp.visited_points[0], 1.0)
        self.assertNotEqual(comp.visited_points[1], 1.0)

    def test_initial_run_CONMIN(self):
        _, local_opt = set_pyoptsparse_opt('CONMIN')
        if local_opt != 'CONMIN':
            raise unittest.SkipTest("pyoptsparse is not providing CONMIN")

        # Make sure all our opts have run the initial point just once.
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', val=1.0))
        comp = model.add_subsystem('comp1', DataSave())
        model.connect('p1.x', 'comp1.x')

        model.add_design_var('p1.x', lower=-100.0, upper=100.0)
        model.add_objective('comp1.y')
        model.add_constraint('p1.x', lower=-200.0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['print_results'] = False
        prob.driver.options['optimizer'] = 'CONMIN'
        prob.driver.opt_settings['IPRINT'] = 2

        prob.setup(mode='auto')
        prob.run_driver()

        self.assertEqual(comp.visited_points[0], 1.0)
        self.assertNotEqual(comp.visited_points[1], 1.0)

    def test_pyoptsparse_missing_objective(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('x', om.IndepVarComp('x', 2.0), promotes=['*'])
        model.add_subsystem('f_x', Paraboloid(), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=0)
        prob.model.add_constraint('x', lower=0)

        prob.setup()

        with self.assertRaises(Exception) as raises_msg:
            prob.run_driver()

        exception = raises_msg.exception

        msg = "Driver requires objective to be declared"

        self.assertEqual(exception.args[0], msg)

    def test_pyoptsparse_invalid_desvar_values(self):

        expected_err = ("The following design variable initial conditions are out of their specified "
                        "bounds:"
                        "\n  paraboloid.x"
                        "\n    val: [100.]"
                        "\n    lower: -50.0"
                        "\n    upper: 50.0"
                        "\n  paraboloid.y"
                        "\n    val: [-200.]"
                        "\n    lower: -50.0"
                        "\n    upper: 50.0"
                        "\nSet the initial value of the design variable to a valid value or set "
                        "the driver option['invalid_desvar_behavior'] to 'ignore'."
                        "\nThis warning will become an error by default in OpenMDAO version 3.25.")

        for option in ['warn', 'raise', 'ignore']:
            with self.subTest(f'invalid_desvar_behavior = {option}'):
                # build the model
                prob = om.Problem()

                prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

                # setup the optimization
                prob.driver = pyOptSparseDriver(print_results=False, invalid_desvar_behavior=option)
                prob.driver.options['optimizer'] = 'SLSQP'

                prob.model.add_design_var('paraboloid.x', lower=-50, upper=50)
                prob.model.add_design_var('paraboloid.y', lower=-50, upper=50)
                prob.model.add_objective('paraboloid.f')

                prob.setup()

                # Set initial values.
                prob.set_val('paraboloid.x', 100.0)
                prob.set_val('paraboloid.y', -200.0)

                # run the optimization
                if option == 'ignore':
                    prob.run_driver()
                elif option == 'raise':
                    with self.assertRaises(ValueError) as ctx:
                        prob.run_driver()
                    self.assertEqual(str(ctx.exception), expected_err)
                else:
                    with assert_warning(om.DriverWarning, expected_err):
                        prob.run_driver()

                if option != 'raise':
                    assert_near_equal(prob.get_val('paraboloid.x'), 6.66666666, tolerance=1.0E-5)
                    assert_near_equal(prob.get_val('paraboloid.y'), -7.33333333, tolerance=1.0E-5)
                    assert_near_equal(prob.get_val('paraboloid.f'), -27.33333333, tolerance=1.0E-5)

    def test_signal_handler_SNOPT(self):
        _, local_opt = set_pyoptsparse_opt('SNOPT')
        if local_opt != 'SNOPT':
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT")

        import pyoptsparse
        if not hasattr(pyoptsparse, '__version__') or \
           Version(pyoptsparse.__version__) < Version('1.1.0'):
            raise unittest.SkipTest("pyoptsparse needs to be updated to 1.1.0")

        class ParaboloidSIG(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_input('y', val=0.0)

                self.add_output('f_xy', val=0.0)

                self.declare_partials('*', '*')

                self.iter_count = 0

            def compute(self, inputs, outputs):
                self.iter_count += 1
                if self.iter_count == 1:
                    # Pretends that this was raised by a signal handler triggered by the user.
                    raise UserRequestedException('This is expected.')
                elif self.iter_count > 3:
                    raise RuntimeError('SNOPT should have stopped.')
                else:
                    # Post optimization run with optimal inputs.
                    pass

            def compute_partials(self, inputs, partials):
                x = inputs['x']
                y = inputs['y']

                partials['f_xy', 'x'] = 2.0*x - 6.0 + y
                partials['f_xy', 'y'] = 2.0*y + 8.0 + x

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('x', om.IndepVarComp('x', 2.0), promotes=['*'])
        model.add_subsystem('f_x', ParaboloidSIG(), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'

        prob.model.add_design_var('x', lower=0)
        model.add_objective('f_xy')
        prob.model.add_constraint('x', lower=0)

        prob.setup()

        prob.run_driver()

        # SNOPT return code 71 is a user-requested termination.
        code = prob.driver.pyopt_solution.optInform['value']
        self.assertEqual(code, 71)

    def test_IPOPT_basic(self):
        _, local_opt = set_pyoptsparse_opt('IPOPT')
        if local_opt != 'IPOPT':
            raise unittest.SkipTest("pyoptsparse is not providing IPOPT")

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "IPOPT"
        prob.driver.opt_settings['print_level'] = 0

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        assert_near_equal(prob['z'][0], 1.9776, 1e-3)

    @require_pyoptsparse('ParOpt')
    def test_ParOpt_basic(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "ParOpt"

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        assert_near_equal(prob['z'][0], 1.9776, 1e-3)
        assert_near_equal(prob['obj_cmp.obj'][0], 3.183, 1e-3)

    def test_error_objfun_reraise(self):
        # Tests that we re-raise any unclassified error encountered during callback eval.

        class EComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', 1.0)
                self.add_output('y', 1.0)

                self.declare_partials('*', '*', method='fd')

            def compute(self, inputs, outputs):
                raise RuntimeError('This comp will fail.')

        p = om.Problem()
        p.model.add_subsystem('comp', EComp())

        p.model.add_objective('comp.y')
        p.model.add_design_var('comp.x')

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.setup()

        with self.assertRaises(RuntimeError) as msg:
            p.run_driver()

        self.assertTrue("This comp will fail." in msg.exception.args[0])

    def test_error_gradfun_reraise(self):
        # Tests that we re-raise any unclassified error encountered during callback eval.

        class EComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', 1.0)
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = 4.8 * inputs['x'] - 3.0

            def compute_partials(self, inputs, partials):
                raise RuntimeError('This gradient will fail.')

        p = om.Problem()
        p.model.add_subsystem('comp', EComp())

        p.model.add_objective('comp.y')
        p.model.add_design_var('comp.x')

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.setup()

        with self.assertRaises(RuntimeError) as msg:
            p.run_driver()

        self.assertTrue("This gradient will fail." in msg.exception.args[0])

    def test_singular_jac_error_responses(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                              'z = 12.0'],),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['singular_jac_behavior'] = 'error'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        # This constraint produces a zero row.
        prob.model.add_constraint('parab.z', equals=12.)

        prob.setup()

        with self.assertRaises(RuntimeError) as msg:
            prob.run_driver()

        self.assertEqual(str(msg.exception),
                         "Constraints or objectives [('parab.z', inds=[0])] cannot be impacted by the design " + \
                         "variables of the problem.")

    def test_singular_jac_error_desvars(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                     om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0 - 0*z',
                                                  ]), #'foo = 0.0 * z'],),
                                     promotes_inputs=['x', 'y', 'z'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['singular_jac_behavior'] = 'error'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)

        # Design var z does not affect any quantities.
        prob.model.add_design_var('z', lower=-50, upper=50)

        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        prob.setup()

        with self.assertRaises(RuntimeError) as msg:
            prob.run_driver()

        self.assertEqual(str(msg.exception),
                         "Design variables [('z', inds=[0])] have no impact on the constraints or objective.")

    def test_singular_jac_ignore(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                              'z = 12.0'],),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['singular_jac_behavior'] = 'ignore'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        # This constraint produces a zero row.
        prob.model.add_constraint('parab.z', equals=12.)

        prob.setup()

        # Will not raise an exception.
        prob.run_driver()

    def test_singular_jac_warn(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                              'z = 12.0'],),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        # Default behavior is 'warn'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        # This constraint produces a zero row.
        prob.model.add_constraint('parab.z', equals=12.)

        prob.setup()

        msg = "Constraints or objectives [('parab.z', inds=[0])] cannot be impacted by the design variables of the problem."

        with assert_warning(UserWarning, msg):
            prob.run_driver()

    def test_singular_jac_tol(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                     om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0 - 1e-20*z',
                                                  ]), #'foo = 0.0 * z'],),
                                     promotes_inputs=['x', 'y', 'z'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)

        # Design var z does not affect any quantities.
        prob.model.add_design_var('z', lower=-50, upper=50)

        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        # Set a very low tolerance to accept '-1e-20*z' as nonzero.
        prob.driver.options['singular_jac_tol'] = 1e-25

        prob.setup()

        prob.run_driver()

    def test_constraint_alias(self):
        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                        a={'shape': (1,)},
                        y={'shape': (101,)},
                        x={'shape': (101,)},
                        z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0], equals=25)
        p.model.add_constraint('exec.z', indices=[-1], lower=20, alias="ALIAS_TEST")

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER

        p.setup()

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_near_equal(p.get_val('exec.z')[0], 25, tolerance=1e-4)
        assert_near_equal(p.get_val('exec.z')[50], -75, tolerance=1e-4)

    def test_multiple_constraints_no_alias(self):
        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                        a={'shape': (1,)},
                        y={'shape': (101,)},
                        x={'shape': (101,)},
                        z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0], equals=25)

        msg = "Constraint 'exec.z' already exists. Use the 'alias' argument to apply a second constraint"
        with self.assertRaises(RuntimeError) as msg:
            p.model.add_constraint('exec.z', indices=[-1], lower=20)

    def test_obj_and_con_same_var_different_indices(self):

        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                           a={'shape': (1,)},
                           y={'shape': (101,)},
                           x={'shape': (101,)},
                           z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.z', index=50)
        p.model.add_constraint('exec.z', indices=[0], equals=25, alias="ALIAS_TEST")

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER

        p.setup()

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_near_equal(p.get_val('exec.z')[0], 25)
        assert_near_equal(p.get_val('exec.z')[50], -75)

    def test_multi_constraint_promotes(self):
        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec, promotes=['*'])

        p.model.add_design_var('a', lower=-1000, upper=1000)
        p.model.add_objective('y', index=50)
        p.model.add_constraint('z', indices=[-1], lower=0)
        p.model.add_constraint('z', indices=[50], equals=-70, alias="ALIAS_TEST")

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER

        p.setup()

        p.set_val('x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_near_equal(p.get_val('z')[-1], 30, tolerance=1e-4)
        assert_near_equal(p.get_val('z')[50], -70, tolerance=1e-4)

    def test_overlapping_response_indices(self):
        p = om.Problem(name='overlapping_response_indices')

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0, 1], equals=25)

        # Need to fix up this test to run right
        with self.assertRaises(RuntimeError) as ctx:
            p.model.add_constraint('exec.z', indices=om.slicer[1:10], lower=20, alias="ALIAS_TEST")
            p.setup()

        self.assertEqual(str(ctx.exception),
           "\nCollected errors for problem 'overlapping_response_indices':"
           "\n   <model> <class Group>: Indices for aliases ['ALIAS_TEST'] are overlapping "
           "constraint/objective 'exec.z'.")

        p = om.Problem(name='overlapping_response_indices2')

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0, 1], equals=25)

        with self.assertRaises(RuntimeError) as ctx:
            p.model.add_constraint('exec.z', indices=[0], lower=20, alias="ALIAS_TEST")
            p.setup()

        self.assertEqual(str(ctx.exception),
           "\nCollected errors for problem 'overlapping_response_indices2':"
           "\n   <model> <class Group>: Indices for aliases ['ALIAS_TEST'] are overlapping "
           "constraint/objective 'exec.z'.")

        p = om.Problem(name='overlapping_response_indices3')

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0, 1], equals=25)

        with self.assertRaises(RuntimeError) as ctx:
            p.model.add_constraint('exec.z', indices=[1, 2], lower=20, alias="ALIAS_TEST")
            p.setup()

        self.assertEqual(str(ctx.exception),
           "\nCollected errors for problem 'overlapping_response_indices3':"
           "\n   <model> <class Group>: Indices for aliases ['ALIAS_TEST'] are overlapping "
           "constraint/objective 'exec.z'.")

        p = om.Problem(name='overlapping_response_indices4')

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0, 100], equals=25)

        with self.assertRaises(RuntimeError) as ctx:
            p.model.add_constraint('exec.z', indices=[-1], lower=20, alias="ALIAS_TEST")
            p.setup()

        self.assertEqual(str(ctx.exception),
            "\nCollected errors for problem 'overlapping_response_indices4':"
            "\n   <model> <class Group>: Indices for aliases ['ALIAS_TEST'] are overlapping "
            "constraint/objective 'exec.z'.")

    def test_constraint_aliases_standalone(self):
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

        model.add_constraint('f_xy', indices=[5], flat_indices=True, alias='a1', lower=10.0)
        model.add_constraint('f_xy', indices=[1], flat_indices=True, alias='a2', lower=0.5)

        model.add_objective('f_sum', index=-1)

        prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(con['a1'], 24.0)
        assert_near_equal(con['a2'], 24.96)

        totals = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(totals)

        # Makes sure relevancy works in this case.
        prob.run_driver()
        assert_near_equal(prob.get_val('f_sum')[-1], 160.6500)

    def test_fwd_rev_obj_constraint(self):
        # Test equality constraint
        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.z', index=50)
        p.model.add_constraint('exec.z', indices=[-1], equals=25, alias="ALIAS_TEST")


        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER

        # FWD
        p.setup(mode='fwd')

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_check_totals(p.check_totals())
        assert_near_equal(p.get_val('exec.z')[0], 25)
        assert_near_equal(p.get_val('exec.z')[50], -75)

        # REV
        p.setup(mode='rev')

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_check_totals(p.check_totals())
        assert_near_equal(p.get_val('exec.z')[0], 25)
        assert_near_equal(p.get_val('exec.z')[50], -75)

        # Test inequality constraint
        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.z', index=50)
        p.model.add_constraint('exec.z', indices=[-1], lower=30., alias="ALIAS_TEST")


        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER

        # FWD
        p.setup(mode='fwd')

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_check_totals(p.check_totals())
        assert_near_equal(p.get_val('exec.z')[0], 30., tolerance=1e-13)
        assert_near_equal(p.get_val('exec.z')[50], -70, tolerance=1e-13)

        # REV
        p.setup(mode='rev')

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_check_totals(p.check_totals())
        assert_near_equal(p.get_val('exec.z')[0], 30., tolerance=1e-13)
        assert_near_equal(p.get_val('exec.z')[50], -70, tolerance=1e-13)

    def test_fwd_rev_multi_constraint(self):
        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0], equals=25)
        p.model.add_constraint('exec.z', indices=[-1], lower=20, alias="ALIAS_TEST")

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER

        # FWD
        p.setup(mode='fwd')

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_check_totals(p.check_totals())
        assert_near_equal(p.get_val('exec.z')[0], 25, tolerance=1e-4)
        assert_near_equal(p.get_val('exec.z')[50], -75, tolerance=1e-4)

        # REV
        p.setup(mode='rev')

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_check_totals(p.check_totals())
        assert_near_equal(p.get_val('exec.z')[0], 25, tolerance=1e-4)
        assert_near_equal(p.get_val('exec.z')[50], -75, tolerance=1e-4)

    def test_fwd_rev_compute_totals_check(self):
        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0], equals=25)
        p.model.add_constraint('exec.z', indices=[-1], lower=20, alias="ALIAS_TEST")

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER

        # FWD
        p.setup(mode='fwd')

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_model()
        p.run_driver()

        J = p.compute_totals()
        assert_near_equal(J[('exec.y', 'exec.a')].flatten(), np.array([-0]))
        assert_near_equal(J[('exec.z', 'exec.a')].flatten(), np.array([1.]))
        assert_near_equal(J[('ALIAS_TEST', 'exec.a')].flatten(), np.array([1.]))

        # REV
        p.setup(mode='rev')

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_model()
        p.run_driver()

        J = p.compute_totals()
        assert_near_equal(J[('exec.y', 'exec.a')].flatten(), np.array([-0]))
        assert_near_equal(J[('exec.z', 'exec.a')].flatten(), np.array([1.]))
        assert_near_equal(J[('ALIAS_TEST', 'exec.a')].flatten(), np.array([1.]))

    def test_dynamic_coloring_w_multi_constraints(self):

        OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT', fallback=False)

        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0], equals=25)
        p.model.add_constraint('exec.z', indices=[-1], lower=20, alias="ALIAS_TEST")

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = "SNOPT"

        p.driver.declare_coloring()

        p.setup(mode='rev')

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_model()
        p.run_driver()

        J = p.compute_totals()

        assert_near_equal(J[('exec.y', 'exec.a')].flatten(), np.array([-0]))
        assert_near_equal(J[('exec.z', 'exec.a')].flatten(), np.array([1.]))
        assert_near_equal(J[('ALIAS_TEST', 'exec.a')].flatten(), np.array([1.]))
        assert_near_equal(p.get_val('exec.z')[0], 25, tolerance=1e-4)
        assert_near_equal(p.get_val('exec.z')[50], -75, tolerance=1e-4)


@unittest.skipIf(OPT is None or OPTIMIZER is None, "only run if pyoptsparse is installed.")
@use_tempdirs
class TestPyoptSparseFeature(unittest.TestCase):

    def setUp(self):
        from openmdao.utils.general_utils import set_pyoptsparse_opt
        import unittest

        OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

    def test_basic(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SLSQP"

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        assert_near_equal(prob.get_val('z', indices=0), 1.9776, 1e-3)

    def test_settings_print(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

        prob.driver.options['print_results'] = False

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        assert_near_equal(prob.get_val('z', indices=0), 1.9776, 1e-3)

    def test_slsqp_atol(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SLSQP"

        prob.driver.opt_settings['ACC'] = 1e-9

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        assert_near_equal(prob.get_val('z', indices=0), 1.9776, 1e-3)

    def test_slsqp_maxit(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped(nonlinear_solver=om.NonlinearBlockGS,
                                                      linear_solver=om.ScipyKrylov)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SLSQP"

        prob.driver.opt_settings['MAXIT'] = 3

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        assert_near_equal(prob.get_val('z', indices=0), 1.98337708, 1e-3)


class TestPyoptSparseSnoptFeature(unittest.TestCase):
    # All of these tests require SNOPT

    def setUp(self):
        OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT', fallback=False)

    def test_snopt_atol(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped(nonlinear_solver=om.NonlinearBlockGS,
                                                      linear_solver=om.ScipyKrylov)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SNOPT"

        prob.driver.opt_settings['Major feasibility tolerance'] = 1e-9

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        assert_near_equal(prob.get_val('z', indices=0), 1.9776, 1e-3)

    def test_snopt_maxit(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped(nonlinear_solver=om.NonlinearBlockGS,
                                                      linear_solver=om.ScipyKrylov)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SNOPT"

        # after upgrading to SNOPT 7.5-1.1, this test failed unless iter limit raised from 4 to 5
        prob.driver.opt_settings['Major iterations limit'] = 5

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')

        prob.run_driver()

        assert_near_equal(prob.get_val('z', indices=0), 1.9780247, 2e-3)

    def test_snopt_fd_solution(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 50.0)
        model.set_input_defaults('y', 50.0)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['gradient_method'] = 'snopt_fd'
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_snopt_fd_is_called(self):

        class ParaboloidApplyLinear(Paraboloid):
            def apply_linear(inputs, outputs, resids):
                raise Exception("OpenMDAO's finite difference has been called."
                                " snopt_fd option has failed.")

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 50.0)
        model.set_input_defaults('y', 50.0)

        model.add_subsystem('comp', ParaboloidApplyLinear(), promotes=['*'])

        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['gradient_method'] = 'snopt_fd'
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_sellar_analysis_error(self):
        # One discipline of Sellar will something raise analysis error. This is to test that
        # the iprinting doesn't get out-of-whack.

        class SellarDis1AE(om.ExplicitComponent):
            def setup(self):
                self.add_input('z', val=np.zeros(2))
                self.add_input('x', val=0.)
                self.add_input('y2', val=1.0)
                self.add_output('y1', val=1.0)
                self.fail_deriv = [2, 4]
                self.count_iter = 0
                self.failed = 0

            def setup_partials(self):
                self.declare_partials('*', '*')

            def compute(self, inputs, outputs):

                z1 = inputs['z'][0]
                z2 = inputs['z'][1]
                x1 = inputs['x']
                y2 = inputs['y2']

                outputs['y1'] = z1**2 + z2 + x1 - 0.2*y2

            def compute_partials(self, inputs, partials):

                self.count_iter += 1
                if self.count_iter in self.fail_deriv:
                    self.failed += 1
                    raise om.AnalysisError('Try again.')

                partials['y1', 'y2'] = -0.2
                partials['y1', 'z'] = np.array([[2.0 * inputs['z'][0], 1.0]])
                partials['y1', 'x'] = 1.0

        class SellarDis2AE(om.ExplicitComponent):
            def setup(self):
                self.add_input('z', val=np.zeros(2))
                self.add_input('y1', val=1.0)
                self.add_output('y2', val=1.0)

            def setup_partials(self):
                self.declare_partials('*', '*')

            def compute(self, inputs, outputs):
                z1 = inputs['z'][0]
                z2 = inputs['z'][1]
                y1 = inputs['y1']

                # Note: this may cause some issues. However, y1 is constrained to be
                # above 3.16, so lets just let it converge, and the optimizer will
                # throw it out
                if y1.real < 0.0:
                    y1 *= -1

                outputs['y2'] = y1**.5 + z1 + z2

            def compute_partials(self, inputs, J):
                y1 = inputs['y1']
                if y1.real < 0.0:
                    y1 *= -1

                J['y2', 'y1'] = .5*y1**-.5
                J['y2', 'z'] = np.array([[1.0, 1.0]])

        class SellarMDAAE(om.Group):
            def setup(self):

                model.set_input_defaults('x', 1.0)
                model.set_input_defaults('z', np.array([5.0, 2.0]))

                cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])

                cycle.add_subsystem('d1', SellarDis1AE(),
                                    promotes_inputs=['x', 'z', 'y2'],
                                    promotes_outputs=['y1'])
                cycle.add_subsystem('d2', SellarDis2AE(),
                                    promotes_inputs=['z', 'y1'],
                                    promotes_outputs=['y2'])

                self.linear_solver = om.LinearBlockGS()
                cycle.linear_solver = om.ScipyKrylov()
                cycle.nonlinear_solver = om.NonlinearBlockGS()

                self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                          z=np.array([0.0, 0.0]), x=0.0),
                                   promotes=['x', 'z', 'y1', 'y2', 'obj'])

                self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
                self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        prob = om.Problem()
        model = prob.model = SellarMDAAE()

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=2)

        prob.setup(check=False, mode='rev')

        failed, output = run_driver(prob)

        self.assertFalse(failed, "Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        assert_near_equal(prob['z'][0], 1.9776, 1e-3)
        assert_near_equal(prob['z'][1], 0.0, 1e-3)
        assert_near_equal(prob['x'], 0.0, 1e-3)

        self.assertEqual(model.cycle.d1.failed, 2)

        # Checking that iprint stack gets routinely cleaned.
        output = output.split('\n')
        self.assertEqual(output[-2], ('NL: NLBGS Converged'))

    def test_signal_set(self):
        import signal

        prob = om.Problem()
        model = prob.model

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SNOPT"
        prob.driver.options['user_terminate_signal'] = signal.SIGUSR1


class MatMultCompExact(om.ExplicitComponent):
    def __init__(self, mat, sparse=False, **kwargs):
        super().__init__(**kwargs)
        self.mat = mat
        self.sparse = sparse

    def setup(self):
        self.add_input('x', val=np.ones(self.mat.shape[1]))
        self.add_output('y', val=np.zeros(self.mat.shape[0]))

        if self.sparse:
            self.rows, self.cols = np.nonzero(self.mat)
            self.declare_partials(of='y', wrt='x', rows=self.rows, cols=self.cols)
        else:
            self.declare_partials(of='y', wrt='x')
        self.num_computes = 0

    def compute(self, inputs, outputs):
        outputs['y'] = self.mat.dot(inputs['x'])
        self.num_computes += 1

    def compute_partials(self, inputs, partials):
        """
        Compute the sparse partials.

        Parameters
        ----------
        inputs: Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials: Jacobian
            Sub-jac components written to partials[output_name, input_name].
        """
        if self.sparse:
            partials['y', 'x'] = self.mat[self.rows, self.cols]
        else:
            partials['y', 'x'] = self.mat


class MyGroup(om.Group):
    def __init__(self, size=5):
        super().__init__()
        self.size = size

    def setup(self):
        size = self.size
        self.add_subsystem('indeps', om.IndepVarComp('x', np.ones(size)))
        A = np.ones((size, size))  # force coloring to fail
        self.add_subsystem('comp1', MatMultCompExact(A))
        self.add_subsystem('comp2', om.ExecComp('y=x-1.0', x=np.zeros(size), y=np.zeros(size), has_diag_partials=True))
        self.connect('indeps.x', 'comp1.x')
        self.connect('comp1.y', 'comp2.x')
        self.add_design_var('indeps.x')
        self.add_objective('comp2.y', index=0)
        self.add_constraint('comp1.y', indices=list(range(1, size)), lower=5., upper=10.)


@unittest.skipIf(OPT is None or OPTIMIZER is None, "only run if pyoptsparse is installed.")
@use_tempdirs
class TestResizingTestCase(unittest.TestCase):
    def test_resize(self):
        # this test just verifies that pyoptsparsedriver doesn't raise an exception due
        # to mismatched sizes in the sparsity definition, so this test passes as long as
        # an exception isn't raised.
        p = om.Problem()
        model = p.model
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()

        G = model.add_subsystem("G", MyGroup(5))
        p.setup()
        p.run_driver()
        p.compute_totals()

        G.size = 10
        p.setup()
        p.run_driver()
        p.compute_totals()


if __name__ == "__main__":
    unittest.main()
