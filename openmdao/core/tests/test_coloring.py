
import os
import sys
import itertools
import pickle

import unittest
import numpy as np

from io import StringIO

from numpy.testing import assert_almost_equal
try:
    from scipy.sparse import load_npz
except ImportError:
    load_npz = None

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.array_utils import array_viz
from openmdao.utils.coloring import _compute_coloring, compute_total_coloring, Coloring
from openmdao.utils.mpi import MPI, multi_proc_exception_check
from openmdao.utils.testing_utils import use_tempdirs, set_env_vars
from openmdao.test_suite.tot_jac_builder import TotJacBuilder
from openmdao.utils.general_utils import run_driver

import openmdao.test_suite

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


# check that pyoptsparse is installed
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class CounterGroup(om.Group):
    def __init__(self, *args, **kwargs):
        self._solve_count = 0
        self._solve_nl_count = 0
        self._apply_nl_count = 0
        super().__init__(*args, **kwargs)

    def _solve_linear(self, *args, **kwargs):
        super()._solve_linear(*args, **kwargs)
        self._solve_count += 1

    def _solve_nonlinear(self, *args, **kwargs):
        super()._solve_nonlinear(*args, **kwargs)
        self._solve_nl_count += 1

    def _apply_nonlinear(self, *args, **kwargs):
        super()._apply_nonlinear(*args, **kwargs)
        self._apply_nl_count += 1


# note: size must be an even number
SIZE = 10


class DynPartialsComp(om.ExplicitComponent):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.num_computes = 0

    def setup(self):
        self.add_input('y', np.ones(self.size))
        self.add_input('x', np.ones(self.size))
        self.add_output('g', np.ones(self.size))

        # turn on dynamic partial coloring
        self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5, num_full_jacs=2, tol=1e-20)

    def compute(self, inputs, outputs):
        outputs['g'] = np.arctan(inputs['y'] / inputs['x'])
        self.num_computes += 1



def run_opt(driver_class, mode, assemble_type=None, color_info=None, derivs=True,
            recorder=None, has_lin_constraint=True, has_diag_partials=True, partial_coloring=False,
            use_vois=True, auto_ivc=False, con_alias=False, check=False, **options):

    p = om.Problem(model=CounterGroup())

    if assemble_type is not None:
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = assemble_type


    # the following were randomly generated using np.random.random(10)*2-1 to randomly
    # disperse them within a unit circle centered at the origin.
    x_init = np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                        0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886])
    y_init = np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                        -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582])
    r_init = .7

    if auto_ivc:
        p.model.set_input_defaults('x', x_init)
        p.model.set_input_defaults('y', y_init)
        p.model.set_input_defaults('r', r_init)

    else:
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['*'])
        indeps.add_output('x', x_init)
        indeps.add_output('y', y_init)
        indeps.add_output('r', r_init)

    if partial_coloring:
        arctan_yox = om.ExecComp('g=arctan(y/x)', shape=SIZE)
        arctan_yox.declare_coloring(wrt='*', method='cs', perturb_size=1e-5, num_full_jacs=2, tol=1e-20)
    else:
        arctan_yox = om.ExecComp('g=arctan(y/x)', shape=SIZE, has_diag_partials=has_diag_partials)

    p.model.add_subsystem('arctan_yox', arctan_yox)

    p.model.add_subsystem('circle', om.ExecComp('area=pi*r**2'))

    p.model.add_subsystem('r_con', om.ExecComp('g=x**2 + y**2 - r', has_diag_partials=has_diag_partials,
                                               g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

    thetas = np.linspace(0, np.pi/4, SIZE)
    p.model.add_subsystem('theta_con', om.ExecComp('g = x - theta', has_diag_partials=has_diag_partials,
                                                   g=np.ones(SIZE), x=np.ones(SIZE),
                                                   theta=thetas))
    p.model.add_subsystem('delta_theta_con', om.ExecComp('g = even - odd', has_diag_partials=has_diag_partials,
                                                         g=np.ones(SIZE//2), even=np.ones(SIZE//2),
                                                         odd=np.ones(SIZE//2)))

    p.model.add_subsystem('l_conx', om.ExecComp('g=x-1', has_diag_partials=has_diag_partials, g=np.ones(SIZE), x=np.ones(SIZE)))

    IND = np.arange(SIZE, dtype=int)
    ODD_IND = IND[1::2]  # all odd indices
    EVEN_IND = IND[0::2]  # all even indices

    if auto_ivc:
        p.model.promotes('circle', inputs=['r'])
        p.model.promotes('r_con', inputs=['r', 'x', 'y'])
        p.model.promotes('l_conx', inputs=['x'])
        p.model.promotes('arctan_yox', inputs=['x', 'y'])
    else:
        p.model.connect('r', ('circle.r', 'r_con.r'))
        p.model.connect('x', ['r_con.x', 'arctan_yox.x', 'l_conx.x'])
        p.model.connect('y', ['r_con.y', 'arctan_yox.y'])

    p.model.connect('arctan_yox.g', 'theta_con.x')
    p.model.connect('arctan_yox.g', 'delta_theta_con.even', src_indices=EVEN_IND)
    p.model.connect('arctan_yox.g', 'delta_theta_con.odd', src_indices=ODD_IND)

    p.driver = driver_class()
    if 'method' in options:
        p.model.approx_totals(method=options['method'])
        del options['method']

    min_improve_pct = options.get('min_improve_pct', 5.)
    if 'min_improve_pct' in options:
        del options['min_improve_pct']

    if 'dynamic_total_coloring' in options:
        p.driver.declare_coloring(tol=1e-15, min_improve_pct=min_improve_pct)
        del options['dynamic_total_coloring']

    p.driver.options.update(options)

    if use_vois:
        p.model.add_design_var('x')
        p.model.add_design_var('y')
        p.model.add_design_var('r', lower=.5, upper=10)

        if con_alias:

            class MuxComp(om.ExplicitComponent):
                """
                This component muxes all of the constraints in the model into a single wide vector.
                This includes a variety of sizes.
                """

                def setup(self):
                    self.add_input('r_con_g', val=np.ones(10))
                    self.add_input('theta_con_g', val=np.ones(5))
                    self.add_input('delta_theta_con_g', val=np.ones(5))
                    self.add_input('l_conx_g', val=1.0)
                    self.add_input('y', val=1.0)
                    self.add_output('con', val=np.ones(22))

                    row_col = np.arange(10)
                    val = np.ones(10)
                    self.declare_partials(of='con', wrt='r_con_g', rows=row_col, cols=row_col, val=val)

                    row = np.arange(5) + 10
                    col = np.arange(5)
                    val = np.ones(5)
                    self.declare_partials(of='con', wrt='theta_con_g', rows=row, cols=col, val=val)

                    row = np.arange(5) + 15
                    self.declare_partials(of='con', wrt='delta_theta_con_g', rows=row, cols=col, val=val)
                    self.declare_partials(of='con', wrt='l_conx_g', rows=np.array([20]), cols=np.array([0]), val=np.array([1.0]))
                    self.declare_partials(of='con', wrt='y', rows=np.array([21]), cols=np.array([0]), val=np.array([1.0]))

                def compute(self, inputs, outputs):
                    con = outputs['con']
                    con[:10] = inputs['r_con_g']
                    con[10:15] = inputs['theta_con_g']
                    con[15:20] = inputs['delta_theta_con_g']
                    con[20] = inputs['l_conx_g']
                    con[21] = inputs['y']

            p.model.add_subsystem('mux', MuxComp())
            p.model.connect('r_con.g', 'mux.r_con_g')
            p.model.connect('theta_con.g', 'mux.theta_con_g', src_indices=[0, 2, 4, 6, 8])
            p.model.connect('delta_theta_con.g', 'mux.delta_theta_con_g')
            p.model.connect('l_conx.g', 'mux.l_conx_g', src_indices=[0, ])
            p.model.connect('y', 'mux.y', src_indices=[0, ])

            p.model.add_constraint('mux.con', indices=om.slicer[0:10], equals=0)
            p.model.add_constraint('mux.con', indices=om.slicer[10:15], lower=-1e-5, upper=1e-5, alias='c2')
            p.model.add_constraint('mux.con', indices=om.slicer[15:20], lower=-1e-5, upper=1e-5, alias='c3')
            p.model.add_constraint('mux.con', indices=[20], equals=0, linear=False, alias='c4')
            p.model.add_constraint('mux.con', indices=[21], equals=0, linear=has_lin_constraint, alias='c5')

        else:
            # nonlinear constraints
            p.model.add_constraint('r_con.g', equals=0)

            p.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND)
            p.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

            # this constrains x[0] to be 1 (see definition of l_conx)
            p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

            # linear constraint (if has_lin_constraint is set)
            p.model.add_constraint('y', equals=0, indices=[0,], linear=has_lin_constraint)

        p.model.add_objective('circle.area', ref=-1)

    # setup coloring
    if color_info is not None:
        p.driver.use_fixed_coloring(color_info)

    if recorder:
        p.driver.add_recorder(recorder)

    p.setup(mode=mode, derivatives=derivs, check=check)
    if use_vois:
        p.run_driver()
    else:
        p.run_model()

    return p


@use_tempdirs
class SimulColoringPyoptSparseTestCase(unittest.TestCase):

    def setUp(self):
        # make sure no default reports run because they'll mess up run counts
        om.clear_reports()

    # turn off TESTFLO_RUNNING so that reports will be active, in order to detect a bug
    # when scaling report and coloring are both active.
    @set_env_vars(TESTFLO_RUNNING='0', OPENMDAO_REPORTS='scaling')
    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_total_coloring_snopt_auto(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False,
                          dynamic_total_coloring=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_count, 21)
        self.assertEqual(p_color.model._solve_count, 5)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_total_coloring_snopt_auto_autoivc(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False,
                    auto_ivc=True)
        p_color = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False,
                          dynamic_total_coloring=True, auto_ivc=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_count, 21)
        self.assertEqual(p_color.model._solve_count, 5)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_total_coloring_snopt_auto_dyn_partials(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False,
                          dynamic_total_coloring=True, partial_coloring=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_count, 21)
        self.assertEqual(p_color.model._solve_count, 5)

        partial_coloring = p_color.model._get_subsystem('arctan_yox')._coloring_info['coloring']
        expected = [
            "self.declare_partials(of='g', wrt='x', rows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])",
            "self.declare_partials(of='g', wrt='y', rows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])",
        ]
        decl_partials_calls = partial_coloring.get_declare_partials_calls().strip()
        for i, d in enumerate(decl_partials_calls.split('\n')):
            self.assertEqual(d.strip(), expected[i])

        fwd_solves, rev_solves = p_color.driver._coloring_info['coloring'].get_row_var_coloring('delta_theta_con.g')
        self.assertEqual(fwd_solves, 4)
        self.assertEqual(rev_solves, 0)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_total_coloring_snopt_auto_dyn_partials_assembled_jac(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', assemble_type='csc', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'auto', assemble_type='csc', optimizer='SNOPT', print_results=False,
                          dynamic_total_coloring=True, partial_coloring=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_count, 21)
        self.assertEqual(p_color.model._solve_count, 5)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_total_coloring_snopt_auto_assembled(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', assemble_type='dense', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'auto', assemble_type='dense', optimizer='SNOPT', print_results=False,
                          dynamic_total_coloring=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_count, 21)
        self.assertEqual(p_color.model._solve_count, 5)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_fwd_simul_coloring_snopt_approx_cs(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False, has_lin_constraint=False, method='cs')
        p_color = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', has_lin_constraint=False,
                          has_diag_partials=True, print_results=False,
                          dynamic_total_coloring=True, method='cs')

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_nl_count = 0
        p_color.model._solve_nl_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_nl_count, 21)
        self.assertEqual(p_color.model._solve_nl_count, 5)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_fwd_simul_coloring_snopt_approx_fd(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False, has_lin_constraint=False, method='cs')
        p_color = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', has_lin_constraint=False,
                          has_diag_partials=True, print_results=False,
                          dynamic_total_coloring=True, method='fd')

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_nl_count = 0
        p_color.model._solve_nl_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_nl_count, 21)
        self.assertEqual(p_color.model._solve_nl_count, 5)

    def test_size_zero_array_in_component(self):
        class DynamicPartialsComp(om.ExplicitComponent):
            def __init__(self, size):
                super().__init__()
                self.size = size
                self.num_computes = 0

            def setup(self):
                self.add_input('y', np.ones(self.size))
                self.add_input('x', np.ones(self.size))
                self.add_output('g', np.ones(self.size))

                self.declare_partials('*', '*', method='cs')

                # turn on dynamic partial coloring
                self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5, num_full_jacs=2, tol=1e-20,
                                    orders=20, show_summary=True, show_sparsity=True)

            def compute(self, inputs, outputs):
                outputs['g'] = np.arctan(inputs['y'] / inputs['x'])
                self.num_computes += 1

        SIZE = 0
        p = om.Problem()

        arctan_yox = p.model.add_subsystem('arctan_yox', DynamicPartialsComp(SIZE))

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['disp'] = False

        p.driver.declare_coloring(show_summary=True, show_sparsity=True)

        p.setup(mode='fwd')

        with self.assertRaises(Exception) as context:
            p.run_driver()
        self.assertEqual(str(context.exception),
                         "'arctan_yox' <class DynamicPartialsComp>: 'arctan_yox.g' is an array of size 0")

    def test_size_zero_array_declare_partials(self):
        class DynamicPartialsComp(om.ExplicitComponent):
            def __init__(self, size):
                super().__init__()
                self.size = size
                self.num_computes = 0

            def setup(self):
                self.add_input('y', np.ones(self.size))
                self.add_input('x', np.ones(self.size))
                self.add_output('g', np.ones(self.size))
                self.add_output('r', np.ones(1))

                self.declare_partials('r', 'y', method='cs')

                # turn on dynamic partial coloring
                self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5, num_full_jacs=2, tol=1e-20,
                                    orders=20, show_summary=True, show_sparsity=True)

            def compute(self, inputs, outputs):
                outputs['g'] = np.arctan(inputs['y'] / inputs['x'])
                self.num_computes += 1

        SIZE = 0
        p = om.Problem()

        arctan_yox = p.model.add_subsystem('arctan_yox', DynamicPartialsComp(SIZE))

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['disp'] = False

        p.driver.declare_coloring(show_summary=True, show_sparsity=True)

        p.setup(mode='fwd')

        with self.assertRaises(Exception) as context:
            p.run_driver()
        self.assertEqual(str(context.exception),
                         "'arctan_yox' <class DynamicPartialsComp>: 'arctan_yox.y' is an array of size 0")


    def test_dynamic_total_coloring_pyoptsparse_slsqp_auto(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        p_color = run_opt(pyOptSparseDriver, 'auto', optimizer='SLSQP', print_results=False,
                          dynamic_total_coloring=True)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_count, 21)
        self.assertEqual(p_color.model._solve_count, 5)

        # test __repr__
        rep = repr(p_color.driver._coloring_info['coloring'])
        self.assertEqual(rep.replace('L', ''), 'Coloring (direction: fwd, ncolors: 5, shape: (22, 21), pct nonzero: 13.42, tol: 1e-15')

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_print_options_total_with_coloring_fwd(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False,
                          dynamic_total_coloring=True, debug_print=['totals'])

        failed, output = run_driver(p_color)

        self.assertFalse(failed, "Optimization failed.")

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        self.assertTrue('In mode: fwd, Solving variable(s) using simul coloring:' in output)
        self.assertTrue("('indeps.y', [1, 3, 5, 7, 9])" in output)
        self.assertTrue('Elapsed Time:' in output)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_print_options_total_with_coloring_rev(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False,
                          dynamic_total_coloring=True, debug_print=['totals'])

        failed, output = run_driver(p_color)

        self.assertFalse(failed, "Optimization failed.")

        self.assertTrue('In mode: rev, Solving variable(s) using simul coloring:' in output)
        self.assertTrue("('r_con.g', [0])" in output)
        self.assertTrue('Elapsed Time:' in output)

@use_tempdirs
@unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
class SimulColoringRecordingTestCase(unittest.TestCase):

    def test_recording(self):
        # coloring involves an underlying call to run_model (and final_setup),
        # this verifies that it is handled properly by the recording setup logic
        recorder = om.SqliteRecorder('cases.sql')

        p = run_opt(pyOptSparseDriver, 'auto', assemble_type='csc', optimizer='SNOPT',
                    dynamic_total_coloring=True, print_results=False, recorder=recorder)

        cr = om.CaseReader('cases.sql')

        self.assertEqual(cr.list_cases(out_stream=None), ['rank0:pyOptSparse_SNOPT|%d' % i for i in range(p.driver.iter_count)])


@use_tempdirs
class SimulColoringPyoptSparseRevTestCase(unittest.TestCase):
    """Reverse coloring tests for pyoptsparse."""
    def setUp(self):
        # make sure no default reports run because they'll mess up run counts
        om.clear_reports()

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_rev_simul_coloring_snopt(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False,
                          dynamic_total_coloring=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 11 solves per driver iter  (11 vs 22)
        self.assertEqual(p.model._solve_count, 22)
        self.assertEqual(p_color.model._solve_count, 11)

        # improve coverage of coloring.py
        coloring = p_color.driver._coloring_info['coloring']
        coloring.display_txt()
        coloring.display(show=False)
        with open(os.devnull, 'w') as f:
            array_viz(coloring.get_dense_sparsity(), prob=p_color, stream=f)
            array_viz(coloring.get_dense_sparsity(), stream=f)

    def test_dynamic_rev_simul_coloring_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        p_color = run_opt(pyOptSparseDriver, 'rev', optimizer='SLSQP', print_results=False,
                          dynamic_total_coloring=True)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # Tests a bug where coloring ran the model when not needed.
        self.assertEqual(p_color.model.iter_count, 9)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 11 solves per driver iter  (11 vs 22)
        self.assertEqual(p.model._solve_count, 22)
        self.assertEqual(p_color.model._solve_count, 11)


@use_tempdirs
class SimulColoringScipyTestCase(unittest.TestCase):

    def setUp(self):
        # make sure no default reports run because they'll mess up run counts
        om.clear_reports()

    def test_bad_mode(self):
        p_color_fwd = run_opt(om.ScipyOptimizeDriver, 'fwd', optimizer='SLSQP', disp=False, dynamic_total_coloring=True)
        coloring = p_color_fwd.driver._coloring_info['coloring']

        with self.assertRaises(Exception) as context:
            p_color = run_opt(om.ScipyOptimizeDriver, 'rev', color_info=coloring, optimizer='SLSQP', disp=False)
        self.assertEqual(str(context.exception),
                         "Simultaneous coloring does forward solves but mode has been set to 'rev'")

    def test_dynamic_total_coloring_auto(self):

        # first, run w/o coloring
        p = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False)
        p_color = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False, dynamic_total_coloring=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_count, 21)
        self.assertEqual(p_color.model._solve_count, 5)

    def test_min_improve_pct(self):
        # first, run w/o coloring
        p = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False)
        p_color = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False, dynamic_total_coloring=True, min_improve_pct=99.)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring should have been aborted since it couldn't improve more than 99% so solve counts should match
        self.assertEqual(p.model._solve_count, 21)
        self.assertEqual(p_color.model._solve_count, 21)

    def test_dynamic_total_coloring_auto_con_alias(self):
        # This test makes sure that coloring works with aliased constraints.

        # first, run w/o coloring
        p = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False, con_alias=True)
        p_color = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False, dynamic_total_coloring=True, con_alias=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (5 vs 21)
        self.assertEqual(p.model._solve_count, 21)
        self.assertEqual(p_color.model._solve_count, 5)

    def test_problem_total_coloring_auto(self):

        p = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False, use_vois=False)
        coloring = compute_total_coloring(p,
                                          of=['r_con.g', 'theta_con.g', 'delta_theta_con.g',
                                              'l_conx.g', 'y', 'circle.area'],
                                          wrt=['x', 'y', 'r'])
        self.assertEqual(coloring.total_solves(), 5)

    def test_problem_total_coloring_auto_mixed_vois(self):

        p = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False,)
        coloring = compute_total_coloring(p,
                                          of=['r_con.g', 'theta_con.g', 'delta_theta_con.g',
                                              'l_conx.g', 'y', 'circle.area'],
                                          wrt=['x', 'y', 'r'])
        self.assertEqual(coloring.total_solves(), 5)
        coloring.display_txt()  # leave this in because at one point it caused an exception

    def test_simul_coloring_example(self):

        SIZE = 10

        p = om.Problem()

        p.model.add_subsystem('arctan_yox', om.ExecComp('g=arctan(y/x)', has_diag_partials=True,
                                                        g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)),
                              promotes_inputs=['x', 'y'])

        p.model.add_subsystem('circle', om.ExecComp('area=pi*r**2'), promotes_inputs=['r'])

        p.model.add_subsystem('r_con', om.ExecComp('g=x**2 + y**2 - r', has_diag_partials=True,
                                                   g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)),
                              promotes_inputs=['r', 'x', 'y'])

        thetas = np.linspace(0, np.pi/4, SIZE)
        p.model.add_subsystem('theta_con', om.ExecComp('g = x - theta', has_diag_partials=True,
                                                       g=np.ones(SIZE), x=np.ones(SIZE),
                                                       theta=thetas))
        p.model.add_subsystem('delta_theta_con', om.ExecComp('g = even - odd', has_diag_partials=True,
                                                             g=np.ones(SIZE//2), even=np.ones(SIZE//2),
                                                             odd=np.ones(SIZE//2)))

        p.model.add_subsystem('l_conx', om.ExecComp('g=x-1', has_diag_partials=True, g=np.ones(SIZE), x=np.ones(SIZE)),
                              promotes_inputs=['x'])

        IND = np.arange(SIZE, dtype=int)
        ODD_IND = IND[1::2]  # all odd indices
        EVEN_IND = IND[0::2]  # all even indices

        p.model.connect('arctan_yox.g', 'theta_con.x')
        p.model.connect('arctan_yox.g', 'delta_theta_con.even', src_indices=EVEN_IND)
        p.model.connect('arctan_yox.g', 'delta_theta_con.odd', src_indices=ODD_IND)

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['disp'] = False

        # set up dynamic total coloring here
        p.driver.declare_coloring()

        p.model.add_design_var('x')
        p.model.add_design_var('y')
        p.model.add_design_var('r', lower=.5, upper=10)

        # nonlinear constraints
        p.model.add_constraint('r_con.g', equals=0)

        p.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND)
        p.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

        # this constrains x[0] to be 1 (see definition of l_conx)
        p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

        # linear constraint
        p.model.add_constraint('y', equals=0, indices=[0,], linear=True)

        p.model.add_objective('circle.area', ref=-1)

        p.setup(mode='fwd')

        # the following were randomly generated using np.random.random(10)*2-1 to randomly
        # disperse them within a unit circle centered at the origin.
        p.set_val('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                  0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        p.set_val('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                 -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        p.set_val('r', .7)

        p.run_driver()

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

    def test_total_and_partial_coloring_example(self):

        class DynamicPartialsComp(om.ExplicitComponent):
            def __init__(self, size):
                super().__init__()
                self.size = size
                self.num_computes = 0

            def setup(self):
                self.add_input('y', np.ones(self.size))
                self.add_input('x', np.ones(self.size))
                self.add_output('g', np.ones(self.size))

                self.declare_partials('*', '*', method='cs')

                # turn on dynamic partial coloring
                self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5, num_full_jacs=2, tol=1e-20,
                                      orders=20, show_summary=True, show_sparsity=True)

            def compute(self, inputs, outputs):
                outputs['g'] = np.arctan(inputs['y'] / inputs['x'])
                self.num_computes += 1


        SIZE = 10

        p = om.Problem()
        model = p.model

        ########################################################################
        # DynamicPartialsComp is set up to do dynamic partial coloring
        arctan_yox = model.add_subsystem('arctan_yox', DynamicPartialsComp(SIZE),
                                         promotes_inputs=['x', 'y'])
        ########################################################################

        model.add_subsystem('circle', om.ExecComp('area=pi*r**2'),
                            promotes_inputs=['r'])

        model.add_subsystem('r_con', om.ExecComp('g=x**2 + y**2 - r', has_diag_partials=True,
                                                 g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)),
                            promotes_inputs=['x', 'y', 'r'])

        thetas = np.linspace(0, np.pi/4, SIZE)
        model.add_subsystem('theta_con', om.ExecComp('g = x - theta', has_diag_partials=True,
                                                       g=np.ones(SIZE), x=np.ones(SIZE),
                                                       theta=thetas))
        model.add_subsystem('delta_theta_con', om.ExecComp('g = even - odd', has_diag_partials=True,
                                                             g=np.ones(SIZE//2), even=np.ones(SIZE//2),
                                                             odd=np.ones(SIZE//2)))

        model.add_subsystem('l_conx', om.ExecComp('g=x-1', has_diag_partials=True,
                                                  g=np.ones(SIZE), x=np.ones(SIZE)),
                            promotes_inputs=['x'])

        IND = np.arange(SIZE, dtype=int)
        ODD_IND = IND[1::2]  # all odd indices
        EVEN_IND = IND[0::2]  # all even indices

        model.connect('arctan_yox.g', 'theta_con.x')
        model.connect('arctan_yox.g', 'delta_theta_con.even', src_indices=EVEN_IND)
        model.connect('arctan_yox.g', 'delta_theta_con.odd', src_indices=ODD_IND)

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['disp'] = False

        #####################################
        # set up dynamic total coloring here
        p.driver.declare_coloring(show_summary=True, show_sparsity=True)
        #####################################

        model.add_design_var('x')
        model.add_design_var('y')
        model.add_design_var('r', lower=.5, upper=10)

        # nonlinear constraints
        model.add_constraint('r_con.g', equals=0)

        model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND)
        model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

        # this constrains x[0] to be 1 (see definition of l_conx)
        model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

        # linear constraint
        model.add_constraint('y', equals=0, indices=[0,], linear=True)

        model.add_objective('circle.area', ref=-1)

        p.setup(mode='fwd')

        # the following were randomly generated using np.random.random(10)*2-1 to randomly
        # disperse them within a unit circle centered at the origin.
        p.set_val('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                  0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        p.set_val('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                  -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        p.set_val('r', .7)

        # coloring info will be displayed during run_driver.  The number of colors in the
        # partial coloring of arctan_yox should be 2 and the number of colors in the
        # total coloring should be 5.
        p.run_driver()

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # Let's see how many calls to compute we need to determine partials for arctan_yox.
        # The partial derivatives are all diagonal, so we should be able to cover them using
        # only 2 colors.
        start_calls = arctan_yox.num_computes
        arctan_yox.run_linearize()
        self.assertEqual(arctan_yox.num_computes - start_calls, 2)


@use_tempdirs
class SimulColoringRevScipyTestCase(unittest.TestCase):
    """Rev mode coloring tests."""

    def setUp(self):
        # make sure no default reports run because they'll mess up run counts
        om.clear_reports()

    def test_summary(self):
        p_color = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False, dynamic_total_coloring=True)
        coloring = p_color.driver._coloring_info['coloring']
        save_out = sys.stdout
        sys.stdout = StringIO()
        try:
            coloring.summary()
            summary = sys.stdout.getvalue()
        finally:
            sys.stdout = save_out

        self.assertTrue('Jacobian shape: (22, 21)  (13.42% nonzero)' in summary)
        self.assertTrue('FWD solves: 5   REV solves: 0' in summary)
        self.assertTrue('Total colors vs. total size: 5 vs 21  (76.2% improvement)' in summary)
        self.assertTrue('Time to compute sparsity:' in summary)
        self.assertTrue('Time to compute coloring:' in summary)

        dense_J = np.ones((50, 50), dtype=bool)
        coloring = _compute_coloring(dense_J, 'auto')
        sys.stdout = StringIO()
        try:
            coloring.summary()
            summary = sys.stdout.getvalue()
        finally:
            sys.stdout = save_out

        self.assertTrue('Jacobian shape: (50, 50)  (100.00% nonzero)' in summary)
        self.assertTrue('FWD solves: 50   REV solves: 0' in summary)
        self.assertTrue('Total colors vs. total size: 50 vs 50  (0.0% improvement)' in summary)
        self.assertFalse('Time to compute sparsity:' in summary)
        self.assertTrue('Time to compute coloring:' in summary)

    def test_repr(self):
        p_color = run_opt(om.ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False, dynamic_total_coloring=True)
        coloring = p_color.driver._coloring_info['coloring']
        rep = repr(coloring)
        self.assertEqual(rep.replace('L', ''), 'Coloring (direction: fwd, ncolors: 5, shape: (22, 21), pct nonzero: 13.42, tol: 1e-15')

        dense_J = np.ones((50, 50), dtype=bool)
        coloring = _compute_coloring(dense_J, 'auto')
        rep = repr(coloring)
        self.assertEqual(rep.replace('L', ''), 'Coloring (direction: fwd, ncolors: 50, shape: (50, 50), pct nonzero: 100.00, tol: None')

    def test_bad_mode(self):
        p_color_rev = run_opt(om.ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False, dynamic_total_coloring=True)
        coloring = p_color_rev.driver._coloring_info['coloring']

        with self.assertRaises(Exception) as context:
            p_color = run_opt(om.ScipyOptimizeDriver, 'fwd', color_info=coloring, optimizer='SLSQP', disp=False)
        self.assertEqual(str(context.exception),
                         "Simultaneous coloring does reverse solves but mode has been set to 'fwd'")

    def test_dynamic_total_coloring(self):

        p_color = run_opt(om.ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False, dynamic_total_coloring=True)
        p = run_opt(om.ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        p.model._solve_count = 0
        p_color.model._solve_count = 0

        J = p.driver._compute_totals()
        J_color = p_color.driver._compute_totals()

        # coloring saves 16 solves per driver iter  (11 vs 22)
        self.assertEqual(p.model._solve_count, 22)
        self.assertEqual(p_color.model._solve_count, 11)

    def test_dynamic_total_coloring_no_derivs(self):
        with self.assertRaises(Exception) as context:
            p_color = run_opt(om.ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False,
                              dynamic_total_coloring=True, derivs=False)
        self.assertEqual(str(context.exception),
                         "Derivative support has been turned off but compute_totals was called.")


def _test_func_name(func, num, param):
    args = []
    for p in param.args:
        try:
            arg = p.__name__
        except:
            arg = str(p)
        args.append(arg)
    return func.__name__ + '_'.join(args)


@use_tempdirs
class BidirectionalTestCase(unittest.TestCase):
    def test_eisenstat(self):
        for n in range(6, 20, 2):
            builder = TotJacBuilder.eisenstat(n)
            builder.color('auto')
            tot_size, tot_colors, fwd_solves, rev_solves, pct = builder.coloring._solves_info()
            self.assertLessEqual(tot_colors, n // 2 + 2,
                                 "Eisenstat's example of size %d required %d colors but shouldn't "
                                 "need more than %d." % (n, tot_colors, n // 2 + 2))

            builder_fwd = TotJacBuilder.eisenstat(n)
            builder_fwd.color('fwd')
            tot_size, tot_colors, fwd_solves, rev_solves, pct = builder_fwd.coloring._solves_info()
            # The columns of Eisenstat's example are pairwise nonorthogonal, so fwd coloring
            # should require n colors.
            self.assertEqual(n, tot_colors,
                             "Eisenstat's example of size %d was not constructed properly. "
                             "fwd coloring required only %d colors but should have required "
                             "%d" % (n, tot_colors, n))

    def test_arrowhead(self):
        for n in [5, 50, 55]:
            builder = TotJacBuilder(n, n)
            builder.add_row(n-1)
            builder.add_col(n-1)
            builder.add_block_diag([(1,1)] * (n-1), 0, 0)
            builder.color('auto')
            tot_size, tot_colors, fwd_solves, rev_solves, pct = builder.coloring._solves_info()
            self.assertEqual(tot_colors, 3)

    @parameterized.expand(itertools.product(
        [('n4c6-b15', 3), ('can_715', 21), ('lp_finnis', 14), ('ash608', 6), ('ash331', 6),
         ('D_6', 27), ('Harvard500', 26), ('illc1033', 5)],
        ), name_func=_test_func_name
    )
    @unittest.skipIf(load_npz is None, "scipy version too old")
    def test_bidir_coloring(self, tup):
        matname, expected_colors = tup
        matdir = os.path.join(os.path.dirname(openmdao.test_suite.__file__), 'matrices')

        # uses matrices from the sparse matrix collection website (sparse.tamu.edu)
        matfile = os.path.join(matdir, matname + '.npz')
        if not os.path.exists(matfile):
            raise unittest.SkipTest("Matrix test file were not included.")

        mat = load_npz(matfile).tocoo()
        mat.data = np.asarray(mat.data, dtype=bool)
        coloring = _compute_coloring(mat, 'auto')
        mat = None

        tot_size, tot_colors, fwd_solves, rev_solves, pct = coloring._solves_info()

        self.assertEqual(tot_colors, expected_colors)


def _get_random_mat(rows, cols):
    if MPI:
        if MPI.COMM_WORLD.rank == 0:
            mat = np.random.random(rows * cols).reshape((rows, cols)) - 0.5
            MPI.COMM_WORLD.bcast(mat, root=0)
            return mat
        else:
            return MPI.COMM_WORLD.bcast(None, root=0)
    else:
        return np.random.random(rows * cols).reshape((rows, cols)) - 0.5


def build_multipoint_problem(size=10, num_pts=4):

    np.random.seed(11)

    p = om.Problem()
    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = OPTIMIZER
    p.driver.declare_coloring()
    if OPTIMIZER == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        # p.driver.opt_settings['iSumm'] = 6

    model = p.model
    for i in range(num_pts):
        model.add_subsystem('indep%d' % i, om.IndepVarComp('x', val=np.ones(size)))
        model.add_design_var('indep%d.x' % i)

    par1 = model.add_subsystem('par1', om.ParallelGroup())
    for i in range(num_pts):
        mat = _get_random_mat(5, size)
        par1.add_subsystem('comp%d' % i, om.ExecComp('y=A.dot(x)', A=mat, x=np.ones(size), y=np.ones(5)))
        model.connect('indep%d.x' % i, 'par1.comp%d.x' % i)

    par2 = model.add_subsystem('par2', om.ParallelGroup())
    for i in range(num_pts):
        mat = _get_random_mat(size, 5)
        par2.add_subsystem('comp%d' % i, om.ExecComp('y=A.dot(x)', A=mat, x=np.ones(5), y=np.ones(size)))
        model.connect('par1.comp%d.y' % i, 'par2.comp%d.x' % i)
        par2.add_constraint('comp%d.y' % i, lower=-1.)

        model.add_subsystem('normcomp%d' % i, om.ExecComp("y=sum(x*x)", x=np.ones(size)))
        model.connect('par2.comp%d.y' % i, 'normcomp%d.x' % i)

    model.add_subsystem('obj', om.ExecComp("y=" + '+'.join(['x%d' % i for i in range(num_pts)])))

    for i in range(num_pts):
        model.connect('normcomp%d.y' % i, 'obj.x%d' % i)

    model.add_objective('obj.y')

    return p


@use_tempdirs
@unittest.skipUnless(OPTIMIZER is not None, "pyOptSparse required.")
class MatMultMultipointTestCase(unittest.TestCase):

    def test_multipoint_with_coloring(self):
        num_pts = 4
        p = build_multipoint_problem(size=10, num_pts=num_pts)
        p.setup()

        p.run_driver()

        J = p.compute_totals()

        for i in range(num_pts):
            with multi_proc_exception_check(p.comm):
                A1 = p.get_val('par1.comp%d.A'%i, get_remote=True)
            with multi_proc_exception_check(p.comm):
                A2 = p.get_val('par2.comp%d.A'%i, get_remote=True)
            norm = np.linalg.norm(J['par2.comp%d.y'%i,'indep%d.x'%i] - A2.dot(A1))
            with multi_proc_exception_check(p.comm):
                self.assertLess(norm, 1.e-7)

        print("final obj:", p['obj.y'])


@use_tempdirs
class SimulColoringVarOutputTestClass(unittest.TestCase):
    def test_multi_variable_coloring_debug_print_totals(self):
        size = 10
        num_pts = 4

        np.random.seed(11)

        p = om.Problem()
        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()
        p.driver.options['debug_print'] = ['totals']
        if OPTIMIZER == 'SNOPT':
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        #     p.driver.opt_settings['iSumm'] = 6

        model = p.model
        for i in range(num_pts):
            model.add_subsystem('indep%d' % i, om.IndepVarComp('x', val=np.ones(size)))
            model.add_design_var('indep%d.x' % i)

        par1 = model.add_subsystem('par1', om.ParallelGroup())
        for i in range(num_pts):
            mat = _get_random_mat(5, size)
            par1.add_subsystem('comp%d' % i, om.ExecComp('y=A.dot(x)', A=mat, x=np.ones(size), y=np.ones(5)))
            model.connect('indep%d.x' % i, 'par1.comp%d.x' % i)

        par2 = model.add_subsystem('par2', om.ParallelGroup())
        for i in range(num_pts):
            mat = _get_random_mat(size, 5)
            par2.add_subsystem('comp%d' % i, om.ExecComp('y=A.dot(x)', A=mat, x=np.ones(5), y=np.ones(size)))
            model.connect('par1.comp%d.y' % i, 'par2.comp%d.x' % i)
            par2.add_constraint('comp%d.y' % i, lower=-1.)

            model.add_subsystem('normcomp%d' % i, om.ExecComp("y=sum(x*x)", x=np.ones(size)))
            model.connect('par2.comp%d.y' % i, 'normcomp%d.x' % i)

        model.add_subsystem('obj', om.ExecComp("y=" + '+'.join(['x%d' % i for i in range(num_pts)])))

        for i in range(num_pts):
            model.connect('normcomp%d.y' % i, 'obj.x%d' % i)

        model.add_objective('obj.y')

        p.setup(check=False)

        failed, output = run_driver(p)

        self.assertFalse(failed, "Optimization failed.")

        self.assertTrue('In mode: fwd, Solving variable(s) using simul coloring:' in output)
        self.assertTrue("('indep0.x', [7])" in output)
        self.assertTrue("('indep1.x', [7])" in output)
        self.assertTrue("('indep2.x', [7])" in output)
        self.assertTrue("('indep3.x', [7])" in output)


class DumbComp(om.ExplicitComponent):
    def __init__(self, inputs, outputs, isizes, osizes, **kwargs):
        super().__init__(**kwargs)
        self._inames = inputs[:]
        self._onames = outputs[:]
        self._isizes = isizes[:]
        self._osizes = osizes[:]

    def setup(self):
        for name, size in zip(self._inames, self._isizes):
            self.add_input(name, val=np.zeros(size))

        for name, size in zip(self._onames, self._osizes):
            self.add_output(name, val=np.zeros(size))

        self.add_output('obj', val=0.0)

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        mult = 1.0
        for iname, oname in zip(self._inames, self._onames):
            outputs[oname] = inputs[iname] * mult

        outputs['obj'] = outputs[self._onames[0]][0]


@use_tempdirs
class SimulColoringConfigCheckTestCase(unittest.TestCase):
    def _build_model(self, ofnames, wrtnames, sizes, color, fixed):
        """
        Build a model consisting of an IndepVarComp and an ExecComp with customizable vars and sizes.
        """
        assert len(ofnames) == len(wrtnames), 'Must have same number of OF and WRT names'
        assert len(ofnames) == len(sizes), 'names and sizes must have same length'

        p = om.Problem()
        model = p.model
        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['disp'] = False

        if color == 'total':
            p.driver.declare_coloring()
            if fixed:
                # NOTE: This call line is embedded in the 2.x->3.x api conversion guide. Do not
                # modify without carefully checking the guide.
                p.driver.use_fixed_coloring()

        indeps = model.add_subsystem('indeps', om.IndepVarComp())
        for name, sz in zip(wrtnames, sizes):
            indeps.add_output(name, val=np.ones(sz))
            model.add_design_var('indeps.' + name)

        for name in ofnames:
            model.add_constraint('comp.' + name, lower=0.0)

        inames = [n + '_in' for n in ofnames]
        comp = model.add_subsystem('comp', DumbComp(inames, ofnames, sizes, sizes))
        model.add_objective('comp.obj')

        if color == 'partial':
            comp.declare_coloring()
            if fixed:
                comp.use_fixed_coloring()

        for ofname, wrtname in zip(ofnames, wrtnames):
            model.connect('indeps.' + wrtname, 'comp.' + ofname + '_in')

        p.setup()
        p.final_setup()

        return p

    def test_good_total(self):
        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='total', fixed=False)
        p.run_driver()

        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='total', fixed=True)
        p.run_driver()

    def test_good_partial(self):
        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='partial', fixed=False)
        p.run_driver()

        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='partial', fixed=True)
        p.run_driver()

    def test_added_name_total(self):
        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='total', fixed=False)
        p.run_driver()

        with self.assertRaises(RuntimeError) as ctx:
            p = self._build_model(ofnames=['w', 'x', 'y', 'z'], wrtnames=['a', 'b', 'c', 'd'],
                                sizes=[3, 4, 5, 6], color='total', fixed=True)

        self.assertEqual(str(ctx.exception),
                         "ScipyOptimizeDriver: Current coloring configuration does not match the configuration of the current model.\n   The following row vars were added: ['comp.z'].\n   The following column vars were added: ['indeps.d'].\nMake sure you don't have different problems that have the same coloring directory. Set the coloring directory by setting the value of problem.options['coloring_dir'].")

    def test_added_name_partial(self):
        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='partial', fixed=False)
        p.run_driver()

        p = self._build_model(ofnames=['w', 'x', 'y', 'z'], wrtnames=['a', 'b', 'c', 'd'],
                                sizes=[3, 4, 5, 6], color='partial', fixed=True)

        with self.assertRaises(RuntimeError) as ctx:
            p.run_driver()

        self.assertEqual(str(ctx.exception), "'comp' <class DumbComp>: Current coloring configuration does not match the configuration of the current model.\n   The following row vars were added: ['z'].\n   The following column vars were added: ['z_in'].\nMake sure you don't have different problems that have the same coloring directory. Set the coloring directory by setting the value of problem.options['coloring_dir'].")

    def test_removed_name_total(self):
        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='total', fixed=False)
        p.run_driver()


        with self.assertRaises(RuntimeError) as ctx:
            p = self._build_model(ofnames=['w', 'y'], wrtnames=['a', 'c'],
                                  sizes=[3, 5], color='total', fixed=True)
        self.assertEqual(str(ctx.exception), "ScipyOptimizeDriver: Current coloring configuration does not match the configuration of the current model.\n   The following row vars were removed: ['comp.x'].\n   The following column vars were removed: ['indeps.b'].\nMake sure you don't have different problems that have the same coloring directory. Set the coloring directory by setting the value of problem.options['coloring_dir'].")

    def test_removed_name_partial(self):
        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='partial', fixed=False)
        p.run_driver()

        p = self._build_model(ofnames=['w', 'y'], wrtnames=['a', 'c'],
                                sizes=[3, 5], color='partial', fixed=True)

        with self.assertRaises(RuntimeError) as ctx:
            p.run_driver()

        self.assertEqual(str(ctx.exception),
                         "'comp' <class DumbComp>: Current coloring configuration does not match the configuration of the current model.\n   The following row vars were removed: ['x'].\n   The following column vars were removed: ['x_in'].\nMake sure you don't have different problems that have the same coloring directory. Set the coloring directory by setting the value of problem.options['coloring_dir'].")

    def test_reordered_name_total(self):
        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='total', fixed=False)
        p.run_driver()

        with self.assertRaises(RuntimeError) as ctx:
            p = self._build_model(ofnames=['w', 'y', 'x'], wrtnames=['a', 'c', 'b'],
                                  sizes=[3, 5, 4], color='total', fixed=True)
        self.assertEqual(str(ctx.exception), "ScipyOptimizeDriver: Current coloring configuration does not match the configuration of the current model.\n   The row vars have changed order.\n   The column vars have changed order.\nMake sure you don't have different problems that have the same coloring directory. Set the coloring directory by setting the value of problem.options['coloring_dir'].")

    def test_reordered_name_partial(self):
        p = self._build_model(ofnames=['x', 'y', 'z'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='partial', fixed=False)
        p.run_driver()

        p = self._build_model(ofnames=['x', 'z', 'y'], wrtnames=['a', 'c', 'b'],
                              sizes=[3, 4, 5], color='partial', fixed=True)

        with self.assertRaises(RuntimeError) as ctx:
            p.run_driver()

        self.assertEqual(str(ctx.exception), "'comp' <class DumbComp>: Current coloring configuration does not match the configuration of the current model.\n   The row vars have changed order.\n   The column vars have changed order.\nMake sure you don't have different problems that have the same coloring directory. Set the coloring directory by setting the value of problem.options['coloring_dir'].")

    def test_size_change_total(self):
        p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='total', fixed=False)
        p.run_driver()

        with self.assertRaises(RuntimeError) as ctx:
            p = self._build_model(ofnames=['w', 'x', 'y'], wrtnames=['a', 'b', 'c'],
                                  sizes=[3, 7, 5], color='total', fixed=True)
        self.assertEqual(str(ctx.exception), "ScipyOptimizeDriver: Current coloring configuration does not match the configuration of the current model.\n   The following variables have changed sizes: ['comp.x', 'indeps.b'].\nMake sure you don't have different problems that have the same coloring directory. Set the coloring directory by setting the value of problem.options['coloring_dir'].")

    def test_size_change_partial(self):
        p = self._build_model(ofnames=['x', 'y', 'z'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 4, 5], color='partial', fixed=False)
        p.run_driver()

        p = self._build_model(ofnames=['x', 'y', 'z'], wrtnames=['a', 'b', 'c'],
                              sizes=[3, 9, 5], color='partial', fixed=True)

        with self.assertRaises(RuntimeError) as ctx:
            p.run_driver()

        self.assertEqual(str(ctx.exception), "'comp' <class DumbComp>: Current coloring configuration does not match the configuration of the current model.\n   The following variables have changed sizes: ['y', 'y_in'].\nMake sure you don't have different problems that have the same coloring directory. Set the coloring directory by setting the value of problem.options['coloring_dir'].")

    def test_bad_format(self):
        with open('_bad_format_', 'w') as f:
            f.write('asdfas asdfasdf;lkjasdflkjas df sadf;jasdf;lkja')
        with self.assertRaises(RuntimeError) as ctx:
            c = Coloring.load('_bad_format_')

        self.assertEqual(ctx.exception.args[0], "File '_bad_format_' is not a valid coloring file.")

    def test_wrong_pickle(self):
        s = 'asdfas asdfasdf;lkjasdflkjas df sadf;jasdf;lkja'
        with open('_bad_pickle_', 'wb') as f:
            pickle.dump(s, f)

        with self.assertRaises(RuntimeError) as ctx:
            c = Coloring.load('_bad_pickle_')

        self.assertEqual(ctx.exception.args[0], "File '_bad_pickle_' is not a valid coloring file.")


if __name__ == '__main__':
    unittest.main()
