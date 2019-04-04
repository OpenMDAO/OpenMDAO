from __future__ import print_function

import os
import shutil
import tempfile

import unittest
import numpy as np
import math

from distutils.version import LooseVersion
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import scipy
try:
    from scipy.sparse import load_npz
except ImportError:
    load_npz = None

from openmdao.api import Problem, IndepVarComp, ExecComp, DirectSolver,\
    ExplicitComponent, LinearRunOnce, ScipyOptimizeDriver, ParallelGroup, Group, \
    SqliteRecorder, CaseReader
from openmdao.utils.assert_utils import assert_rel_error, assert_warning
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.coloring import Coloring, _compute_coloring
from openmdao.utils.mpi import MPI
from openmdao.test_suite.tot_jac_builder import TotJacBuilder

import openmdao.test_suite

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


# check that pyoptsparse is installed
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class CounterGroup(Group):
    def __init__(self, *args, **kwargs):
        self._solve_count = 0
        super(CounterGroup, self).__init__(*args, **kwargs)

    def _solve_linear(self, *args, **kwargs):
        super(CounterGroup, self)._solve_linear(*args, **kwargs)
        self._solve_count += 1


# note: size must be an even number
SIZE = 10

def run_opt(driver_class, mode, assemble_type=None, color_info=None, sparsity=None, derivs=True,
            recorder=None, **options):

    p = Problem(model=CounterGroup())

    if assemble_type is not None:
        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = assemble_type

    indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['*'])

    # the following were randomly generated using np.random.random(10)*2-1 to randomly
    # disperse them within a unit circle centered at the origin.
    indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                      0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
    indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                     -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
    indeps.add_output('r', .7)

    p.model.add_subsystem('arctan_yox', ExecComp('g=arctan(y/x)', vectorize=True,
                                                 g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

    p.model.add_subsystem('circle', ExecComp('area=pi*r**2'))

    p.model.add_subsystem('r_con', ExecComp('g=x**2 + y**2 - r', vectorize=True,
                                            g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

    thetas = np.linspace(0, np.pi/4, SIZE)
    p.model.add_subsystem('theta_con', ExecComp('g = x - theta', vectorize=True,
                                                g=np.ones(SIZE), x=np.ones(SIZE),
                                                theta=thetas))
    p.model.add_subsystem('delta_theta_con', ExecComp('g = even - odd', vectorize=True,
                                                      g=np.ones(SIZE//2), even=np.ones(SIZE//2),
                                                      odd=np.ones(SIZE//2)))

    p.model.add_subsystem('l_conx', ExecComp('g=x-1', vectorize=True, g=np.ones(SIZE), x=np.ones(SIZE)))

    IND = np.arange(SIZE, dtype=int)
    ODD_IND = IND[1::2]  # all odd indices
    EVEN_IND = IND[0::2]  # all even indices

    p.model.connect('r', ('circle.r', 'r_con.r'))
    p.model.connect('x', ['r_con.x', 'arctan_yox.x', 'l_conx.x'])
    p.model.connect('y', ['r_con.y', 'arctan_yox.y'])
    p.model.connect('arctan_yox.g', 'theta_con.x')
    p.model.connect('arctan_yox.g', 'delta_theta_con.even', src_indices=EVEN_IND)
    p.model.connect('arctan_yox.g', 'delta_theta_con.odd', src_indices=ODD_IND)

    p.driver = driver_class()
    if 'approx' in options:
        p.model.approx_totals(method=options['method'])
        del options['approx']
        del options['method']

    p.driver.options.update(options)

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

    # # setup coloring
    if color_info is not None:
        p.driver.set_coloring_spec(color_info)
    elif sparsity is not None:
        p.driver.set_total_jac_sparsity(sparsity)

    if recorder:
        p.driver.add_recorder(recorder)

    p.setup(mode=mode, derivatives=derivs)
    p.run_driver()

    return p


class SimulColoringPyoptSparseTestCase(unittest.TestCase):

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_simul_coloring_snopt_fwd(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False)
        color_info = Coloring()
        color_info._fwd = [[
           [20],   # uncolored columns
           [0, 2, 4, 6, 8],   # color 1
           [1, 3, 5, 7, 9],   # color 2
           [10, 12, 14, 16, 18],   # color 3
           [11, 13, 15, 17, 19]   # color 4
        ],
        [
           [1, 11, 12, 17],   # column 0
           [2, 17],   # column 1
           [3, 13, 18],   # column 2
           [4, 18],   # column 3
           [5, 14, 19],   # column 4
           [6, 19],   # column 5
           [7, 15, 20],   # column 6
           [8, 20],   # column 7
           [9, 16, 21],   # column 8
           [10, 21],   # column 9
           [1, 12, 17],   # column 10
           [2, 17],   # column 11
           [3, 13, 18],   # column 12
           [4, 18],   # column 13
           [5, 14, 19],   # column 14
           [6, 19],   # column 15
           [7, 15, 20],   # column 16
           [8, 20],   # column 17
           [9, 16, 21],   # column 18
           [10, 21],   # column 19
           None   # column 20
        ]]
        color_info._subjac_sparsity = {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            }
        }
        p_color = run_opt(pyOptSparseDriver, 'fwd', color_info=color_info, optimizer='SNOPT', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model._solve_count - 21) / 21,
                         (p_color.model._solve_count - 21) / 5)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_total_coloring_snopt_auto(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False,
                          dynamic_simul_derivs=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model._solve_count - 21) / 21,
                         (p_color.model._solve_count - 21 * 4) / 5)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_total_coloring_snopt_auto_assembled(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', assemble_type='dense', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'auto', assemble_type='dense', optimizer='SNOPT', print_results=False,
                          dynamic_simul_derivs=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model._solve_count - 21) / 21,
                         (p_color.model._solve_count - 21 * 4) / 5)

    def test_simul_coloring_pyoptsparse_slsqp_fwd(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        color_info = Coloring()
        color_info._fwd = [[
           [20],   # uncolored columns
           [0, 2, 4, 6, 8],   # color 1
           [1, 3, 5, 7, 9],   # color 2
           [10, 12, 14, 16, 18],   # color 3
           [11, 13, 15, 17, 19]   # color 4
        ],
        [
           [1, 11, 12, 17],   # column 0
           [2, 17],   # column 1
           [3, 13, 18],   # column 2
           [4, 18],   # column 3
           [5, 14, 19],   # column 4
           [6, 19],   # column 5
           [7, 15, 20],   # column 6
           [8, 20],   # column 7
           [9, 16, 21],   # column 8
           [10, 21],   # column 9
           [1, 12, 17],   # column 10
           [2, 17],   # column 11
           [3, 13, 18],   # column 12
           [4, 18],   # column 13
           [5, 14, 19],   # column 14
           [6, 19],   # column 15
           [7, 15, 20],   # column 16
           [8, 20],   # column 17
           [9, 16, 21],   # column 18
           [10, 21],   # column 19
           None   # column 20
        ]]
        color_info._subjac_sparsity = {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            }
        }

        p_color = run_opt(pyOptSparseDriver, 'fwd', color_info=color_info, optimizer='SLSQP', print_results=False)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model._solve_count - 21) / 21,
                         (p_color.model._solve_count - 21) / 5)

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
                          dynamic_simul_derivs=True)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model._solve_count - 21) / 21,
                         (p_color.model._solve_count - 21 * 4) / 5)


@unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
class SimulColoringRecordingTestCase(unittest.TestCase):

    def setUp(self):
        from tempfile import mkdtemp
        self.dir = mkdtemp()
        self.original_path = os.getcwd()
        os.chdir(self.dir)

    def tearDown(self):
        os.chdir(self.original_path)
        try:
            shutil.rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_recording(self):
        # coloring involves an underlying call to run_model (and final_setup),
        # this verifies that it is handled properly by the recording setup logic
        recorder = SqliteRecorder('cases.sql')

        p = run_opt(pyOptSparseDriver, 'auto', assemble_type='csc', optimizer='SNOPT',
                    dynamic_simul_derivs=True, print_results=False, recorder=recorder)

        cr = CaseReader('cases.sql')

        self.assertEqual(cr.list_cases(), ['rank0:pyOptSparse_SNOPT|%d' % i for i in range(p.driver.iter_count)])


class SimulColoringPyoptSparseRevTestCase(unittest.TestCase):
    """Reverse coloring tests for pyoptsparse."""

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_simul_coloring_snopt(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False)

        color_info = Coloring()
        color_info._rev = [[
            [4, 5, 6, 7, 8, 9, 10],   # uncolored rows
            [1, 18, 19, 20, 21],   # color 1
            [0, 17, 13, 14, 15, 16],   # color 2
            [2, 11],   # color 3
            [3, 12]   # color 4
            ],
            [
            [20],   # row 0
            [0, 10, 20],   # row 1
            [1, 11, 20],   # row 2
            [2, 12, 20],   # row 3
            None,   # row 4
            None,   # row 5
            None,   # row 6
            None,   # row 7
            None,   # row 8
            None,   # row 9
            None,   # row 10
            [0],   # row 11
            [0, 10],   # row 12
            [2, 12],   # row 13
            [4, 14],   # row 14
            [6, 16],   # row 15
            [8, 18],   # row 16
            [0, 1, 10, 11],   # row 17
            [2, 3, 12, 13],   # row 18
            [4, 5, 14, 15],   # row 19
            [6, 7, 16, 17],   # row 20
            [8, 9, 18, 19]   # row 21
            ]]
        color_info._subjac_sparsity = {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            }
        }
        p_color = run_opt(pyOptSparseDriver, 'rev', color_info=color_info, optimizer='SNOPT', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - (total_solves - 1) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model._solve_count - 1) / 22,
                         (p_color.model._solve_count - 1) / 11)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_rev_simul_coloring_snopt(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False,
                          dynamic_simul_derivs=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - rev coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 22 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 1 for the uncolored case and 22 * 3 + 1 for the dynamic colored case.
        self.assertEqual((p.model._solve_count - 1) / 22,
                         (p_color.model._solve_count - 1 - 22 * 3) / 11)

    @unittest.expectedFailure
    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_fwd_simul_coloring_snopt_approx(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False,
                          dynamic_simul_derivs=True, approx=True, method='fd')

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - fwd coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 22 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 1 for the uncolored case and 22 * 3 + 1 for the dynamic colored case.
        self.assertEqual((p.model._solve_count - 1) / 22,
                         (p_color.model._solve_count - 1 - 22 * 3) / 11)

    def test_simul_coloring_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        color_info = Coloring()
        color_info._rev = [[
           [1, 4, 5, 6, 7, 8, 9, 10],
           [3, 17],
           [0, 11, 13, 14, 15, 16],
           [2, 12, 18, 19, 20, 21]
        ],
        [
           [20],
           None,
           [1, 11, 20],
           [2, 12, 20],
           None,
           None,
           None,
           None,
           None,
           None,
           None,
           [0],
           [0, 10],
           [2, 12],
           [4, 14],
           [6, 16],
           [8, 18],
           [0, 1, 10, 11],
           [2, 3, 12, 13],
           [4, 5, 14, 15],
           [6, 7, 16, 17],
           [8, 9, 18, 19]
        ]]
        color_info._subjac_sparsity = {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            }
        }

        p_color = run_opt(pyOptSparseDriver, 'rev', color_info=color_info, optimizer='SLSQP', print_results=False)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - (total_solves - 1) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model._solve_count - 1) / 22,
                         (p_color.model._solve_count - 1) / 11)

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
                          dynamic_simul_derivs=True)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # Tests a bug where coloring ran the model when not needed.
        self.assertEqual(p_color.model.iter_count, 9)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 22 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 1 for the uncolored case and 22 * 3 + 1 for the dynamic colored case.
        self.assertEqual((p.model._solve_count - 1) / 22,
                         (p_color.model._solve_count - 1 - 22 * 3) / 11)


class SimulColoringScipyTestCase(unittest.TestCase):

    def setUp(self):
        self.color_info = Coloring()
        self.color_info._fwd = [[
               [20],   # uncolored columns
               [0, 2, 4, 6, 8],   # color 1
               [1, 3, 5, 7, 9],   # color 2
               [10, 12, 14, 16, 18],   # color 3
               [11, 13, 15, 17, 19]   # color 4
            ],
            [
               [1, 11, 16, 21],   # column 0
               [2, 16],   # column 1
               [3, 12, 17],   # column 2
               [4, 17],   # column 3
               [5, 13, 18],   # column 4
               [6, 18],   # column 5
               [7, 14, 19],   # column 6
               [8, 19],   # column 7
               [9, 15, 20],   # column 8
               [10, 20],   # column 9
               [1, 11, 16],   # column 10
               [2, 16],   # column 11
               [3, 12, 17],   # column 12
               [4, 17],   # column 13
               [5, 13, 18],   # column 14
               [6, 18],   # column 15
               [7, 14, 19],   # column 16
               [8, 19],   # column 17
               [9, 15, 20],   # column 18
               [10, 20],   # column 19
               None   # column 20
            ]]

    def test_simul_coloring_fwd(self):

        # first, run w/o coloring
        p = run_opt(ScipyOptimizeDriver, 'fwd', optimizer='SLSQP', disp=False)
        p_color = run_opt(ScipyOptimizeDriver, 'fwd', color_info=self.color_info, optimizer='SLSQP', disp=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model._solve_count - 21) / 21,
                         (p_color.model._solve_count - 21) / 5)

        # check for proper handling if someone calls compute_totals on Problem with different set or different order
        # of desvars/responses than were used to define the coloring.  Behavior should be that coloring is turned off
        # and a warning is issued.
        msg = "compute_totals called using a different list of design vars and/or responses than those used " \
              "to define coloring, so coloring will be turned off.\ncoloring design vars: " \
              "['indeps.x', 'indeps.y', 'indeps.r'], current design vars: ['indeps.x', 'indeps.y', 'indeps.r']\n" \
              "coloring responses: ['circle.area', 'r_con.g', 'theta_con.g', 'delta_theta_con.g', 'l_conx.g'], " \
              "current responses: ['delta_theta_con.g', 'circle.area', 'r_con.g', 'theta_con.g', 'l_conx.g']."

        with assert_warning(UserWarning, msg):
            p_color.compute_totals(of=['delta_theta_con.g', 'circle.area', 'r_con.g', 'theta_con.g', 'l_conx.g'],
                                   wrt=['x', 'y', 'r'])

    def test_bad_mode(self):
        with self.assertRaises(Exception) as context:
            p_color = run_opt(ScipyOptimizeDriver, 'rev', color_info=self.color_info, optimizer='SLSQP', disp=False)
        self.assertEqual(str(context.exception),
                         "Simultaneous coloring does forward solves but mode has been set to 'rev'")

    def test_dynamic_total_coloring_auto(self):

        # first, run w/o coloring
        p = run_opt(ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False)
        p_color = run_opt(ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False, dynamic_simul_derivs=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - bidirectional coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model._solve_count - 21) / 21,
                         (p_color.model._solve_count - 21 * 4) / 5)

    def test_simul_coloring_example(self):

        from openmdao.api import Problem, IndepVarComp, ExecComp, ScipyOptimizeDriver
        import numpy as np

        SIZE = 10

        p = Problem()

        indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['*'])

        # the following were randomly generated using np.random.random(10)*2-1 to randomly
        # disperse them within a unit circle centered at the origin.
        indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                          0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                          -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        indeps.add_output('r', .7)

        p.model.add_subsystem('arctan_yox', ExecComp('g=arctan(y/x)', vectorize=True,
                                                    g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

        p.model.add_subsystem('circle', ExecComp('area=pi*r**2'))

        p.model.add_subsystem('r_con', ExecComp('g=x**2 + y**2 - r', vectorize=True,
                                                g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

        thetas = np.linspace(0, np.pi/4, SIZE)
        p.model.add_subsystem('theta_con', ExecComp('g = x - theta', vectorize=True,
                                                    g=np.ones(SIZE), x=np.ones(SIZE),
                                                    theta=thetas))
        p.model.add_subsystem('delta_theta_con', ExecComp('g = even - odd', vectorize=True,
                                                        g=np.ones(SIZE//2), even=np.ones(SIZE//2),
                                                        odd=np.ones(SIZE//2)))

        p.model.add_subsystem('l_conx', ExecComp('g=x-1', vectorize=True, g=np.ones(SIZE), x=np.ones(SIZE)))

        IND = np.arange(SIZE, dtype=int)
        ODD_IND = IND[1::2]  # all odd indices
        EVEN_IND = IND[0::2]  # all even indices

        p.model.connect('r', ('circle.r', 'r_con.r'))
        p.model.connect('x', ['r_con.x', 'arctan_yox.x', 'l_conx.x'])
        p.model.connect('y', ['r_con.y', 'arctan_yox.y'])
        p.model.connect('arctan_yox.g', 'theta_con.x')
        p.model.connect('arctan_yox.g', 'delta_theta_con.even', src_indices=EVEN_IND)
        p.model.connect('arctan_yox.g', 'delta_theta_con.odd', src_indices=ODD_IND)

        p.driver = ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['disp'] = False

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

        # setup coloring
        color_info = Coloring()
        color_info._fwd = [[
           [20],   # uncolored column list
           [0, 2, 4, 6, 8],   # color 1
           [1, 3, 5, 7, 9],   # color 2
           [10, 12, 14, 16, 18],   # color 3
           [11, 13, 15, 17, 19],   # color 4
        ],
        [
           [1, 11, 16, 21],   # column 0
           [2, 16],   # column 1
           [3, 12, 17],   # column 2
           [4, 17],   # column 3
           [5, 13, 18],   # column 4
           [6, 18],   # column 5
           [7, 14, 19],   # column 6
           [8, 19],   # column 7
           [9, 15, 20],   # column 8
           [10, 20],   # column 9
           [1, 11, 16],   # column 10
           [2, 16],   # column 11
           [3, 12, 17],   # column 12
           [4, 17],   # column 13
           [5, 13, 18],   # column 14
           [6, 18],   # column 15
           [7, 14, 19],   # column 16
           [8, 19],   # column 17
           [9, 15, 20],   # column 18
           [10, 20],   # column 19
           None,   # column 20
        ]]

        p.driver.set_coloring_spec(color_info)

        p.setup(mode='fwd')
        p.run_driver()

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)


class SimulColoringRevScipyTestCase(unittest.TestCase):
    """Rev mode coloring tests."""

    def setUp(self):
        self.color_info = Coloring()
        self.color_info._rev = [[
               [4, 5, 6, 7, 8, 9, 10],   # uncolored rows
               [2, 21],   # color 1
               [3, 16],   # color 2
               [1, 17, 18, 19, 20],   # color 3
               [0, 11, 12, 13, 14, 15]   # color 4
            ],
            [
               [20],   # row 0
               [0, 10, 20],   # row 1
               [1, 11, 20],   # row 2
               [2, 12, 20],   # row 3
               None,   # row 4
               None,   # row 5
               None,   # row 6
               None,   # row 7
               None,   # row 8
               None,   # row 9
               None,   # row 10
               [0, 10],   # row 11
               [2, 12],   # row 12
               [4, 14],   # row 13
               [6, 16],   # row 14
               [8, 18],   # row 15
               [0, 1, 10, 11],   # row 16
               [2, 3, 12, 13],   # row 17
               [4, 5, 14, 15],   # row 18
               [6, 7, 16, 17],   # row 19
               [8, 9, 18, 19],   # row 20
               [0]   # row 21
            ]]

    def test_simul_coloring(self):

        color_info = self.color_info

        p = run_opt(ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False)
        p_color = run_opt(ScipyOptimizeDriver, 'rev', color_info=color_info, optimizer='SLSQP', disp=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - rev coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - (total_solves - 1) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model._solve_count - 1) / 22,
                         (p_color.model._solve_count - 1) / 11)

    def test_bad_mode(self):
        with self.assertRaises(Exception) as context:
            p_color = run_opt(ScipyOptimizeDriver, 'fwd', color_info=self.color_info, optimizer='SLSQP', disp=False)
        self.assertEqual(str(context.exception),
                         "Simultaneous coloring does reverse solves but mode has been set to 'fwd'")

    def test_dynamic_total_coloring(self):

        p_color = run_opt(ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False, dynamic_simul_derivs=True)
        p = run_opt(ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - rev coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 22 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 1 for the uncolored case and 22 * 3 + 1 for the dynamic colored case.
        self.assertEqual((p.model._solve_count - 1) / 22,
                         (p_color.model._solve_count - 1 - 22 * 3) / 11)

    def test_dynamic_total_coloring_no_derivs(self):
        with self.assertRaises(Exception) as context:
            p_color = run_opt(ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False,
                              dynamic_simul_derivs=True, derivs=False)
        self.assertEqual(str(context.exception),
                         "Derivative support has been turned off but compute_totals was called.")


class SparsityTestCase(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='SparsityTestCase-')
        os.chdir(self.tempdir)

        self.sparsity = {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            }
        }

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_sparsity_snopt(self):
        # first, run without sparsity
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False)

        # run with dynamic sparsity
        p_dynamic = run_opt(pyOptSparseDriver, 'fwd', dynamic_derivs_sparsity=True,
                            optimizer='SNOPT', print_results=False)

        # run with provided sparsity
        p_sparsity = run_opt(pyOptSparseDriver, 'fwd', sparsity=self.sparsity,
                             optimizer='SNOPT', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_dynamic['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_sparsity['circle.area'], np.pi, decimal=7)

    def test_sparsity_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        # first, run without sparsity
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SLSQP', print_results=False)

        # run with dynamic sparsity
        p_dynamic = run_opt(pyOptSparseDriver, 'fwd', dynamic_derivs_sparsity=True,
                            optimizer='SLSQP', print_results=False)

        # run with provided sparsity
        p_sparsity = run_opt(pyOptSparseDriver, 'fwd', sparsity=self.sparsity,
                             optimizer='SLSQP', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_dynamic['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_sparsity['circle.area'], np.pi, decimal=7)


class BidirectionalTestCase(unittest.TestCase):
    def test_eisenstat(self):
        for n in range(6, 20, 2):
            builder = TotJacBuilder.eisenstat(n)
            builder.color('auto')
            tot_size, tot_colors, fwd_solves, rev_solves, pct = builder.coloring._solves_info()
            if tot_colors == n // 2 + 3:
                raise unittest.SkipTest("Current bicoloring algorithm requires n/2 + 3 solves, so skipping for now.")
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
        for n in range(5, 50, 55):
            builder = TotJacBuilder(n, n)
            builder.add_row(0)
            builder.add_col(0)
            builder.add_block_diag([(1,1)] * (n-1), 1, 1)
            builder.color('auto')
            tot_size, tot_colors, fwd_solves, rev_solves, pct = builder.coloring._solves_info()
            self.assertEqual(tot_colors, 3)

    @unittest.skipIf(LooseVersion(scipy.__version__) < LooseVersion("0.19.1"), "scipy version too old")
    def test_can_715(self):
        # this test is just to show the superiority of bicoloring vs. single coloring in
        # either direction.  Bicoloring gives only 21 colors in this case vs. 105 for either
        # fwd or rev.
        matdir = os.path.join(os.path.dirname(openmdao.test_suite.__file__), 'matrices')

        # uses matrix can_715 from the sparse matrix collection website
        mat = load_npz(os.path.join(matdir, 'can_715.npz')).toarray()
        mat = np.asarray(mat, dtype=bool)
        coloring = _compute_coloring(mat, 'auto')

        tot_size, tot_colors, fwd_solves, rev_solves, pct = coloring._solves_info()

        self.assertEqual(tot_colors, 21)

        # verify that unidirectional colorings are much worse (105 vs 21 for bidirectional)
        coloring = _compute_coloring(mat, 'fwd')

        tot_size, tot_colors, fwd_solves, rev_solves, pct = coloring._solves_info()

        self.assertEqual(tot_colors, 105)

        coloring = _compute_coloring(mat, 'rev')

        tot_size, tot_colors, fwd_solves, rev_solves, pct = coloring._solves_info()

        self.assertEqual(tot_colors, 105)


def _get_mat(rows, cols):
    if MPI:
        if MPI.COMM_WORLD.rank == 0:
            mat = np.random.random(rows * cols).reshape((rows, cols)) - 0.5
        else:
            mat = None
        return MPI.COMM_WORLD.bcast(mat, root=0)
    else:
        return np.random.random(rows * cols).reshape((rows, cols)) - 0.5


@unittest.skipUnless(PETScVector is not None and OPTIMIZER is not None, "PETSc and pyOptSparse required.")
class MatMultMultipointTestCase(unittest.TestCase):
    N_PROCS = 4

    def test_multipoint_with_coloring(self):
        size = 10
        num_pts = self.N_PROCS

        np.random.seed(11)

        p = Problem()
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER
        p.driver.options['dynamic_simul_derivs'] = True
        if OPTIMIZER == 'SNOPT':
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
            p.driver.opt_settings['iSumm'] = 6

        model = p.model
        for i in range(num_pts):
            model.add_subsystem('indep%d' % i, IndepVarComp('x', val=np.ones(size)))
            model.add_design_var('indep%d.x' % i)

        par1 = model.add_subsystem('par1', ParallelGroup())
        for i in range(num_pts):
            mat = _get_mat(5, size)
            par1.add_subsystem('comp%d' % i, ExecComp('y=A.dot(x)', A=mat, x=np.ones(size), y=np.ones(5)))
            model.connect('indep%d.x' % i, 'par1.comp%d.x' % i)

        par2 = model.add_subsystem('par2', ParallelGroup())
        for i in range(num_pts):
            mat = _get_mat(size, 5)
            par2.add_subsystem('comp%d' % i, ExecComp('y=A.dot(x)', A=mat, x=np.ones(5), y=np.ones(size)))
            model.connect('par1.comp%d.y' % i, 'par2.comp%d.x' % i)
            par2.add_constraint('comp%d.y' % i, lower=-1.)

            model.add_subsystem('normcomp%d' % i, ExecComp("y=sum(x*x)", x=np.ones(size)))
            model.connect('par2.comp%d.y' % i, 'normcomp%d.x' % i)

        model.add_subsystem('obj', ExecComp("y=" + '+'.join(['x%d' % i for i in range(num_pts)])))

        for i in range(num_pts):
            model.connect('normcomp%d.y' % i, 'obj.x%d' % i)

        model.add_objective('obj.y')

        p.setup()

        p.run_driver()

        J = p.compute_totals()

        for i in range(num_pts):
            vname = 'par2.comp%d.A' % i
            if vname in model._var_abs_names['input']:
                norm = np.linalg.norm(J['par2.comp%d.y'%i,'indep%d.x'%i] -
                                      getattr(par2, 'comp%d'%i)._inputs['A'].dot(getattr(par1, 'comp%d'%i)._inputs['A']))
                self.assertLess(norm, 1.e-7)
            elif vname not in model._var_allprocs_abs_names['input']:
                self.fail("Can't find variable par2.comp%d.A" % i)

        # print("final obj:", p['obj.y'])


if __name__ == '__main__':
    unittest.main()
