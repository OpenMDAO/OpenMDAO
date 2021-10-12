import openmdao.api as om
import dymos as dm
import numpy as np
from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE


# def solution():
#     sqrt_two = np.sqrt(2)
#     val = sqrt_two * tf
#     c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
#     c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

#     ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
#     uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
#     J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) * np.exp(-2 * val) -
#                (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)
#     return ui, uf, J

# tf = np.float128(10)

# # Initialize the problem and assign the driver
# p = om.Problem(model=om.Group())
# p.options['opt_dashboard'] = True
# p.driver = om.pyOptSparseDriver()
# p.driver.options['optimizer'] = 'SNOPT'
# p.driver.opt_settings['iSumm'] = 6
# p.driver.declare_coloring()

# recorder = om.SqliteRecorder("cases.sql")
# p.driver.add_recorder(recorder)
# p.add_recorder(recorder)

# p.driver.recording_options['includes'] = []
# p.driver.recording_options['record_objectives'] = True
# p.driver.recording_options['record_constraints'] = True
# p.driver.recording_options['record_desvars'] = True
# p.driver.recording_options['record_inputs'] = True

# # Setup the trajectory and its phase
# traj = p.model.add_subsystem('traj', dm.Trajectory())

# transcription = dm.Radau(num_segments=30, order=3, compressed=False)

# phase = traj.add_phase('phase0',
#                        dm.Phase(ode_class=HyperSensitiveODE, transcription=transcription))

# phase.set_time_options(fix_initial=True, fix_duration=True)
# phase.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
# phase.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
# phase.add_control('u', opt=True, targets=['u'])

# phase.add_boundary_constraint('x', loc='final', equals=1)

# phase.add_objective('xL', loc='final')

# p.setup(check=True)

# p.set_val('traj.phase0.states:x', phase.interp('x', [1.5, 1]))
# p.set_val('traj.phase0.states:xL', phase.interp('xL', [0, 1]))
# p.set_val('traj.phase0.t_initial', 0)
# p.set_val('traj.phase0.t_duration', tf)
# p.set_val('traj.phase0.controls:u', phase.interp('u', [-0.6, 2.4]))

# #
# # Solve the problem.
# #

# dm.run_problem(p)
# # om.OptViewer("cases.sql")

import errno
import os
import unittest
from io import StringIO
import sqlite3

import numpy as np

import openmdao.api as om

from openmdao.test_suite.scripts.circuit_analysis import Resistor, Diode, Node
from openmdao.test_suite.components.ae_tests import AEComp
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesGrouped, \
    SellarProblem, SellarStateConnection, SellarProblemWithArrays, SellarDis1, SellarDis2
from openmdao.test_suite.components.sellar_feature import SellarMDAWithUnits, SellarMDA
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_problem import ParaboloidProblem
from openmdao.solvers.linesearch.tests.test_backtracking import ImplCompTwoStates

from openmdao.recorders.tests.sqlite_recorder_test_utils import assertMetadataRecorded, \
    assertDriverIterDataRecorded, assertSystemIterDataRecorded, assertSolverIterDataRecorded, \
    assertViewerDataRecorded, assertSystemMetadataIdsRecorded, assertSystemIterCoordsRecorded, \
    assertDriverDerivDataRecorded, assertProblemDerivDataRecorded

from openmdao.recorders.tests.recorder_test_utils import run_driver
from openmdao.utils.assert_utils import assert_near_equal, assert_equal_arrays, \
    assert_warning, assert_no_warning
from openmdao.utils.general_utils import determine_adder_scaler
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.om_warnings import OMDeprecationWarning

# check that pyoptsparse is installed. if it is, try to use SLSQP.
from openmdao.utils.general_utils import set_pyoptsparse_opt

OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class Cycle(om.Group):

    def setup(self):
        self.add_subsystem('d1', SellarDis1())
        self.add_subsystem('d2', SellarDis2())
        self.connect('d1.y1', 'd2.y1')

        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 20
        self.linear_solver = om.DirectSolver()

        # paths are relative, not absolute like for Driver and Problem
        self.nonlinear_solver.recording_options['includes'] = ['d1*']
        self.nonlinear_solver.recording_options['excludes'] = ['*z']


class SellarMDAConnect(om.Group):

    def setup(self):
        indeps = self.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 1.0)
        indeps.add_output('z', np.array([5.0, 2.0]))

        self.add_subsystem('cycle', Cycle())

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0))

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

        self.connect('indeps.x', ['cycle.d1.x', 'obj_cmp.x'])
        self.connect('indeps.z', ['cycle.d1.z', 'cycle.d2.z', 'obj_cmp.z'])
        self.connect('cycle.d1.y1', ['obj_cmp.y1', 'con_cmp1.y1'])
        self.connect('cycle.d2.y2', ['obj_cmp.y2', 'con_cmp2.y2'])


prob = om.Problem()

prob.model = SellarMDAConnect()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-8

prob.set_solver_print(level=0)

prob.model.add_design_var('indeps.x', lower=0, upper=10)
prob.model.add_design_var('indeps.z', lower=0, upper=10)
prob.model.add_objective('obj_cmp.obj')
prob.model.add_constraint('con_cmp1.con1', upper=0)
prob.model.add_constraint('con_cmp2.con2', upper=0)

prob.setup()

nl = prob.model._get_subsystem('cycle').nonlinear_solver
# Default includes and excludes
nl.recording_options['includes'] = ['*']
nl.recording_options['excludes'] = []

filename = "sqlite2"
recorder = om.SqliteRecorder('cases3.sql', record_viewer_data=False)
nl.add_recorder(recorder)

prob['indeps.x'] = 2.
prob['indeps.z'] = [-1., -1.]

prob.run_driver()