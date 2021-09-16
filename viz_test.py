import openmdao.api as om
import dymos as dm
import numpy as np
from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE
from test import OptViewer


def solution():
    sqrt_two = np.sqrt(2)
    val = sqrt_two * tf
    c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
    c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

    ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
    uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
    J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) * np.exp(-2 * val) -
               (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)
    return ui, uf, J

tf = np.float128(10)

# Initialize the problem and assign the driver
p = om.Problem(model=om.Group())
p.options['opt_dashboard'] = True
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['iSumm'] = 6
p.driver.declare_coloring()

p.driver.add_recorder(om.SqliteRecorder("cases.sql"))

p.driver.recording_options['includes'] = []
p.driver.recording_options['record_objectives'] = True
p.driver.recording_options['record_constraints'] = True
p.driver.recording_options['record_desvars'] = True

# Setup the trajectory and its phase
traj = p.model.add_subsystem('traj', dm.Trajectory())

transcription = dm.Radau(num_segments=30, order=3, compressed=False)

phase = traj.add_phase('phase0',
                       dm.Phase(ode_class=HyperSensitiveODE, transcription=transcription))

phase.set_time_options(fix_initial=True, fix_duration=True)
phase.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
phase.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
phase.add_control('u', opt=True, targets=['u'])

phase.add_boundary_constraint('x', loc='final', equals=1)

phase.add_objective('xL', loc='final')

p.setup(check=True)

p.set_val('traj.phase0.states:x', phase.interp('x', [1.5, 1]))
p.set_val('traj.phase0.states:xL', phase.interp('xL', [0, 1]))
p.set_val('traj.phase0.t_initial', 0)
p.set_val('traj.phase0.t_duration', tf)
p.set_val('traj.phase0.controls:u', phase.interp('u', [-0.6, 2.4]))

#
# Solve the problem.
#

OptViewer(p)


