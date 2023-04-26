import openmdao.api as om
import dymos as dm
from dymos.examples.plotting import plot_results
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
# from dymos.examples.brachistochrone import BrachistochroneODE
import matplotlib.pyplot as plt

#
# Initialize the Problem and the optimization driver
#
p = om.Problem(model=om.Group())
# p.driver = om.ScipyOptimizeDriver()
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['iSumm'] = 6
p.driver.opt_settings['Verify level'] = 3
# p.driver.declare_coloring()

#
# Create a trajectory and add a phase to it
#
# traj = p.model.add_subsystem('traj', dm.Trajectory())

# phase = traj.add_phase('phase0',
#                        dm.Phase(ode_class=BrachistochroneODE,
#                                 transcription=dm.GaussLobatto(num_segments=10)))

phase = dm.Phase(ode_class=BrachistochroneODE,
                 transcription=dm.GaussLobatto(num_segments=200, solve_segments='forward'))

#
# Set the variables
#
phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

phase.add_state('x', input_initial=True)

phase.add_state('y', input_initial=True)

phase.add_state('v', input_initial=True)

phase.add_polynomial_control('theta', order=1, #continuity=True, rate_continuity=False,
                  units='deg', lower=0.01, upper=179.9)

phase.add_parameter('g', units='m/s**2', val=9.80665)

#
# Minimize time at the end of the phase
#
phase.add_objective('time', loc='final', scaler=10)


submodel_phase = om.SubmodelComp(model=phase)


submodel_phase.add_input('t_initial')
submodel_phase.add_input('t_duration')
submodel_phase.add_input('initial_states:x')
submodel_phase.add_input('initial_states:y')
submodel_phase.add_input('initial_states:v')
# submodel_phase.add_input('states:x')
# submodel_phase.add_input('states:y')
# submodel_phase.add_input('states:v')
submodel_phase.add_input('polynomial_controls:theta')
submodel_phase.add_input('parameters:g')

submodel_phase.add_output('timeseries.time', name='timeseries:time')
submodel_phase.add_output('timeseries.time_phase', name='timeseries:time_phase')
submodel_phase.add_output('timeseries.polynomial_controls:theta', name='timeseries:polynomial_controls:theta')
# submodel_phase.add_output('timeseries.control_rates:theta_rate', name='timeseries:control_rates:theta_rate')
# submodel_phase.add_output('timeseries.control_rates:theta_rate2', name='timeseries:control_rates:theta_rate2')
submodel_phase.add_output('parameter_vals:g', name='parameter_vals:g')
submodel_phase.add_output('timeseries.states:x', name='timeseries:states:x')
submodel_phase.add_output('timeseries.states:y', name='timeseries:states:y')
submodel_phase.add_output('timeseries.states:v', name='timeseries:states:v')
submodel_phase.add_output('timeseries.state_rates:x', name='timeseries:state_rates:x')
submodel_phase.add_output('timeseries.state_rates:y', name='timeseris:state_rates:y')
submodel_phase.add_output('timeseries.state_rates:v', name='timeseries:state_rates:v')
# submodel_phase.add_output('collocation_constraint.defects:x', name='collocation_constraint:defects:x')
# submodel_phase.add_output('collocation_constraint.defects:y', name='collocation_constraint:defects:y')
# submodel_phase.add_output('collocation_constraint.defects:v', name='collocation_constraint:defects:v')
# submodel_phase.add_output('continuity_comp.defect_control_rates:theta_rate', name='continuity_comp:defect_control_rates:theta_rate')

p.model.add_subsystem('phase0', submodel_phase)
p.model.add_objective('phase0.timeseries:time', index=-1, ref=1.0)

p.model.add_design_var('phase0.t_duration')
# p.model.add_design_var('phase0.initial_states:x')
# p.model.add_design_var('phase0.initial_states:y')
# p.model.add_design_var('phase0.initial_states:v')
# p.model.add_design_var('phase0.states:x', indices=om.slicer[1:-1])
# p.model.add_design_var('phase0.states:y', indices=om.slicer[1:-1])
# p.model.add_design_var('phase0.states:v', indices=om.slicer[1:])
p.model.add_design_var('phase0.polynomial_controls:theta')

# NOTE boundary and path constraints should already be taken care of
# p.model.add_constraint('phase0.collocation_constraint:defects:x', equals=0.0)
# p.model.add_constraint('phase0.collocation_constraint:defects:y', equals=0.0)
# p.model.add_constraint('phase0.collocation_constraint:defects:v', equals=0.0)
# p.model.add_constraint('phase0.continuity_comp:defect_control_rates:theta_rate', equals=0.0)

p.model.add_constraint('phase0.timeseries:states:x', indices=-1, equals=0.0)
p.model.add_constraint('phase0.timeseries:states:y', indices=-1, equals=10.0)

p.model.linear_solver = om.DirectSolver()


#
# Setup the Problem
#
p.setup()

#
# Set the initial values
#
p['phase0.t_initial'] = 0.0
p['phase0.t_duration'] = 2.0

p.set_val('phase0.initial_states:x', 0.0)#, phase.interp('x', ys=[0, 10]))
p.set_val('phase0.initial_states:y', 10.0)#, phase.interp('y', ys=[10, 5]))
p.set_val('phase0.initial_states:v', 0.0)#, phase.interp('v', ys=[0, 9.9]))
p.set_val('phase0.polynomial_controls:theta', phase.interp('theta', ys=[5, 100.5]))

#
# Solve for the optimal trajectory
#
# dm.run_problem(p)

p.run_driver()

# Check the results
print(p.get_val('phase0.timeseries:time')[-1])