"""
This is not a real run_file. It is only used to make the n2
diagram for the notional aerostructural problem used for demonstration in the docs.
"""

from openmdao.api import Problem, Group, ExecComp, IndepVarComp, view_model, ImplicitComponent

p = Problem()
dvs = p.model.add_subsystem('design_vars', IndepVarComp(), promotes=['*'])
dvs.add_output('x_aero')
dvs.add_output('x_struct')
aerostruct = p.model.add_subsystem('aerostruct_cycle', Group(), promotes=['*'])
#note the equations don't matter... just need a simple way to get inputs and outputs there
aerostruct.add_subsystem('aero',
                         ExecComp(['w = u+x_aero', 'Cl=u+x_aero', 'Cd = u + x_aero']),
                         promotes=['*'])
aerostruct.add_subsystem('struct', ExecComp(['u = w+x_struct', 'mass=x_struct']),
                         promotes=['*'])

p.model.add_subsystem('objective', ExecComp('f=mass+Cl/Cd'), promotes=['*'])
p.model.add_subsystem('constraint', ExecComp('g=Cl'), promotes=['*'])

p.setup()

view_model(p, outfile='aerostruct_n2.html', embeddable=True, draw_potential_connections=False, show_browser=False)




