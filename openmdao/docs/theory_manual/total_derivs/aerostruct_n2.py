"""
This is not a real run_file. It is only used to make the n2
diagram for the notional aerostructural problem used for demonstration in the docs.
"""
import openmdao.api as om

p = om.Problem()
dvs = p.model.add_subsystem('design_vars', om.IndepVarComp(), promotes=['*'])
dvs.add_output('x_aero')
dvs.add_output('x_struct')
aerostruct = p.model.add_subsystem('aerostruct_cycle', om.Group(), promotes=['*'])
#note the equations don't matter... just need a simple way to get inputs and outputs there
aerostruct.add_subsystem('aero',
                         om.ExecComp(['w = u+x_aero', 'Cl=u+x_aero', 'Cd = u + x_aero']),
                         promotes=['*'])
aerostruct.add_subsystem('struct', om.ExecComp(['u = w+x_struct', 'mass=x_struct']),
                         promotes=['*'])

p.model.add_subsystem('objective', om.ExecComp('f=mass+Cl/Cd'), promotes=['*'])
p.model.add_subsystem('constraint', om.ExecComp('g=Cl'), promotes=['*'])

p.setup()

om.view_model(p, outfile='aerostruct_n2.html', embeddable=True, show_browser=False)




