from numpy import pi
import openmdao.api as om
from openmdao.core.SubproblemComp import SubproblemComp


prob = om.Problem()

model = om.ExecComp('z = x**2 + y')
submodel1 = om.ExecComp('x = r*cos(theta)')
submodel2 = om.ExecComp('y = r*sin(theta)')

subprob1 = SubproblemComp(model=submodel1, inputs=['r', 'theta'],
                          outputs=['x'])
subprob2 = SubproblemComp(model=submodel2, inputs =['r', 'theta'],
                          outputs=['y'])

prob.model.add_subsystem('sub1', subprob1, promotes_inputs=['r','theta'],
                            promotes_outputs=['x'])
prob.model.add_subsystem('sub2', subprob2, promotes_inputs=['r','theta'],
                            promotes_outputs=['y'])
prob.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                            promotes_outputs=['z'])

prob.setup(force_alloc_complex=True)

prob.set_val('r', 1)
prob.set_val('theta', pi)

prob.run_model()
cpd = prob.check_partials(method='cs')    
print(f"x = {prob.get_val('x')}")
print(f"y = {prob.get_val('y')}") 
print(f"z = {prob.get_val('z')}")

# om.n2(prob)