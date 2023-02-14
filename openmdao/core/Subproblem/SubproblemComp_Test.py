import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import unittest
from sys import exit

from SubproblemComp import SubproblemComp

prob = om.Problem()

model = om.ExecComp('z = x**2 + y')
sub_model1 = om.ExecComp('x = r*cos(theta)') #promotes_inputs=['r', 'theta'],
                        #  promotes_outputs=['x'])
sub_model2 = om.ExecComp('y = r*sin(theta)') #promotes_inputs=['r','theta'],
                        #  promotes_outputs=['y'])
# print(sub_model1.list_inputs(out_stream=None, prom_name=True, units=True, shape=True, desc=True))
# exit()

subprob1 = SubproblemComp(model=sub_model1, driver=None, comm=None,
                            name=None, reports=False, prob_kwargs=None,
                            inputs=['r', 'theta'],
                            outputs=['x'])
subprob2 = SubproblemComp(model=sub_model2, driver=None, comm=None,
                            name=None, reports=False, prob_kwargs=None,
                            inputs=['r', 'theta'],
                            outputs=['y'])

prob.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                            promotes_outputs=['z'])
prob.model.add_subsystem('sub1', subprob1, promotes_inputs=['r','theta'],
                            promotes_outputs=['x'])
prob.model.add_subsystem('sub2', subprob2, promotes_inputs=['r','theta'],
                            promotes_outputs=['y'])

# prob.model.connect('supModel.x', 'sub1.x')
# prob.model.connect('supModel.y', 'sub2.y')
# prob.model.connect('sub1.r', 'sub2.r')
# prob.model.connect('sub1.theta', 'sub2.theta')

prob.setup(force_alloc_complex=True)

prob.set_val('sub1.r', 1)
prob.set_val('sub1.theta', 0.5)

prob.run_model()
cpd = prob.check_partials(method='fd')     
print(cpd)

om.n2(prob.model.sub1._subprob)