import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import unittest
from sys import exit

from numpy import pi

from SubproblemComp import SubproblemComp

prob = om.Problem()

model = om.ExecComp('z = x**2 + y')
sub_model1 = om.ExecComp('x = r*cos(theta)') #promotes_inputs=['r', 'theta'],
                        #  promotes_outputs=['x'])
sub_model2 = om.ExecComp('y = r*sin(theta)') #promotes_inputs=['r','theta'],
                        #  promotes_outputs=['y'])

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

prob.setup(force_alloc_complex=True)

prob.set_val('r', 1)
prob.set_val('theta', pi)

cpd = prob.check_partials(method='fd')     
print(prob.get_val('z'))

om.n2(prob)