from numpy import pi

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import unittest

from openmdao.components.subproblem_comp import SubproblemComp

class TestSubproblemComp(unittest.TestCase):
    def test_subproblem_comp(self):
        prob = om.Problem()

        model = om.ExecComp('z = x**2 + y')
        submodel1 = om.ExecComp('x = r*cos(theta)')
        submodel2 = om.ExecComp('y = r*sin(theta)')

        subprob1 = SubproblemComp(model=submodel1, inputs=['r', 'theta'],
                                outputs=['x'])
        subprob2 = SubproblemComp(model=submodel2, inputs=['r', 'theta'],
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
        cpd = prob.check_partials(method='cs', out_stream=None)
        
        assert_near_equal(prob.get_val('z'), 1.0) 
