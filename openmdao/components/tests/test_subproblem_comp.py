from numpy import pi

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import unittest

# from openmdao.components.subproblem_comp import SubproblemComp

class TestSubproblemComp(unittest.TestCase):
    def test_subproblem_comp(self):
        prob = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        submodel1 = om.Group()
        submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                                promotes_outputs=['r'])
        submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', pi),
                                promotes_outputs=['theta'])
        submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['x'])

        submodel2 = om.Group()
        submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                                promotes_outputs=['r'])
        submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', pi/2),
                                promotes_outputs=['theta'])
        submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['y'])

        subprob1 = om.SubproblemComp(model=submodel1, inputs=['r', 'theta'],
                                  outputs=['x'])
        subprob2 = om.SubproblemComp(model=submodel2, inputs=['r', 'theta'],
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

    def test_variable_alias(self):
        p = om.Problem()
        model = om.Group()

        model.add_subsystem('subsys', om.ExecComp('z = x**2 + y**2'))
        subprob = om.SubproblemComp(model=model, inputs=[('subsys.x', 'a'), ('subsys.y', 'b')],
                                    outputs=[('subsys.z', 'c')])

        p.model.add_subsystem('prob', subprob, promotes_inputs=['a', 'b'], promotes_outputs=['c'])
        p.setup()

        p.set_val('a', 1)
        p.set_val('b', 2)

        p.run_model()

        inputs = p.model.prob.list_inputs()
        outputs = p.model.prob.list_outputs()

        inputs = {inputs[i][0]: inputs[i][1]['val'] for i in range(len(inputs))}
        outputs = {outputs[i][0]: outputs[i][1]['val'] for i in range(len(outputs))}

        assert(inputs['a'] == 1)
        assert(inputs['b'] == 2)
        assert(outputs['c'] == 5)

    def test_unconnected_same_var(self):
        p = om.Problem()

        model = om.Group()

        model.add_subsystem('x1Comp', om.ExecComp('x1 = x*3'))
        model.add_subsystem('x2Comp', om.ExecComp('x2 = x**3'))
        model.connect('x1Comp.x1', 'model.x1')
        model.connect('x2Comp.x2', 'model.x2')
        model.add_subsystem('model', om.ExecComp('z = x1**2 + x2**2'))

        subprob = om.SubproblemComp(model=model, inputs=[('x1Comp.x', 'x'), ('x2Comp.x', 'y')],
                                    outputs=[('model.z', 'z')])

        p.model.add_subsystem('subprob', subprob, promotes=['*'])
        p.setup()

        p.set_val('x1Compx', 1)
        p.set_val('x2Compx', 2)

        p.run_model()

        inputs = p.model.subprob.list_inputs()
        outputs = p.model.subprob.list_outputs()

        inputs = {inputs[i][0]: inputs[i][1]['val'] for i in range(len(inputs))}
        outputs = {outputs[i][0]: outputs[i][1]['val'] for i in range(len(outputs))}

        assert(inputs['x'] == 1)
        assert(inputs['y'] == 2)
        assert(outputs['z'] == 73)
