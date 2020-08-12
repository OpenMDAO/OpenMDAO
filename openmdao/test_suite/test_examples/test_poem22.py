import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

# class L2(om.ExplicitComponent):
#     """takes the 2 norm of the input"""

#     def setup(self):
#         # Inputs
#         self.add_input('vec', shape_by_conn=True, desc="Induced Velocity Factor")

#         # Outputs
#         self.add_output('val', 0.0, units="m/s",
#                         desc="Air velocity at rotor exit plane")

#     def compute(self, inputs, outputs):
#         """ Considering the entire rotor as a single disc that extracts
#         velocity uniformly from the incoming flow and converts it to
#         power."""

#         outputs['val'] = np.linalg.norm(inputs['vec'])




# class TestAdder(unittest.TestCase):

#     def test_adder(self):
#         import openmdao.api as om
#         from openmdao.test_suite.components.sellar_feature import SellarMDA

#         prob = om.Problem()
#         prob.model = om.Group()

#         indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
#         indeps.add_output('x', np.ones(10), tags="advanced")

#         prob.model.add_subsystem('L2norm', L2())
#         prob.model.connect('x', ['L2norm.vec'])
#         prob.setup()
#         prob.run_model()


# class TestParallel(unittest.TestCase):

#     def test_adder(self):
#         # import openmdao.api as om
#         # from openmdao.test_suite.components.sellar_feature import SellarMDA

#         prob = om.Problem()
#         prob.model = om.Group()

#         indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
#         indeps.add_output('x', np.ones(10), tags="advanced")

#         prob.model.add_subsystem('L2norm', L2())
#         prob.model.connect('x', ['L2norm.vec'])
#         prob.setup()
#         prob.run_model()




class B(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('x', copy_shape='y')
        self.add_output('y', shape_by_conn=True)

    def compute(self, inputs, outputs):
        pass




class C(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('x', shape=3)
        self.add_output('y', shape=9)

    def compute(self, inputs, outputs):
        pass



class D(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('x', shape_by_conn=True)
        self.add_output('y', copy_shape='x')

    def compute(self, inputs, outputs):
        pass


class E(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('x', shape_by_conn=True)

    def compute(self, inputs, outputs):
        pass





class TestPassSize(unittest.TestCase):

    def test_sys(self):

        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('y', shape_by_conn=True)

        prob.model.add_subsystem('B', B())
        prob.model.connect('A.y', ['B.x'])

        prob.model.add_subsystem('C', C())
        prob.model.connect('B.y', ['C.x'])

        prob.model.add_subsystem('D', D())
        prob.model.connect('C.y', ['D.x'])
        
        prob.model.add_subsystem('E', E())
        prob.model.connect('D.y', ['E.x'])

        prob.setup()


if __name__ == "__main__":
    unittest.main()
