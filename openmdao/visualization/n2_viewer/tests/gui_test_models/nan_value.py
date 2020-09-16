import numpy as np
import openmdao.api as om

class SomeComponent(om.ExplicitComponent):
    def setup(self):
        self.add_input("foo", val=np.nan)
        self.add_output("bar")
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["bar"] = inputs["foo"]

problem = om.Problem()
ivc = om.IndepVarComp()
ivc.add_output("foo", 1.0)
problem.model.add_subsystem("ivc", ivc, promotes=["*"])
problem.model.add_subsystem("comp", SomeComponent(), promotes=["*"])
problem.setup()
problem.run_model()

