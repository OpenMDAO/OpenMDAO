import numpy as np
import openmdao.api as om


class MyInterp(om.MetaModelUnStructuredComp):
    def setup(self):
        # Training Data
        x_train = np.linspace(0, 10, 20)
        y_train = np.linspace(0, 20, 20)

        # Inputs
        self.add_input('simple_x', 0., training_data=x_train)
        self.add_input('sin_x', 0., training_data=x_train)

        # Outputs
        self.add_output('cos_x', 0., training_data=.5*np.cos(y_train))

        # Surrogate Model
        self.options['default_surrogate'] = om.ResponseSurface()


# define model with two metamodel components
model = om.Group()
model.add_subsystem('interp1', MyInterp())
model.add_subsystem('interp2', MyInterp())

# setup a problem using our dual metamodel model
prob = om.Problem(model)
prob.setup()
prob.final_setup()
