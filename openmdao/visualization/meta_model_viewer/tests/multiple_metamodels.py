import numpy as np
import openmdao.api as om


class CosMetaModel(om.MetaModelUnStructuredComp):
    def setup(self):
        # Training Data
        x_train = np.linspace(0, 10, 20)
        y_train = np.linspace(0, 20, 20)

        # Inputs
        self.add_input('x', 0., training_data=x_train)
        self.add_input('y', 0., training_data=y_train)

        # Outputs
        self.add_output('cos_x', 0., training_data=np.cos(x_train + y_train))

        # Surrogate Model
        self.options['default_surrogate'] = om.ResponseSurface()


class SinMetaModel(om.MetaModelUnStructuredComp):
    def setup(self):
        # Training Data
        x_train = np.linspace(0, 10, 20)
        y_train = np.linspace(0, 20, 20)

        # Inputs
        self.add_input('x', 0., training_data=x_train)
        self.add_input('y', 0., training_data=y_train)

        # Outputs
        self.add_output('sin_x', 0., training_data=np.sin(x_train + y_train))

        # Surrogate Model
        self.options['default_surrogate'] = om.ResponseSurface()


# define model with two metamodel components
model = om.Group()
cos_mm = model.add_subsystem('cos_mm', CosMetaModel())
sin_mm = model.add_subsystem('sin_mm', SinMetaModel())

# setup a problem using our dual metamodel model
prob = om.Problem(model)
prob.setup()
prob.final_setup()

