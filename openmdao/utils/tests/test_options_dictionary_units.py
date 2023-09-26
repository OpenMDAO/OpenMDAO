import unittest

import openmdao.api as om

# TODO: Turn this into a test.

def units_setter(opt_dict, value):
    """
    Check and convert new units tuple into 

    Parameters
    ----------
    opt_dict : dict
        Dictionary of entries for the option.
    value : any
        New value for the option.

    Returns
    -------
    any
        Post processed value to set into the option.
    """
    new_val, new_unit = value

class AviaryComp(om.ExplicitComponent):

    def setup(self):

        self.add_input('x', 3.0)
        self.add_output('y', 3.0)

    def initialize(self):
        self.options.declare('length', default=(12.0, 'inch'), set_function=None)

    def compute(self, inputs, outputs):
        length = self.options['length'][0]

        x = inputs['x']
        outputs['y'] = length * x


class Fakeviary(om.Group):

    def setup(self):
        self.add_subsystem('mass', AviaryComp())


prob = om.Problem()
model = prob.model

model.add_subsystem('statics', Fakeviary())

prob.model_options['*'] = {'length': (2.0, 'ft')}
prob.setup()

prob.run_model()
print('The following should be 72 if the units convert correctly.')
print(prob.get_val('statics.mass.y'))
print('done')