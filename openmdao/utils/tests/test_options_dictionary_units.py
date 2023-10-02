import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.units import convert_units


def units_setter(opt_meta, value):
    """
    Check and convert new units tuple into

    Parameters
    ----------
    opt_meta : dict
        Dictionary of entries for the option.
    value : any
        New value for the option.

    Returns
    -------
    any
        Post processed value to set into the option.
    """
    new_val, new_units = value
    old_val, units = opt_meta['val']

    converted_val = convert_units(new_val, new_units, units)
    return (converted_val, units)


class AviaryComp(om.ExplicitComponent):

    def setup(self):

        self.add_input('x', 3.0)
        self.add_output('y', 3.0)

    def initialize(self):
        self.options.declare('length', default=(12.0, 'inch'),
                             set_function=units_setter)

    def compute(self, inputs, outputs):
        length = self.options['length'][0]

        x = inputs['x']
        outputs['y'] = length * x


class Fakeviary(om.Group):

    def setup(self):
        self.add_subsystem('mass', AviaryComp())


class TestOptionsDictionaryUnits(unittest.TestCase):

    def test_simple(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('statics', Fakeviary())

        prob.model_options['*'] = {'length': (2.0, 'ft')}
        prob.setup()

        prob.run_model()

        y = prob.get_val('statics.mass.y')
        assert_near_equal(y, 72, 1e-6)


if __name__ == "__main__":
    unittest.main()
