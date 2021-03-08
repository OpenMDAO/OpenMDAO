import unittest

import openmdao.api as om

from openmdao.utils.testing_utils import use_tempdirs


class BadOptionComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('bad', recordable=False)

    def setup(self):
        self.add_input('x')
        self.add_output('y')


@use_tempdirs
class SerializeTestCase(unittest.TestCase):
    def test_serialize_n2(self):
        p = om.Problem()

        p.model.add_subsystem('foo', BadOptionComp(bad=object()))

        p.setup()
        p.final_setup()

        om.n2(p, show_browser=False)


if __name__ == '__main__':
    unittest.main()
