"""
A component that computes y = func(x), where func
is a function given as an option.
"""

from types import FunctionType

import openmdao.api as om


class UnitaryFunctionComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('func', types=FunctionType, recordable=False)

    def setup(self):
        self.add_input('x')
        self.add_output('y')

    def setup_partials(self):
        self.declare_partials('y', 'x', method='fd')

    def compute(self, inputs, outputs):
        func = self.options['func']
        outputs['y'] = func(inputs['x'])
