"""Define the explicit test component (simple)."""
import scipy.sparse

from openmdao.core.explicitcomponent import ExplicitComponent


class TestExplCompSimple(ExplicitComponent):

    def setup(self):
        self.add_input('length', val=1., desc='length of rectangle')
        self.add_input('width', val=1., desc='width of rectangle')
        self.add_output('area', val=1., desc='area of rectangle')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['area'] = inputs['length'] * inputs['width']


class TestExplCompSimpleDense(TestExplCompSimple):

    def compute_partials(self, inputs, partials):
        partials['area', 'length'] = inputs['width']
        partials['area', 'width'] = inputs['length']


class TestExplCompSimpleJacVec(TestExplCompSimple):

    def setup_partials(self):
        pass  # prevent declared partials from base class

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        length = inputs['length']
        width = inputs['width']
        d_area = d_outputs['area']

        if mode == 'fwd':

            # TODO: Assignment back into the results vector doesn't work with
            # intermediate variables (see commented out line).

            if 'width' in d_inputs:
                #d_area += d_inputs['width']*length
                d_outputs['area'] += d_inputs['width']*length
            if 'length' in d_inputs:
                #d_area += d_inputs['length']*width
                d_outputs['area'] += d_inputs['length']*width
        else:
            if 'width' in d_inputs:
                d_inputs['width'] += d_area*length
            if 'length' in d_inputs:
                d_inputs['length'] += d_area*width
