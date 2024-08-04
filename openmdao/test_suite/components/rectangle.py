
import openmdao.api as om

# Note: The following class definitions are used in feature docs


class RectangleComp(om.ExplicitComponent):
    """
    A simple Explicit Component that computes the area of a rectangle.
    """

    def setup(self):
        self.add_input('length', val=1.)
        self.add_input('width', val=1.)
        self.add_output('area', val=1.)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['area'] = inputs['length'] * inputs['width']


class RectangleCompWithTags(om.ExplicitComponent):
    """
    A simple Explicit Component that also has input and output with tags.
    """

    def setup(self):
        self.add_input('length', val=1., tags=["tag1"])
        self.add_input('width', val=1., tags=["tag2"])
        self.add_output('area', val=1., tags=["tag1"])

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['area'] = inputs['length'] * inputs['width']


class RectanglePartial(RectangleComp):

    def compute_partials(self, inputs, partials):
        partials['area', 'length'] = inputs['width']
        partials['area', 'width'] = inputs['length']


class RectangleJacVec(RectangleComp):

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'area' in d_outputs:
                if 'length' in d_inputs:
                    d_outputs['area'] += inputs['width'] * d_inputs['length']
                if 'width' in d_inputs:
                    d_outputs['area'] += inputs['length'] * d_inputs['width']
        elif mode == 'rev':
            if 'area' in d_outputs:
                if 'length' in d_inputs:
                    d_inputs['length'] += inputs['width'] * d_outputs['area']
                if 'width' in d_inputs:
                    d_inputs['width'] += inputs['length'] * d_outputs['area']


class RectangleGroup(om.Group):

    def setup(self):
        self.add_subsystem('comp1', RectanglePartial(), promotes_inputs=['width', 'length'])
        self.add_subsystem('comp2', RectangleJacVec(), promotes_inputs=['width', 'length'])
