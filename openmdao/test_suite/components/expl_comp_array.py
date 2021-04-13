"""Define the explicit test component (array)."""
import numpy as np
import scipy.sparse

import openmdao.api as om


class TestExplCompArray(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('thickness', default=1.)

    def setup(self):
        self.add_input('lengths', val=np.ones((2, 2)))
        self.add_input('widths', val=np.ones((2, 2)))
        self.add_output('areas', val=np.ones((2, 2)))
        self.add_output('total_volume', val=1.)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        thk = self.options['thickness']

        outputs['areas'] = inputs['lengths'] * inputs['widths']
        outputs['total_volume'] = np.sum(outputs['areas']) * thk


class TestExplCompArrayDense(TestExplCompArray):

    def compute_partials(self, inputs, partials):
        thk = self.options['thickness']

        partials['areas', 'lengths'] = np.diag(inputs['widths'].flatten())
        partials['areas', 'widths'] = np.diag(inputs['lengths'].flatten())
        partials['total_volume', 'lengths'] = inputs['widths'].flatten() * thk
        partials['total_volume', 'widths'] = inputs['lengths'].flatten() * thk


class TestExplCompArraySparse(TestExplCompArray):

    def setup_partials(self):
        self.declare_partials('areas', ['lengths', 'widths'], rows=np.arange(4), cols=np.arange(4))
        self.declare_partials('total_volume', ['lengths', 'widths'])

    def compute_partials(self, inputs, partials):
        thk = self.options['thickness']

        partials['areas', 'lengths'] = inputs['widths'].flatten()
        partials['areas', 'widths'] = inputs['lengths'].flatten()
        partials['total_volume', 'lengths'] = inputs['widths'].flatten() * thk
        partials['total_volume', 'widths'] = inputs['lengths'].flatten() * thk


class TestExplCompArrayJacVec(TestExplCompArray):

    def setup_partials(self):
        pass  # prevent declared partials from base class

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        if mode == 'fwd':
            if 'areas' in d_outputs:
                if 'widths' in d_inputs:
                    d_outputs['areas'] += d_inputs['widths']*inputs['lengths']
                if 'lengths' in d_inputs:
                    d_outputs['areas'] += d_inputs['lengths']*inputs['widths']
            if 'total_volume' in d_outputs:
                if 'widths' in d_inputs:
                    d_outputs['total_volume'] += np.sum(d_inputs['widths']*inputs['lengths'])
                if 'lengths' in d_inputs:
                    d_outputs['total_volume'] += np.sum(d_inputs['lengths']*inputs['widths'])
        else:
            if 'widths' in d_inputs:
                d_inputs['widths'] += d_outputs['areas']*inputs['lengths']
                d_inputs['widths'] += d_outputs['total_volume']*inputs['lengths']
            if 'lengths' in d_inputs:
                d_inputs['lengths'] += d_outputs['areas']*inputs['widths']
                d_inputs['lengths'] += d_outputs['total_volume']*inputs['widths']
