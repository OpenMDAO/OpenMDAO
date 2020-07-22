import numpy as np

import openmdao.api as om

from openmdao.test_suite.test_examples.beam_optimization.components.moment_comp import MomentOfInertiaComp
from openmdao.test_suite.test_examples.beam_optimization.components.local_stiffness_matrix_comp import LocalStiffnessMatrixComp
from openmdao.test_suite.test_examples.beam_optimization.components.states_comp import StatesComp
from openmdao.test_suite.test_examples.beam_optimization.components.compliance_comp import ComplianceComp
from openmdao.test_suite.test_examples.beam_optimization.components.volume_comp import VolumeComp


class BeamGroup(om.Group):

    def initialize(self):
        self.options.declare('E')
        self.options.declare('L')
        self.options.declare('b')
        self.options.declare('volume')
        self.options.declare('num_elements', int)

    def setup(self):
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        volume = self.options['volume']
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        force_vector = np.zeros(2 * num_nodes)
        force_vector[-2] = -1.

        I_comp = MomentOfInertiaComp(num_elements=num_elements, b=b)
        self.add_subsystem('I_comp', I_comp, promotes_inputs=['h'])

        comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
        self.add_subsystem('local_stiffness_matrix_comp', comp)

        comp = StatesComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('states_comp', comp)

        comp = ComplianceComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('compliance_comp', comp)

        comp = VolumeComp(num_elements=num_elements, b=b, L=L)
        self.add_subsystem('volume_comp', comp, promotes_inputs=['h'])

        self.connect('I_comp.I', 'local_stiffness_matrix_comp.I')
        self.connect('local_stiffness_matrix_comp.K_local', 'states_comp.K_local')
        self.connect('states_comp.d', 'compliance_comp.displacements',
                     src_indices=np.arange(2 *num_nodes))

        self.add_design_var('h', lower=1e-2, upper=10.)
        self.add_objective('compliance_comp.compliance')
        self.add_constraint('volume_comp.volume', equals=volume)
