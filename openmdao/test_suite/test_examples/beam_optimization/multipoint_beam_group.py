"""
This is a multipoint implementation of the beam optimization problem.


"""

from __future__ import division
import numpy as np

from openmdao.api import Group, IndepVarComp, ParallelGroup, ExecComp

from openmdao.test_suite.test_examples.beam_optimization.components.compliance_comp import ComplianceComp
from openmdao.test_suite.test_examples.beam_optimization.components.displacements_comp import DisplacementsComp
from openmdao.test_suite.test_examples.beam_optimization.components.global_stiffness_matrix_comp import GlobalStiffnessMatrixComp
from openmdao.test_suite.test_examples.beam_optimization.components.interp import BsplinesComp
from openmdao.test_suite.test_examples.beam_optimization.components.local_stiffness_matrix_comp import LocalStiffnessMatrixComp
from openmdao.test_suite.test_examples.beam_optimization.components.moment_comp import MomentOfInertiaComp
from openmdao.test_suite.test_examples.beam_optimization.components.states_comp import StatesComp
from openmdao.test_suite.test_examples.beam_optimization.components.volume_comp import VolumeComp


class MultipointBeamGroup(Group):

    def initialize(self):
        self.metadata.declare('E')
        self.metadata.declare('L')
        self.metadata.declare('b')
        self.metadata.declare('volume')
        self.metadata.declare('num_elements', 5)
        self.metadata.declare('num_cp', 50)
        self.metadata.declare('num_load_cases', 1)

    def setup(self):
        E = self.metadata['E']
        L = self.metadata['L']
        b = self.metadata['b']
        volume = self.metadata['volume']
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        num_cp = self.metadata['num_cp']
        num_load_cases = self.metadata['num_load_cases']

        inputs_comp = IndepVarComp()
        inputs_comp.add_output('h_cp', shape=num_cp)
        self.add_subsystem('inputs_comp', inputs_comp)

        comp = BsplinesComp(num_control_points=num_cp, num_points=num_elements, in_name='h_cp',
                            out_name='h')
        self.add_subsystem('interp', comp)

        I_comp = MomentOfInertiaComp(num_elements=num_elements, b=b)
        self.add_subsystem('I_comp', I_comp)

        comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
        self.add_subsystem('local_stiffness_matrix_comp', comp)

        comp = GlobalStiffnessMatrixComp(num_elements=num_elements)
        self.add_subsystem('global_stiffness_matrix_comp', comp)

        # Parallel Subsystem for load cases.
        par = self.add_subsystem('parallel', ParallelGroup())

        obj_terms = []
        for j in range(num_load_cases):

            name = 'sub_%d' % j
            sub = par.add_subsystem(name, Group())

            # Load is a sinusoidal distributed force of varying spatial frequency.
            end = 1.5 * np.pi
            if num_load_cases > 1:
                end += j * 0.5 * np.pi / (num_load_cases - 1)

            x = np.linspace(0, end, num_nodes)
            f = - np.sin(x)
            force_vector = np.zeros(2 * num_nodes)
            force_vector[0:-1:2] = f

            comp = StatesComp(num_elements=num_elements, force_vector=force_vector)
            sub.add_subsystem('states_comp', comp)

            comp = DisplacementsComp(num_elements=num_elements)
            sub.add_subsystem('displacements_comp', comp)

            comp = ComplianceComp(num_elements=num_elements, force_vector=force_vector)
            sub.add_subsystem('compliance_comp', comp)

            self.connect(
                'global_stiffness_matrix_comp.K',
                'parallel.%s.states_comp.K' % name)
            sub.connect(
                'states_comp.d',
                'displacements_comp.d')
            sub.connect(
                'displacements_comp.displacements',
                'compliance_comp.displacements')

            obj_terms.append('compliance_%d' % j)

        comp = VolumeComp(num_elements=num_elements, b=b, L=L)
        self.add_subsystem('volume_comp', comp)

        comp = ExecComp(['obj = ' + ' + '.join(obj_terms)])
        self.add_subsystem('obj_sum', comp)

        for j in range(num_load_cases):
            self.connect('parallel.sub_%d.compliance_comp.compliance' % j,
                         'obj_sum.compliance_%d' % j)

        self.connect('inputs_comp.h_cp', 'interp.h_cp')
        self.connect('interp.h', 'I_comp.h')
        self.connect('I_comp.I', 'local_stiffness_matrix_comp.I')
        self.connect(
            'local_stiffness_matrix_comp.K_local',
            'global_stiffness_matrix_comp.K_local')
        self.connect(
            'interp.h',
            'volume_comp.h')

        self.add_design_var('inputs_comp.h_cp', lower=1e-2, upper=10.)
        self.add_constraint('volume_comp.volume', equals=volume)
        self.add_objective('obj_sum.obj')
