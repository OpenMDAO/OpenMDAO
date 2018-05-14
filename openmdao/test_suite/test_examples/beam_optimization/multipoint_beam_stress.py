"""
This is a multipoint implementation of the beam optimization problem.

This version minimizes volume while satisfying a max bending stress constraint in each element
for each loadcase.
"""
from __future__ import division
from six.moves import range

import numpy as np

from openmdao.api import Group, IndepVarComp, ParallelGroup, ExecComp
from openmdao.components.bsplines_comp import BsplinesComp
from openmdao.components.ks_comp import KSComp

from openmdao.test_suite.test_examples.beam_optimization.components.displacements_comp import MultiDisplacementsComp
from openmdao.test_suite.test_examples.beam_optimization.components.global_stiffness_matrix_comp import GlobalStiffnessMatrixComp
from openmdao.test_suite.test_examples.beam_optimization.components.local_stiffness_matrix_comp import LocalStiffnessMatrixComp
from openmdao.test_suite.test_examples.beam_optimization.components.moment_comp import MomentOfInertiaComp
from openmdao.test_suite.test_examples.beam_optimization.components.states_comp import MultiStatesComp
from openmdao.test_suite.test_examples.beam_optimization.components.stress_comp import MultiStressComp
from openmdao.test_suite.test_examples.beam_optimization.components.volume_comp import VolumeComp


def divide_cases(ncases, nprocs):
    """
    Divide up load cases among available procs.

    Parameters
    ----------
    ncases : int
        Number of load cases.
    nprocs : int
        Number of processors.

    Returns
    -------
    list of list of int
        Integer case numbers for each proc.
    """
    data = []
    for j in range(nprocs):
        data.append([])

    wrap = 0
    for j in range(ncases):
        idx = j - wrap
        if idx >= nprocs:
            idx = 0
            wrap = j

        data[idx].append(j)

    return data


class MultipointBeamGroup(Group):
    """
    System setup for minimization of volume (i.e., mass) subject to KS aggregated bending stress constraints.
    """

    def initialize(self):
        self.options.declare('E')
        self.options.declare('L')
        self.options.declare('b')
        self.options.declare('volume')
        self.options.declare('max_bending')
        self.options.declare('num_elements', 5)
        self.options.declare('num_cp', 50)
        self.options.declare('num_load_cases', 1)
        self.options.declare('parallel_derivs', False, types=bool, allow_none=True)

    def setup(self):
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        volume = self.options['volume']
        max_bending = self.options['max_bending']
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        num_cp = self.options['num_cp']
        num_load_cases = self.options['num_load_cases']
        parallel_derivs = self.options['parallel_derivs']

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

        # Parallel Subsystem for load cases.
        par = self.add_subsystem('parallel', ParallelGroup())

        # Determine how to split cases up over the available procs.
        nprocs = self.comm.size
        divide = divide_cases(num_load_cases, nprocs)

        for j, this_proc in enumerate(divide):
            num_rhs = len(this_proc)

            name = 'sub_%d' % j
            sub = par.add_subsystem(name, Group())

            # Load is a sinusoidal distributed force of varying spatial frequency.
            force_vector = np.zeros((2 * num_nodes, num_rhs))
            for i, k in enumerate(this_proc):

                end = 1.5 * np.pi
                if num_load_cases > 1:
                    end += k * 0.5 * np.pi / (num_load_cases - 1)

                x = np.linspace(0, end, num_nodes)
                f = - np.sin(x)
                force_vector[0:-1:2, i] = f

            comp = GlobalStiffnessMatrixComp(num_elements=num_elements)
            sub.add_subsystem('global_stiffness_matrix_comp', comp)

            comp = MultiStatesComp(num_elements=num_elements, force_vector=force_vector,
                                   num_rhs=num_rhs)
            sub.add_subsystem('states_comp', comp)

            comp = MultiDisplacementsComp(num_elements=num_elements, num_rhs=num_rhs)
            sub.add_subsystem('displacements_comp', comp)

            comp = MultiStressComp(num_elements=num_elements, E=E, num_rhs=num_rhs)
            sub.add_subsystem('stress_comp', comp)

            self.connect(
                'local_stiffness_matrix_comp.K_local',
                'parallel.%s.global_stiffness_matrix_comp.K_local' % name)

            sub.connect(
                'global_stiffness_matrix_comp.K',
                'states_comp.K')

            for k in range(num_rhs):
                sub.connect(
                    'states_comp.d_%d' % k,
                    'displacements_comp.d_%d' % k)
                sub.connect(
                    'displacements_comp.displacements_%d' % k,
                    'stress_comp.displacements_%d' % k)

                comp = KSComp(width=num_elements)
                comp.options['upper'] = max_bending
                sub.add_subsystem('KS_%d' % k, comp)

                sub.connect(
                    'stress_comp.stress_%d' % k,
                    'KS_%d.g' % k)

                if parallel_derivs:
                    color = 'red_%d' % k
                else:
                    color = None

                sub.add_constraint('KS_%d.KS' % k, upper=0.0,
                                   parallel_deriv_color=color)

        comp = VolumeComp(num_elements=num_elements, b=b, L=L)
        self.add_subsystem('volume_comp', comp)

        self.connect('inputs_comp.h_cp', 'interp.h_cp')
        self.connect('interp.h', 'I_comp.h')
        self.connect('I_comp.I', 'local_stiffness_matrix_comp.I')
        self.connect('interp.h', 'volume_comp.h')

        self.add_design_var('inputs_comp.h_cp', lower=1e-2, upper=10.)
        self.add_objective('volume_comp.volume')
