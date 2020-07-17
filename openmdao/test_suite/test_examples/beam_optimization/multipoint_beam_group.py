"""
This is a multipoint implementation of the beam optimization problem.


"""
import numpy as np

import openmdao.api as om

from openmdao.test_suite.test_examples.beam_optimization.components.local_stiffness_matrix_comp import LocalStiffnessMatrixComp
from openmdao.test_suite.test_examples.beam_optimization.components.moment_comp import MomentOfInertiaComp
from openmdao.test_suite.test_examples.beam_optimization.components.multi_compliance_comp import MultiComplianceComp
from openmdao.test_suite.test_examples.beam_optimization.components.multi_states_comp import MultiStatesComp
from openmdao.test_suite.test_examples.beam_optimization.components.volume_comp import VolumeComp
from openmdao.utils.spline_distributions import sine_distribution


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


class MultipointBeamGroup(om.Group):

    def initialize(self):
        self.options.declare('E')
        self.options.declare('L')
        self.options.declare('b')
        self.options.declare('volume')
        self.options.declare('num_elements', 5)
        self.options.declare('num_cp', 50)
        self.options.declare('num_load_cases', 1)

    def setup(self):
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        volume = self.options['volume']
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        num_cp = self.options['num_cp']
        num_load_cases = self.options['num_load_cases']

        x_interp = sine_distribution(num_elements)
        comp = om.SplineComp(method='bsplines', num_cp=num_cp, x_interp_val=x_interp)
        comp.add_spline(y_cp_name='h_cp', y_interp_name='h')
        self.add_subsystem('interp', comp)

        I_comp = MomentOfInertiaComp(num_elements=num_elements, b=b)
        self.add_subsystem('I_comp', I_comp)

        comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
        self.add_subsystem('local_stiffness_matrix_comp', comp)

        # Parallel Subsystem for load cases.
        par = self.add_subsystem('parallel', om.ParallelGroup())

        # Determine how to split cases up over the available procs.
        nprocs = self.comm.size
        divide = divide_cases(num_load_cases, nprocs)

        obj_srcs = []
        for j, this_proc in enumerate(divide):
            num_rhs = len(this_proc)

            name = 'sub_%d' % j
            sub = par.add_subsystem(name, om.Group())

            # Load is a sinusoidal distributed force of varying spatial frequency.
            force_vector = np.zeros((2 * num_nodes, num_rhs))
            for i, k in enumerate(this_proc):

                end = 1.5 * np.pi
                if num_load_cases > 1:
                    end += k * 0.5 * np.pi / (num_load_cases - 1)

                x = np.linspace(0, end, num_nodes)
                f = - np.sin(x)
                force_vector[0:-1:2, i] = f

            comp = MultiStatesComp(num_elements=num_elements, force_vector=force_vector, num_rhs=num_rhs)
            sub.add_subsystem('states_comp', comp)

            comp = MultiComplianceComp(num_elements=num_elements, force_vector=force_vector,
                                       num_rhs=num_rhs)
            sub.add_subsystem('compliance_comp', comp)

            self.connect(
                'local_stiffness_matrix_comp.K_local',
                'parallel.%s.states_comp.K_local' % name)

            for k in range(num_rhs):
                sub.connect('states_comp.d_%d' % k,
                            'compliance_comp.displacements_%d' % k,
                            src_indices=np.arange(2 *num_nodes))

                obj_srcs.append('parallel.%s.compliance_comp.compliance_%d' % (name, k))

        comp = VolumeComp(num_elements=num_elements, b=b, L=L)
        self.add_subsystem('volume_comp', comp)

        comp = om.ExecComp(['obj = ' + ' + '.join(['compliance_%d' % i for i in range(num_load_cases)])])
        self.add_subsystem('obj_sum', comp)

        for j, src in enumerate(obj_srcs):
            self.connect(src, 'obj_sum.compliance_%d' % j)

        self.connect('interp.h', 'I_comp.h')
        self.connect('I_comp.I', 'local_stiffness_matrix_comp.I')
        self.connect('interp.h', 'volume_comp.h')

        self.add_design_var('interp.h_cp', lower=1e-2, upper=10.)
        self.add_constraint('volume_comp.volume', equals=volume)
        self.add_objective('obj_sum.obj')
