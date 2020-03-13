"""
Contains test groups for cycles with easily verified values/derivatives.

The Group contains the following components:
    - 'first'
    - (num_comp - 2 copies of) 'middle_i'
    - 'last'

Components other than the 'last' component take an input vector and rotate that vector by a
specified angle in the plane spanned by [1, 0, ..., 0, 1] and [0, 1, ..., 1, 0]. 'first' takes a
parameter 'psi' that corresponds to the initial rotation. The middle components use the rotation
angle specified by 'theta'. All components (including 'first' and 'last') have an output 'theta_out'
that is used to pass the 'theta' angle to the next component. The goal of this group is to determine
a value for 'theta' such that
    psi + (num_comp-1)*theta = 0 (mod 2*pi),
i.e., an angle 'theta' such that rotations by the non-first components would return the vector to
its original location. The 'last' component computes a new value for 'theta_out' that has 1/2 the
error as the input 'theta'. 'last.theta_out' is connected to 'first.theta' to form a cycle and the
reduction of error by 1/2 ensures Gauss-Seidel type methods will converge.

To provide support for multiple variables/variable sizes, the vector is constructed from the inputs
by taking the inputs in order (e.g. x_0, x_1, ...), flattening the array, and concatenating those
arrays into one vector. The outputs y_i are constructed in the reverse manner.

Note: 'theta' is unique only up to equivalence mod (2*pi)/(num_comp - 1). Test authors should not
depend on particular values of 'theta' (or 'x_i'/'y_i' values) without taking this in to account.
"""

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.cycle_comps import PSI, \
    ExplicitCycleComp, ExplicitFirstComp, ExplicitLastComp
from openmdao.test_suite.groups.parametric_group import ParametericTestGroup


class CycleGroup(ParametericTestGroup):
    """
    Group with a cycle. Derivatives and values are known.
    """

    def initialize(self):
        self.default_params.update({
            'connection_type': ['implicit', 'explicit'],
            'partial_type': ['array', 'sparse', 'aij'],
            'partial_method': ['exact', 'fd', 'cs'],
            'num_comp': [3, 2],
            'num_var': [3, 1],
            'var_shape': [(2, 3), (3,)],
        })

        self.options.declare('num_comp', types=int, default=2,
                             desc='Total number of components')
        self.options.declare('num_var', types=int, default=1,
                             desc='Number of variables per component')
        self.options.declare('var_shape', default=(3,),
                             desc='Shape of each variable')
        self.options.declare('connection_type', default='explicit',
                             values=['explicit', 'implicit'],
                             desc='How to connect variables.')
        self.options.declare('partial_type', default='array',
                             values=['array', 'sparse', 'aij'],
                             desc='type of partial derivatives')
        self.options.declare('partial_method', default='exact',
                             values=('exact', 'fd', 'cs'),
                             desc='Method used to solve derivatives (exact, fd, cs).')

    def setup(self):
        num_comp = self.options['num_comp']
        if num_comp < 2:
            raise ValueError('Number of components must be at least 2.')

        self.num_var = num_var = self.options['num_var']
        self.var_shape = var_shape = self.options['var_shape']

        self.size = num_var * np.prod(var_shape)
        if self.size < 3:
            raise ValueError('Product of num_var and var_shape must be at least 3.')

        connection_type = self.options['connection_type']

        first_class = ExplicitFirstComp
        middle_class = ExplicitCycleComp
        last_class = ExplicitLastComp

        self._generate_components(connection_type, first_class, middle_class, last_class, num_comp)

        theta_name = 'last.theta_out' if connection_type == 'explicit' else \
            'theta_{}'.format(num_comp)

        self.total_of = ['last.x_norm2', theta_name]
        self.total_wrt = ['psi_comp.psi']
        self.expected_totals = {
            ('last.x_norm2', 'psi_comp.psi'): [[0.]],
            (theta_name, 'psi_comp.psi'): [[-1. / (num_comp - 1)]],
        }

        expected_theta = (2 * np.pi - PSI) / (num_comp - 1)
        self.expected_values = {
            theta_name: expected_theta,
            'last.x_norm2': 0.5 * self.size,
        }

    def _generate_components(self, conn_type, first_class, middle_class, last_class, num_comp):
        first_name = 'first'
        last_name = 'last'
        var_shape = self.options['var_shape']
        num_var = self.options['num_var']
        comp_args = {
            'var_shape': var_shape,
            'num_var': num_var,
            'jacobian_type': self.options['jacobian_type'],
            'partial_type': self.options['partial_type'],
            'connection_type': conn_type,
            'partial_method': self.options['partial_method'],
            'num_comp': self.options['num_comp']
        }

        self.add_subsystem('psi_comp', om.IndepVarComp('psi', PSI))
        indep_var_comp = self.add_subsystem('x0_comp', om.IndepVarComp())
        for i in range(num_var):
            indep_var_comp.add_output('x_{0}'.format(i), np.ones(var_shape))

        idx = 0

        first_comp = first_class(index=idx, **comp_args)
        first_comp._init_parameterized()
        self.add_subsystem(first_name, first_comp,
                           promotes_inputs=first_comp._cycle_promotes_in,
                           promotes_outputs=first_comp._cycle_promotes_out)
        prev_name = first_name

        connection_variables = [('y_{0}'.format(i), 'x_{0}'.format(i)) for i in range(num_var)]
        connection_variables.append(('theta_out', 'theta'))

        # Middle Subsystems
        for idx in range(1, num_comp - 1):
            current_name = 'middle_{0}'.format(idx)
            middle_comp = middle_class(index=idx, **comp_args)
            middle_comp._init_parameterized()
            self.add_subsystem(current_name, middle_comp,
                               promotes_inputs=middle_comp._cycle_promotes_in,
                               promotes_outputs=middle_comp._cycle_promotes_out)

            if conn_type == 'explicit':
                self._explicit_connections(prev_name, current_name, connection_variables)

            prev_name = current_name

        # Final Subsystem
        last_comp = last_class(index=idx+1, **comp_args)
        last_comp._init_parameterized()
        self.add_subsystem(last_name, last_comp,
                           promotes_inputs=last_comp._cycle_promotes_in,
                           promotes_outputs=last_comp._cycle_promotes_out)

        if conn_type == 'explicit':
            self._explicit_connections(prev_name, last_name, connection_variables)
            self._explicit_connections(last_name, first_name, [('theta_out', 'theta')])
        else:
            theta_out = last_comp._cycle_names['theta_out']
            theta_in = first_comp._cycle_names['theta']
            self.connect(theta_out, theta_in)

        self.connect('psi_comp.psi', first_name + '.psi')
        self.connect('psi_comp.psi', last_name + '.psi')

        if conn_type == 'explicit':
            var_name = first_name + '.x_{0}'
        else:
            var_name = 'x_0_{0}'

        for i in range(num_var):
            self.connect('x0_comp.x_{0}'.format(i), var_name.format(i))

    def _explicit_connections(self, prev_name, current_name, vars):
        for out_var, in_var in vars:
            self.connect(
                '{0}.{1}'.format(prev_name, out_var),
                '{0}.{1}'.format(current_name, in_var)
            )
