"""Define the test group classes."""
from __future__ import division, print_function

from six.moves import range

from openmdao.api import Group


class TestGroupFlat(Group):
    """Test group flat, with only 1 level of hierarchy."""

    def initialize(self):
        self.metadata.declare('num_comp', typ=int, value=2,
                              desc='total number of components')
        self.metadata.declare('num_var', typ=int, value=2,
                              desc='number of output variables per component')
        self.metadata.declare('var_shape', value=(1,),
                              desc='input/output variable shapes')
        self.metadata.declare('connection_type', typ=str, value='explicit',
                              values=['explicit', 'implicit'],
                              desc='how to connect variables')
        self.metadata.declare('component_class',
                              desc='Component class to instantiate')
        self.metadata.declare('jacobian_type', value='matvec',
                              values=['matvec', 'dense', 'sparse-coo',
                                      'sparse-csr'],
                              desc='method of assembling derivatives')
        self.metadata.declare('partial_type', value='array',
                              values=['array', 'sparse', 'aij'],
                              desc='type of partial derivatives')

        num_comp = self.metadata['num_comp']
        num_var = self.metadata['num_var']
        component_class = self.metadata['component_class']
        for icomp in range(num_comp):
            kwargs = {
                'num_input': num_var * (num_comp - 1),
                'num_output': num_var,
                'var_shape': self.metadata['var_shape'],
            }
            if self.metadata['connection_type'] == 'explicit':
                self.add_subsystem('comp_%i' % icomp, component_class(**kwargs))
            elif self.metadata['connection_type'] == 'implicit':
                renames_inputs = {}
                renames_outputs = {}

                index = 0
                for icomp2 in range(num_comp):
                    if icomp != icomp2:
                        for ivar in range(num_var):
                            index2 = icomp2 * num_var + ivar
                            old_name = 'input_%i' % index
                            new_name = 'var_%i' % index2
                            renames_inputs[old_name] = new_name
                            index += 1
                    else:
                        for ivar in range(num_var):
                            index2 = icomp2 * num_var + ivar
                            old_name = 'output_%i' % ivar
                            new_name = 'var_%i' % index2
                            renames_outputs[old_name] = new_name

                self.add_subsystem('comp_%i' % icomp, component_class(**kwargs),
                                   renames_inputs=renames_inputs,
                                   renames_outputs=renames_outputs)

        if self.metadata['connection_type'] == 'explicit':
            for icomp in range(num_comp):
                index = 0
                for icomp2 in range(num_comp):
                    if icomp != icomp2:
                        for ivar in range(num_var):
                            ip_name = 'comp_%i.input_%i' % (icomp, index)
                            op_name = 'comp_%i.output_%i' % (icomp2, ivar)
                            self.connect(op_name, ip_name)
                            index += 1
