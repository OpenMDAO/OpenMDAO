import numpy as np

from openmdao.core.group import Group
from openmdao.test_suite.components.explicit_components import TestExplCompNondLinear
from openmdao.test_suite.components.implicit_components import TestImplCompNondLinear


class TestMeshGroup(Group):
    """Test group flat, with only 1 level of hierarchy.
    Every component is connected to every other component."""

    def initialize(self):
        self.metadata.declare('num_comp', type_=int, value=2,
                              desc='total number of components')
        self.metadata.declare('num_var', type_=int, value=2,
                              desc='number of output variables per component')
        self.metadata.declare('var_shape', value=(1,),
                              desc='input/output variable shapes')
        self.metadata.declare('connection_type', type_=str, value='explicit',
                              values=['explicit', 'implicit'],
                              desc='how to connect variables')
        self.metadata.declare('component_class', value='explicit',
                              values=['explicit', 'implicit'],
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


        self.expected_d_input = {}
        self.expected_d_output = {}
        self._vars_of_interest = []

        shape = self.metadata['var_shape']
        size = np.prod(shape)

        d_value = 0.01 * size * (num_var * (num_var + 1)) / 2 * (num_comp - 1)
        self.value = 1 - d_value
        if self.metadata['component_class'] == 'explicit':
            component_class = TestExplCompNondLinear
        elif self.metadata['component_class'] == 'implicit':
            component_class = TestImplCompNondLinear
        else:
            raise ValueError('Component class must be "explicit" or "implicit"')

        for icomp in range(num_comp):
            kwargs = {
                'num_input': num_var * (num_comp - 1),
                'num_output': num_var,
                'var_shape': shape,
            }
            if self.metadata['connection_type'] == 'explicit':
                self.add_subsystem('comp_%i' % icomp, component_class(**kwargs))
                base_name = 'comp_{0}.output_{1}'
                for output_num in range(num_var):
                    key_name = base_name.format(icomp, output_num)
                    self.expected_d_input[key_name] = np.ones(shape) * (output_num + 1)
                    self.expected_d_output[key_name] = np.ones(shape) * (output_num + self.value)
            elif self.metadata['connection_type'] == 'implicit':
                renames_inputs = {}
                renames_outputs = {}
                base_name = 'var_{0}'

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
                            key_name = base_name.format(index2)
                            self.expected_d_input[key_name] = np.ones(shape) * (ivar + 1)
                            self.expected_d_output[key_name] = np.ones(shape) * (ivar + self.value)

                self.add_subsystem('comp_%i' % icomp, component_class(**kwargs),
                                   renames_inputs=renames_inputs,
                                   renames_outputs=renames_outputs)



        if self.metadata['connection_type'] == 'explicit':
            for icomp in range(num_comp):
                index = 0
                for icomp2 in range(num_comp):
                    if icomp != icomp2:
                        for ivar in range(num_var):
                            in_name = 'comp_%i.input_%i' % (icomp, index)
                            out_name = 'comp_%i.output_%i' % (icomp2, ivar)
                            self.connect(out_name, in_name)
                            index += 1