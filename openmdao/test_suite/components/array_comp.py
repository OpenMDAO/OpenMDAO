import numpy as np

import openmdao.api as om


class ArrayComp(om.ExplicitComponent):

    def setup(self):

        J1 = np.array([[1.0, 3.0, -2.0, 7.0],
                        [6.0, 2.5, 2.0, 4.0],
                        [-1.0, 0.0, 8.0, 1.0],
                        [1.0, 4.0, -5.0, 6.0]])

        self.J1 = J1
        self.J2 = J1 * 3.3
        self.Jb = J1.T

        # Inputs
        self.add_input('x1', np.zeros([4]))
        self.add_input('x2', np.zeros([4]))
        self.add_input('bb', np.zeros([4]))

        # Outputs
        self.add_output('y1', np.zeros([4]))

        self.exec_count = 0
        self.set_check_partial_options('x*', directional=True)

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')


    def compute(self, inputs, outputs):
        """
        Execution.
        """
        outputs['y1'] = self.J1.dot(inputs['x1']) + self.J2.dot(inputs['x2']) + self.Jb.dot(inputs['bb'])
        self.exec_count += 1

    def compute_partials(self, inputs, partials):
        """
        Analytical derivatives.
        """
        partials[('y1', 'x1')] = self.J1
        partials[('y1', 'x2')] = self.J2
        partials[('y1', 'bb')] = self.Jb
