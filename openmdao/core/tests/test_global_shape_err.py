import unittest
import numpy as np
from openmdao.utils.mpi import MPI

import openmdao.api as om


class TwoDArrayAdder(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n0')
        self.options.declare('n1')

    def setup(self):
        n0 = self.options['n0']
        n1 = self.options['n1']
        self.add_input('x', shape=((n0, n1)), distributed=True)
        self.add_output('x_sum', shape=((n0, n1)), distributed=True)

    def compute(self, inputs, outputs):
        outputs['x_sum'] = np.sum(inputs['x'], axis=0)


@unittest.skipUnless(MPI, "MPI is required.")
class GlobalShapeErr(unittest.TestCase):
    N_PROCS = 2

    def check_global_shape(self, n1_test_multiple, expect_error):
        """
        Run Problem.setup() and test for an error cause by an improperly-sized variable shape.

        Parameters
        ----------
        n1_test_multiple : int
            Multiple the upper array dimension by this value.
        expect_error: bool
            Whether this test is intended to produce an error or not.
        """
        n0 = 3
        n1 = 100 if MPI.COMM_WORLD.rank == 0 else 100 * n1_test_multiple
        prob = om.Problem()
        ivc = prob.model.add_subsystem('ivc',om.IndepVarComp())
        ivc.add_output('x', val = np.ones((n0, n1)), distributed=True)

        prob.model.add_subsystem('adder',TwoDArrayAdder(n0=n0, n1=n1))
        prob.model.connect('ivc.x','adder.x')

        if expect_error:
            with self.assertRaises(RuntimeError):
                prob.setup()
        else:
            prob.setup()

    def test_global_shape_success(self):
        self.check_global_shape(1, False)

    def test_global_shape_failure(self):
        self.check_global_shape(2, True)

if __name__ == '__main__':
    unittest.main()
