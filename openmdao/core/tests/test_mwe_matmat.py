import unittest

from openmdao.api import ExplicitComponent
from openmdao.api import Problem, IndepVarComp

from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class ADFLOWComp(ExplicitComponent):

    def setup(self):
        # self.options['distributed'] = True
        self.add_input('shape', shape=8 * 12 * 2)

        self.add_output('cl')
        self.add_output('cd')

        self.add_output('lete_1', shape=8)
        self.add_output('lete_2', shape=8)

        self.add_output('vol_constraint', val=0.)

        self.add_output('thickness', shape=30 * 25)

    def compute(self, inputs, outputs):
        pass

    def compute_partials(self, inputs, partials):
        pass


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
