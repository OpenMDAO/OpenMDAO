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


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MatMatTestCase(unittest.TestCase):
    N_PROCS = 4

    def test_matmat(self):
        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('alpha', 2.2)

        indep_var_comp.add_output('shape', 0., shape=8*12*2)
        indep_var_comp.add_output('twist', 0., shape=8-1)

        prob = Problem()
        prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        prob.model.add_subsystem('adflow_comp', ADFLOWComp(), promotes=['*'])

        prob.model.add_design_var('shape', lower=-2., upper=2., scaler=100.)
        prob.model.add_design_var('twist', lower=-8., upper=8.)
        prob.model.add_design_var('alpha', lower=-5., upper=5.)
        prob.model.add_objective('cd')
        prob.model.add_constraint('cl', equals=.5)
        prob.model.add_constraint('thickness', lower=.5, upper=5., vectorize_derivs=True)

        prob.setup(mode='rev')

        # this will hang if the bug is present.
        prob.compute_totals(of=['cd', 'cl', 'thickness'],
                            wrt=['shape', 'twist', 'alpha'])


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
