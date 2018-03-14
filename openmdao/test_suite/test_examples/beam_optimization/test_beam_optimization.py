from __future__ import print_function, division, absolute_import

import unittest

from openmdao.utils.assert_utils import assert_rel_error

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress import MultipointBeamGroup as MultipointBeamStress

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class TestCase(unittest.TestCase):

    def test(self):
        import numpy as np

        from openmdao.api import Problem, ScipyOptimizeDriver

        from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50

        prob = Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup()

        prob.run_driver()

        print(prob['inputs_comp.h'])

    def test_multipoint(self):
        import numpy as np

        from openmdao.api import Problem, ScipyOptimizeDriver

        from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_cp = 5
        num_elements = 50
        num_load_cases = 2

        prob = Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                                 num_elements=num_elements, num_cp=num_cp,
                                                 num_load_cases=num_load_cases))

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup()

        prob.run_driver()

        print(prob['interp.h'])

    def test_multipoint_stress(self):
        import numpy as np

        from openmdao.api import Problem, ScipyOptimizeDriver

        from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01
        max_bending = 100.0

        num_cp = 5
        num_elements = 25
        num_load_cases = 2

        prob = Problem(model=MultipointBeamStress(E=E, L=L, b=b, volume=volume, max_bending = max_bending,
                                                  num_elements=num_elements, num_cp=num_cp,
                                                  num_load_cases=num_load_cases))

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup(mode='rev')

        prob.run_driver()

        print(prob['interp.h'])


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestParallelGroups(unittest.TestCase):

    N_PROCS = 2

    def test_multipoint(self):
        import numpy as np

        from openmdao.api import Problem, ScipyOptimizeDriver

        from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_cp = 5
        num_elements = 50
        num_load_cases = 2

        prob = Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                                 num_elements=num_elements, num_cp=num_cp,
                                                 num_load_cases=num_load_cases))

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup(vector_class=PETScVector)

        prob.run_driver()

        print(prob['interp.h'])

if __name__ == "__main__":
    unittest.main()
