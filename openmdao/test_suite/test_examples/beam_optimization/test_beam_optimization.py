from __future__ import print_function, division, absolute_import

import unittest

from openmdao.utils.assert_utils import assert_rel_error

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup

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

        assert_rel_error(self, prob['inputs_comp.h'],
                         [0.14915754,  0.14764328,  0.14611321,  0.14456715,  0.14300421,  0.14142417,
                          0.13982611,  0.13820976,  0.13657406,  0.13491866,  0.13324268,  0.13154528,
                          0.12982575,  0.12808305,  0.12631658,  0.12452477,  0.12270701,  0.12086183,
                          0.11898809,  0.11708424,  0.11514904,  0.11318072,  0.11117762,  0.10913764,
                          0.10705891,  0.10493903,  0.10277539,  0.10056526,  0.09830546,  0.09599246,
                          0.09362243,  0.09119084,  0.08869265,  0.08612198,  0.08347229,  0.08073573,
                          0.07790323,  0.07496382,  0.07190453,  0.06870925,  0.0653583,   0.06182632,
                          0.05808044,  0.05407658,  0.04975295,  0.0450185,   0.03972912,  0.03363155,
                          0.02620192,  0.01610863], 1e-4)

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

        prob.setup()

        prob.run_driver()

        print(prob['interp.h'])

    def test_multipoint_stress(self):
        import numpy as np

        from openmdao.api import Problem, ScipyOptimizeDriver

        from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress import MultipointBeamGroup

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01
        max_bending = 100.0

        num_cp = 5
        num_elements = 25
        num_load_cases = 2

        prob = Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume, max_bending = max_bending,
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
