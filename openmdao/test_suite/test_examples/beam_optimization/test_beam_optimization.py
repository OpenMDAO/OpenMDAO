import unittest

import openmdao.api as om
from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class TestCase(unittest.TestCase):

    def test(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50

        prob = om.Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob['h'],
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

        import openmdao.api as om
        from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_cp = 5
        num_elements = 50
        num_load_cases = 2

        model = MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                    num_elements=num_elements, num_cp=num_cp,
                                    num_load_cases=num_load_cases)

        prob = om.Problem(model=model)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob['interp.h'][0],
                         [ 0.14122705,  0.14130706,  0.14154096,  0.1419107,   0.14238706,  0.14293095,
                           0.14349514,  0.14402636,  0.1444677,   0.14476123,  0.14485062,  0.14468388,
                           0.14421589,  0.1434107,   0.14224356,  0.14070252,  0.13878952,  0.13652104,
                           0.13392808,  0.13105565,  0.1279617,   0.12471547,  0.1213954,   0.11808665,
                           0.11487828,  0.11185599,  0.10900669,  0.10621949,  0.10338308,  0.10039485,
                           0.09716531,  0.09362202,  0.08971275,  0.08540785,  0.08070168,  0.07561313,
                           0.0701851,   0.06448311,  0.05859294,  0.05261756,  0.0466733,   0.04088557,
                           0.03538417,  0.03029845,  0.02575245,  0.02186027,  0.01872173,  0.01641869,
                           0.0150119,   0.01453876], 1e-4)

    def test_multipoint_stress(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress import MultipointBeamGroup

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01
        max_bending = 100.0

        num_cp = 5
        num_elements = 25
        num_load_cases = 2

        model = MultipointBeamGroup(E=E, L=L, b=b, volume=volume, max_bending = max_bending,
                                    num_elements=num_elements, num_cp=num_cp,
                                    num_load_cases=num_load_cases)

        prob = om.Problem(model=model)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup(mode='rev')

        prob.run_driver()

        assert_near_equal(prob['interp.h'][0],
                         [ 0.45632323,  0.45612552,  0.45543324,  0.45397058,  0.45134629,  0.44714397,
                           0.4410258,   0.43283139,  0.42265378,  0.41087801,  0.3981731,   0.3854358,
                           0.37369202,  0.36342186,  0.35289066,  0.34008777,  0.32362887,  0.30300358,
                           0.27867837,  0.25204063,  0.22519409,  0.20063906,  0.18088818,  0.16807856,
                           0.16364104], 1e-4)

    def test_complex_step(self):
        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50

        prob = om.Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        derivs = prob.check_totals(method='cs', out_stream=None)
        assert_near_equal(derivs[('compliance_comp.compliance', 'h')]['rel error'][0],
                         0.0, 1e-8)
        assert_near_equal(derivs[('volume_comp.volume', 'h')]['rel error'][0],
                         0.0, 1e-8)

        derivs = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(derivs, rtol=1e-15)

    def test_complex_step_multipoint(self):
        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_cp = 5
        num_elements = 50
        num_load_cases = 2

        model = MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                        num_elements=num_elements, num_cp=num_cp,
                                        num_load_cases=num_load_cases)

        prob = om.Problem(model=model)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        derivs = prob.check_totals(method='cs', out_stream=None)
        assert_near_equal(derivs[('obj_sum.obj', 'interp.h_cp')]['rel error'][0],
                         0.0, 1e-8)
        assert_near_equal(derivs[('volume_comp.volume', 'interp.h_cp')]['rel error'][0],
                         0.0, 1e-8)

        derivs = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(derivs, rtol=1e-15)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestParallelGroups(unittest.TestCase):

    N_PROCS = 2

    def test_multipoint(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_cp = 5
        num_elements = 50
        num_load_cases = 2

        model = MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                    num_elements=num_elements, num_cp=num_cp,
                                    num_load_cases=num_load_cases)

        prob = om.Problem(model=model)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob['interp.h'][0],
                         [ 0.14122705,  0.14130706,  0.14154096,  0.1419107,   0.14238706,  0.14293095,
                           0.14349514,  0.14402636,  0.1444677,   0.14476123,  0.14485062,  0.14468388,
                           0.14421589,  0.1434107,   0.14224356,  0.14070252,  0.13878952,  0.13652104,
                           0.13392808,  0.13105565,  0.1279617,   0.12471547,  0.1213954,   0.11808665,
                           0.11487828,  0.11185599,  0.10900669,  0.10621949,  0.10338308,  0.10039485,
                           0.09716531,  0.09362202,  0.08971275,  0.08540785,  0.08070168,  0.07561313,
                           0.0701851,   0.06448311,  0.05859294,  0.05261756,  0.0466733,   0.04088557,
                           0.03538417,  0.03029845,  0.02575245,  0.02186027,  0.01872173,  0.01641869,
                           0.0150119,   0.01453876], 1e-4)


if __name__ == "__main__":
    unittest.main()
