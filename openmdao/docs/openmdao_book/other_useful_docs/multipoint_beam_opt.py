
import numpy as np

import openmdao.api as om
from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    print("PETSc is not installed. Skipping beam optimization.")
    PETScVector = None


if __name__ == '__main__' and PETScVector is not None:
    E = 1.
    L = 1.
    b = 0.1
    volume = 0.01

    num_cp = 5
    num_elements = 50
    num_load_cases = 2

    prob = om.Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                                num_elements=num_elements, num_cp=num_cp,
                                                num_load_cases=num_load_cases))

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = True

    prob.setup()

    prob.run_driver()

    h = prob['interp.h']
    expected = np.array([ 0.14122705,  0.14130706,  0.14154096,  0.1419107,   0.14238706,  0.14293095,
                          0.14349514,  0.14402636,  0.1444677,   0.14476123,  0.14485062,  0.14468388,
                          0.14421589,  0.1434107,   0.14224356,  0.14070252,  0.13878952,  0.13652104,
                          0.13392808,  0.13105565,  0.1279617,   0.12471547,  0.1213954,   0.11808665,
                          0.11487828,  0.11185599,  0.10900669,  0.10621949,  0.10338308,  0.10039485,
                          0.09716531,  0.09362202,  0.08971275,  0.08540785,  0.08070168,  0.07561313,
                          0.0701851,   0.06448311,  0.05859294,  0.05261756,  0.0466733,   0.04088557,
                          0.03538417,  0.03029845,  0.02575245,  0.02186027,  0.01872173,  0.01641869,
                          0.0150119,   0.01453876])

    assert np.linalg.norm(h - expected) < 1e-6
