import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import force_check_partials


class TestDenseNNComp(unittest.TestCase):

    def test_dense_nn_comp(self):

        prob = om.Problem()

        prob.model.add_subsystem(
            "NeuralNetwork",
            om.DenseNNComp(
                weights_file="mazur_nn_weights.npz",
            ),
            promotes_inputs=['x'],
            promotes_outputs=['y']
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val("x", [0.05, 0.10])

        prob.run_model()

        assert_near_equal(
            prob.get_val("x"),
            [0.05, 0.10],
            tolerance=1.0e-12,
        )

        assert_near_equal(
            prob.get_val("y"),
            [1.10590597, 1.22492140],
            tolerance=1.0e-8,
        )

        partials = force_check_partials(
            prob,
            method="cs",
            compact_print=True,
            out_stream=None,
        )

        assert_check_partials(
            partials,
            atol=1.0e-8,
            rtol=1.0e-8,
        )