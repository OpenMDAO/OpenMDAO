import unittest
import numpy as np

try:
    import flax.linen as nn
except ImportError:
    nn = None

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import force_check_partials


class SimpleFlaxModel(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x = inputs["x"]
        y = nn.Dense(features=2, use_bias=True, name="dense")(x)
        return {"y": y}


@unittest.skipIf(nn is None, "Flax is not installed.")
class TestJaxFlaxComp(unittest.TestCase):

    def test_jax_flax_comp(self):

        params = {
            "dense": {
                "kernel": np.array([
                    [1.0, 3.0],
                    [2.0, 4.0],
                ]),
                "bias": np.array([0.5, -0.5]),
            }
        }

        prob = om.Problem()

        prob.model.add_subsystem(
            "NeuralNetwork",
            om.JaxFlaxComp(
                model=SimpleFlaxModel(),
                params=params,
                input_shapes={"x": (2,)},
                output_shapes={"y": (2,)},
            ),
            promotes_inputs=["x"],
            promotes_outputs=["y"],
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
            [0.75, 0.05],
            tolerance=1.0e-12,
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


if __name__ == "__main__":
    unittest.main()