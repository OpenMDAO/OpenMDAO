"""Test serial derivatives in implicit group when running in parallel"""
import unittest

import numpy as np
from mpi4py import MPI

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


dist_shape = 20 if MPI.COMM_WORLD.rank > 0 else 2


class SerialImplicitDerivConsist(unittest.TestCase):
    N_PROCS = 2

    def test_serialImplicitDerivConsist(self):
        class MixedSerialInComp(om.ExplicitComponent):
            def setup(self):
                self.add_input("aoa_serial")
                self.add_output(
                    "flow_state_dist", shape=dist_shape, distributed=True
                )

            def compute(self, inputs, outputs):
                outputs["flow_state_dist"][:] = 0.5 * inputs["aoa_serial"]

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == "fwd":
                    if "flow_state_dist" in d_outputs:
                        if "aoa_serial" in d_inputs:
                            d_outputs["flow_state_dist"] += (
                                0.5 * d_inputs["aoa_serial"]
                            )
                if mode == "rev":
                    if "flow_state_dist" in d_outputs:
                        if "aoa_serial" in d_inputs:
                            d_inputs["aoa_serial"] += 0.5 * self.comm.allreduce(
                                np.sum(d_outputs["flow_state_dist"])
                            )

        class MixedSerialOutComp(om.ExplicitComponent):
            def setup(self):
                self.add_input("aoa_serial")
                self.add_input("force_dist", shape=dist_shape, distributed=True)
                self.add_output("lift_serial")

            def compute(self, inputs, outputs):
                outputs["lift_serial"] = 2.0 * inputs[
                    "aoa_serial"
                ] + self.comm.allreduce(3.0 * np.sum(inputs["force_dist"]))

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == "fwd":
                    if "lift_serial" in d_outputs:
                        if "aoa_serial" in d_inputs:
                            d_outputs["lift_serial"] += (
                                2.0 * d_inputs["aoa_serial"]
                            )
                        if "force_dist" in d_inputs:
                            d_outputs[
                                "lift_serial"
                            ] += 3.0 * self.comm.allreduce(
                                np.sum(d_inputs["force_dist"])
                            )
                if mode == "rev":
                    if "lift_serial" in d_outputs:
                        if "aoa_serial" in d_inputs:
                            d_inputs["aoa_serial"] += (
                                2.0 * d_outputs["lift_serial"]
                            )
                        if "force_dist" in d_inputs:
                            d_inputs["force_dist"] += (
                                3.0 * d_outputs["lift_serial"]
                            )

        class DistComp(om.ImplicitComponent):
            def setup(self):
                self.add_output(
                    "force_dist", shape=dist_shape, distributed=True
                )
                self.add_input(
                    "flow_state_dist", shape=dist_shape, distributed=True
                )

                self.add_input("aoa_serial")

            def solve_nonlinear(self, inputs, outputs):
                outputs["force_dist"] = np.linspace(0, dist_shape, dist_shape)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals["force_dist"] = outputs["force_dist"] - np.linspace(
                    0, dist_shape, dist_shape
                )

            def apply_linear(
                self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode
            ):
                if mode == "fwd":
                    if "force_dist" in d_outputs:
                        if "flow_state_dist" in d_inputs:
                            d_outputs["force_dist"] += 1.0
                        if "aoa_serial" in d_inputs:
                            d_outputs["force_dist"] += 0.0
                if mode == "rev":
                    if "force_dist" in d_outputs:
                        if "flow_state_dist" in d_inputs:
                            d_inputs["flow_state_dist"] += 1.0
                        if "aoa_serial" in d_inputs:
                            d_inputs["aoa_serial"] += 0.0

        p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output("dv", val=1.0)

        p.model.add_subsystem("ivc", ivc)
        p.model.add_subsystem("mixed_in_comp", MixedSerialInComp())
        p.model.add_subsystem("dist_comp", DistComp())
        p.model.add_subsystem("mixed_out_comp", MixedSerialOutComp())
        p.model.connect("ivc.dv", "mixed_in_comp.aoa_serial")
        p.model.connect(
            "mixed_in_comp.flow_state_dist", "dist_comp.flow_state_dist"
        )
        p.model.connect("ivc.dv", "dist_comp.aoa_serial")
        p.model.connect("ivc.dv", "mixed_out_comp.aoa_serial")
        p.model.connect("dist_comp.force_dist", "mixed_out_comp.force_dist")

        p.model.add_design_var("ivc.dv")
        p.model.add_objective("mixed_out_comp.lift_serial")
        p.setup(mode="rev")
        p.run_model()
        totals = p.check_totals()
        for var, err in totals.items():
            rel_err = err["rel error"]
            assert_near_equal(rel_err.forward, 0.0, 5e-3)


if __name__ == "__main__":
    unittest.main()
