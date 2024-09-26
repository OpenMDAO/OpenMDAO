"""Test serial derivatives in implicit group when running in parallel"""
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI  # MPI will be None here if MPI is not active

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class SerialImplicitDerivConsist(unittest.TestCase):
    N_PROCS = 2

    def test_serialImplicitDerivConsist(self):
        class Imp(om.ImplicitComponent):
            def setup(self):
                self.add_input("x", shape_by_conn=True, distributed=True)
                self.add_input("s0")
                self.add_input("s1")
                self.add_output(
                    "y", shape=5 if self.comm.rank == 0 else 0, distributed=True
                )

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals["y"] = outputs["y"] - (
                    np.sum(inputs["x"]) + inputs["s0"] + inputs["s1"]
                )

            def solve_nonlinear(self, inputs, outputs):
                outputs["y"] = np.sum(inputs["x"]) + inputs["s0"] + inputs["s1"]

            def apply_linear(
                self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode
            ):
                if mode == "rev":
                    if "y" in d_residuals:
                        if "x" in d_inputs:
                            d_inputs["x"] = -np.sum(d_residuals["y"])
                        if "s0" in d_inputs:
                            d_inputs["s0"] = self.comm.allreduce(
                                -np.sum(d_residuals["y"])
                            )
                        if "s1" in d_inputs:
                            d_inputs["s1"] = self.comm.allreduce(
                                -np.sum(d_residuals["y"])
                            )
                        if "y" in d_outputs:
                            d_outputs["y"] = d_residuals["y"]

        class Exp(om.ExplicitComponent):
            def setup(self):
                self.add_input("y", distributed=True, shape_by_conn=True)
                self.add_output("total")

            def compute(self, inputs, outputs):
                outputs["total"] = self.comm.allreduce(np.sum(inputs["y"]))

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == "rev":
                    d_inputs["y"] += d_outputs["total"]

        prob = om.Problem()
        ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
        ivc.add_output("x", val=1, distributed=True)
        ivc.add_output("s0", val=2.0)
        ivc.add_output("s1", val=3.0)

        prob.model.add_subsystem("Imp", Imp(), promotes=["*"])
        prob.model.add_subsystem("Exp", Exp(), promotes=["*"])

        prob.setup(mode="rev")
        prob.run_model()
        totals = prob.check_totals(of="total", wrt="s0")
        for var, err in totals.items():
            assert_near_equal(err["rel error"].reverse, 0.0, 5e-3)


if __name__ == "__main__":
    unittest.main()
