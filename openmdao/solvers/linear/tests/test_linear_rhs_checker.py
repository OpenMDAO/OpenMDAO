"""
Here are the behaviours we'd like to test:
- When supplied a RHS vector that is equal to one already stored in the cache, that cached solution should be returned.
- When supplied a RHS vector that is the negative of one already stored in the cache, the negative of that cached solution should be returned.
- When supplied a RHS vector that is a scaled version of one already stored in the cache, the scaled version of that cached solution should be returned.
- All the above should work when running in parallel and, on some processors, the local block of the vectors are entirely zero.
- When supplied a RHS vector that is not parallel to any already stored in the cache, no solution should be returned from the cache. This includes the case where, in parallel, each processor's portion of the RHS vector is parallel to the corresponding portion of a cached RHS vector, but the full vectors are not parallel (i.e on one proc the blocks are equal, while on another proc they are negative of each other).
"""

import unittest
import numpy as np
from parameterized import parameterized_class

import openmdao.api as om
from openmdao.solvers.linear.linear_rhs_checker import LinearRHSChecker
try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

VEC_SIZE = 2


class DistComp(om.ExplicitComponent):
    """A distributed component that computes y = x."""

    def setup(self):
        self.add_input("x", val=np.ones(VEC_SIZE), distributed=True)
        self.add_output("y", val=np.ones(VEC_SIZE), distributed=True)

    def setup_partials(self):
        self.declare_partials("y", "x", val=np.eye(VEC_SIZE))

    def compute(self, inputs, outputs):
        outputs["y"] = inputs["x"]


class Summer(om.ExplicitComponent):
    """Sums a distributed input to a scalar."""

    def setup(self):
        self.add_input("x", val=np.ones(VEC_SIZE), distributed=True)
        self.add_output("sum", val=0.0)

    def compute(self, inputs, outputs):
        local_sum = np.sum(inputs["x"])
        if self.comm.size > 1:
            outputs["sum"] = self.comm.allreduce(local_sum)
        else:
            outputs["sum"] = local_sum

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "rev":
            if "sum" in d_outputs and "x" in d_inputs:
                d_inputs["x"] += d_outputs["sum"]


class DistIVC(om.IndepVarComp):
    """A distributed IndepVarComp."""

    def setup(self):
        self.add_output("x", val=np.ones(VEC_SIZE), distributed=True)

# Run tests with 1 and 2 procs
@parameterized_class(
    ("N_PROCS",),
    [
        (1,),
        (2,),
    ]
)
class TestLinearRHSChecker(unittest.TestCase):
    """
    Unit tests for LinearRHSChecker that directly test the add_solution/get_solution logic.

    These tests manually add solutions to the cache and check that get_solution returns
    the correct cached solution. We bypass the relevance checks by subclassing.
    """

    def setUp(self):
        # Create a problem with redundant adjoint systems:
        # G.y connects to both sum1 and sum2, which are both responses.
        # This creates redundant RHS vectors during adjoint solves.
        self.prob = om.Problem()
        model = self.prob.model

        G = model.add_subsystem("G", om.Group())
        G.add_subsystem("ivc", DistIVC(), promotes=["x"])
        G.add_subsystem("comp", DistComp(), promotes=["x", "y"])

        # If running in parallel, we can only use PETSc linear solver. Otherwise we can use direct
        self.isSerial = self.N_PROCS == 1
        if self.isSerial:
            G.linear_solver = om.DirectSolver()
        else:
            if PETScVector is None:
                raise unittest.SkipTest("PETSc is required for parallel tests.")
            G.linear_solver = om.PETScKrylov()

        # Two downstream components that depend on G.y - creates redundant adjoints
        model.add_subsystem("sum1", Summer())
        model.add_subsystem("sum2", Summer())
        model.connect("G.y", "sum1.x")
        model.connect("G.y", "sum2.x")

        model.add_design_var("G.x")
        model.add_objective("sum1.sum")
        model.add_constraint("sum2.sum", upper=100.0)

        self.prob.setup(mode="rev")
        self.prob.final_setup()

        self.system = G
        self.comm = self.system.comm

        # Create checker with stats collection enabled
        self.checker = LinearRHSChecker(self.system, check_zero=True, verbose=True)

        # Set seed_vars to enable cache lookups in get_solution().
        # This is normally set during compute_totals() but we need to set it manually
        # to unit test the cache logic directly.
        self.system._problem_meta["seed_vars"] = {"sum1.sum", "sum2.sum"}

        # Mock the redundant adjoint systems to include our system.
        # This allows us to test the cache logic without requiring a problem structure
        # that naturally creates redundant adjoint systems.
        self._orig_get_redundant = self.system._relevance.get_redundant_adjoint_systems
        self.system._relevance.get_redundant_adjoint_systems = lambda: {
            "G": {"sum1.sum", "sum2.sum"}
        }

        self.rng = np.random.default_rng(0)
        self.cached_rhs = self.rng.random(VEC_SIZE)
        self.cached_sol = self.rng.random(VEC_SIZE)

        self.checker.add_solution(self.cached_rhs, self.cached_sol, self.system, copy=True)

    def test_exact_match(self):
        """
        Given a cached RHS/solution pair,
        when get_solution is called with the same RHS,
        then the cached solution is returned.
        """
        found_sol, is_zero = self.checker.get_solution(self.cached_rhs, self.system)
        self.assertIsNotNone(found_sol)
        np.testing.assert_allclose(found_sol, self.cached_sol)

    def test_negative_match(self):
        """
        Given a cached RHS/solution pair,
        when get_solution is called with the negated RHS,
        then the negated cached solution is returned.
        """
        found_sol, is_zero = self.checker.get_solution(-self.cached_rhs, self.system)
        self.assertIsNotNone(found_sol)
        np.testing.assert_allclose(found_sol, -self.cached_sol)

    def test_scaled_match(self):
        """
        Given a cached RHS/solution pair,
        when get_solution is called with a scaled RHS,
        then the correspondingly scaled solution is returned.
        """
        scale = 2.5
        found_sol, is_zero = self.checker.get_solution(self.cached_rhs * scale, self.system)
        self.assertIsNotNone(found_sol)
        np.testing.assert_allclose(found_sol, self.cached_sol * scale)

    def test_parallel_zero_blocks(self):
        """
        Given a cached RHS/solution pair with all-zero blocks on some ranks,
        when get_solution is called with a scaled version of that RHS,
        then the scaled solution is returned (norms computed over full distributed vector).
        """
        if self.isSerial:
            raise unittest.SkipTest("Test requires MPI with > 1 proc")

        # Construct a case where the input vector matches a cached vector, and both have entirely zero blocks on some procs
        if self.comm.rank == 0:
            new_cached_rhs = self.cached_rhs
            new_cached_sol = self.cached_sol
        else:
            new_cached_rhs = np.zeros(VEC_SIZE)
            new_cached_sol = np.zeros(VEC_SIZE)

        self.checker.add_solution(new_cached_rhs, new_cached_sol, self.system, copy=True)

        # Test scaled match where one proc has zeros
        scale = self.rng.uniform(0.1, 10.0)
        found_sol, is_zero = self.checker.get_solution(new_cached_rhs * scale, self.system)

        self.assertIsNotNone(found_sol)
        np.testing.assert_allclose(found_sol, new_cached_sol * scale)

    def test_parallel_inconsistency(self):
        """
        Given a cached RHS/solution pair,
        when get_solution is called with an RHS that matches locally but with
        inconsistent scales across ranks (e.g. scale=1 on rank 0, scale=-1 on rank 1),
        then None is returned (no false cache hit).
        """
        if self.isSerial:
            raise unittest.SkipTest("Test requires MPI with > 1 proc")

        # Construct a case where local checks pass but global check should fail
        # Rank 0: rhs_new = rhs_old (scale = 1)
        # Rank >=1: rhs_new = -rhs_old (scale = -1)
        if self.comm.rank == 0:
            rhs_new = self.cached_rhs  # Matches with scale 1
        else:
            rhs_new = -self.cached_rhs  # Matches with scale -1

        found_sol, is_zero = self.checker.get_solution(rhs_new, self.system)

        # Should return None because scales are inconsistent across ranks
        self.assertIsNone(found_sol)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests

    mpirun_tests()
