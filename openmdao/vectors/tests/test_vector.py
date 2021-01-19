import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None


class TestVector(unittest.TestCase):

    def test_keys(self):
        p = om.Problem()
        comp = om.IndepVarComp()
        comp.add_output('v1', val=1.0)
        comp.add_output('v2', val=2.0)
        p.model.add_subsystem('des_vars', comp, promotes=['*'])
        p.setup()
        p.final_setup()

        keys = sorted(p.model._outputs.keys())
        expected = ['des_vars.v1', 'des_vars.v2']

        self.assertListEqual(keys, expected, msg='keys() is not returning the expected names')

    def test_iter(self):
        p = om.Problem()
        comp = om.IndepVarComp()
        comp.add_output('v1', val=1.0)
        comp.add_output('v2', val=2.0)
        p.model.add_subsystem('des_vars', comp, promotes=['*'])
        p.setup()
        p.final_setup()

        outputs = [n for n in p.model._outputs]
        expected = ['des_vars.v1', 'des_vars.v2']

        self.assertListEqual(outputs, expected, msg='Iter is not returning the expected names')

    def test_dot(self):
        p = om.Problem()
        comp = om.IndepVarComp()
        comp.add_output('v1', val=1.0)
        comp.add_output('v2', val=2.0)
        p.model.add_subsystem('des_vars', comp, promotes=['*'])
        p.setup()
        p.final_setup()

        p.model._residuals.set_val(3.)

        self.assertEqual(p.model._residuals.dot(p.model._outputs), 9.)


A = np.array([[1.0, 8.0, 0.0], [-1.0, 10.0, 2.0], [3.0, 100.5, 1.0]])


class DistribQuadtric(om.ImplicitComponent):
    def initialize(self):
        self.options['distributed'] = True
        self.options.declare('size', types=int, default=1,
            desc="Size of input and output vectors.")

    def setup(self):
        comm = self.comm
        rank = comm.rank

        size_total = self.options['size']

        # Distribute x and y vectors across each processor as evenly as possible
        sizes, offsets = evenly_distrib_idxs(comm.size, size_total)
        start = offsets[rank]
        end = start + sizes[rank]
        self.size_local = size_local = sizes[rank]

        # Get the local slice of A that this processor will be working with
        self.A_local = A[start:end,:]

        self.add_input('x', np.ones(size_local, float),
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('y', np.ones(size_local, float))

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs['x']
        y = outputs['y']
        r = residuals['y']
        for i in range(self.size_local):
            r[i] = self.A_local[i, 0] * y[i]**2 + self.A_local[i, 1] * y[i] \
            + self.A_local[i, 2] - x[i]

    def solve_nonlinear(self, inputs, outputs):
        x = inputs['x']
        y = outputs['y']
        for i in range(self.size_local):
            a = self.A_local[i, 0]
            b = self.A_local[i, 1]
            c = self.A_local[i, 2] - x[i]
            y[i] = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)


class SerialLinear(om.ImplicitComponent):
    def initialize(self):

        self.options.declare('size', types=int, default=1,
                             desc="Size of input and output vectors.")

    def setup(self):
        size = self.options['size']
        self.add_input('y', np.ones(size, float))
        self.add_output('x', np.ones(size, float))
        self.A = A

    def apply_nonlinear(self, inputs, outputs, residuals):
        y = inputs['y']
        x = outputs['x']
        residuals['x'] = y - A.dot(x)

    def solve_nonlinear(self, inputs, outputs):
        y = inputs['y']
        x = outputs['x']
        x[:] = np.linalg.inv(A).dot(y)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestPETScVector2Proc(unittest.TestCase):

    N_PROCS = 2

    def test_distributed_norm_distcomp(self):
        prob = om.Problem()
        top_group = prob.model
        top_group.add_subsystem("distributed_quad", DistribQuadtric(size=3))
        top_group.add_subsystem("serial_linear", SerialLinear(size=3))

        # Connect variables between components
        top_group.connect('serial_linear.x', 'distributed_quad.x')
        top_group.connect('distributed_quad.y', 'serial_linear.y')

        # Need a nonlinear solver since the model is coupled
        top_group.nonlinear_solver = om.NonlinearBlockGS(iprint=0, maxiter=20)

        prob.setup()
        prob.run_model()

        vec = prob.model._vectors['output']['nonlinear']
        norm_val = vec.get_norm()

        assert_near_equal(norm_val, 0.22595230821097395, 1e-10)

        # test petsc dot while we're at it
        vec.set_val(3.)
        vec2 = prob.model._vectors['residual']['linear']
        vec2.set_val(4.)
        assert_near_equal(vec.dot(vec2), 12.*6, 1e-10)

@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestPETScVector3Proc(unittest.TestCase):

    N_PROCS = 3

    def test_distributed_norm_distcomp(self):
        prob = om.Problem()
        top_group = prob.model
        top_group.add_subsystem("distributed_quad", DistribQuadtric(size=3))
        top_group.add_subsystem("serial_linear", SerialLinear(size=3))

        # Connect variables between components
        top_group.connect('serial_linear.x', 'distributed_quad.x')
        top_group.connect('distributed_quad.y', 'serial_linear.y')

        # Need a nonlinear solver since the model is coupled
        top_group.nonlinear_solver = om.NonlinearBlockGS(iprint=0, maxiter=20)

        prob.setup()
        prob.run_model()

        vec = prob.model._vectors['output']['nonlinear']
        norm_val = vec.get_norm()

        assert_near_equal(norm_val, 0.22595230821097395, 1e-10)

        # test petsc dot while we're at it
        vec.set_val(3.)
        vec2 = prob.model._vectors['residual']['linear']
        vec2.set_val(4.)
        assert_near_equal(vec.dot(vec2), 12.*6, 1e-10)

    def test_distributed_norm_parallel_group(self):
        prob = om.Problem()
        model = prob.model

        comp = om.IndepVarComp()
        comp.add_output('v1', val=np.array([3.0, 5.0, 8.0]))
        comp.add_output('v2', val=np.array([17.0]))
        model.add_subsystem('des_vars', comp)

        sub = model.add_subsystem('pp', om.ParallelGroup())
        sub.add_subsystem('calc1', om.ExecComp('y = 2.0*x', x=np.ones((3, )), y=np.ones((3, ))))
        sub.add_subsystem('calc2', om.ExecComp('y = 5.0*x', x=np.ones((3, )), y=np.ones((3, ))))
        sub.add_subsystem('calc3', om.ExecComp('y = 7.0*x', x=np.ones((3, )), y=np.ones((3, ))))

        model.connect('des_vars.v1', 'pp.calc1.x')
        model.connect('des_vars.v1', 'pp.calc2.x')
        model.connect('des_vars.v1', 'pp.calc3.x')

        model.linear_solver = om.LinearBlockGS()

        prob.setup()

        prob.run_model()

        vec = prob.model._vectors['output']['nonlinear']
        norm_val = vec.get_norm()
        assert_near_equal(norm_val, 89.61584681293817, 1e-10)

        J = prob.compute_totals(of=['pp.calc1.y', 'pp.calc2.y', 'pp.calc3.y'], wrt=['des_vars.v1'])

        vec = prob.model._vectors['output']['linear']
        norm_val = vec.get_norm()
        assert_near_equal(norm_val, 8.888194417315589, 1e-10)

        # test petsc dot while we're at it
        vec.set_val(3.)
        vec2 = prob.model._vectors['residual']['linear']
        vec2.set_val(4.)
        assert_near_equal(vec.dot(vec2), 12.*13, 1e-10)


if __name__ == '__main__':
    unittest.main()
