import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class Inner(om.Group):
    def setup(self):
        comp = om.ExecComp('y=2*x', x=np.zeros((3, 2)), y=np.zeros((3, 2)))
        self.add_subsystem('comp', comp)


class Outer(om.Group):
    def setup(self):
        self.add_subsystem('inner', Inner())
    def configure(self):
        self.promotes('inner', inputs=[('comp.x', 'desvar_x')],
                      src_indices=np.array([[0, 1], [0, 1], [0, 1]]), flat_src_indices=True, src_shape=2)


class SrcIndicesTestCase(unittest.TestCase):
    def test_one_nesting(self):
        prob = om.Problem()
        model = prob.model
        comp = model.add_subsystem('src', om.ExecComp('y=3*x', x=np.zeros((7)), y=np.zeros((7))))
        model.add_subsystem('outer', Outer())
        model.connect('src.y', 'outer.desvar_x', src_indices=[2, 4], flat_src_indices=True)
        prob.setup()
        prob.set_val('src.x', np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]))
        prob.run_model()
        expected = np.array([[15., 27.],
                             [15., 27.],
                             [15., 27.]])
        assert_near_equal(prob.get_val('outer.desvar_x'), expected, 1e-6)

    def test_broadcast_scalar_connection(self):
        """
        OpenMDAO should allow promotion to an input that appears to have a certain shape.
        Users should be allowed to connect an arbitrary variable with compatible src_indices to that input.
        """
        p = om.Problem()

        # c1 contains scalar calculations
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('y', val=5.0*np.ones(4,))
        ivc.add_output('x', val=np.arange(4, dtype=float) + 1)

        g1 = p.model.add_subsystem('g1', om.Group())
        # c2 is vectorized calculations
        c2 = g1.add_subsystem('c2', om.ExecComp('z = a * y', shape=(4,)))

        # The ultimate source of a and y may be scalar, or have some other arbitrary shape
        g1.promotes('c2', inputs=['a', 'y'], src_indices=[0, 0, 0, 0], src_shape=(1,))

        p.model.connect('ivc.y', 'g1.y')

        # Now connect only a portion of some other output to a, which appears as a scalar input
        # (This currently breaks because we're specifying the src_indices of an input twice.)
        p.model.connect('ivc.x', 'g1.a', src_indices=[-1])

        p.setup()

        p.run_model()

        assert_near_equal(p['g1.c2.z'], [20.] * 4)

    def test_multiple_inputs_different_src_indices(self):
        """
        When different variables get promoted to the same name, but have different src_indices, this
        should be supported as long as the src_shape of the different promotions is compatible or
        made so by the set_input_defaults call.
        """
        p = om.Problem()

        g1 = p.model.add_subsystem('g1', om.Group(), promotes_inputs=['b'])
        # c1 contains scalar calculations
        c1 = g1.add_subsystem('c1', om.ExecComp('y = a0 + b', shape=(1,)),
                              promotes_inputs=[('a0', 'a'), 'b'], promotes_outputs=['y'])
        g2 = g1.add_subsystem('g2', om.Group())
        # c2 is vectorized calculations
        c2 = g2.add_subsystem('c2', om.ExecComp('z = a * y', shape=(4,)), promotes_inputs=['a', 'y'])

        g1.promotes('g2', inputs=['y'], src_indices=[0, 0, 0, 0], src_shape=(1,))
        g1.promotes('g2', inputs=['a'], src_indices=[0, 0, 0, 0], src_shape=(1,))

        p.model.promotes('g1', inputs=['a'], src_indices=[0], src_shape=(1,))

        p.setup()

        p['a'] = 99
        p['b'] = 2

        p.run_model()

        assert_near_equal(p['g1.y'], 101.)
        assert_near_equal(p['g1.g2.c2.z'], [9999.] * 4)

    def test_src_indices_nested_promotes(self):
        """
        Promoting a variable up multiple levels with different src_indices at each level.
        """
        p = om.Problem()

        g1 = p.model.add_subsystem('g1', om.Group(), promotes_inputs=['b'])
        # c1 contains scalar calculations
        c1 = g1.add_subsystem('c1', om.ExecComp('y = a0 + b', shape=(1,)),
                              promotes_inputs=[('a0', 'a'), 'b'], promotes_outputs=['y'])

        g2 = g1.add_subsystem('g2', om.Group())
        # c2 is vectorized calculations
        c2 = g2.add_subsystem('c2',  om.ExecComp('z = a * y', shape=(4,)), promotes_inputs=['a', 'y'])

        g1.promotes('g2', inputs=['a'], src_indices=[0, 0, 0, 0], src_shape=(4,))
        g1.promotes('g2', inputs=['y'], src_indices=[0, 0, 0, 0], src_shape=(1,))

        p.model.promotes('g1', inputs=['a'], src_indices=[0], src_shape=(1,))

        p.setup()

        p['a'] = 99
        p['b'] = 2

        p.run_model()

        assert_near_equal(p['g1.y'], 101.)
        assert_near_equal(p['g1.g2.c2.z'], [9999.] * 4)

class SrcIndicesMPITestCase(unittest.TestCase):
    N_PROCS = 2

    def test_multi_promotes_mpi(self):
        p = om.Problem()
        par = p.model.add_subsystem('par', om.ParallelGroup())
        g1 = par.add_subsystem('g1', om.Group())
        g2 = par.add_subsystem('g2', om.Group())
        g1.add_subsystem('C1', om.ExecComp('y = 3*x', shape=3))
        g2.add_subsystem('C2', om.ExecComp('y = 2*x', shape=2))
        g1.promotes('C1', inputs=['x'], src_indices=om.slicer[:, 1], src_shape=(3,2))
        g2.promotes('C2', inputs=['x'], src_indices=[0,-1], src_shape=(3,2), flat_src_indices=True)
        par.promotes('g1', inputs=['x'])
        par.promotes('g2', inputs=['x'])

        # we want the auto_ivc connected to x to have a shape of (3,2), which differs from the
        # shapes of either of the connected absolute inputs.
        par.set_input_defaults('x', src_shape=(3,2))

        #import wingdbstub
        p.setup()
        p.run_model()


if __name__ == '__main__':
    unittest.main()