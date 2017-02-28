import unittest
from six import assertRaisesRegex
import itertools

import numpy as np
from nose_parameterized import parameterized

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ExplicitComponent
from openmdao.devtools.testutil import assert_rel_error
try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None


class SimpleGroup(Group):

    def initialize(self):
        self.add_subsystem('comp1', IndepVarComp('x', 5.0))
        self.add_subsystem('comp2', ExecComp('b=2*a'))
        self.connect('comp1.x', 'comp2.a')


class BranchGroup(Group):

    def initialize(self):
        b1 = self.add_subsystem('Branch1', Group())
        g1 = b1.add_subsystem('G1', Group())
        g2 = g1.add_subsystem('G2', Group())
        g2.add_subsystem('comp1', ExecComp('b=2.0*a', a=3.0, b=6.0))

        b2 = self.add_subsystem('Branch2', Group())
        g3 = b2.add_subsystem('G3', Group())
        g3.add_subsystem('comp2', ExecComp('b=3.0*a', a=4.0, b=12.0))


class TestGroup(unittest.TestCase):

    def test_same_sys_name(self):
        """Test error checking for the case where we add two subsystems with the same name."""
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp('x', 5.0))
        p.model.add_subsystem('comp2', ExecComp('b=2*a'))

        try:
            p.model.add_subsystem('comp2', ExecComp('b=2*a'))
        except Exception as err:
            self.assertEqual(str(err), "Subsystem name 'comp2' is already used.")
        else:
            self.fail('Exception expected.')

    def test_group_simple(self):
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', ExecComp('b=2.0*a', a=3.0, b=6.0))

        p.setup()

        self.assertEqual(p['comp1.a'], 3.0)
        self.assertEqual(p['comp1.b'], 6.0)

    def test_group_simple_promoted(self):
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', ExecComp('b=2.0*a', a=3.0, b=6.0),
                              promotes_inputs=['a'], promotes_outputs=['b'])

        p.setup()

        self.assertEqual(p['comp1.a'], 3.0) # still use unpromoted name
        self.assertEqual(p['b'], 6.0)

    def test_group_nested(self):
        p = Problem(model=Group())
        g1 = p.model.add_subsystem('G1', Group())
        g1.add_subsystem('comp1', ExecComp('b=2.0*a', a=3.0, b=6.0))
        g1.add_subsystem('comp2', ExecComp('b=3.0*a', a=4.0, b=12.0))

        p.setup()

        self.assertEqual(p['G1.comp1.a'], 3.0)
        self.assertEqual(p['G1.comp1.b'], 6.0)
        self.assertEqual(p['G1.comp2.a'], 4.0)
        self.assertEqual(p['G1.comp2.b'], 12.0)

    def test_group_getsystem_top(self):
        p = Problem(model=BranchGroup())
        p.setup()

        c1 = p.model.get_subsystem('Branch1.G1.G2.comp1')
        self.assertEqual(c1.pathname, 'Branch1.G1.G2.comp1')

        c2 = p.model.get_subsystem('Branch2.G3.comp2')
        self.assertEqual(c2.pathname, 'Branch2.G3.comp2')

    def test_group_getsystem_middle(self):
        p = Problem(model=BranchGroup())
        p.setup()

        grp = p.model.get_subsystem('Branch1.G1')
        c1 = grp.get_subsystem('G2.comp1')
        self.assertEqual(c1.pathname, 'Branch1.G1.G2.comp1')

    def test_group_nested_promoted1(self):
        # promotes from bottom level up 1
        p = Problem(model=Group())
        g1 = p.model.add_subsystem('G1', Group())
        g1.add_subsystem('comp1', ExecComp('b=2.0*a', a=3.0, b=6.0),
                         promotes_inputs=['a'], promotes_outputs=['b'])
        g1.add_subsystem('comp2', ExecComp('b=3.0*a', a=4.0, b=12.0),
                         promotes_inputs=['a'])
        p.setup()

        # output G1.comp1.b is promoted
        self.assertEqual(p['G1.b'], 6.0)
        # output G1.comp2.b is not promoted
        self.assertEqual(p['G1.comp2.b'], 12.0)

        # use unpromoted names for the following 2 promoted inputs
        self.assertEqual(p['G1.comp1.a'], 3.0)
        self.assertEqual(p['G1.comp2.a'], 4.0)

    def test_group_nested_promoted2(self):
        # promotes up from G1 level
        p = Problem(model=Group())
        g1 = Group()
        g1.add_subsystem('comp1', ExecComp('b=2.0*a', a=3.0, b=6.0))
        g1.add_subsystem('comp2', ExecComp('b=3.0*a', a=4.0, b=12.0))

        # use glob pattern 'comp?.a' to promote both comp1.a and comp2.a
        # use glob pattern 'comp?.b' to promote both comp1.b and comp2.b
        p.model.add_subsystem('G1', g1,
                              promotes_inputs=['comp?.a'],
                              promotes_outputs=['comp?.b'])
        p.setup()

        # output G1.comp1.b is promoted
        self.assertEqual(p['comp1.b'], 6.0)
        # output G1.comp2.b is promoted
        self.assertEqual(p['comp2.b'], 12.0)

        # access both promoted inputs using unpromoted names.
        self.assertEqual(p['G1.comp1.a'], 3.0)
        self.assertEqual(p['G1.comp2.a'], 4.0)

    def test_group_promotes(self):
        """Promoting a single variable."""
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp([
                ('a', 2.0),
                ('x', 5.0),
            ]),
            promotes_outputs='x')
        p.model.add_subsystem('comp2', ExecComp('y=2*x'), promotes_inputs='x')
        p.setup()

        p.model.suppress_solver_output = True
        p.run_model()

        self.assertEqual(p['comp1.a'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_promotes_multiple(self):
        """Promoting multiple variables."""
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp([
                ('a', 2.0),
                ('x', 5.0),
            ]),
            promotes_outputs=['a', 'x'])
        p.model.add_subsystem('comp2', ExecComp('y=2*x'), promotes_inputs='x')
        p.setup()

        p.model.suppress_solver_output = True
        p.run_model()

        self.assertEqual(p['a'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_promotes_all(self):
        """Promoting all variables with asterisk."""
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp([
                ('a', 2.0),
                ('x', 5.0),
            ]),
            promotes_outputs='*')
        p.model.add_subsystem('comp2', ExecComp('y=2*x'), promotes_inputs=['x'])
        p.setup()

        p.model.suppress_solver_output = True
        p.run_model()

        self.assertEqual(p['a'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_nested_conn(self):
        """Example of adding subsystems and issuing connections with nested groups."""
        g1 = Group()
        c1_1 = g1.add_subsystem('comp1', IndepVarComp('x', 5.0))
        c1_2 = g1.add_subsystem('comp2', ExecComp('b=2*a'))
        g1.connect('comp1.x', 'comp2.a')
        g2 = Group()
        c2_1 = g2.add_subsystem('comp1', ExecComp('b=2*a'))
        c2_2 = g2.add_subsystem('comp2', ExecComp('b=2*a'))
        g2.connect('comp1.b', 'comp2.a')

        model = Group()
        model.add_subsystem('group1', g1)
        model.add_subsystem('group2', g2)
        model.connect('group1.comp2.b', 'group2.comp1.a')

        p = Problem(model=model)
        p.setup()

        c1_1 = p.model.get_subsystem('group1.comp1')
        c1_2 = p.model.get_subsystem('group1.comp2')
        c2_1 = p.model.get_subsystem('group2.comp1')
        c2_2 = p.model.get_subsystem('group2.comp2')
        self.assertEqual(c1_1.name, 'comp1')
        self.assertEqual(c1_2.name, 'comp2')
        self.assertEqual(c2_1.name, 'comp1')
        self.assertEqual(c2_2.name, 'comp2')

        c1_1 = p.model.get_subsystem('group1').get_subsystem('comp1')
        c1_2 = p.model.get_subsystem('group1').get_subsystem('comp2')
        c2_1 = p.model.get_subsystem('group2').get_subsystem('comp1')
        c2_2 = p.model.get_subsystem('group2').get_subsystem('comp2')
        self.assertEqual(c1_1.name, 'comp1')
        self.assertEqual(c1_2.name, 'comp2')
        self.assertEqual(c2_1.name, 'comp1')
        self.assertEqual(c2_2.name, 'comp2')

        s = p.model.get_subsystem('')
        self.assertEqual(s, None)

        p.model.suppress_solver_output = True
        p.run_model()

        self.assertEqual(p['group1.comp1.x'],  5.0)
        self.assertEqual(p['group1.comp2.b'], 10.0)
        self.assertEqual(p['group2.comp1.b'], 20.0)
        self.assertEqual(p['group2.comp2.b'], 40.0)

    def test_reused_output_promoted_names(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0))
        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem("C1", ExecComp("y=2.0*x"), promotes=['y'])
        G1.add_subsystem("C2", ExecComp("y=2.0*x"), promotes=['y'])
        msg = "Output name 'y' refers to multiple outputs: \['G1.C2.y', 'G1.C1.y'\]."
        with assertRaisesRegex(self, Exception, msg):
            prob.setup(check=False)

    def test_basic_connect_units(self):
        p = Problem(model=Group())
        indep = p.model.add_subsystem('indep', IndepVarComp())
        indep.add_output('x', np.ones(5), units='ft')
        p.model.add_subsystem('C1', ExecComp('y=sum(x)', x=np.zeros(5),
                                             units={'x': 'inch', 'y': 'inch'}))
        p.model.connect('indep.x', 'C1.x')
        p.model.suppress_solver_output = True
        p.setup()
        p.run_model()
        assert_rel_error(self, p['indep.x'], np.ones(5))
        assert_rel_error(self, p['C1.x'], np.ones(5)*12.)
        assert_rel_error(self, p['C1.y'], 60.)

    def test_connect_1_to_many(self):
        p = Problem(model=Group())
        p.model.add_subsystem('indep', IndepVarComp('x', np.ones(5)))
        p.model.add_subsystem('C1', ExecComp('y=sum(x)*2.0', x=np.zeros(5)))
        p.model.add_subsystem('C2', ExecComp('y=sum(x)*4.0', x=np.zeros(5)))
        p.model.add_subsystem('C3', ExecComp('y=sum(x)*6.0', x=np.zeros(5)))
        p.model.connect('indep.x', ['C1.x', 'C2.x', 'C3.x'])
        p.model.suppress_solver_output = True
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.y'], 10.)
        assert_rel_error(self, p['C2.y'], 20.)
        assert_rel_error(self, p['C3.y'], 30.)

    def test_double_src_indices(self):
        class MyComp1(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('x', np.ones(3), src_indices=[0, 1, 2])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        p = Problem(model=Group())

        p.model.add_subsystem('indep', IndepVarComp('x', np.ones(5)))
        p.model.add_subsystem('C1', MyComp1())
        p.model.connect('indep.x', 'C1.x', src_indices=[1, 0, 2])

        with self.assertRaises(Exception) as context:
            p.setup(check=False)
        self.assertEqual(str(context.exception),
                         ": src_indices has been defined in both "
                         "connect('indep.x', 'C1.x') and add_input('C1.x', ...).")

    def test_connect_src_indices(self):
        p = Problem(model=Group())
        p.model.add_subsystem('indep', IndepVarComp('x', np.ones(5)))
        p.model.add_subsystem('C1', ExecComp('y=sum(x)*2.0', x=np.zeros(3)))
        p.model.add_subsystem('C2', ExecComp('y=sum(x)*4.0', x=np.zeros(2)))

        # connect C1.x to the first 3 entries of indep.x
        p.model.connect('indep.x', 'C1.x', src_indices=[0, 1, 2])

        # connect C2.x to the last 2 entries of indep.x
        p.model.connect('indep.x', 'C2.x', src_indices=[3, 4])

        p.model.suppress_solver_output = True
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.x'], np.ones(3))
        assert_rel_error(self, p['C1.y'], 6.)
        assert_rel_error(self, p['C2.x'], np.ones(2))
        assert_rel_error(self, p['C2.y'], 8.)

    def test_connect_src_indices_noflat(self):
        p = Problem(model=Group())
        p.model.add_subsystem('indep', IndepVarComp('x', np.arange(12).reshape((4,3))))
        p.model.add_subsystem('C1', ExecComp('y=numpy.sum(x)*2.0', x=np.zeros((2,2))))

        # connect C1.x to entries (0,0), (3,1), (2,1), (1,1) of indep.x
        p.model.connect('indep.x', 'C1.x',
                        src_indices=[[(0,0), (3,1)],
                                     [(2,1), (1,1)]])

        p.model.suppress_solver_output = True
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.x'], np.array([[0., 10.],
                                                    [7., 4.]]))
        assert_rel_error(self, p['C1.y'], 42.)

    def test_promote_not_found1(self):
        p = Problem(model=Group())
        p.model.add_subsystem('indep', IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', ExecComp('y=x'), promotes_inputs=['x'])
        p.model.add_subsystem('C2', ExecComp('y=x'), promotes_outputs=['x*'])

        with self.assertRaises(Exception) as context:
            p.setup(check=False)
        self.assertEqual(str(context.exception),
                         "C2: no variables were promoted based on promotes_outputs=['x*']")
        
    def test_promote_not_found2(self):
        p = Problem(model=Group())
        p.model.add_subsystem('indep', IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', ExecComp('y=x'), promotes_inputs=['x'])
        p.model.add_subsystem('C2', ExecComp('y=x'), promotes_inputs=['xx'])

        with self.assertRaises(Exception) as context:
            p.setup(check=False)
        self.assertEqual(str(context.exception),
                         "C2: no variables were promoted based on promotes_inputs=['xx']")

    def test_promote_not_found3(self):
        p = Problem(model=Group())
        p.model.add_subsystem('indep', IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', ExecComp('y=x'), promotes=['x'])
        p.model.add_subsystem('C2', ExecComp('y=x'), promotes=['xx'])

        with self.assertRaises(Exception) as context:
            p.setup(check=False)
        self.assertEqual(str(context.exception),
                         "C2: no variables were promoted based on promotes=['xx']")

    def test_promote_src_indices(self):
        class MyComp1(ExplicitComponent):
            def initialize_variables(self):
                # this input will connect to entries 0, 1, and 2 of its source
                self.add_input('x', np.ones(3), src_indices=[0, 1, 2])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        class MyComp2(ExplicitComponent):
            def initialize_variables(self):
                # this input will connect to entries 3 and 4 of its source
                self.add_input('x', np.ones(2), src_indices=[3, 4])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*4.0

        p = Problem(model=Group())

        # by promoting the following output and inputs to 'x', they will
        # be automatically connected
        p.model.add_subsystem('indep', IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp1(), promotes_inputs=['x'])
        p.model.add_subsystem('C2', MyComp2(), promotes_inputs=['x'])

        p.model.suppress_solver_output = True
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.x'], np.ones(3))
        assert_rel_error(self, p['C1.y'], 6.)
        assert_rel_error(self, p['C2.x'], np.ones(2))
        assert_rel_error(self, p['C2.y'], 8.)

    def test_promote_src_indices_nonflat(self):
        class MyComp(ExplicitComponent):
            def initialize_variables(self):
                # We want to pull the following 4 values out of the source:
                # [(0,0), (3,1), (2,1), (1,1)].
                # Because our input is also non-flat we must arrange the
                # source index tuples into an array having the same shape
                # as our input.
                self.add_input('x', np.ones((2,2)),
                               src_indices=[[(0,0), (3,1)],
                                            [(2,1), (1,1)]])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

        p = Problem(model=Group())

        # by promoting the following output and inputs to 'x', they will
        # be automatically connected
        p.model.add_subsystem('indep',
                              IndepVarComp('x', np.arange(12).reshape((4,3))),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        p.model.suppress_solver_output = True
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.x'],
                         np.array([[0., 10.],
                                   [7., 4.]]))
        assert_rel_error(self, p['C1.y'], 21.)

    def test_promote_src_indices_nonflat_to_scalars(self):
        class MyComp(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('x', 1.0, src_indices=[(3,1)], shape=(1,))
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x']*2.0

        p = Problem(model=Group())

        p.model.add_subsystem('indep',
                              IndepVarComp('x', np.arange(12).reshape((4,3))),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        p.model.suppress_solver_output = True
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.x'], 10.)
        assert_rel_error(self, p['C1.y'], 20.)

    def test_promote_src_indices_nonflat_error(self):
        class MyComp(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('x', 1.0, src_indices=[(3,1)])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

        p = Problem(model=Group())

        p.model.add_subsystem('indep',
                              IndepVarComp('x', np.arange(12).reshape((4,3))),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        with self.assertRaises(Exception) as context:
            p.setup(check=False)
        self.assertEqual(str(context.exception),
                         "src_indices for 'x' is not flat, so its input shape "
                         "must be provided. src_indices may contain an extra "
                         "dimension if the connected source is not flat, making "
                         "the input shape ambiguous.")

    @parameterized.expand(itertools.product(
        [((4,3),  [(0,0), (3,1), (2,1), (1,1)]),
         ((1,12), [(0,0), (0,10), (0,7), (0,4)]),
         ((12,),  [0, 10, 7, 4]),
         ((12,1), [(0,0), (10,0), (7,0), (4,0)])],
        [(2,2), (4,), (4,1), (1,4)],
        ), testcase_func_name=lambda f, n, p: 'test_promote_src_indices_'+'_'.join(str(a) for a in p.args)
    )
    def test_promote_src_indices_param(self, src_info, tgt_shape):
        src_shape, idxvals = src_info
        class MyComp(ExplicitComponent):
            def initialize_variables(self):
                if len(tgt_shape) == 1:
                    tshape = None  # don't need to set shape if input is flat
                    sidxs = idxvals
                else:
                    tshape = tgt_shape
                    sidxs = []
                    i = 0
                    for r in range(tgt_shape[0]):
                        sidxs.append([])
                        for c in range(tgt_shape[1]):
                            sidxs[-1].append(idxvals[i])
                            i += 1

                self.add_input('x', np.ones(4).reshape(tgt_shape),
                               src_indices=sidxs, shape=tshape)
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

        p = Problem(model=Group())

        p.model.add_subsystem('indep',
                              IndepVarComp('x', np.arange(12).reshape(src_shape)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        p.model.suppress_solver_output = True
        p.setup(check=False)
        p.run_model()
        assert_rel_error(self, p['C1.x'],
                         np.array([0., 10., 7., 4.]).reshape(tgt_shape))
        assert_rel_error(self, p['C1.y'], 21.)


class TestGroupMPI(unittest.TestCase):
    # FIXME: fix MPI stuff so this can run as an MPI test
    #N_PROCS = 2

    def test_promote_distrib(self):
        if PETScVector is None:
            raise unittest.SkipTest("PETSc is not installed")

        class MyComp(ExplicitComponent):
            def initialize_variables(self):
                # decide what parts of the array we want based on our rank
                if self.comm.rank == 0:
                    idxs = [0, 1, 2]
                else:
                    idxs = [3, 4]

                self.add_input('x', np.ones(len(idxs)), src_indices=idxs)
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        p = Problem(model=Group())

        p.model.add_subsystem('indep', IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        p.model.suppress_solver_output = True
        p.setup(PETScVector)
        p.run_model()
        if p.model.comm.rank == 0:
            assert_rel_error(self, p['C1.x'], np.ones(3))
            assert_rel_error(self, p['C1.y'], 6.)
        else:
            assert_rel_error(self, p['C1.x'], np.ones(2))
            assert_rel_error(self, p['C1.y'], 4.)


class TestConnect(unittest.TestCase):

    def setUp(self):
        prob = Problem(Group())

        sub = prob.model.add_subsystem('sub', Group())
        sub.add_subsystem('src', IndepVarComp('x', np.zeros(5,)))
        sub.add_subsystem('tgt', ExecComp('y = x'))
        sub.add_subsystem('cmp', ExecComp('z = x'))

        self.sub = sub
        self.prob = prob

    def test_src_indices_as_int_list(self):
        self.sub.connect('src.x', 'tgt.x', src_indices=[1])

    def test_src_indices_as_int_array(self):
        self.sub.connect('src.x', 'tgt.x', src_indices=np.zeros(1, dtype=int))

    def test_src_indices_as_float_list(self):
        msg = "src_indices must contain integers, but src_indices for " + \
              "connection from 'src.x' to 'tgt.x' is <.* 'numpy.float64'>."

        with assertRaisesRegex(self, TypeError, msg):
            self.sub.connect('src.x', 'tgt.x', src_indices=[1.0])

    def test_src_indices_as_float_array(self):
        msg = "src_indices must contain integers, but src_indices for " + \
              "connection from 'src.x' to 'tgt.x' is <.* 'numpy.float64'>."

        with assertRaisesRegex(self, TypeError, msg):
            self.sub.connect('src.x', 'tgt.x', src_indices=np.zeros(1))

    def test_src_indices_as_str(self):
        msg = "src_indices must be an index array, " + \
              "did you mean connect('src.x', [tgt.x, cmp.x])?"

        with assertRaisesRegex(self, TypeError, msg):
            self.sub.connect('src.x', 'tgt.x', 'cmp.x')

    def test_already_connected(self):
        msg = "Input 'tgt.x' is already connected to 'src.x'."

        self.sub.connect('src.x', 'tgt.x', src_indices=[1])
        with assertRaisesRegex(self, RuntimeError, msg):
            self.sub.connect('cmp.x', 'tgt.x', src_indices=[1])

    def test_invalid_source(self):
        msg = "Output 'src.z' does not exist for connection " + \
              "in 'sub' from 'src.z' to 'tgt.x'."

        # source and target names can't be checked until setup
        # because initialize_variables is not called until then
        self.sub.connect('src.z', 'tgt.x', src_indices=[1])
        with assertRaisesRegex(self, NameError, msg):
            self.prob.setup(check=False)

    def test_invalid_target(self):
        msg = "Input 'tgt.z' does not exist for connection " + \
              "in 'sub' from 'src.x' to 'tgt.z'."

        # source and target names can't be checked until setup
        # because initialize_variables is not called until then
        self.sub.connect('src.x', 'tgt.z', src_indices=[1])
        with assertRaisesRegex(self, NameError, msg):
            self.prob.setup(check=False)

    def test_connect_within_system(self):
        msg = "Output and input are in the same System for connection " + \
              "from 'tgt.y' to 'tgt.x'."

        with assertRaisesRegex(self, RuntimeError, msg):
            self.sub.connect('tgt.y', 'tgt.x', src_indices=[1])

    def test_connect_within_system_with_promotes(self):
        prob = Problem(Group())

        sub = prob.model.add_subsystem('sub', Group())
        sub.add_subsystem('tgt', ExecComp('y = x'), promotes_outputs=['y'])
        sub.connect('y', 'tgt.x', src_indices=[1])

        msg = "Output and input are in the same System for connection " + \
              "in 'sub' from 'y' to 'tgt.x'."

        with assertRaisesRegex(self, RuntimeError, msg):
            prob.setup(check=False)

    def test_connect_units_with_unitless(self):
        msg = "Output units of 'degC' for 'src.x2' are incompatible with input units of 'unitless' for 'tgt.x'."

        prob = Problem(Group())
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0))
        prob.model.add_subsystem('src', ExecComp('x2 = 2 * x1', units={'x2': 'degC'}))
        prob.model.add_subsystem('tgt', ExecComp('y = 3 * x', units={'x': 'unitless'}))

        prob.model.connect('px1.x1', 'src.x1')
        prob.model.connect('src.x2', 'tgt.x')

        with assertRaisesRegex(self, RuntimeError, msg):
            prob.setup(check=False)

    def test_connect_incompatible_units(self):
        msg = "Output units of 'degC' for 'src.x2' are incompatible with input units of 'm' for 'tgt.x'."

        prob = Problem(Group())
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0))
        prob.model.add_subsystem('src', ExecComp('x2 = 2 * x1', units={'x2': 'degC'}))
        prob.model.add_subsystem('tgt', ExecComp('y = 3 * x', units={'x': 'm'}))

        prob.model.connect('px1.x1', 'src.x1')
        prob.model.connect('src.x2', 'tgt.x')

        with assertRaisesRegex(self, RuntimeError, msg):
            prob.setup(check=False)

    def test_connect_units_with_nounits(self):
        prob = Problem(Group())
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0))
        prob.model.add_subsystem('src', ExecComp('x2 = 2 * x1', units={'x2': 'degC'}))
        prob.model.add_subsystem('tgt', ExecComp('y = 3 * x'))

        prob.model.connect('px1.x1', 'src.x1')
        prob.model.connect('src.x2', 'tgt.x')
        prob.model.suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['tgt.y'], 600.)


if __name__ == "__main__":
    unittest.main()
