"""
Unit tests for Group.
"""
from __future__ import print_function

import itertools
import unittest

from six import assertRaisesRegex, iteritems
from six.moves import range

import numpy as np

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDis2
from openmdao.utils.assert_utils import assert_rel_error, assert_warning


class SimpleGroup(om.Group):

    def __init__(self):
        super(SimpleGroup, self).__init__()

        self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
        self.add_subsystem('comp2', om.ExecComp('b=2*a'))
        self.connect('comp1.x', 'comp2.a')


class BranchGroup(om.Group):

    def __init__(self):
        super(BranchGroup, self).__init__()

        b1 = self.add_subsystem('Branch1', om.Group())
        g1 = b1.add_subsystem('G1', om.Group())
        g2 = g1.add_subsystem('G2', om.Group())
        g2.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0))

        b2 = self.add_subsystem('Branch2', om.Group())
        g3 = b2.add_subsystem('G3', om.Group())
        g3.add_subsystem('comp2', om.ExecComp('b=3.0*a', a=4.0, b=12.0))


class SetOrderGroup(om.Group):
    def setup(self):
        self.add_subsystem('C1', om.ExecComp('y=2.0*x'))
        self.add_subsystem('C2', om.ExecComp('y=2.0*x'))
        self.add_subsystem('C3', om.ExecComp('y=2.0*x'))

        self.set_order(['C1', 'C3', 'C2'])

        self.connect('C1.y', 'C3.x')
        self.connect('C3.y', 'C2.x')


class ReportOrderComp(om.ExplicitComponent):
    def __init__(self, order_list):
        super(ReportOrderComp, self).__init__()
        self._order_list = order_list

    def setup(self):
        self.add_input('x', 0.0)
        self.add_output('y', 0.0)

    def compute(self, inputs, outputs):
        self._order_list.append(self.pathname)


class TestGroup(unittest.TestCase):

    def test_add_subsystem_class(self):
        p = om.Problem()
        try:
            p.model.add_subsystem('comp', om.IndepVarComp)
        except TypeError as err:
            self.assertEqual(str(err), "Group: Subsystem 'comp' should be an instance, "
                                       "but a IndepVarComp class object was found.")
        else:
            self.fail('Exception expected.')

    def test_same_sys_name(self):
        """Test error checking for the case where we add two subsystems with the same name."""
        p = om.Problem()
        p.model.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
        p.model.add_subsystem('comp2', om.ExecComp('b=2*a'))

        try:
            p.model.add_subsystem('comp2', om.ExecComp('b=2*a'))
        except Exception as err:
            self.assertEqual(str(err), "Group: Subsystem name 'comp2' is already used.")
        else:
            self.fail('Exception expected.')

    def test_deprecated_runonce(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', 5.0))
        p.model.add_subsystem('comp', om.ExecComp('b=2*a'))

        msg = "NonLinearRunOnce is deprecated.  Use NonlinearRunOnce instead."

        with assert_warning(DeprecationWarning, msg):
            p.model.nonlinear_solver = om.NonLinearRunOnce()

    def test_group_simple(self):
        import openmdao.api as om

        p = om.Problem()
        p.model.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0))

        p.setup()

        self.assertEqual(p['comp1.a'], 3.0)
        self.assertEqual(p['comp1.b'], 6.0)

    def test_group_add(self):
        model = om.Group()
        ecomp = om.ExecComp('b=2.0*a', a=3.0, b=6.0)

        msg = "The 'add' method provides backwards compatibility with OpenMDAO <= 1.x ; " \
              "use 'add_subsystem' instead."

        with assert_warning(DeprecationWarning, msg):
            comp1 = model.add('comp1', ecomp)

        self.assertTrue(ecomp is comp1)

    def test_group_simple_promoted(self):
        import openmdao.api as om

        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('a', 3.0),
                              promotes_outputs=['a'])
        p.model.add_subsystem('comp1', om.ExecComp('b=2.0*a'),
                              promotes_inputs=['a'])

        p.setup()
        p.run_model()

        self.assertEqual(p['a'], 3.0)
        self.assertEqual(p['comp1.b'], 6.0)

    def test_inner_connect_w_extern_promote(self):
        p = om.Problem()
        g = p.model.add_subsystem('g', om.Group(), promotes_inputs=['c0.x'])
        g.add_subsystem('ivc', om.IndepVarComp('x', 2.))
        g.add_subsystem('c0', om.ExecComp('y = 2*x'))
        g.connect('ivc.x', 'c0.x')

        p.setup()
        p.final_setup()

        from openmdao.error_checking.check_config import _get_promoted_connected_ins
        ins = _get_promoted_connected_ins(p.model)
        self.assertEqual(len(ins), 1)
        inp, tup = list(ins.items())[0]
        in_proms, mans = tup
        self.assertEqual(inp, 'g.c0.x')
        self.assertEqual(in_proms, ['g'])
        self.assertEqual(mans, [('c0.x', 'g')])

    def test_inner_connect_w_2extern_promotes(self):
        p = om.Problem()
        g0 = p.model.add_subsystem('g0', om.Group(), promotes_inputs=['c0.x'])
        g = g0.add_subsystem('g', om.Group(), promotes_inputs=['c0.x'])
        g.add_subsystem('ivc', om.IndepVarComp('x', 2.))
        g.add_subsystem('c0', om.ExecComp('y = 2*x'))
        g.connect('ivc.x', 'c0.x')

        p.setup()
        p.final_setup()

        from openmdao.error_checking.check_config import _get_promoted_connected_ins
        ins = _get_promoted_connected_ins(p.model)
        self.assertEqual(len(ins), 1)
        inp, tup = list(ins.items())[0]
        in_proms, mans = tup
        self.assertEqual(inp, 'g0.g.c0.x')
        self.assertEqual(list(sorted(in_proms)), ['g0', 'g0.g'])
        self.assertEqual(mans, [('c0.x', 'g0.g')])

    def test_group_rename_connect(self):
        import openmdao.api as om

        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('aa', 3.0),
                              promotes=['aa'])
        p.model.add_subsystem('comp1', om.ExecComp('b=2.0*aa'),
                              promotes_inputs=['aa'])

        # here we alias 'a' to 'aa' so that it will be automatically
        # connected to the independent variable 'aa'.
        p.model.add_subsystem('comp2', om.ExecComp('b=3.0*a'),
                              promotes_inputs=[('a', 'aa')])

        p.setup()
        p.run_model()

        self.assertEqual(p['comp1.b'], 6.0)
        self.assertEqual(p['comp2.b'], 9.0)

    def test_subsys_attributes(self):
        p = om.Problem()

        class MyGroup(om.Group):
            def setup(self):
                # two subsystems added during setup
                self.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0))
                self.add_subsystem('comp2', om.ExecComp('b=3.0*a', a=4.0, b=12.0))

        # subsystems become attributes
        my_group = p.model.add_subsystem('gg', MyGroup())
        self.assertTrue(p.model.gg is my_group)

        # after calling setup(), MyGroup's subsystems are also attributes
        p.setup()
        self.assertTrue(hasattr(p.model.gg, 'comp1'))
        self.assertTrue(hasattr(p.model.gg, 'comp2'))

        # calling setup() again doesn't break anything
        p.setup()
        self.assertTrue(p.model.gg is my_group)
        self.assertTrue(hasattr(p.model.gg, 'comp1'))
        self.assertTrue(hasattr(p.model.gg, 'comp2'))

        # name cannot start with an underscore
        with self.assertRaises(Exception) as err:
            p.model.add_subsystem('_bad_name', om.Group())
        self.assertEqual(str(err.exception),
                         "Group (<top>): '_bad_name' is not a valid sub-system name.")

        # 'name', 'pathname', 'comm' and 'options' are reserved names
        for reserved in ['name', 'pathname', 'comm', 'options']:
            with self.assertRaises(Exception) as err:
                p.model.add_subsystem(reserved, om.Group())
            self.assertEqual(str(err.exception),
                             "Group (<top>): Can't add subsystem '%s' because an attribute with that name already exits." %
                             reserved)

    def test_group_nested(self):
        import openmdao.api as om

        p = om.Problem()
        p.model.add_subsystem('G1', om.Group())
        p.model.G1.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0))
        p.model.G1.add_subsystem('comp2', om.ExecComp('b=3.0*a', a=4.0, b=12.0))

        p.setup()

        self.assertEqual(p['G1.comp1.a'], 3.0)
        self.assertEqual(p['G1.comp1.b'], 6.0)
        self.assertEqual(p['G1.comp2.a'], 4.0)
        self.assertEqual(p['G1.comp2.b'], 12.0)

    def test_group_getsystem_top(self):
        import openmdao.api as om
        from openmdao.core.tests.test_group import BranchGroup

        p = om.Problem(model=BranchGroup())
        p.setup()

        c1 = p.model.Branch1.G1.G2.comp1
        self.assertEqual(c1.pathname, 'Branch1.G1.G2.comp1')

        c2 = p.model.Branch2.G3.comp2
        self.assertEqual(c2.pathname, 'Branch2.G3.comp2')

    def test_group_nested_promoted1(self):
        import openmdao.api as om

        # promotes from bottom level up 1
        p = om.Problem()
        g1 = p.model.add_subsystem('G1', om.Group())
        g1.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0),
                         promotes_inputs=['a'], promotes_outputs=['b'])
        g1.add_subsystem('comp2', om.ExecComp('b=3.0*a', a=4.0, b=12.0),
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
        import openmdao.api as om

        # promotes up from G1 level
        p = om.Problem()
        g1 = om.Group()
        g1.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0))
        g1.add_subsystem('comp2', om.ExecComp('b=3.0*a', a=4.0, b=12.0))

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
        p = om.Problem()
        p.model.add_subsystem('comp1', om.IndepVarComp([('a', 2.0), ('x', 5.0)]),
                              promotes_outputs=['x'])
        p.model.add_subsystem('comp2', om.ExecComp('y=2*x'), promotes_inputs=['x'])
        p.setup()

        p.set_solver_print(level=0)
        p.run_model()

        self.assertEqual(p['comp1.a'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_renames(self):
        p = om.Problem()
        p.model.add_subsystem('comp1', om.IndepVarComp('x', 5.0),
                              promotes_outputs=[('x', 'foo')])
        p.model.add_subsystem('comp2', om.ExecComp('y=2*foo'), promotes_inputs=['foo'])
        p.setup()

        p.set_solver_print(level=0)
        p.run_model()

        self.assertEqual(p['foo'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_renames_errors_single_string(self):
        p = om.Problem()
        with self.assertRaises(Exception) as err:
            p.model.add_subsystem('comp1', om.IndepVarComp('x', 5.0),
                                  promotes_outputs='x')
        self.assertEqual(str(err.exception),
                         "Group: promotes must be an iterator of strings and/or tuples.")

    def test_group_renames_errors_not_found(self):
        p = om.Problem()
        p.model.add_subsystem('comp1', om.IndepVarComp('x', 5.0),
                              promotes_outputs=[('xx', 'foo')])
        p.model.add_subsystem('comp2', om.ExecComp('y=2*foo'), promotes_inputs=['foo'])

        with self.assertRaises(Exception) as err:
            p.setup()
        self.assertEqual(str(err.exception),
                         "IndepVarComp (comp1): 'promotes_outputs' failed to find any matches for "
                         "the following names or patterns: ['xx'].")

    def test_group_renames_errors_bad_tuple(self):
        p = om.Problem()
        p.model.add_subsystem('comp1', om.IndepVarComp('x', 5.0),
                              promotes_outputs=[('x', 'foo', 'bar')])
        p.model.add_subsystem('comp2', om.ExecComp('y=2*foo'), promotes_inputs=['foo'])

        with self.assertRaises(Exception) as err:
            p.setup()
        self.assertEqual(str(err.exception),
                         "when adding subsystem 'comp1', entry '('x', 'foo', 'bar')' "
                         "is not a string or tuple of size 2")

    def test_group_promotes_multiple(self):
        """Promoting multiple variables."""
        p = om.Problem()
        p.model.add_subsystem('comp1', om.IndepVarComp([('a', 2.0), ('x', 5.0)]),
                              promotes_outputs=['a', 'x'])
        p.model.add_subsystem('comp2', om.ExecComp('y=2*x'),
                              promotes_inputs=['x'])
        p.setup()

        p.set_solver_print(level=0)
        p.run_model()

        self.assertEqual(p['a'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_promotes_all(self):
        """Promoting all variables with asterisk."""
        p = om.Problem()
        p.model.add_subsystem('comp1', om.IndepVarComp([('a', 2.0), ('x', 5.0)]),
                              promotes_outputs=['*'])
        p.model.add_subsystem('comp2', om.ExecComp('y=2*x'),
                              promotes_inputs=['x'])
        p.setup()

        p.set_solver_print(level=0)
        p.run_model()

        self.assertEqual(p['a'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_promotes2(self):

        class Sellar(om.Group):
            def setup(self):
                dv = self.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
                dv.add_output('x', 1.0)
                dv.add_output('z', np.array([5.0, 2.0]))

                self.add_subsystem('d1', SellarDis2(),
                                   promotes_inputs=['y1'], promotes_outputs=['foo'])
                self.add_subsystem('d2', SellarDis2())

        p = om.Problem()
        p.model = Sellar()

        with self.assertRaises(Exception) as err:
            p.setup()
        self.assertEqual(str(err.exception),
                         "SellarDis2 (d1): 'promotes_outputs' failed to find any matches for "
                         "the following names or patterns: ['foo'].")

    def test_group_nested_conn(self):
        """Example of adding subsystems and issuing connections with nested groups."""
        g1 = om.Group()
        c1_1 = g1.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
        c1_2 = g1.add_subsystem('comp2', om.ExecComp('b=2*a'))
        g1.connect('comp1.x', 'comp2.a')
        g2 = om.Group()
        c2_1 = g2.add_subsystem('comp1', om.ExecComp('b=2*a'))
        c2_2 = g2.add_subsystem('comp2', om.ExecComp('b=2*a'))
        g2.connect('comp1.b', 'comp2.a')

        model = om.Group()
        model.add_subsystem('group1', g1)
        model.add_subsystem('group2', g2)
        model.connect('group1.comp2.b', 'group2.comp1.a')

        p = om.Problem(model=model)
        p.setup()

        c1_1 = p.model.group1.comp1
        c1_2 = p.model.group1.comp2
        c2_1 = p.model.group2.comp1
        c2_2 = p.model.group2.comp2
        self.assertEqual(c1_1.name, 'comp1')
        self.assertEqual(c1_2.name, 'comp2')
        self.assertEqual(c2_1.name, 'comp1')
        self.assertEqual(c2_2.name, 'comp2')

        c1_1 = p.model.group1.comp1
        c1_2 = p.model.group1.comp2
        c2_1 = p.model.group2.comp1
        c2_2 = p.model.group2.comp2
        self.assertEqual(c1_1.name, 'comp1')
        self.assertEqual(c1_2.name, 'comp2')
        self.assertEqual(c2_1.name, 'comp1')
        self.assertEqual(c2_2.name, 'comp2')

        s = p.model._get_subsystem('')
        self.assertEqual(s, None)

        p.set_solver_print(level=0)
        p.run_model()

        self.assertEqual(p['group1.comp1.x'], 5.0)
        self.assertEqual(p['group1.comp2.b'], 10.0)
        self.assertEqual(p['group2.comp1.b'], 20.0)
        self.assertEqual(p['group2.comp2.b'], 40.0)

    def test_reused_output_promoted_names(self):
        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0))
        G1 = prob.model.add_subsystem('G1', om.Group())
        G1.add_subsystem("C1", om.ExecComp("y=2.0*x"), promotes=['y'])
        G1.add_subsystem("C2", om.ExecComp("y=2.0*x"), promotes=['y'])
        msg = r"Output name 'y' refers to multiple outputs: \['G1.C1.y', 'G1.C2.y'\]."
        with assertRaisesRegex(self, Exception, msg):
            prob.setup()

    def test_basic_connect_units(self):
        import numpy as np

        import openmdao.api as om

        p = om.Problem()

        indep_comp = om.IndepVarComp()
        indep_comp.add_output('x', np.ones(5), units='ft')

        exec_comp = om.ExecComp('y=sum(x)',
                                x={'value': np.zeros(5), 'units': 'inch'},
                                y={'units': 'inch'})

        p.model.add_subsystem('indep', indep_comp)
        p.model.add_subsystem('comp1', exec_comp)
        p.model.connect('indep.x', 'comp1.x')

        p.setup()
        p.run_model()

        assert_rel_error(self, p['indep.x'], np.ones(5))
        assert_rel_error(self, p['comp1.x'], np.ones(5)*12.)
        assert_rel_error(self, p['comp1.y'], 60.)

    def test_connect_1_to_many(self):
        import numpy as np

        import openmdao.api as om

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)))
        p.model.add_subsystem('C1', om.ExecComp('y=sum(x)*2.0', x=np.zeros(5)))
        p.model.add_subsystem('C2', om.ExecComp('y=sum(x)*4.0', x=np.zeros(5)))
        p.model.add_subsystem('C3', om.ExecComp('y=sum(x)*6.0', x=np.zeros(5)))

        p.model.connect('indep.x', ['C1.x', 'C2.x', 'C3.x'])

        p.setup()
        p.run_model()

        assert_rel_error(self, p['C1.y'], 10.)
        assert_rel_error(self, p['C2.y'], 20.)
        assert_rel_error(self, p['C3.y'], 30.)

    def test_double_src_indices(self):
        class MyComp1(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(3), src_indices=[0, 1, 2])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)))
        p.model.add_subsystem('C1', MyComp1())
        p.model.connect('indep.x', 'C1.x', src_indices=[1, 0, 2])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "Group (<top>): src_indices has been defined in both "
                         "connect('indep.x', 'C1.x') and add_input('C1.x', ...).")

    def test_connect_src_indices(self):
        import numpy as np

        import openmdao.api as om

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)))
        p.model.add_subsystem('C1', om.ExecComp('y=sum(x)*2.0', x=np.zeros(3)))
        p.model.add_subsystem('C2', om.ExecComp('y=sum(x)*4.0', x=np.zeros(2)))

        # connect C1.x to the first 3 entries of indep.x
        p.model.connect('indep.x', 'C1.x', src_indices=[0, 1, 2])

        # connect C2.x to the last 2 entries of indep.x
        # use -2 (same as 3 in this case) to show that negative indices work.
        p.model.connect('indep.x', 'C2.x', src_indices=[-2, 4])

        p.setup()
        p.run_model()

        assert_rel_error(self, p['C1.x'], np.ones(3))
        assert_rel_error(self, p['C1.y'], 6.)
        assert_rel_error(self, p['C2.x'], np.ones(2))
        assert_rel_error(self, p['C2.y'], 8.)

    def test_connect_src_indices_noflat(self):
        import numpy as np

        import openmdao.api as om

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', np.arange(12).reshape((4, 3))))
        p.model.add_subsystem('C1', om.ExecComp('y=sum(x)*2.0', x=np.zeros((2, 2))))

        # connect C1.x to entries (0,0), (-1,1), (2,1), (1,1) of indep.x
        p.model.connect('indep.x', 'C1.x',
                        src_indices=[[(0, 0), (-1, 1)],
                                     [(2, 1), (1, 1)]], flat_src_indices=False)

        p.setup()
        p.run_model()

        assert_rel_error(self, p['C1.x'], np.array([[0., 10.],
                                                    [7., 4.]]))
        assert_rel_error(self, p['C1.y'], 42.)

    def test_promote_not_found1(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', om.ExecComp('y=x'), promotes_inputs=['x'])
        p.model.add_subsystem('C2', om.ExecComp('y=x'), promotes_outputs=['x*'])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "ExecComp (C2): 'promotes_outputs' failed to find any matches for "
                         "the following names or patterns: ['x*'].")

    def test_promote_not_found2(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', om.ExecComp('y=x'), promotes_inputs=['x'])
        p.model.add_subsystem('C2', om.ExecComp('y=x'), promotes_inputs=['xx'])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "ExecComp (C2): 'promotes_inputs' failed to find any matches for "
                         "the following names or patterns: ['xx'].")

    def test_promote_not_found3(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', om.ExecComp('y=x'), promotes=['x'])
        p.model.add_subsystem('C2', om.ExecComp('y=x'), promotes=['xx'])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "ExecComp (C2): 'promotes' failed to find any matches for "
                         "the following names or patterns: ['xx'].")

    def test_missing_promote_var(self):
        p = om.Problem()

        indep_var_comp = om.IndepVarComp('z', val=2.)
        p.model.add_subsystem('indep_vars', indep_var_comp, promotes=['*'])

        p.model.add_subsystem('d1', om.ExecComp("y1=z+bar"),
                              promotes_inputs=['z', 'foo'])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "ExecComp (d1): 'promotes_inputs' failed to find any matches for "
                         "the following names or patterns: ['foo'].")

    def test_missing_promote_var2(self):
        p = om.Problem()

        indep_var_comp = om.IndepVarComp('z', val=2.)
        p.model.add_subsystem('indep_vars', indep_var_comp, promotes=['*'])

        p.model.add_subsystem('d1', om.ExecComp("y1=z+bar"),
                              promotes_outputs=['y1', 'blammo', ('bar', 'blah')])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "ExecComp (d1): 'promotes_outputs' failed to find any matches for "
                         "the following names or patterns: ['bar', 'blammo'].")

    def test_promote_src_indices(self):
        import numpy as np

        import openmdao.api as om

        class MyComp1(om.ExplicitComponent):
            def setup(self):
                # this input will connect to entries 0, 1, and 2 of its source
                self.add_input('x', np.ones(3), src_indices=[0, 1, 2])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        class MyComp2(om.ExplicitComponent):
            def setup(self):
                # this input will connect to entries 3 and 4 of its source
                self.add_input('x', np.ones(2), src_indices=[3, 4])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*4.0

        p = om.Problem()

        # by promoting the following output and inputs to 'x', they will
        # be automatically connected
        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp1(), promotes_inputs=['x'])
        p.model.add_subsystem('C2', MyComp2(), promotes_inputs=['x'])

        p.setup()
        p.run_model()

        assert_rel_error(self, p['C1.x'], np.ones(3))
        assert_rel_error(self, p['C1.y'], 6.)
        assert_rel_error(self, p['C2.x'], np.ones(2))
        assert_rel_error(self, p['C2.y'], 8.)

    def test_promote_src_indices_nonflat(self):
        import numpy as np

        import openmdao.api as om

        class MyComp(om.ExplicitComponent):
            def setup(self):
                # We want to pull the following 4 values out of the source:
                # [(0,0), (3,1), (2,1), (1,1)].
                # Because our input is also non-flat we arrange the
                # source index tuples into an array having the same shape
                # as our input.  If we didn't set flat_src_indices to False,
                # we could specify src_indices as a 1D array of indices into
                # the flattened source.
                self.add_input('x', np.ones((2, 2)),
                               src_indices=[[(0, 0), (3, 1)],
                                            [(2, 1), (1, 1)]],
                               flat_src_indices=False)
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

        p = om.Problem()

        # by promoting the following output and inputs to 'x', they will
        # be automatically connected
        p.model.add_subsystem('indep',
                              om.IndepVarComp('x', np.arange(12).reshape((4, 3))),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(),
                              promotes_inputs=['x'])

        p.setup()
        p.run_model()

        assert_rel_error(self, p['C1.x'],
                         np.array([[0., 10.],
                                   [7., 4.]]))
        assert_rel_error(self, p['C1.y'], 21.)

    def test_promote_src_indices_nonflat_to_scalars(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', 1.0, src_indices=[(3, 1)], shape=(1,))
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x']*2.0

        p = om.Problem()

        p.model.add_subsystem('indep',
                              om.IndepVarComp('x', np.arange(12).reshape((4, 3))),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        p.set_solver_print(level=0)
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.x'], 10.)
        assert_rel_error(self, p['C1.y'], 20.)

    def test_promote_src_indices_nonflat_error(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', 1.0, src_indices=[(3, 1)])
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

        p = om.Problem()

        p.model.add_subsystem('indep',
                              om.IndepVarComp('x', np.arange(12).reshape((4, 3))),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "src_indices for 'x' is not flat, so its input shape "
                         "must be provided. src_indices may contain an extra "
                         "dimension if the connected source is not flat, making "
                         "the input shape ambiguous.")

    @parameterized.expand(itertools.product(
        [((4, 3),  [(0, 0), (3, 1), (2, 1), (1, 1)]),
         ((1, 12), [(0, 0), (0, 10), (0, 7), (0, 4)]),
         ((12,),   [0, 10, 7, 4]),
         ((12, 1), [(0, 0), (10, 0), (7, 0), (4, 0)])],
        [(2, 2), (4,), (4, 1), (1, 4)],
    ), name_func=lambda f, n, p: 'test_promote_src_indices_'+'_'.join(str(a) for a in p.args))
    def test_promote_src_indices_param(self, src_info, tgt_shape):
        src_shape, idxvals = src_info

        class MyComp(om.ExplicitComponent):
            def setup(self):
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

        p = om.Problem()

        p.model.add_subsystem('indep',
                              om.IndepVarComp('x', np.arange(12).reshape(src_shape)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        p.set_solver_print(level=0)
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.x'],
                         np.array([0., 10., 7., 4.]).reshape(tgt_shape))
        assert_rel_error(self, p['C1.y'], 21.)

    def test_set_order_feature(self):
        import openmdao.api as om

        class ReportOrderComp(om.ExplicitComponent):
            """Adds name to list."""

            def __init__(self, order_list):
                super(ReportOrderComp, self).__init__()
                self._order_list = order_list

            def compute(self, inputs, outputs):
                self._order_list.append(self.pathname)

        # this list will record the execution order of our C1, C2, and C3 components
        order_list = []

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', 1.))
        model.add_subsystem('C1', ReportOrderComp(order_list))
        model.add_subsystem('C2', ReportOrderComp(order_list))
        model.add_subsystem('C3', ReportOrderComp(order_list))

        prob.setup()
        prob.run_model()

        self.assertEqual(order_list, ['C1', 'C2', 'C3'])

        # reset the shared order list
        order_list[:] = []

        # now swap C2 and C1 in the order
        model.set_order(['indeps', 'C2', 'C1', 'C3'])

        # after changing the order, we must call setup again
        prob.setup()
        prob.run_model()

        self.assertEqual(order_list, ['C2', 'C1', 'C3'])

    def test_set_order(self):

        order_list = []
        prob = om.Problem()
        model = prob.model
        model.nonlinear_solver = om.NonlinearRunOnce()
        model.add_subsystem('indeps', om.IndepVarComp('x', 1.))
        model.add_subsystem('C1', ReportOrderComp(order_list))
        model.add_subsystem('C2', ReportOrderComp(order_list))
        model.add_subsystem('C3', ReportOrderComp(order_list))
        model.connect('indeps.x', 'C1.x')
        model.connect('C1.y', 'C2.x')
        model.connect('C2.y', 'C3.x')
        prob.set_solver_print(level=0)

        self.assertEqual(['indeps', 'C1', 'C2', 'C3'],
                         [s.name for s in model._static_subsystems_allprocs])

        prob.setup()
        prob.run_model()

        self.assertEqual(['C1', 'C2', 'C3'], order_list)

        order_list[:] = []

        # Big boy rules
        model.set_order(['indeps', 'C2', 'C1', 'C3'])

        prob.setup()
        prob.run_model()
        self.assertEqual(['C2', 'C1', 'C3'], order_list)

        # Extra
        with self.assertRaises(ValueError) as cm:
            model.set_order(['indeps', 'C2', 'junk', 'C1', 'C3'])

        self.assertEqual(str(cm.exception),
                         "Group (<top>): subsystem(s) ['junk'] found in subsystem order but don't exist.")

        # Missing
        with self.assertRaises(ValueError) as cm:
            model.set_order(['indeps', 'C2', 'C3'])

        self.assertEqual(str(cm.exception),
                         "Group (<top>): ['C1'] expected in subsystem order and not found.")

        # Extra and Missing
        with self.assertRaises(ValueError) as cm:
            model.set_order(['indeps', 'C2', 'junk', 'C1', 'junk2'])

        self.assertEqual(str(cm.exception),
                         "Group (<top>): ['C3'] expected in subsystem order and not found.\n"
                         "Group (<top>): subsystem(s) ['junk', 'junk2'] found in subsystem order "
                         "but don't exist.")

        # Dupes
        with self.assertRaises(ValueError) as cm:
            model.set_order(['indeps', 'C2', 'C1', 'C3', 'C1'])

        self.assertEqual(str(cm.exception),
                         "Group (<top>): Duplicate name(s) found in subsystem order list: ['C1']")

    def test_set_order_init_subsystems(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('indeps', om.IndepVarComp('x', 1.))
        model.add_subsystem('G1', SetOrderGroup())
        prob.setup()
        prob.run_model()

        # this test passes if it doesn't raise an exception

    def test_guess_nonlinear_feature(self):
        import openmdao.api as om

        class Discipline(om.Group):

            def setup(self):
                self.add_subsystem('comp0', om.ExecComp('y=x**2'))
                self.add_subsystem('comp1', om.ExecComp('z=2*external_input'),
                                   promotes_inputs=['external_input'])

                self.add_subsystem('balance', om.BalanceComp('x', lhs_name='y', rhs_name='z'),
                                   promotes_outputs=['x'])

                self.connect('comp0.y', 'balance.y')
                self.connect('comp1.z', 'balance.z')

                self.connect('x', 'comp0.x')

                self.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
                self.linear_solver = om.DirectSolver()

            def guess_nonlinear(self, inputs, outputs, residuals):
                # inputs are addressed using full path name, regardless of promotion
                external_input = inputs['comp1.external_input']

                # balance drives x**2 = 2*external_input
                x_guess = (2*external_input)**.5

                # outputs are addressed by the their promoted names
                outputs['x'] = x_guess # perfect guess should converge in 0 iterations

        p = om.Problem()

        p.model.add_subsystem('parameters', om.IndepVarComp('input_value', 1.))
        p.model.add_subsystem('discipline', Discipline())

        p.model.connect('parameters.input_value', 'discipline.external_input')

        p.setup()
        p.run_model()

        self.assertEqual(p.model.nonlinear_solver._iter_count, 0)

        assert_rel_error(self, p['discipline.x'], 1.41421356, 1e-6)

    def test_guess_nonlinear_complex_step(self):

        class Discipline(om.Group):

            def setup(self):
                self.add_subsystem('comp0', om.ExecComp('y=x**2'))
                self.add_subsystem('comp1', om.ExecComp('z=2*external_input'),
                                   promotes_inputs=['external_input'])

                self.add_subsystem('balance', om.BalanceComp('x', lhs_name='y', rhs_name='z'),
                                   promotes_outputs=['x'])

                self.connect('comp0.y', 'balance.y')
                self.connect('comp1.z', 'balance.z')

                self.connect('x', 'comp0.x')

                self.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
                self.linear_solver = om.DirectSolver()

            def guess_nonlinear(self, inputs, outputs, residuals):

                if outputs._data.dtype == np.complex:
                    raise RuntimeError('Vector should not be complex when guess_nonlinear is called.')

                # inputs are addressed using full path name, regardless of promotion
                external_input = inputs['comp1.external_input']

                # balance drives x**2 = 2*external_input
                x_guess = (2*external_input)**.5

                # outputs are addressed by the their promoted names
                outputs['x'] = x_guess # perfect guess should converge in 0 iterations

        p = om.Problem()

        p.model.add_subsystem('parameters', om.IndepVarComp('input_value', 1.))
        p.model.add_subsystem('discipline', Discipline())

        p.model.connect('parameters.input_value', 'discipline.external_input')

        p.setup(force_alloc_complex=True)
        p.run_model()

        self.assertEqual(p.model.nonlinear_solver._iter_count, 0)

        assert_rel_error(self, p['discipline.x'], 1.41421356, 1e-6)

        totals = p.check_totals(of=['discipline.comp1.z'], wrt=['parameters.input_value'], method='cs', out_stream=None)

        for key, val in iteritems(totals):
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-15)


class MyComp(om.ExplicitComponent):
    def __init__(self, input_shape, src_indices=None, flat_src_indices=False):
        super(MyComp, self).__init__()
        self._input_shape = input_shape
        self._src_indices = src_indices
        self._flat_src_indices = flat_src_indices

    def setup(self):
        self.add_input('x', val=np.zeros(self._input_shape),
                       src_indices=self._src_indices, flat_src_indices=self._flat_src_indices)
        self.add_output('y', val=np.zeros(self._input_shape))

    def compute(self, inputs, outputs):
        outputs['y'] = 2.0 * inputs['x']


def src_indices_model(src_shape, tgt_shape, src_indices=None, flat_src_indices=False,
                      promotes=None):
    prob = om.Problem()
    prob.model.add_subsystem('indeps', om.IndepVarComp('x', shape=src_shape),
                             promotes=promotes)
    prob.model.add_subsystem('C1', MyComp(tgt_shape,
                                          src_indices=src_indices if promotes else None,
                                          flat_src_indices=flat_src_indices),
                             promotes=promotes)
    if promotes is None:
        prob.model.connect('indeps.x', 'C1.x', src_indices=src_indices,
                           flat_src_indices=flat_src_indices)
    prob.setup()
    return prob


class TestConnect(unittest.TestCase):

    def setUp(self):
        prob = om.Problem(om.Group())

        sub = prob.model.add_subsystem('sub', om.Group())

        idv = sub.add_subsystem('src', om.IndepVarComp())
        idv.add_output('x', np.arange(15).reshape((5, 3)))  # array
        idv.add_output('s', 3.)                             # scalar

        sub.add_subsystem('tgt', om.ExecComp('y = x'))
        sub.add_subsystem('cmp', om.ExecComp('z = x'))
        sub.add_subsystem('arr', om.ExecComp('a = x', x=np.zeros(2)))

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
        # because setup is not called until then
        self.sub.connect('src.z', 'tgt.x', src_indices=[1])
        with assertRaisesRegex(self, NameError, msg):
            self.prob.setup()

    def test_invalid_target(self):
        msg = "Group (sub): Input 'tgt.z' does not exist for connection from 'src.x' to 'tgt.z'."

        # source and target names can't be checked until setup
        # because setup is not called until then
        self.sub.connect('src.x', 'tgt.z', src_indices=[1])
        with self.assertRaises(NameError) as ctx:
            self.prob.setup()

        self.assertEqual(str(ctx.exception), msg)

    def test_connect_within_system(self):
        msg = "Output and input are in the same System for connection " + \
              "from 'tgt.y' to 'tgt.x'."

        with assertRaisesRegex(self, RuntimeError, msg):
            self.sub.connect('tgt.y', 'tgt.x', src_indices=[1])

    def test_connect_within_system_with_promotes(self):
        prob = om.Problem()

        sub = prob.model.add_subsystem('sub', om.Group())
        sub.add_subsystem('tgt', om.ExecComp('y = x'), promotes_outputs=['y'])
        sub.connect('y', 'tgt.x', src_indices=[1])

        msg = "Group (sub): Output and input are in the same System for connection from 'y' to 'tgt.x'."

        with self.assertRaises(RuntimeError) as ctx:
            prob.setup()

        self.assertEqual(str(ctx.exception), msg)

    def test_connect_units_with_unitless(self):
        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0))
        prob.model.add_subsystem('src', om.ExecComp('x2 = 2 * x1', x2={'units': 'degC'}))
        prob.model.add_subsystem('tgt', om.ExecComp('y = 3 * x', x={'units': 'unitless'}))

        prob.model.connect('px1.x1', 'src.x1')
        prob.model.connect('src.x2', 'tgt.x')

        msg = "Group (<top>): Output 'src.x2' with units of 'degC' is connected " \
              "to input 'tgt.x' which has no units."

        with assert_warning(UserWarning, msg):
            prob.setup()

    def test_connect_incompatible_units(self):
        msg = "Output units of 'degC' for 'src.x2' are incompatible " \
              "with input units of 'm' for 'tgt.x'."

        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0))
        prob.model.add_subsystem('src', om.ExecComp('x2 = 2 * x1', x2={'units': 'degC'}))
        prob.model.add_subsystem('tgt', om.ExecComp('y = 3 * x', x={'units': 'm'}))

        prob.model.connect('px1.x1', 'src.x1')
        prob.model.connect('src.x2', 'tgt.x')

        with assertRaisesRegex(self, RuntimeError, msg):
            prob.setup()

    def test_connect_units_with_nounits(self):
        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0))
        prob.model.add_subsystem('src', om.ExecComp('x2 = 2 * x1'))
        prob.model.add_subsystem('tgt', om.ExecComp('y = 3 * x', x={'units': 'degC'}))

        prob.model.connect('px1.x1', 'src.x1')
        prob.model.connect('src.x2', 'tgt.x')

        prob.set_solver_print(level=0)

        msg = "Group (<top>): Input 'tgt.x' with units of 'degC' is " \
              "connected to output 'src.x2' which has no units."

        with assert_warning(UserWarning, msg):
            prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['tgt.y'], 600.)

    def test_connect_units_with_nounits_prom(self):
        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x', 100.0), promotes_outputs=['x'])
        prob.model.add_subsystem('src', om.ExecComp('y = 2 * x'), promotes=['x', 'y'])
        prob.model.add_subsystem('tgt', om.ExecComp('z = 3 * y', y={'units': 'degC'}), promotes=['y'])

        prob.set_solver_print(level=0)

        msg = "Group (<top>): Input 'tgt.y' with units of 'degC' is " \
              "connected to output 'src.y' which has no units."

        with assert_warning(UserWarning, msg):
            prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['tgt.z'], 600.)

    def test_mix_promotes_types(self):
        prob = om.Problem()
        prob.model.add_subsystem('src', om.ExecComp(['y = 2 * x', 'y2 = 3 * x']),
                                 promotes=['x', 'y'], promotes_outputs=['y2'])

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "ExecComp (src): 'promotes' cannot be used at the same time as "
                         "'promotes_inputs' or 'promotes_outputs'.")

    def test_mix_promotes_types2(self):
        prob = om.Problem()
        prob.model.add_subsystem('src', om.ExecComp(['y = 2 * x', 'y2 = 3 * x2']),
                                 promotes=['x', 'y'], promotes_inputs=['x2'])
        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "ExecComp (src): 'promotes' cannot be used at the same time as "
                         "'promotes_inputs' or 'promotes_outputs'.")

    def test_nested_nested_conn(self):
        prob = om.Problem()
        root = prob.model

        root.add_subsystem('p', om.IndepVarComp('x', 1.0))

        G1 = root.add_subsystem('G1', om.Group())
        par1 = G1.add_subsystem('par1', om.Group())

        par1.add_subsystem('c2', om.ExecComp('y = x * 2.0'))
        par1.add_subsystem('c4', om.ExecComp('y = x * 4.0'))

        prob.model.add_design_var('p.x')
        prob.model.add_constraint('G1.par1.c4.y', upper=0.0)

        root.connect('p.x', 'G1.par1.c2.x')
        root.connect('G1.par1.c2.y', 'G1.par1.c4.x')

        prob.setup()
        prob.run_driver()

        assert_rel_error(self, prob['G1.par1.c4.y'], 8.0)

    def test_bad_shapes(self):
        self.sub.connect('src.s', 'arr.x')

        msg = ("The source and target shapes do not match or are ambiguous for the connection "
               "'sub.src.s' to 'sub.arr.x'.")

        with assertRaisesRegex(self, ValueError, msg):
            self.prob.setup()

    def test_bad_indices_shape(self):
        p = om.Problem()
        p.model.add_subsystem('IV', om.IndepVarComp('x', np.arange(12).reshape((4, 3))))
        p.model.add_subsystem('C1', om.ExecComp('y=sum(x)*2.0', x=np.zeros((2, 2))))

        p.model.connect('IV.x', 'C1.x', src_indices=[(1, 1)])

        msg = (r"The source indices \[\[1 1\]\] do not specify a valid shape for "
               r"the connection 'IV.x' to 'C1.x'. The target "
               r"shape is \(2.*, 2.*\) but indices are \(1.*, 2.*\).")

        with assertRaisesRegex(self, ValueError, msg):
            p.setup()

    def test_bad_indices_dimensions(self):
        self.sub.connect('src.x', 'arr.x', src_indices=[(2, -1, 2), (2, 2, 2)],
                         flat_src_indices=False)

        msg = ("Group (sub): The source indices [[ 2 -1  2] [ 2  2  2]] do not specify a "
               "valid shape for the connection 'sub.src.x' to 'sub.arr.x'. "
               "The source has 2 dimensions but the indices expect 3.")

        try:
            self.prob.setup()
        except ValueError as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail('Exception expected.')

    def test_bad_indices_index(self):
        # the index value within src_indices is outside the valid range for the source
        self.sub.connect('src.x', 'arr.x', src_indices=[(2, -1), (4, 4)],
                         flat_src_indices=False)

        msg = ("Group (sub): The source indices do not specify a valid index for the "
               "connection 'sub.src.x' to 'sub.arr.x'. Index '4' "
               "is out of range for source dimension of size 3.")

        try:
            self.prob.setup()
        except ValueError as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail('Exception expected.')

    def test_src_indices_shape(self):
        src_indices_model(src_shape=(3, 3), tgt_shape=(2, 2),
                          src_indices=[[4, 5], [7, 8]],
                          flat_src_indices=True)

    def test_src_indices_shape_bad_idx_flat(self):
        try:
            src_indices_model(src_shape=(3, 3), tgt_shape=(2, 2),
                              src_indices=[[4, 5], [7, 9]],
                              flat_src_indices=True)
        except Exception as err:
            self.assertEqual(str(err), "Group (<top>): The source indices do not specify a valid index "
                                       "for the connection 'indeps.x' to 'C1.x'. "
                                       "Index '9' is out of range for a flat source of size 9.")
        else:
            self.fail("Exception expected.")

    def test_src_indices_shape_bad_idx_flat_promotes(self):
        try:
            src_indices_model(src_shape=(3, 3), tgt_shape=(2, 2),
                              src_indices=[[4, 5], [7, 9]],
                              flat_src_indices=True, promotes=['x'])
        except Exception as err:
            self.assertEqual(str(err), "Group (<top>): The source indices do not specify a valid index "
                                       "for the connection 'indeps.x' to 'C1.x'. "
                                       "Index '9' is out of range for a flat source of size 9.")
        else:
            self.fail("Exception expected.")

    def test_src_indices_shape_bad_idx_flat_neg(self):
        try:
            src_indices_model(src_shape=(3, 3), tgt_shape=(2, 2),
                              src_indices=[[-10, 5], [7, 8]],
                              flat_src_indices=True)
        except Exception as err:
            self.assertEqual(str(err), "Group (<top>): The source indices do not specify a valid index "
                                       "for the connection 'indeps.x' to 'C1.x'. "
                                       "Index '-10' is out of range for a flat source of size 9.")
        else:
            self.fail("Exception expected.")


if __name__ == "__main__":
    unittest.main()
