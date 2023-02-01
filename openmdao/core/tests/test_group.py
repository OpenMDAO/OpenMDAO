"""
Unit tests for Group.
"""
import itertools
import unittest
from collections import defaultdict

import numpy as np

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDis2
from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_no_warning
from openmdao.utils.logger_utils import TestLogger
from openmdao.utils.om_warnings import PromotionWarning, OMDeprecationWarning
from openmdao.utils.name_maps import name2abs_names
from openmdao.utils.testing_utils import set_env_vars_context

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


arr_order_1x1 = np.array([1, 2, 3, 4])
arr_2x4 = np.array([[0, 1, 2, 3], [10, 11, 12, 13]])
arr_order_3x3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
arr_order_4x4 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
arr_large_4x4 = np.array([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])

class SimpleGroup(om.Group):

    def setup(self):
        self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
        self.add_subsystem('comp2', om.ExecComp('b=2*a'))
        self.connect('comp1.x', 'comp2.a')


class BranchGroup(om.Group):

    def setup(self):
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
        super().__init__()
        self._order_list = order_list

    def setup(self):
        self.add_input('x', 0.0)
        self.add_output('y', 0.0)

    def compute(self, inputs, outputs):
        self._order_list.append(self.pathname)


class TestSubsystemConfigError(unittest.TestCase):

    def test_add_subsystem_error_on_config(self):
        class SimpleGroup(om.Group):

            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))
                self.connect('comp1.x', 'comp2.a')

            def configure(self):
                self.add_subsystem('comp3', om.IndepVarComp('y', 10.0))

        top = om.Problem(model=SimpleGroup())

        with self.assertRaises(RuntimeError) as cm:
            top.setup()

        self.assertEqual(str(cm.exception),
                         "<model> <class SimpleGroup>: Cannot call add_subsystem in the configure method.")

class SlicerComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', np.ones(4))
        self.add_output('y', 1.0)

    def compute(self, inputs, outputs):
        outputs['y'] = np.sum(inputs['x'])**2.0


class TestGroup(unittest.TestCase):

    def test_add_subsystem_class(self):
        p = om.Problem()
        try:
            p.model.add_subsystem('comp', om.IndepVarComp)
        except TypeError as err:
            self.assertEqual(str(err), "<class Group>: Subsystem 'comp' should be an instance, "
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
            self.assertEqual(str(err), "<class Group>: Subsystem name 'comp2' is already used.")
        else:
            self.fail('Exception expected.')

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

    def test_double_promote_conns(self):
        p = om.Problem(name='double_promote_conns')
        gouter = p.model.add_subsystem('gouter', om.Group())
        gouter.add_subsystem('couter', om.ExecComp('xx = a * 3.'), promotes_outputs=['xx'])
        g = gouter.add_subsystem('g', om.Group(), promotes_inputs=[('x', 'xx')])
        g.add_subsystem('ivc', om.IndepVarComp('x', 2.), promotes_outputs=['x'])
        g.add_subsystem('c0', om.ExecComp('y = 2*x'), promotes_inputs=['x'])

        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'double_promote_conns':"
           "\n   'gouter' <class Group>: The following inputs have multiple connections: "
           "gouter.g.c0.x from ['gouter.couter.xx', 'gouter.g.ivc.x'].")

    def test_double_promote_one_conn(self):
        p = om.Problem()
        gouter = p.model.add_subsystem('gouter', om.Group())
        gouter.add_subsystem('couter', om.ExecComp('xx = a * 3.'))
        g = gouter.add_subsystem('g', om.Group(), promotes_inputs=[('x', 'xx')])
        g.add_subsystem('ivc', om.IndepVarComp('x', 2.), promotes_outputs=['x'])
        g.add_subsystem('c0', om.ExecComp('y = 2*x'), promotes_inputs=['x'])

        p.setup()

        self.assertEqual(p.model._conn_global_abs_in2out['gouter.g.c0.x'], 'gouter.g.ivc.x')

    def test_hide_group_input(self):
        p = om.Problem()
        g1 = p.model.add_subsystem('g1', om.Group())
        g2 = g1.add_subsystem('g2', om.Group(), promotes=['g3.c1.x'])  # make g2 disappear using promotes
        g3 = g2.add_subsystem('g3', om.Group())
        c1 = g3.add_subsystem('c1', om.ExecComp('y=2.*x', x=2.))

        g3_ = g1.add_subsystem('g3', om.Group(), promotes=['x'])  # second g3, but directly under g1
        c1_ = g3_.add_subsystem('c1', om.ExecComp('y=3.*x', x=3.), promotes=['x'])

        with self.assertRaises(Exception) as cm:
            p.setup()
        self.assertEqual(cm.exception.args[0], f"{p.model.msginfo}: Absolute variable name 'g1.g3.c1.x'"
                                               " is masked by a matching promoted name. Try"
                                               " promoting to a different name. This can be caused"
                                               " by promoting '*' at group level or promoting using"
                                               " dotted names.")

    def test_hide_group_output(self):
        p = om.Problem()
        g1 = p.model.add_subsystem('g1', om.Group())
        g2 = g1.add_subsystem('g2', om.Group(), promotes=['g3.c1.y'])  # make g2 disappear using promotes
        g3 = g2.add_subsystem('g3', om.Group())
        c1 = g3.add_subsystem('c1', om.ExecComp('y=2.*x', x=2.))

        g3_ = g1.add_subsystem('g3', om.Group(), promotes=['y'])  # second g3, but directly under g1
        c1_ = g3_.add_subsystem('c1', om.ExecComp('y=3.*x', x=3.), promotes=['y'])

        with self.assertRaises(Exception) as cm:
            p.setup()
        self.assertEqual(cm.exception.args[0], f"{p.model.msginfo}: Absolute variable name 'g1.g3.c1.y'"
                                               " is masked by a matching promoted name. Try"
                                               " promoting to a different name. This can be caused"
                                               " by promoting '*' at group level or promoting using"
                                               " dotted names.")

    def test_invalid_subsys_name(self):
        p = om.Problem()

        # name cannot start with an underscore
        with self.assertRaises(Exception) as err:
            p.model.add_subsystem('_bad_name', om.Group())
        self.assertEqual(str(err.exception),
                         "<class Group>: '_bad_name' is not a valid sub-system name.")

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

        # 'name', 'pathname', 'comm' and 'options' are reserved names
        p = om.Problem()
        for reserved in ['name', 'pathname', 'comm', 'options']:
            with self.assertRaises(Exception) as err:
                p.model.add_subsystem(reserved, om.Group())
            self.assertEqual(str(err.exception),
                             "<class Group>: Can't add subsystem '%s' because an attribute with that name already exits." %
                             reserved)

    def test_group_promotes(self):
        """Promoting a single variable."""
        p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output('a', 2.0)
        ivc.add_output('x', 5.0)

        p.model.add_subsystem('comp1', ivc, promotes_outputs=['x'])
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
                         "<class Group>: promotes must be an iterator of strings and/or tuples.")

    def test_group_renames_errors_not_found(self):
        p = om.Problem()
        p.model.add_subsystem('comp1', om.IndepVarComp('x', 5.0),
                              promotes_outputs=[('xx', 'foo')])
        p.model.add_subsystem('comp2', om.ExecComp('y=2*foo'), promotes_inputs=['foo'])

        with self.assertRaises(Exception) as err:
            p.setup()
        self.assertEqual(str(err.exception),
                         "'comp1' <class IndepVarComp>: 'promotes_outputs' failed to find any matches for "
                         "the following names or patterns: [('xx', 'foo')].")

    def test_group_renames_errors_bad_tuple(self):
        p = om.Problem()
        p.model.add_subsystem('comp1', om.IndepVarComp('x', 5.0),
                              promotes_outputs=[('x', 'foo', 'bar')])
        p.model.add_subsystem('comp2', om.ExecComp('y=2*foo'), promotes_inputs=['foo'])

        with self.assertRaises(Exception) as err:
            p.setup()
        self.assertEqual(str(err.exception),
                         "when adding subsystem 'comp1', entry '('x', 'foo', 'bar')' "
                         "is not a string or tuple of size 2.")

    def test_group_promotes_multiple(self):
        """Promoting multiple variables."""
        p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output('a', 2.0)
        ivc.add_output('x', 5.0)

        p.model.add_subsystem('comp1', ivc, promotes_outputs=['a', 'x'])
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

        ivc = om.IndepVarComp()
        ivc.add_output('a', 2.0)
        ivc.add_output('x', 5.0)

        p.model.add_subsystem('comp1', ivc, promotes_outputs=['*'])
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
                         "'d1' <class SellarDis2>: 'promotes_outputs' failed to find any matches for "
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
        self.assertEqual(s, p.model)

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
        with self.assertRaisesRegex(Exception, msg):
            prob.setup()

    def test_unconnected_input_units_no_mismatch(self):
        p = om.Problem()

        p.model.add_subsystem('comp1', om.ExecComp('y=sum(x)',
                                                   x={'val': np.zeros(5), 'units': 'ft'},
                                                   y={'units': 'inch'}), promotes=['x'])
        p.model.add_subsystem('comp2', om.ExecComp('y=sum(x)',
                                                   x={'val': np.zeros(5), 'units': 'ft'},
                                                   y={'units': 'inch'}), promotes=['x'])

        p.setup()
        p['comp2.x'] = np.ones(5)
        p.run_model()
        np.testing.assert_allclose(p['comp1.y'], 5.)
        np.testing.assert_allclose(p['comp2.y'], 5.)

    def test_unconnected_input_units_mismatch(self):
        p = om.Problem()

        p.model.add_subsystem('comp1', om.ExecComp('y=sum(x)',
                                                   x={'val': np.zeros(5), 'units': 'inch'},
                                                   y={'units': 'inch'}), promotes=['x'])
        p.model.add_subsystem('comp2', om.ExecComp('y=sum(x)',
                                                   x={'val': np.zeros(5), 'units': 'ft'},
                                                   y={'units': 'inch'}), promotes=['x'])

        p.model.set_input_defaults('x', units='ft')

        p.setup()
        p['comp2.x'] = np.ones(5)

        p.run_model()
        np.testing.assert_allclose(p['comp1.y'], 60.)
        np.testing.assert_allclose(p['comp2.y'], 5.)

    def test_double_src_indices(self):
        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(3), src_indices=[0, 1, 2])

        p = om.Problem(name='double_src_indices')

        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x', src_indices=[1, 0, 2])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
            "\nCollected errors for problem 'double_src_indices':"
            "\n   <model> <class Group>: src_indices has been defined in both connect('indep.x', 'C1.x') "
            "and add_input('C1.x', ...).")


    def test_incompatible_src_indices_error(self):
        class ControlInterpComp(om.ExplicitComponent):

            def setup(self):
                self.add_output('x', shape=(3, 1))

        class CollocationComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', shape=(1, 2))

        class Phase(om.Group):

            def setup(self):
                self.add_subsystem('comp1', ControlInterpComp())
                self.add_subsystem('comp2', CollocationComp())

                self.connect('comp1.x', 'comp2.x', src_indices=[1], flat_src_indices=True)

        p = om.Problem(name='src_indices_error')

        p.model.add_subsystem('phase', Phase())

        with self.assertRaises(Exception) as context:
            p.setup()

        self.assertEqual(str(context.exception),
           "\nCollected errors for problem 'src_indices_error':"
           "\n   'phase' <class Phase>: src_indices shape (1,) does not match phase.comp2.x "
           "shape (1, 2).")

    def test_connect_to_flat_array_with_slice(self):
        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones((12,)))
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x']) ** 2.0

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_large_4x4))
        p.model.add_subsystem('row123_comp', SlicerComp())

        idxs = np.array([0, 2, 3], dtype=int)

        p.model.connect('indep.x', 'row123_comp.x', src_indices=om.slicer[idxs, ...])

        p.setup()
        p.run_model()

        assert_near_equal(p['row123_comp.x'], arr_large_4x4[(0, 2, 3), ...].ravel())
        assert_near_equal(p['row123_comp.y'], np.sum(arr_large_4x4[(0, 2, 3), ...]) ** 2.0)

    def test_connect_to_flat_src_indices_with_slice(self):
        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones((12,)))
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x']) ** 2.0

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_large_4x4))
        p.model.add_subsystem('row123_comp', SlicerComp())

        idxs = np.array([0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15], dtype=int)

        # the ellipsis in this case is basically ignored
        p.model.connect('indep.x', 'row123_comp.x', src_indices=om.slicer[idxs, ...],
                        flat_src_indices=True)

        p.setup()
        p.run_model()

        assert_near_equal(p['row123_comp.x'], arr_large_4x4.ravel()[idxs, ...])
        assert_near_equal(p['row123_comp.y'], np.sum(arr_large_4x4.ravel()[idxs, ...]) ** 2.0)

    def test_connect_to_flat_array(self):
        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones((4,)))
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', val=arr_large_4x4))
        p.model.add_subsystem('trace_comp', SlicerComp())

        idxs = np.array([0, 5, 10, 15], dtype=int)

        p.model.connect('indep.x', 'trace_comp.x', src_indices=idxs, flat_src_indices=True)

        p.setup()
        p.run_model()

        assert_near_equal(p['trace_comp.x'], np.diag(arr_large_4x4))
        assert_near_equal(p['trace_comp.y'], np.sum(np.diag(arr_large_4x4)))

    def test_om_slice_in_connect(self):

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_order_4x4))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x', src_indices=om.slicer[:, 1])

        p.setup()
        p.run_model()

        assert_near_equal(p['C1.x'], np.array([2, 2, 2, 2]))

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_order_4x4))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x', src_indices=om.slicer[:, 1])

        p.setup()
        p.run_model()

        assert_near_equal(p['C1.x'], np.array([2, 2, 2, 2]))

    def test_om_slice_in_promotes(self):

        p = om.Problem()

        model = p.model
        model.add_subsystem('indep', om.IndepVarComp('a', arr_order_3x3), promotes=['*'])
        model.add_subsystem('comp1', om.ExecComp('b=2*a', a=np.ones(3), b=np.ones(3)))
        model.promotes('comp1', inputs=['a'], src_indices=om.slicer[:, 1])

        p.setup()
        p.run_model()

        assert_near_equal(p['comp1.a'], [2, 2, 2])

    def test_om_slice_in_promotes_flat(self):
        p = om.Problem(name='slice_in_promotes_flat')

        model = p.model
        model.add_subsystem('indep', om.IndepVarComp('a', arr_order_3x3), promotes=['*'])
        model.add_subsystem('comp1', om.ExecComp('b=2*a', a=np.ones(3), b=np.ones(3)))

        msg = "<class Group>: When promoting ['a'] from 'comp1': Can't index into a flat array with an indexer expecting 2 dimensions."

        with set_env_vars_context(OPENMDAO_FAIL_FAST='1'):
            with self.assertRaises(Exception) as context:
                model.promotes('comp1', inputs=['a'], src_indices=om.slicer[:, 1], flat_src_indices=True)

        self.assertEqual(context.exception.args[0], msg)

    def test_desvar_indice_slice(self):

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_order_1x1))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x')
        p.model.add_design_var('indep.x', indices=om.slicer[2:])

        p.model.add_objective('C1.y')

        p.setup()
        p.run_model()

        assert_near_equal(arr_order_1x1[p.model._design_vars['indep.x']['indices']()], np.array([3., 4]))

    def test_om_slice_in_add_response(self):

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_order_1x1))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x')
        p.model.add_response('indep.x', type_='con', indices=om.slicer[2:])

        p.model.add_objective('C1.y')

        p.setup()
        p.run_model()

        assert_near_equal(arr_order_1x1[p.model._responses['indep.x']['indices']()], np.array([3, 4]))
        self.assertTrue(p.model._responses['indep.x']['indices'](), slice(2, None, None))

    def test_om_slice_in_add_constraint(self):

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_order_1x1))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x')
        p.model.add_constraint('indep.x', indices=om.slicer[2:])

        p.model.add_objective('C1.y')

        p.setup()
        p.run_model()

        assert_near_equal(arr_order_1x1[p.model._responses['indep.x']['indices'].flat()], np.array([3, 4]))
        self.assertTrue(p.model._responses['indep.x']['indices'](), slice(2, None, None))

    def test_om_slice_in_add_input(self):
        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(4), src_indices=om.slicer[:, 2])

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_order_4x4))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x')

        p.setup()
        p.run_model()

        assert_near_equal(p['C1.x'], np.array([3, 3, 3, 3]))

    def test_om_slice_negative_stop(self):
        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(4), src_indices=om.slicer[:, -1])

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_order_4x4))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x')

        p.setup()
        p.run_model()

        assert_near_equal(p['C1.x'], np.array([4, 4, 4, 4]))

    def test_om_slice_3d(self):
        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(4), src_indices=om.slicer[:, 1, 2])

        arr = np.arange(64, dtype=int).reshape(4, 4, 4)

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x')

        p.setup()
        p.run_model()

        assert_near_equal(p['C1.x'], np.array([6, 22, 38, 54]))

    def test_om_slice_with_ellipsis_in_connect(self):

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_large_4x4))
        p.model.add_subsystem('row1_comp', SlicerComp())
        p.model.add_subsystem('row4_comp', SlicerComp())

        p.model.connect('indep.x', 'row1_comp.x', src_indices=om.slicer[0, ...])
        p.model.connect('indep.x', 'row4_comp.x', src_indices=om.slicer[3, ...])

        p.setup()
        p.run_model()

        assert_near_equal(p['row1_comp.x'], arr_large_4x4[0, ...])
        assert_near_equal(p['row4_comp.x'], arr_large_4x4[3, ...])

    def test_om_slice_4D_with_ellipsis(self):

        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(shape=(5, 3)))

        # Connect
        p = om.Problem()

        arr = np.random.randint(5, size=(3, 5, 3, 2))

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr))
        p.model.add_subsystem('row1_comp', SlicerComp())
        p.model.add_subsystem('row4_comp', SlicerComp())

        p.model.connect('indep.x', 'row1_comp.x', src_indices=om.slicer[1, ..., 1])
        p.model.connect('indep.x', 'row4_comp.x', src_indices=om.slicer[2, ..., 1])

        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('row1_comp.x'), arr[1, ..., 1])
        assert_near_equal(p.get_val('row4_comp.x'), arr[2, ..., 1])

        # Promotes
        p = om.Problem()

        arr = np.random.randint(5, size=(3, 5, 3, 2))

        p.model.add_subsystem('indep', om.IndepVarComp('a', arr), promotes=['*'])
        p.model.add_subsystem('row1_comp', om.ExecComp('b=2*a', a=np.ones((5,3)), b=np.ones((5,3))))

        p.model.promotes('row1_comp', inputs=['a'], src_indices=om.slicer[1, ..., 1])

        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('row1_comp.a'), arr[1, ..., 1])

        # Design Variable
        p = om.Problem()

        arr = np.random.randint(5, size=(3, 5, 3, 2))

        p.model.add_subsystem('indep', om.IndepVarComp('a', arr))
        p.model.add_subsystem('row1_comp', om.ExecComp('b=2*a', a=np.ones((3,5,3)), b=np.ones((3,5,3))))
        p.model.connect('indep.a', 'row1_comp.a', src_indices=om.slicer[..., 1])
        p.model.add_design_var('indep.a', indices=om.slicer[1, ..., 1])

        p.setup()
        p.run_model()

        assert_near_equal(arr[p.model._design_vars['indep.a']['indices']()], arr[1, ..., 1])

        # Response
        p = om.Problem()

        arr = np.random.randint(5, size=(3, 5, 3, 2))

        p.model.add_subsystem('indep', om.IndepVarComp('a', arr))
        p.model.add_subsystem('row1_comp', om.ExecComp('b=2*a', a=np.ones((3,5,3)), b=np.ones((3,5,3))))
        p.model.connect('indep.a', 'row1_comp.a', src_indices=om.slicer[..., 1])
        p.model.add_response('indep.a', type_='con', indices=om.slicer[1, ..., 1])

        p.setup()
        p.run_model()

        assert_near_equal(arr[p.model._responses['indep.a']['indices']()], arr[1, ..., 1])

        # Constraint
        p = om.Problem()

        arr = np.random.randint(5, size=(3, 5, 3, 2))

        p.model.add_subsystem('indep', om.IndepVarComp('a', arr))
        p.model.add_subsystem('row1_comp', om.ExecComp('b=2*a', a=np.ones((3,5,3)), b=np.ones((3,5,3))))
        p.model.connect('indep.a', 'row1_comp.a', src_indices=om.slicer[..., 1])
        p.model.add_constraint('indep.a', indices=om.slicer[1, ..., 1])

        p.setup()
        p.run_model()

        assert_near_equal(arr[p.model._responses['indep.a']['indices']()], arr[1, ..., 1])

    def test_om_slice_with_ellipsis_in_promotes(self):

        p = om.Problem()

        model = p.model
        model.add_subsystem('indep', om.IndepVarComp('a', arr_large_4x4), promotes=['*'])
        model.add_subsystem('comp1', om.ExecComp('b=2*a', a=np.ones(4), b=np.ones(4)))
        model.add_subsystem('comp2', om.ExecComp('b=2*a', a=np.ones(4), b=np.ones(4)))
        model.add_subsystem('comp3', om.ExecComp('b=2*a', a=np.ones(4), b=np.ones(4)))

        model.promotes('comp1', inputs=['a'], src_indices=om.slicer[0, ...])
        model.promotes('comp2', inputs=['a'], src_indices=om.slicer[3, ...])
        model.promotes('comp3', inputs=['a'], src_indices=om.slicer[..., 3])

        p.setup()
        p.run_model()

        assert_near_equal(p['comp1.a'], np.array([0,  1,  2,  3]))
        assert_near_equal(p['comp2.a'], np.array([30, 31, 32, 33]))
        assert_near_equal(p['comp3.a'], np.array([ 3, 13, 23, 33]))

    def test_om_slice_with_ellipsis_in_desvar(self):

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_2x4))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x', src_indices=om.slicer[1, ...])
        p.model.add_design_var('indep.x', indices=om.slicer[1, ...])

        p.model.add_objective('C1.y')

        p.setup()
        p.run_model()

        assert_near_equal(arr_2x4[p.model._design_vars['indep.x']['indices']()][0], 10)

    def test_om_slice_with_ellipsis_in_add_response(self):

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_2x4))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x', src_indices=om.slicer[1, ...])
        p.model.add_response('indep.x', type_='con', indices=om.slicer[1, ...])

        p.model.add_objective('C1.y')

        p.setup()
        p.run_model()

        assert_near_equal(arr_2x4[p.model._responses['indep.x']['indices']()],
                          np.array([10, 11, 12, 13]))
        self.assertTrue(p.model._responses['indep.x']['indices']()[0], 1)

    def test_om_slice_with_ellipsis_in_add_constraint(self):

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_2x4))
        p.model.add_subsystem('C1', SlicerComp())
        p.model.connect('indep.x', 'C1.x', src_indices=om.slicer[1, ...])
        p.model.add_constraint('indep.x', indices=om.slicer[1, ...])

        p.model.add_objective('C1.y')

        p.setup()
        p.run_model()

        assert_near_equal(arr_2x4[p.model._responses['indep.x']['indices']()],
                          np.array([10, 11, 12, 13]))
        self.assertTrue(p.model._responses['indep.x']['indices']()[0], 1)

    def test_om_slice_with_ellipsis_auto_ivc(self):

        # Add_constraint
        p = om.Problem()

        p.model.add_subsystem('C1', SlicerComp(), promotes_inputs=['x'])
        p.model.add_constraint('x', indices=om.slicer[1, ...])

        p.model.add_objective('C1.y')

        p.setup()
        p.set_val('x', arr_2x4, indices=om.slicer[1, ...])
        p.run_model()

        assert_near_equal(arr_2x4[p.model._responses['x']['indices']()],
                          np.array([10, 11, 12, 13]))
        self.assertTrue(p.model._responses['x']['indices']()[0], 1)

        # Add_response
        p = om.Problem()

        p.model.add_subsystem('C1', SlicerComp(), promotes_inputs=['x'])
        p.model.add_response('x', type_='con', indices=om.slicer[1, ...])

        p.model.add_objective('C1.y')

        p.setup()
        p.set_val('x', arr_2x4, indices=om.slicer[1, ...])
        p.run_model()

        assert_near_equal(arr_2x4[p.model._responses['x']['indices']()],
                          np.array([10, 11, 12, 13]))
        self.assertTrue(p.model._responses['x']['indices']()[0], 1)

        # Add_design_var
        p = om.Problem()

        p.model.add_subsystem('C1', SlicerComp(), promotes_inputs=['x'])
        p.model.add_design_var('x', indices=om.slicer[1, ...])

        p.setup()
        p.set_val('x', arr_2x4, indices=om.slicer[1, ...])
        p.run_model()

        assert_near_equal(arr_2x4[p.model._design_vars['x']['indices']()],
                          np.array([10, 11, 12, 13]))
        self.assertTrue(p.model._design_vars['x']['indices']()[0], 1)
        self.assertTrue(p.driver.get_design_var_values()['x'], np.array(11.))

    def test_om_slice_with_indices_and_ellipsis_in_connect(self):
        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones((3, 4)))
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x']) ** 2.0

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_large_4x4))
        p.model.add_subsystem('row134_comp', SlicerComp())

        idxs = np.array([0, 2, 3], dtype=int)

        p.model.connect('indep.x', 'row134_comp.x', src_indices=om.slicer[idxs, ...])

        p.setup()
        p.run_model()

        assert_near_equal(p['row134_comp.x'], arr_large_4x4[(0, 2, 3), ...])
        assert_near_equal(p['row134_comp.y'], np.sum(arr_large_4x4[(0, 2, 3), ...])**2)

    def test_om_slice_get_val(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('c1', om.ExecComp('y=x', x=np.ones(2), y=np.ones(2)))
        model.add_subsystem('c2', om.ExecComp('y=x', x=np.ones(2), y=np.ones(2)))

        model.connect('c1.y', 'c2.x', om.slicer[:])

        prob.setup()
        prob.set_val('c1.x', 3.5*np.ones(2))
        prob.run_model()

        val = prob.get_val('c2.x')
        assert_near_equal(val, np.array([3.5, 3.5]))

    def test_promote_not_found1(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', om.ExecComp('y=x'), promotes_inputs=['x'])
        p.model.add_subsystem('C2', om.ExecComp('y=x'), promotes_outputs=['x*'])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "'C2' <class ExecComp>: 'promotes_outputs' failed to find any matches for the "
                         "following names or patterns: ['x*'].")

    def test_promote_not_found2(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', np.ones(5)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', om.ExecComp('y=x'), promotes_inputs=['x'])
        p.model.add_subsystem('C2', om.ExecComp('y=x'), promotes_inputs=['xx'])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "'C2' <class ExecComp>: 'promotes_inputs' failed to find any matches for "
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
                         "'C2' <class ExecComp>: 'promotes' failed to find any matches for "
                         "the following names or patterns: ['xx'].")

    def test_empty_group(self):
        p = om.Problem()
        g1 = p.model.add_subsystem('G1', om.Group(), promotes=['*'])

        p.setup()

    def test_missing_promote_var(self):
        p = om.Problem()

        indep_var_comp = om.IndepVarComp('z', val=2.)
        p.model.add_subsystem('indep_vars', indep_var_comp, promotes=['*'])

        p.model.add_subsystem('d1', om.ExecComp("y1=z+bar"),
                              promotes_inputs=['z', 'foo'])

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "'d1' <class ExecComp>: 'promotes_inputs' failed to find any matches for "
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
                         "'d1' <class ExecComp>: 'promotes_outputs' failed to find any matches for "
                         "the following names or patterns: [('bar', 'blah'), 'blammo'].")

    def test_promote_src_indices_nonflat_to_scalars(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', 1.0, src_indices=[[3],[1]], shape=(1,))
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
        assert_near_equal(p['C1.x'], 10.)
        assert_near_equal(p['C1.y'], 20.)

    @parameterized.expand(itertools.product(
        [((4, 3),  [[0,3,2,1],[0,1,1,1]]),
         ((1, 12), [[0,0,0,0],[0,10,7,4]]),
         ((12,),   [0, 10, 7, 4]),
         ((12, 1), [[0,10,7,4],[0,0,0,0]])],
        [(2, 2), (4,), (4, 1), (1, 4)],
    ), name_func=lambda f, n, p: 'test_promote_src_indices_'+'_'.join(str(a) for a in p.args))
    def test_promote_src_indices_param(self, src_info, tgt_shape):
        src_shape, idxvals = src_info
        flat = np.atleast_1d(idxvals).ndim == 1

        class MyComp(om.ExplicitComponent):
            def setup(self):
                sidxs = idxvals
                if len(tgt_shape) == 1:
                    tshape = None  # don't need to set shape if input is flat
                else:
                    tshape = tgt_shape

                self.add_input('x', np.ones(4).reshape(tgt_shape),
                               src_indices=sidxs, flat_src_indices=flat, shape=tshape)
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
        assert_near_equal(p['C1.x'],
                         np.array([0., 10., 7., 4.]).reshape(tgt_shape))
        assert_near_equal(p['C1.y'], 21.)

    def test_set_order(self):

        order_list = []
        prob = om.Problem()
        model = prob.model
        model.nonlinear_solver = om.NonlinearRunOnce()
        model.add_subsystem('C1', ReportOrderComp(order_list), promotes_inputs=['x'])
        model.add_subsystem('C2', ReportOrderComp(order_list))
        model.add_subsystem('C3', ReportOrderComp(order_list))
        model.connect('C1.y', 'C2.x')
        model.connect('C2.y', 'C3.x')
        prob.set_solver_print(level=0)

        self.assertEqual(['C1', 'C2', 'C3'],
                         [s.name for s, _ in model._static_subsystems_allprocs.values()])

        prob.setup()
        prob.set_val('x', 1.)
        prob.run_model()

        self.assertEqual(['C1', 'C2', 'C3'], order_list)

        order_list[:] = []

        # Big boy rules
        model.set_order(['C2', 'C1', 'C3'])

        prob.setup()
        prob.set_val('x', 1.)
        prob.run_model()
        self.assertEqual(['C2', 'C1', 'C3'], order_list)

        # Extra
        with self.assertRaises(ValueError) as cm:
            model.set_order(['C2', 'junk', 'C1', 'C3'])

        self.assertEqual(str(cm.exception),
                         "<model> <class Group>: subsystem(s) ['junk'] found in subsystem order but don't exist.")

        # Missing
        with self.assertRaises(ValueError) as cm:
            model.set_order(['C2', 'C3'])

        self.assertEqual(str(cm.exception),
                         "<model> <class Group>: ['C1'] expected in subsystem order and not found.")

        # Extra and Missing
        with self.assertRaises(ValueError) as cm:
            model.set_order(['C2', 'junk', 'C1', 'junk2'])

        self.assertEqual(str(cm.exception),
                         "<model> <class Group>: ['C3'] expected in subsystem order and not found.\n"
                         "<model> <class Group>: subsystem(s) ['junk', 'junk2'] found in subsystem order "
                         "but don't exist.")

        # Dupes
        with self.assertRaises(ValueError) as cm:
            model.set_order(['C2', 'C1', 'C3', 'C1'])

        self.assertEqual(str(cm.exception),
                         "<model> <class Group>: Duplicate name(s) found in subsystem order list: ['C1']")

    def test_set_order_init_subsystems(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('indeps', om.IndepVarComp('x', 1.))
        model.add_subsystem('G1', SetOrderGroup())
        prob.setup()
        prob.run_model()

        # this test passes if it doesn't raise an exception

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

                if outputs.asarray().dtype == complex:
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

        assert_near_equal(p['discipline.x'], 1.41421356, 1e-6)

        totals = p.check_totals(of=['discipline.comp1.z'], wrt=['parameters.input_value'], method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-15)

    def test_set_order_in_config_error(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))

            def configure(self):
                self.set_order(['C2', 'C1'])

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('C1', SimpleGroup())
        model.add_subsystem('C2', SimpleGroup())

        msg = "'C1' <class SimpleGroup>: Cannot call set_order in the configure method."
        with self.assertRaises(RuntimeError) as cm:
            prob.setup()

        self.assertEqual(str(cm.exception), msg)

    def test_set_order_after_setup(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('C1', SimpleGroup())
        model.add_subsystem('C2', SimpleGroup())

        prob.setup()
        prob.model.set_order(['C2', 'C1'])

        msg = "Problem .*: Cannot call set_order without calling setup after"
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.run_model()

    def test_set_order_normal(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('C1', SimpleGroup())
        model.add_subsystem('C2', SimpleGroup())

        prob.model.set_order(['C2', 'C1'])
        prob.setup()
        prob.run_model()

    def test_double_setup_for_set_order(self):
        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('C1', SimpleGroup())
        model.add_subsystem('C2', SimpleGroup())

        prob.setup()
        model.set_order(['C2', 'C1'])
        prob.setup()
        prob.run_model()

    def test_promote_units_and_none(self):
        p = om.Problem(name='promote_units_and_none')

        p.model.add_subsystem('c1', om.ExecComp('y1=x', x={'units': None}),
                              promotes=['*'])
        p.model.add_subsystem('c2', om.ExecComp('y2=x', x={'units': 's'}),
                              promotes=['*'])

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
                         "\nCollected errors for problem 'promote_units_and_none':\n   <model> <class Group>: The following inputs, ['c1.x', 'c2.x'], promoted to 'x', are connected but their metadata entries ['units', 'val'] differ. Call <group>.set_input_defaults('x', units=?, val=?), where <group> is the model to remove the ambiguity.")

    def test_double_set_input_defaults(self):
        problem = om.Problem()
        problem.model.add_subsystem("foo", om.ExecComp("a=b+c"), promotes=["*"])

        problem.model.set_input_defaults("b", 5)
        problem.model.set_input_defaults("c", 10)

        msg = ("<class Group>: Setting input defaults for input 'b' which override "
               "previously set defaults for ['auto', 'prom', 'src_shape', 'val'].")
        with assert_warning(PromotionWarning, msg):
            problem.model.set_input_defaults("b", 4)

    def test_set_input_defaults_promotes_error(self):

        class Foo(om.ExplicitComponent):

            def setup(self):
                nn = 5

                self.add_input('test_param', val=np.zeros(nn))
                self.add_output('bar', val=np.ones(nn))

            def compute(self, inputs, outputs):
                outputs['bar'] = inputs['test_param'] ** 2


        p = om.Problem(name='input_defaults_promotes_error')

        g = p.model.add_subsystem('G', om.Group())

        g.add_subsystem('foo', Foo())

        g.promotes('foo', ['test_param'])

        p.model.set_input_defaults('G.test_param', val=7.0)

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'input_defaults_promotes_error':"
           "\n   <model> <class Group>: The source and target shapes do not match or are ambiguous "
           "for the connection '_auto_ivc.v0' to 'G.foo.test_param'. The source shape is (1,) but "
           "the target shape is (5,).")

    def test_set_input_defaults_keyerror(self):
        class Sub(om.Group):
            def setup(self):
                comp = om.ExecComp("z = x + y")
                self.add_subsystem('comp', comp, promotes_inputs=['*'])

        prob = om.Problem(name='set_input_def_key_error')
        model = prob.model

        model.add_subsystem('sub', Sub(), promotes=['*'])

        model.set_input_defaults('bad_name', 3.0)
        with self.assertRaises(Exception) as cm:
            prob.setup()

        msg = ("\nCollected errors for problem 'set_input_def_key_error':\n"
               "   <model> <class Group>: The following group inputs, passed to set_input_defaults(), could not be found: ['bad_name'].")
        self.assertEqual(cm.exception.args[0], msg)

@unittest.skipUnless(MPI, "MPI is required.")
class TestGroupMPISlice(unittest.TestCase):
    N_PROCS = 2

    def test_om_slice_2d_mpi(self):
        class MyComp1(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(4), src_indices=om.slicer[:, 2], distributed=True)
                self.add_output('y', 1.0, distributed=True)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        arr = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr))
        p.model.add_subsystem('C1', MyComp1())
        p.model.connect('indep.x', 'C1.x')

        p.setup()
        p.run_model()

        val = p.get_val('C1.x', get_remote=False)
        assert_near_equal(val, np.array([3, 3, 3, 3]))

    def test_om_slice_3d_mpi(self):
        class MyComp1(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(4), src_indices=om.slicer[:, 1, 2], distributed=True)
                self.add_output('y', 1.0, distributed=True)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        arr = np.arange(64, dtype=int).reshape(4, 4, 4)

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr))
        p.model.add_subsystem('C1', MyComp1())
        p.model.connect('indep.x', 'C1.x')

        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('C1.x', get_remote=False), np.array([6, 22, 38, 54]))

    def test_om_slice_4D_with_ellipsis_mpi(self):

        class SlicerComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(shape=(5, 3)))

        # Connect
        p = om.Problem()

        arr = np.random.randint(5, size=(3, 5, 3, 2))

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr))
        p.model.add_subsystem('row1_comp', SlicerComp())
        p.model.add_subsystem('row4_comp', SlicerComp())

        p.model.connect('indep.x', 'row1_comp.x', src_indices=om.slicer[1, ..., 1])
        p.model.connect('indep.x', 'row4_comp.x', src_indices=om.slicer[2, ..., 1])

        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('row1_comp.x'), arr[1, ..., 1])
        assert_near_equal(p.get_val('row4_comp.x'), arr[2, ..., 1])

    def test_om_slice_negative_stop_mpi(self):
        class MyComp1(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(4), src_indices=om.slicer[:,-1], distributed=True)
                self.add_output('y', 1.0, distributed=True)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        arr = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr))
        p.model.add_subsystem('C1', MyComp1())
        p.model.connect('indep.x', 'C1.x')

        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('C1.x', get_remote=False), np.array([4, 4, 4, 4]))

class TestGroupPromotes(unittest.TestCase):

    def test_promotes_outputs_in_config(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))

            def configure(self):
                self.promotes('comp2', outputs=['b'])

        top = om.Problem(model=SimpleGroup())
        top.setup()

        self.assertEqual(top['b'], 1)
        with self.assertRaises(KeyError) as cm:
            top['a']

        self.assertEqual(str(cm.exception),
                         "'<model> <class SimpleGroup>: Variable \"a\" not found.'")

    def test_promotes_inputs_in_config(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))

            def configure(self):
                self.promotes('comp2', inputs=['a'])

        top = om.Problem(model=SimpleGroup())
        top.setup()

        self.assertEqual(top['a'], 1)
        with self.assertRaises(KeyError) as cm:
            top['b']

        self.assertEqual(str(cm.exception),
                         "'<model> <class SimpleGroup>: Variable \"b\" not found.'")

    def test_promotes_any_in_config(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))

            def configure(self):
                self.promotes('comp1', any=['*'])

        top = om.Problem(model=SimpleGroup())
        top.setup()

        self.assertEqual(top['x'], 5)
        with self.assertRaises(KeyError) as cm:
            top['a']

        self.assertEqual(str(cm.exception),
                         "'<model> <class SimpleGroup>: Variable \"a\" not found.'")

    def test_promotes_alias(self):
        class SubGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.ExecComp('x=2.0*a+3.0*b', a=3.0, b=4.0))

            def configure(self):
                self.promotes('comp1', inputs=['a'])

        class TopGroup(om.Group):
            def setup(self):
                self.add_subsystem('sub', SubGroup())

            def configure(self):
                self.sub.promotes('comp1', inputs=['b'])
                self.promotes('sub', inputs=[('b', 'bb')])

        top = om.Problem(model=TopGroup())
        top.setup()

        self.assertEqual(top['bb'], 4.0)

    def test_promotes_alias_from_parent(self):
        class SubGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.ExecComp('x=2.0*a+3.0*b+c', a=3.0, b=4.0))

            def configure(self):
                self.promotes('comp1', inputs=[('b', 'bb')])

        class TopGroup(om.Group):
            def setup(self):
                self.add_subsystem('sub', SubGroup())

            def configure(self):
                self.sub.promotes('comp1', inputs=['b'])

        top = om.Problem(model=TopGroup(), name='alias_from_parent')

        with self.assertRaises(RuntimeError) as context:
            top.setup()

        self.assertEqual(str(context.exception),
            "\nCollected errors for problem 'alias_from_parent':"
            "\n   'sub' <class SubGroup>: Trying to promote 'b' when it has been aliased to 'bb'."
            "\n   'sub' <class SubGroup>: Trying to promote 'b' when it has been aliased to 'b'."
            "\n   'sub.comp1' <class ExecComp>: Can't alias promoted input 'b' to 'b' because 'b' "
            "has already been promoted as '('bb', 'b')'.")

    def test_promotes_alias_src_indices(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('b=2*a', a=np.ones(3), b=np.ones(3)))
                self.add_subsystem('comp2', om.ExecComp('b=4*a', a=np.ones(2), b=np.ones(2)))

            def configure(self):
                self.indep.add_output('x', np.array(range(5)))
                self.promotes('comp1', inputs=[('a', 'x')], src_indices=[0, 1, 2])
                self.promotes('comp2', inputs=[('a', 'x')], src_indices=[3, 4])

        p = om.Problem(model=SimpleGroup())

        p.setup()
        p.run_model()

        assert_near_equal(p['comp1.b'], np.array([0, 2, 4]))
        assert_near_equal(p['comp2.b'], np.array([12, 16]))

    def test_promotes_wildcard_rename(self):
        class SubGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.ExecComp('x=2.0+bb', bb=4.0))

            def configure(self):
                self.promotes('comp1', inputs=["b*"])

        class TopGroup(om.Group):
            def setup(self):
                self.add_subsystem('sub', SubGroup())

            def configure(self):
                self.sub.promotes('comp1', inputs=[('bb', 'xx')])

        top = om.Problem(model=TopGroup())

        msg = "'sub.comp1' <class ExecComp>: input variable 'bb', promoted using ('bb', 'xx'), was already promoted using 'b*'."
        with assert_warning(UserWarning, msg):
            top.setup()

    def test_promotes_wildcard_name(self):
        class SubGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.ExecComp('x=2.0+bb', bb=4.0))

            def configure(self):
                self.promotes('comp1', inputs=["b*"])

        class TopGroup(om.Group):
            def setup(self):
                self.add_subsystem('sub', SubGroup())

            def configure(self):
                self.sub.promotes('comp1', inputs=['bb'])

        top = om.Problem(model=TopGroup())

        msg = "'sub.comp1' <class ExecComp>: input variable 'bb', promoted using 'bb', was already promoted using 'b*'."
        with assert_warning(UserWarning, msg):
            top.setup()

    def test_promotes_src_indcies_in_second_prommote(self):
        # Make sure we can call `promotes` on and already-promoted input and add src_indices.

        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(4))
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])

        p = om.Problem()

        p.model.add_subsystem('indep',
                              om.IndepVarComp('x', np.arange(12)),
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', MyComp(), promotes_inputs=['x'])

        p.model.promotes('C1', inputs=['x'], src_indices=np.arange(4))

        # Runs without exception.
        p.setup()

    def test_multiple_promotes(self):

        class BranchGroup(om.Group):
            def setup(self):
                b1 = self.add_subsystem('Branch1', om.Group())
                g1 = b1.add_subsystem('G1', om.Group())
                g2 = g1.add_subsystem('G2', om.Group())
                g2.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0))

            def configure(self):
                self.Branch1.G1.G2.promotes('comp1', inputs=['a'])
                self.Branch1.G1.promotes('G2', any=['*'])

        top = om.Problem(model=BranchGroup())
        top.setup()

        self.assertEqual(top['Branch1.G1.a'], 3)
        self.assertEqual(top['Branch1.G1.comp1.b'], 6)
        with self.assertRaises(KeyError) as cm:
            top['Branch1.G1.comp1.a']

        self.assertEqual(str(cm.exception),
                         "'<model> <class BranchGroup>: Variable \"Branch1.G1.comp1.a\" not found.'")

    def test_multiple_promotes_collision(self):

        class SubGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.ExecComp('x=2.0*a+3.0*b', a=3.0, b=4.0))

            def configure(self):
                self.promotes('comp1', inputs=['a'])

        class TopGroup(om.Group):
            def setup(self):
                self.add_subsystem('sub', SubGroup())

            def configure(self):
                self.sub.promotes('comp1', inputs=['b'])
                self.promotes('sub', inputs=['b'])

        top = om.Problem(model=TopGroup())
        top.setup()

        self.assertEqual(top['sub.a'], 3)
        self.assertEqual(top['b'], 4)

    def test_multiple_promotes_src_indices(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('c=2*a*b', a=np.ones(3), b=np.ones(3), c=np.ones(3)))

            def configure(self):
                self.indep.add_output('x', np.array(range(5)))
                self.promotes('comp1', inputs=[('a', 'x'), ('b', 'x')], src_indices=[0, 2, 4])

        p = om.Problem(model=SimpleGroup())
        p.setup()
        p.run_model()

        assert_near_equal(p['x'], np.array([0, 1, 2, 3, 4]))
        assert_near_equal(p['comp1.a'], np.array([0, 2, 4]))
        assert_near_equal(p['comp1.b'], np.array([0, 2, 4]))
        assert_near_equal(p['comp1.c'], np.array([0, 8, 32]))

    def test_promotes_src_indices_flat(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp('a', np.array(range(9)).reshape(3, 3)), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('b=2*a', a=np.ones(3), b=np.ones(3)))

            def configure(self):
                self.promotes('comp1', inputs=['a'], src_indices=[0, 4, 8], flat_src_indices=True)


        p = om.Problem(model=SimpleGroup())
        p.setup()
        p.run_model()

        assert_near_equal(p['a'], np.array([[0, 1, 2],
                                            [3, 4, 5],
                                            [6, 7, 8]]))

        assert_near_equal(p['comp1.a'], np.array([0, 4, 8]))
        assert_near_equal(p['comp1.b'], np.array([0, 8, 16]))

    def test_promotes_bad_spec(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp', om.ExecComp('b=2*a', a=np.zeros(5), b=np.zeros(5)))

            def configure(self):
                self.promotes('comp', inputs='a')

        top = om.Problem(model=SimpleGroup(), name='promotes_bad_spec')

        with self.assertRaises(Exception) as cm:
            top.setup()

        self.assertEqual(str(cm.exception),
            "\nCollected errors for problem 'promotes_bad_spec':"
            "\n   <model> <class SimpleGroup>: Trying to promote inputs='a', "
            "but an iterator of strings and/or tuples is required.")

    def test_promotes_src_indices_bad_type(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp2', om.ExecComp('b=2*a', a=np.zeros(5), b=np.zeros(5)))

            def configure(self):
                self.promotes('comp2', inputs=['a'], src_indices=1.0)

        top = om.Problem(model=SimpleGroup(), name='src_indices_bad_type')

        with self.assertRaises(Exception) as cm:
            top.setup()

        self.assertEqual(str(cm.exception),
            "\nCollected errors for problem 'src_indices_bad_type':"
            "\n   <model> <class SimpleGroup>: When promoting ['a'] from 'comp2': Can't create an "
            "index array using indices of non-integral type 'float64'.")

    def test_promotes_src_indices_bad_dtype(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp2', om.ExecComp('b=2*a', a=np.zeros(5), b=np.zeros(5)))

            def configure(self):
                self.promotes('comp2', inputs=['a'], src_indices=np.array([0], dtype=complex))

        top = om.Problem(model=SimpleGroup(), name='promotes_src_indices_bad_dtype')

        with self.assertRaises(Exception) as cm:
            top.setup()

        self.assertEqual(str(cm.exception),
            "\nCollected errors for problem 'promotes_src_indices_bad_dtype':"
            "\n   <model> <class SimpleGroup>: When promoting ['a'] from 'comp2': Can't create "
            "an index array using indices of non-integral type 'complex128'.")

    def test_promotes_src_indices_bad_shape(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('b=2*a', a=np.ones(5), b=np.ones(5)))

            def configure(self):
                self.indep.add_output('a1', np.ones(3))
                self.promotes('comp1', inputs=['a'], src_indices=[0, 1, 2], src_shape=3)

        p = om.Problem(model=SimpleGroup(), name='src_indices_bad_shape')

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(str(cm.exception),
            "\nCollected errors for problem 'src_indices_bad_shape':"
            "\n   <model> <class SimpleGroup>: The source indices [0 1 2] do not specify a "
            "valid shape for the connection '_auto_ivc.v0' to 'comp1.a'. (target shape=(5,), "
            "indices_shape=(3,)): shape mismatch: value array of shape (5,) could not be "
            "broadcast to indexing result of shape (3,)")

    def test_promotes_src_indices_different(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('z=x+y', x=np.ones(3), y=np.ones(3), z=np.ones(3)))

            def configure(self):
                self.indep.add_output('x', 2*np.array(range(5)))
                self.indep.add_output('y', 3*np.array(range(5)))
                self.promotes('comp1', inputs=['x'], src_indices=[0, 2, 4])
                self.promotes('comp1', inputs=['y'], src_indices=[1, 2, 3])

        p = om.Problem(model=SimpleGroup())

        p.setup()
        p.run_model()

        assert_near_equal(p['indep.x'], np.array([0, 2, 4, 6, 8]))
        assert_near_equal(p['indep.y'], np.array([0, 3, 6, 9, 12]))

        assert_near_equal(p['comp1.x'], np.array([0, 4, 8]))
        assert_near_equal(p['comp1.y'], np.array([3, 6, 9]))
        assert_near_equal(p['comp1.z'], np.array([3, 10, 17]))

    def test_promotes_src_indices_mixed(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('z=x+y',
                                                        x=np.ones(3),
                                                        y={'val': np.ones(3),
                                                           'src_indices': [1, 2, 3]},
                                                        z=np.ones(3)),
                                    promotes_inputs=['y'])

            def configure(self):
                self.indep.add_output('x', 2*np.array(range(5)))
                self.indep.add_output('y', 3*np.array(range(5)))
                self.promotes('comp1', inputs=['x'], src_indices=[0, 2, 4])

        p = om.Problem(model=SimpleGroup())

        p.setup()
        p.run_model()

        assert_near_equal(p['indep.x'], np.array([0, 2, 4, 6, 8]))
        assert_near_equal(p['indep.y'], np.array([0, 3, 6, 9, 12]))

        assert_near_equal(p['comp1.x'], np.array([0, 4, 8]))
        assert_near_equal(p['comp1.y'], np.array([3, 6, 9]))
        assert_near_equal(p['comp1.z'], np.array([3, 10, 17]))

    def test_promotes_src_indices_mixed_array(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('z=x+y',
                                                        x=np.ones(3),
                                                        y={'val': np.ones(3),
                                                           'src_indices': [1, 2, 3]},
                                                        z=np.ones(3)),
                                    promotes_inputs=['y'])

            def configure(self):
                self.indep.add_output('x', 2*np.array(range(5)))
                self.indep.add_output('y', 3*np.array(range(5)))
                self.promotes('comp1', inputs=['x'],
                              src_indices=np.array([0, 2, 4]))

        p = om.Problem(model=SimpleGroup())

        p.setup()
        p.run_model()

        assert_near_equal(p['indep.x'], np.array([0, 2, 4, 6, 8]))
        assert_near_equal(p['indep.y'], np.array([0, 3, 6, 9, 12]))

        assert_near_equal(p['comp1.x'], np.array([0, 4, 8]))
        assert_near_equal(p['comp1.y'], np.array([3, 6, 9]))
        assert_near_equal(p['comp1.z'], np.array([3, 10, 17]))

    def test_promotes_src_indices_wildcard(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('z=x+y', x=np.ones(3), y=np.ones(3), z=np.ones(3)))

            def configure(self):
                self.indep.add_output('x', 2*np.array(range(5)))
                self.indep.add_output('y', 3*np.array(range(5)))
                self.promotes('comp1', inputs=['*'], src_indices=[0, 2, 4])

        p = om.Problem(model=SimpleGroup())

        p.setup()
        p.run_model()

        assert_near_equal(p['indep.x'], np.array([0, 2, 4, 6, 8]))
        assert_near_equal(p['indep.y'], np.array([0, 3, 6, 9, 12]))

        assert_near_equal(p['comp1.x'], np.array([0, 4, 8]))
        assert_near_equal(p['comp1.y'], np.array([0, 6, 12]))
        assert_near_equal(p['comp1.z'], np.array([0, 10, 20]))

    def test_promotes_src_indices_wildcard_any(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('z=x+y', x=np.ones(3), y=np.ones(3), z=np.ones(3)))

            def configure(self):
                self.indep.add_output('x', 2*np.array(range(5)))
                self.indep.add_output('y', 3*np.array(range(5)))
                self.promotes('comp1', any=['*'], src_indices=[0, 2, 4])

        p = om.Problem(model=SimpleGroup())

        p.setup()

        p.run_model()

        assert_near_equal(p['indep.x'], np.array([0, 2, 4, 6, 8]))
        assert_near_equal(p['indep.y'], np.array([0, 3, 6, 9, 12]))

        assert_near_equal(p['comp1.x'], np.array([0, 4, 8]))
        assert_near_equal(p['comp1.y'], np.array([0, 6, 12]))
        assert_near_equal(p['comp1.z'], np.array([0, 10, 20]))

    def test_promotes_src_indices_wildcard_output(self):

        class SimpleGroup(om.Group):
            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('comp1', om.ExecComp('z=x+y', x=np.ones(3), y=np.ones(3), z=np.ones(3)))

            def configure(self):
                self.indep.add_output('x', 2*np.array(range(5)))
                self.indep.add_output('y', 3*np.array(range(5)))
                self.promotes('comp1', outputs=['*'], src_indices=[0, 2, 4])

        p = om.Problem(model=SimpleGroup(), name='src_indices_wildcard_output')

        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        self.assertEqual(str(cm.exception),
            "\nCollected errors for problem 'src_indices_wildcard_output':"
            "\n   <model> <class SimpleGroup>: Trying to promote outputs ['*'] "
            "while specifying src_indices [0 2 4] is not meaningful.")

    def test_promotes_src_indices_collision(self):

        class SubGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp', om.ExecComp('x=2.0*a+3.0*b',
                                                       a=np.ones(3),
                                                       b=np.ones(3),
                                                       x=np.ones(3)))

            def configure(self):
                self.promotes('comp', inputs=['a'], src_indices=[0, 2, 4])

        class TopGroup(om.Group):
            def setup(self):
                self.add_subsystem('ind', om.IndepVarComp())
                self.add_subsystem('sub', SubGroup())

            def configure(self):
                self.ind.add_output('a', val=np.ones(5))

                self.promotes('ind', outputs=['a'])
                self.promotes('sub', inputs=['a'], src_indices=[0, 1, 2])

        p = om.Problem(model=TopGroup(), name='promotes_src_indices_collision')

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(str(cm.exception),
            "\nCollected errors for problem 'promotes_src_indices_collision':"
            "\n   <model> <class TopGroup>: When connecting 'ind.a' to 'sub.comp.a': input 'sub.a' "
            "src_indices are [0 1 2] and indexing into those failed using src_indices [0 2 4] "
            "from input 'sub.comp.a'. Error was: index 4 is out of bounds for source dimension of "
            "size 3.")

    def test_promotes_list_order(self):
        # This test verifies that the order we promote in the arguments to add_subsystem doesn't
        # trigger any of our exceptions when we have wildcards with promote_as.
        class AllPatterns(om.Group):
            def setup(self):
                comp = om.ExecComp(['a1=b1+c1', 'g1=b1-c1'])
                self.add_subsystem('comp1', comp,
                                   promotes_inputs=['*',('c1','ialias1')],
                                   promotes_outputs=['*',('g1','oalias1')])

                comp = om.ExecComp(['a2=b2+c2', 'g2=b2-c2'])
                self.add_subsystem('comp2', comp,
                                   promotes_inputs=[('c2','ialias2'), '*'],
                                   promotes_outputs=[('g2','oalias2'), '*'])

                comp = om.ExecComp(['a3=b3+c3', 'g3=b3-c3'])
                self.add_subsystem('comp3', comp,
                                   promotes=[('c3','ialias3'), ('g3','oalias3'), '*'])

                comp = om.ExecComp(['a4=b4+c4', 'g4=b4-c4'])
                self.add_subsystem('comp4', comp,
                                   promotes=['*', ('c4','ialias4'), ('g4','oalias4')])

        p = om.Problem(model=AllPatterns())
        p.setup()
        p.run_model()
        # If working correctly, no exception raised.


class MyComp(om.ExplicitComponent):
    def __init__(self, input_shape, src_indices=None, flat_src_indices=False):
        super().__init__()
        self._input_shape = input_shape
        self._src_indices = src_indices
        self._flat_src_indices = flat_src_indices

    def setup(self):
        self.add_input('x', val=np.zeros(self._input_shape),
                       src_indices=self._src_indices, flat_src_indices=self._flat_src_indices)
        self.add_output('y', val=np.zeros(self._input_shape))

    def compute(self, inputs, outputs):
        outputs['y'] = 2.0 * inputs['x']


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

        with set_env_vars_context(OPENMDAO_FAIL_FAST='1'):
            with self.assertRaises(Exception) as cm:
                self.sub.connect('src.x', 'tgt.x', src_indices=[1.0])

        msg = "'sub' <class Group>: When connecting from 'src.x' to 'tgt.x': Can't create an index array using indices of non-integral type 'float64'."
        self.assertEquals(str(cm.exception), msg)

    def test_src_indices_as_float_array(self):
        self.prob._name = 'src_indices_as_float_array'
        self.sub.connect('src.x', 'tgt.x', src_indices=np.zeros(1))
        with self.assertRaises(Exception) as cm:
            self.prob.setup()
            self.prob.run_model()

        self.assertEquals(str(cm.exception),
           "\nCollected errors for problem 'src_indices_as_float_array':"
           "\n   'sub' <class Group>: When connecting from 'src.x' to 'tgt.x': Can't create an "
           "index array using indices of non-integral type 'float64'.")

    def test_src_indices_as_str(self):
        msg = "'sub' <class Group>: src_indices must be a slice, int, or index array. Did you mean connect('src.x', ['tgt.x', 'cmp.x'])?"
        with set_env_vars_context(OPENMDAO_FAIL_FAST='1'):
            with self.assertRaisesRegex(Exception, msg):
                self.sub.connect('src.x', 'tgt.x', 'cmp.x')

    def test_already_connected(self):
        msg = "'sub' <class Group>: Input 'tgt.x' is already connected to 'src.x'."

        with set_env_vars_context(OPENMDAO_FAIL_FAST='1'):
            self.sub.connect('src.x', 'tgt.x', src_indices=[1])
            with self.assertRaises(Exception) as cm:
                self.sub.connect('cmp.x', 'tgt.x', src_indices=[1])

        self.assertEquals(str(cm.exception), msg)

    def test_invalid_source(self):
        self.prob._name = 'invalid_source'
        # source and target names can't be checked until setup
        # because setup is not called until then
        self.sub.connect('src.z', 'tgt.x', src_indices=[1])

        with self.assertRaises(Exception) as context:
            self.prob.setup()

        self.assertEqual(str(context.exception),
           "\nCollected errors for problem 'invalid_source':"
           "\n   'sub' <class Group>: Attempted to connect from 'src.z' to 'tgt.x', but 'src.z' "
           "doesn't exist. Perhaps you meant to connect to one of the following outputs: "
           "['src.x', 'src.s', 'cmp.z']."
)
    def test_connect_to_output(self):
        self.prob._name = 'connect_to_output'
        msg = "\nCollected errors for problem 'connect_to_output':\n   'sub' <class Group>: " + \
              "Attempted to connect from 'tgt.y' to 'cmp.z', but 'cmp.z' is an output. " + \
              "All connections must be from an output to an input."

        # source and target names can't be checked until setup
        # because setup is not called until then
        self.sub.connect('tgt.y', 'cmp.z')

        with self.assertRaises(Exception) as context:
            self.prob.setup()

        self.assertEqual(str(context.exception), msg)

    def test_connect_from_input(self):
        self.prob._name = 'connect_from_input'
        msg = "\nCollected errors for problem 'connect_from_input':\n   'sub' <class Group>: " + \
              "Attempted to connect from 'tgt.x' to 'cmp.x', but 'tgt.x' is an input. " + \
              "All connections must be from an output to an input."

        # source and target names can't be checked until setup
        # because setup is not called until then
        self.sub.connect('tgt.x', 'cmp.x')
        with self.assertRaises(Exception) as context:
            self.prob.setup()

        self.assertEqual(str(context.exception), msg)

    def test_explicit_conn_to_prom_inputs(self):
        p = om.Problem()
        p.model.add_subsystem('indeps', om.IndepVarComp('foo', val=10., units='ft'))
        p.model.add_subsystem('C1', om.ExecComp('y=3*x', x={'units': 'ft', 'val': 1.}), promotes_inputs=['x'])
        p.model.add_subsystem('C2', om.ExecComp('y=4*x', x={'units': 'ft', 'val': 1.}), promotes_inputs=['x'])
        p.model.connect('indeps.foo', 'x')
        p.setup()
        p.final_setup()

        # before bug fix, the following generated an ambiguity error
        p['x']

    def test_invalid_target(self):
        self.prob._name = 'invalid_target'
        msg = "\nCollected errors for problem 'invalid_target':\n   'sub' <class Group>: " + \
              "Attempted to connect from 'src.x' to 'tgt.z', but 'tgt.z' doesn't exist. " + \
              "Perhaps you meant to connect to one of the following inputs: ['tgt.x']."

        # source and target names can't be checked until setup
        # because setup is not called until then
        self.sub.connect('src.x', 'tgt.z', src_indices=[1])

        with self.assertRaises(Exception) as context:
            self.prob.setup()

        self.assertEqual(str(context.exception), msg)

    def test_connect_within_system(self):
        msg = "Output and input are in the same System for connection " + \
              "from 'tgt.y' to 'tgt.x'."

        with set_env_vars_context(OPENMDAO_FAIL_FAST='1'):
            with self.assertRaisesRegex(Exception, msg):
                self.sub.connect('tgt.y', 'tgt.x', src_indices=[1])

    def test_connect_within_system_with_promotes(self):
        prob = om.Problem(name='connect_within_system_with_promotes')

        sub = prob.model.add_subsystem('sub', om.Group())
        sub.add_subsystem('tgt', om.ExecComp('y = x'), promotes_outputs=['y'])
        sub.connect('y', 'tgt.x', src_indices=[1])

        msg = "\nCollected errors for problem 'connect_within_system_with_promotes':" + \
            "\n   'sub' <class Group>: Output and input are in the same System for " + \
              "connection from 'y' to 'tgt.x'."

        with self.assertRaises(Exception) as ctx:
            prob.setup()

        self.assertEqual(str(ctx.exception), msg)

    def test_connect_units_with_unitless(self):
        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0))
        prob.model.add_subsystem('src', om.ExecComp('x2 = 2 * x1', x2={'units': 'degC'}))
        prob.model.add_subsystem('tgt', om.ExecComp('y = 3 * x', x={'units': None}))

        prob.model.connect('px1.x1', 'src.x1')
        prob.model.connect('src.x2', 'tgt.x')

        msg = "<model> <class Group>: Output 'src.x2' with units of 'degC' is connected " \
              "to input 'tgt.x' which has no units."

        with assert_warning(UserWarning, msg):
            prob.setup()

        with assert_warning(UserWarning, msg):
            prob.setup()

    def test_connect_incompatible_units(self):
        msg = "\nCollected errors for problem 'connect_incompatible_units':" + \
            "\n   <model> <class Group>: Output units of 'degC' for 'src.x2' are incompatible with " + \
            "input units of 'm' for 'tgt.x'."


        prob = om.Problem(name='connect_incompatible_units')
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0))
        prob.model.add_subsystem('src', om.ExecComp('x2 = 2 * x1', x2={'units': 'degC'}))
        prob.model.add_subsystem('tgt', om.ExecComp('y = 3 * x', x={'units': 'm'}))

        prob.model.connect('px1.x1', 'src.x1')
        prob.model.connect('src.x2', 'tgt.x')

        with self.assertRaises(Exception) as context:
            prob.setup()

        self.assertEqual(str(context.exception), msg)

    def test_connect_units_with_nounits(self):
        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0))
        prob.model.add_subsystem('src', om.ExecComp('x2 = 2 * x1'))
        prob.model.add_subsystem('tgt', om.ExecComp('y = 3 * x', x={'units': 'degC'}))

        prob.model.connect('px1.x1', 'src.x1')
        prob.model.connect('src.x2', 'tgt.x')

        prob.set_solver_print(level=0)

        msg = "<model> <class Group>: Input 'tgt.x' with units of 'degC' is " \
              "connected to output 'src.x2' which has no units."

        with assert_warning(UserWarning, msg):
            prob.setup()

        prob.run_model()

        assert_near_equal(prob['tgt.y'], 600.)

    def test_connect_units_with_nounits_prom(self):
        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x', 100.0), promotes_outputs=['x'])
        prob.model.add_subsystem('src', om.ExecComp('y = 2 * x'), promotes=['x', 'y'])
        prob.model.add_subsystem('tgt', om.ExecComp('z = 3 * y', y={'units': 'degC'}), promotes=['y'])

        prob.set_solver_print(level=0)

        msg = "<model> <class Group>: Input 'tgt.y' with units of 'degC' is " \
              "connected to output 'src.y' which has no units."

        with assert_warning(UserWarning, msg):
            prob.setup()

        prob.run_model()

        assert_near_equal(prob['tgt.z'], 600.)

    def test_mix_promotes_types(self):
        prob = om.Problem()
        prob.model.add_subsystem('src', om.ExecComp(['y = 2 * x', 'y2 = 3 * x']),
                                 promotes=['x', 'y'], promotes_outputs=['y2'])

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "'src' <class ExecComp>: 'promotes' cannot be used at the same time as "
                         "'promotes_inputs' or 'promotes_outputs'.")

    def test_mix_promotes_types2(self):
        prob = om.Problem()
        prob.model.add_subsystem('src', om.ExecComp(['y = 2 * x', 'y2 = 3 * x2']),
                                 promotes=['x', 'y'], promotes_inputs=['x2'])
        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "'src' <class ExecComp>: 'promotes' cannot be used at the same time as "
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

        assert_near_equal(prob['G1.par1.c4.y'], 8.0)

    def test_bad_shapes(self):
        self.prob._name = 'bad_shapes'
        self.sub.connect('src.s', 'arr.x')

        msg = "\nCollected errors for problem 'bad_shapes':" + \
              "\n   'sub' <class Group>: The source and target shapes do not match or are ambiguous " + \
              "for the connection 'sub.src.s' to 'sub.arr.x'. The source shape is (1,) " + \
              "but the target shape is (2,)."

        with self.assertRaises(Exception) as context:
            self.prob.setup()

        self.assertEqual(str(context.exception), msg)

    def test_bad_indices_shape(self):
        p = om.Problem(name='bad_indices_shape')
        p.model.add_subsystem('IV', om.IndepVarComp('x', np.arange(12).reshape((4, 3))))
        p.model.add_subsystem('C1', om.ExecComp('y=sum(x)*2.0', x=np.zeros((2, 2))))

        p.model.connect('IV.x', 'C1.x', src_indices=[[1], [1]])

        msg = "\nCollected errors for problem 'bad_indices_shape':\n   <model> <class Group>: " + \
              "The source indices ([1], [1]) do not specify a valid shape " + \
              "for the connection 'IV.x' to 'C1.x'. The target shape is (2, 2) but " + \
              "indices are shape (1,)."

        with self.assertRaises(Exception) as context:
            p.setup()

        self.assertEqual(str(context.exception), msg)

    def test_bad_indices_dimensions(self):
        self.prob._name = 'bad_indices_dimensions'
        self.sub.connect('src.x', 'arr.x', src_indices=[[2,2],[-1,2],[2,2]],
                         flat_src_indices=False)

        msg = "\nCollected errors for problem 'bad_indices_dimensions':\n   <model> <class Group>: " + \
              "When connecting 'sub.src.x' to 'sub.arr.x': Can't set source shape to (5, 3) because " + \
              "indexer ([2, 2], [-1, 2], [2, 2]) expects 3 dimensions."
        try:
            self.prob.setup()
        except Exception as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail('Exception expected.')

    def test_bad_indices_index(self):
        self.prob._name = 'bad_indices_index'
        # the index value within src_indices is outside the valid range for the source
        self.sub.connect('src.x', 'arr.x', src_indices=[[2, 4],[-1, 4]],
                         flat_src_indices=False)

        msg = "\nCollected errors for problem 'bad_indices_index':\n   <model> <class Group>: " + \
              "When connecting 'sub.src.x' to 'sub.arr.x': index 4 is out of bounds for source " + \
              "dimension of size 3."

        try:
            self.prob.setup()
        except Exception as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail('Exception expected.')


class TestSrcIndices(unittest.TestCase):

    def create_problem(self, src_shape, tgt_shape, src_indices=None, flat_src_indices=False,
                       promotes=None, name=None):
        prob = om.Problem(name=name)
        prob.model.add_subsystem('indeps', om.IndepVarComp('x', shape=src_shape),
                                 promotes=promotes)
        prob.model.add_subsystem('C1', MyComp(tgt_shape,
                                              src_indices=src_indices if promotes else None,
                                              flat_src_indices=flat_src_indices),
                                 promotes=promotes)
        if promotes is None:
            prob.model.connect('indeps.x', 'C1.x', src_indices=src_indices,
                               flat_src_indices=flat_src_indices)

        return prob

    def test_src_indices_shape(self):
        self.create_problem(src_shape=(3, 3), tgt_shape=(2, 2),
                            src_indices=[4, 5, 7, 8],
                            flat_src_indices=True)

    def test_src_indices_shape_bad_idx_flat(self):
        msg = "\nCollected errors for problem 'src_indices_shape_bad_idx_flat':" + \
              "\n   <model> <class Group>: When connecting 'indeps.x' to 'C1.x': index 9 is out " + \
              "of bounds for source dimension of size 9."

        p = self.create_problem(src_shape=(3, 3), tgt_shape=(2, 2),
                            src_indices=[4, 7, 5, 9],
                            flat_src_indices=True, name='src_indices_shape_bad_idx_flat')
        try:
            p.setup()
        except Exception as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail("Exception expected.")

    def test_src_indices_shape_bad_idx_flat_promotes(self):
        p = self.create_problem(src_shape=(3, 3), tgt_shape=(2, 2),
                            src_indices=[4, 5, 7, 9],
                            flat_src_indices=True, promotes=['x'], name='src_indices_shape_bad_idx_flat_promotes')

        msg = "\nCollected errors for problem 'src_indices_shape_bad_idx_flat_promotes':" + \
              "\n   'C1' <class MyComp>: When accessing 'indeps.x' with src_shape (3, 3) from 'x' " + \
              "using src_indices [4 5 7 9]: index 9 is out of bounds for source dimension of size 9."
        try:
            p.setup()
        except Exception as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail("Exception expected.")

    def test_src_indices_shape_bad_idx_flat_neg(self):
        msg = "\nCollected errors for problem 'src_indices_shape_bad_idx_flat_neg':" + \
              "\n   <model> <class Group>: When connecting 'indeps.x' to 'C1.x': index -10 is out " + \
              "of bounds for source dimension of size 9."
        p = self.create_problem(src_shape=(3, 3), tgt_shape=(2, 2),
                            src_indices=[-10, 5, 7, 8],
                            flat_src_indices=True, name='src_indices_shape_bad_idx_flat_neg')
        try:
            p.setup()
        except Exception as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail("Exception expected.")

    def test_om_slice_with_ellipsis_error_in_connect(self):

        p = om.Problem(name='slice_with_ellipsis_error_in_connect')

        p.model.add_subsystem('indep', om.IndepVarComp('x', arr_large_4x4))
        p.model.add_subsystem('row4_comp', SlicerComp())

        # the 4 should be a 3
        p.model.connect('indep.x', 'row4_comp.x', src_indices=om.slicer[4, ...])

        with self.assertRaises(Exception) as err:
            p.setup()

        expected_error_msg = "\nCollected errors for problem 'slice_with_ellipsis_error_in_connect':" + \
            "\n   <model> <class Group>: When connecting 'indep.x' to 'row4_comp.x': index 4 is out " + \
            "of bounds of the source shape (4,)."
        self.assertEqual(str(err.exception), expected_error_msg)


class TestGroupAddInput(unittest.TestCase):

    def _make_tree_model(self, diff_units=False, diff_vals=False, name=None):
        p = om.Problem(name=name)
        model = p.model

        if diff_units:
            units1 = 'ft'
            units2 = 'inch'
        else:
            units1 = units2 = 'ft'

        val = 1.0

        g1 = model.add_subsystem("G1", om.Group(), promotes_inputs=['x'])

        g2 = g1.add_subsystem("G2", om.Group(), promotes_inputs=['x'])
        g2.add_subsystem("C1", om.ExecComp("y = 2. * x",
                                            x={'val': val, 'units': units2},
                                            y={'val': 1.0, 'units': units2}),
                                            promotes_inputs=['x'])
        g2.add_subsystem("C2", om.ExecComp("y = 3. * x",
                                            x={'val': val, 'units': units1},
                                            y={'val': 1.0, 'units': units1}),
                                            promotes_inputs=['x'])

        g3 = g1.add_subsystem("G3", om.Group(), promotes_inputs=['x'])
        if diff_vals: val = 2.0
        g3.add_subsystem("C3", om.ExecComp("y = 4. * x",
                                            x={'val': val, 'units': units1},
                                            y={'val': 1.0, 'units': units1}),
                                            promotes_inputs=['x'])
        g3.add_subsystem("C4", om.ExecComp("y = 5. * x",
                                            x={'val': val, 'units': units2},
                                            y={'val': 1.0, 'units': units2}),
                                            promotes_inputs=['x'])

        par = model.add_subsystem("par", om.ParallelGroup(), promotes_inputs=['x'])

        g4 = par.add_subsystem("G4", om.Group(), promotes_inputs=['x'])
        if diff_vals: val = 3.0
        g4.add_subsystem("C5", om.ExecComp("y = 6. * x",
                                            x={'val': val, 'units': units2},
                                            y={'val': 1.0, 'units': units2}),
                                            promotes_inputs=['x'])
        g4.add_subsystem("C6", om.ExecComp("y = 7. * x",
                                            x={'val': val, 'units': units1},
                                            y={'val': 1.0, 'units': units1}),
                                            promotes_inputs=['x'])

        g5 = par.add_subsystem("G5", om.Group(), promotes_inputs=['x'])
        if diff_vals: val = 4.0
        g5.add_subsystem("C7", om.ExecComp("y = 8. * x",
                                            x={'val': val, 'units': units1},
                                            y={'val': 1.0, 'units': units1}),
                                            promotes_inputs=['x'])
        g5.add_subsystem("C8", om.ExecComp("y = 9. * x",
                                            x={'val': val, 'units': units2},
                                            y={'val': 1.0, 'units': units2}),
                                            promotes_inputs=['x'])

        return p

    def test_missing_diff_units(self):
        p = om.Problem(name="missing_diff_units")
        model = p.model

        par = model.add_subsystem('par', om.ParallelGroup(), promotes_inputs=['x'])
        par.add_subsystem('C1', om.ExecComp('y = 3. * x',
                                            x={'val': 1.0, 'units': 'ft'},
                                            y={'val': 1.0, 'units': 'ft'}),
                                            promotes_inputs=['x'])
        par.add_subsystem('C2', om.ExecComp('y = 5. * x',
                                            x={'val': 1.0, 'units': 'inch'},
                                            y={'val': 1.0, 'units': 'inch'}),
                                            promotes_inputs=['x'])

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
            "\nCollected errors for problem 'missing_diff_units':"
            "\n   <model> <class Group>: The following inputs, ['par.C1.x', 'par.C2.x'], promoted "
            "to 'x', are connected but their metadata entries ['units', 'val'] differ. "
            "Call <group>.set_input_defaults('x', units=?, val=?), where <group> is the Group named "
            "'par' to remove the ambiguity.")

    def test_missing_diff_vals(self):
        p = om.Problem(name="missing_diff_vals")
        model = p.model

        par = model.add_subsystem('par', om.ParallelGroup(), promotes_inputs=['x'])
        par.add_subsystem('C1', om.ExecComp('y = 3. * x', x=1.0), promotes_inputs=['x'])
        par.add_subsystem('C2', om.ExecComp('y = 5. * x', x=1.1), promotes_inputs=['x'])

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
            "\nCollected errors for problem 'missing_diff_vals':"
            "\n   <model> <class Group>: The following inputs, ['par.C1.x', 'par.C2.x'], promoted "
            "to 'x', are connected but their metadata entries ['val'] differ. "
            "Call <group>.set_input_defaults('x', val=?), where <group> is the Group named 'par' "
            "to remove the ambiguity.")

    def test_conflicting_units(self):
        # multiple Group.set_input_defaults calls at same tree level with conflicting units args
        p = self._make_tree_model(name="conflicting_units")
        model = p.model
        g2 = model._get_subsystem('G1.G2')
        g2.set_input_defaults('x', units='ft')

        g3 = model._get_subsystem('G1.G3')
        g3.set_input_defaults('x', units='ft')

        g4 = model._get_subsystem('par.G4')
        g4.set_input_defaults('x', units='inch')

        g5 = model._get_subsystem('par.G5')
        g5.set_input_defaults('x', units='ft')

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
            "\nCollected errors for problem 'conflicting_units':"
            "\n   <model> <class Group>: The subsystems G1.G2 and par.G4 called set_input_defaults "
            "for promoted input 'x' with conflicting values for 'units'. "
            "Call <group>.set_input_defaults('x', units=?), where <group> is the model to remove "
            "the ambiguity.")

    def test_conflicting_units_multi_level(self):
        # multiple Group.set_input_defaults calls at different tree levels with conflicting units args
        p = self._make_tree_model(diff_units=True, name="conflicting_units_multi_level")
        model = p.model
        g2 = model._get_subsystem('G1.G2')
        g2.set_input_defaults('x', units='km')

        g3 = model._get_subsystem('G1.G3')
        g3.set_input_defaults('x', units='ft')

        g4 = model._get_subsystem('par.G4')
        g4.set_input_defaults('x', units='ft')

        g5 = model._get_subsystem('par.G5')
        g5.set_input_defaults('x', units='ft')

        g1 = model._get_subsystem('G1')
        g1.set_input_defaults('x', units='inch')

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'conflicting_units_multi_level':"
           "\n   <model> <class Group>: The subsystems G1 and par.G4 called set_input_defaults "
           "for promoted input 'x' with conflicting values for 'units'. "
           "Call <group>.set_input_defaults('x', units=?), where <group> is the model to remove the ambiguity."
           "\n   <model> <class Group>: The subsystems G1 and par.G5 called set_input_defaults for "
           "promoted input 'x' with conflicting values for 'units'. "
           "Call <group>.set_input_defaults('x', units=?), where <group> is the model to remove the ambiguity."
           "\n   <model> <class Group>: The following inputs, ['G1.G2.C1.x', 'G1.G2.C2.x', "
           "'G1.G3.C3.x', 'G1.G3.C4.x', 'par.G4.C5.x', 'par.G4.C6.x', 'par.G5.C7.x', 'par.G5.C8.x'], "
           "promoted to 'x', are connected but their metadata entries ['val'] differ. "
           "Call <group>.set_input_defaults('x', val=?), where <group> is the model to remove the ambiguity.")

    def test_override_units(self):
        # multiple Group.set_input_defaults calls at different tree levels with conflicting units args
        p = self._make_tree_model()
        model = p.model
        g2 = model._get_subsystem('G1.G2')
        g2.set_input_defaults('x', units='km')

        g1 = model._get_subsystem('G1')
        g1.set_input_defaults('x', units='inch', val=2.)

        msg = "Groups 'G1' and 'G1.G2' called set_input_defaults for the input 'x' with conflicting 'units'. The value (inch) from 'G1' will be used."
        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        self.assertEqual(testlogger.get('warning')[1], msg)

    def test_units_checking(self):
        p = om.Problem()
        model = p.model
        G1 = model.add_subsystem('G1', om.Group())
        G1.add_subsystem('C1', om.ExecComp('y = 3.*x', x={'units': 'm'}), promotes=['x'])

        with self.assertRaises(ValueError) as cm:
            G1.set_input_defaults('x', units='junk')

        msg = "'G1' <class Group>: The units 'junk' are invalid."
        self.assertEqual(cm.exception.args[0], msg)

        with self.assertRaises(TypeError) as cm:
            G1.set_input_defaults('x', units=3)

        msg = "'G1' <class Group>: The units argument should be a str or None"
        self.assertEqual(cm.exception.args[0], msg)

        # Simplification
        G1.set_input_defaults('x', units='ft*ft/ft')
        self.assertEqual(G1._static_group_inputs['x'][0]['units'], 'ft')

    def test_sub_sub_override(self):
        p = om.Problem()
        model = p.model
        G1 = model.add_subsystem('G1', om.Group())
        G1.set_input_defaults('x', units='mm', val=1.)
        G2 = G1.add_subsystem('G2', om.Group(), promotes=['x'])
        G3 = G2.add_subsystem('G3', om.Group(), promotes=['x'])
        G3.add_subsystem('C1', om.ExecComp('y = 3.*x', x={'units': 'm'}), promotes=['x'])
        G3.add_subsystem('C2', om.ExecComp('y = 4.*x', x={'units': 'cm'}), promotes=['x'])
        G3.set_input_defaults('x', units='cm')
        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        self.assertEqual(testlogger.get('warning')[1],
                        "Groups 'G1' and 'G1.G2.G3' called set_input_defaults for the input 'x' "
                        "with conflicting 'units'. The value (mm) from 'G1' will be used.")

    def test_sub_sets_parent_meta(self):
        p = om.Problem()
        model = p.model
        G1 = model.add_subsystem('G1', om.Group())
        G1.set_input_defaults('x', val=2.)
        G2 = G1.add_subsystem('G2', om.Group(), promotes=['x'])
        G2.add_subsystem('C1', om.ExecComp('y = 3.*x', x={'units': 'm'}), promotes=['x'])
        G2.set_input_defaults('x', units='cm')
        msg = "Group 'G1' did not set a default 'units' for input 'x', so the value of (cm) from group 'G1.G2' will be used."
        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        self.assertEqual(testlogger.get('warning')[1], msg)

    def test_sub_sub_override2(self):
        p = om.Problem()
        model = p.model
        G1 = model.add_subsystem('G1', om.Group())
        G1.set_input_defaults('x', units='mm', val=1.)
        G2 = G1.add_subsystem('G2', om.Group(), promotes=['x'])
        G2.set_input_defaults('x', units='km')
        G3 = G2.add_subsystem('G3', om.Group(), promotes=['x'])
        G3.add_subsystem('C1', om.ExecComp('y = 3.*x', x={'units': 'm'}), promotes=['x'])
        G3.add_subsystem('C2', om.ExecComp('y = 4.*x', x={'units': 'cm'}), promotes=['x'])
        G3.set_input_defaults('x', units='cm')
        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        msgs = [
            "Groups 'G1' and 'G1.G2' called set_input_defaults for the input 'x' with conflicting 'units'. The value (mm) from 'G1' will be used.",
            "Groups 'G1' and 'G1.G2.G3' called set_input_defaults for the input 'x' with conflicting 'units'. The value (mm) from 'G1' will be used."
        ]
        p.final_setup()

        self.assertEqual(testlogger.get('warning')[1], msgs[0])
        self.assertEqual(testlogger.get('warning')[2], msgs[1])

    def test_conflicting_units_multi_level_par(self):
        # multiple Group.set_input_defaults calls at different tree levels with conflicting units args
        p = self._make_tree_model(diff_units=True, name="conflicting_units_multi_level_par")
        model = p.model
        g2 = model._get_subsystem('G1.G2')
        g2.set_input_defaults('x', units='ft')

        g3 = model._get_subsystem('G1.G3')
        g3.set_input_defaults('x', units='ft')

        g4 = model._get_subsystem('par.G4')
        g4.set_input_defaults('x', units='ft')

        g5 = model._get_subsystem('par.G5')
        g5.set_input_defaults('x', units='ft')

        par = model._get_subsystem('par')
        par.set_input_defaults('x', units='inch')

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
            "\nCollected errors for problem 'conflicting_units_multi_level_par':"
            "\n   <model> <class Group>: The subsystems G1.G2 and par called set_input_defaults "
            "for promoted input 'x' with conflicting values for 'units'. "
            "Call <group>.set_input_defaults('x', units=?), where <group> is the model to remove "
            "the ambiguity."
            "\n   <model> <class Group>: The following inputs, ['G1.G2.C1.x', 'G1.G2.C2.x', "
            "'G1.G3.C3.x', 'G1.G3.C4.x', 'par.G4.C5.x', 'par.G4.C6.x', 'par.G5.C7.x', 'par.G5.C8.x'], "
            "promoted to 'x', are connected but their metadata entries ['val'] differ. "
            "Call <group>.set_input_defaults('x', val=?), where <group> is the model to remove the "
            "ambiguity.")

    def test_group_input_not_found(self):
        p = self._make_tree_model(diff_units=True, name='group_input_not_found')
        model = p.model
        g2 = model._get_subsystem('G1.G2')
        g2.set_input_defaults('xx', units='ft')

        g3 = model._get_subsystem('G1.G3')
        g3.set_input_defaults('x', units='ft')

        g4 = model._get_subsystem('par.G4')
        g4.set_input_defaults('x', units='ft')

        g5 = model._get_subsystem('par.G5')
        g5.set_input_defaults('x', units='ft')

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'group_input_not_found':"
           "\n   'G1.G2' <class Group>: The following group inputs, passed to set_input_defaults(), "
           "could not be found: ['xx'].\n   'G1' <class Group>: The following group inputs, passed "
           "to set_input_defaults(), could not be found: ['G2.xx']."
           "\n   <model> <class Group>: The following group inputs, passed to set_input_defaults(), "
           "could not be found: ['G1.G2.xx']."
           "\n   <model> <class Group>: The following inputs, ['G1.G2.C1.x', 'G1.G2.C2.x', "
           "'G1.G3.C3.x', 'G1.G3.C4.x', 'par.G4.C5.x', 'par.G4.C6.x', 'par.G5.C7.x', 'par.G5.C8.x'],"
           " promoted to 'x', are connected but their metadata entries ['val'] differ. "
           "Call <group>.set_input_defaults('x', val=?), where <group> is the model to remove the "
           "ambiguity.")

    def test_conflicting_val(self):
        p = self._make_tree_model(diff_vals=True, name='conflicting_val')
        model = p.model
        g2 = model._get_subsystem('G1.G2')
        g2.set_input_defaults('x', val=3.0)

        g3 = model._get_subsystem('G1.G3')
        g3.set_input_defaults('x', val=3.0)

        g4 = model._get_subsystem('par.G4')
        g4.set_input_defaults('x', val=3.0)

        g5 = model._get_subsystem('par.G5')
        g5.set_input_defaults('x', val=3.0)

        g1 = model._get_subsystem('G1')
        g1.set_input_defaults('x', val=4.0)

        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'conflicting_val':"
           "\n   <model> <class Group>: The subsystems G1 and par.G4 called set_input_defaults for "
           "promoted input 'x' with conflicting values for 'val'. Call "
           "<group>.set_input_defaults('x', val=?), where <group> is the model to remove the ambiguity."
           "\n   <model> <class Group>: The subsystems G1 and par.G5 called set_input_defaults for "
           "promoted input 'x' with conflicting values for 'val'. Call <group>.set_input_defaults('x', val=?), "
           "where <group> is the model to remove the ambiguity.")


class MultComp(om.ExplicitComponent):
    """
    This class just performs a list of simple multiplications. It also keeps track of the number
    of times _setup_var_data is called.
    """
    def __init__(self, mults=(), inits=None, **kwargs):
        super().__init__(**kwargs)
        self.mults = list(mults)
        self.var_setup_count = 0
        if inits is None:
            inits = {}
        self.inits = inits

    def _setup_var_data(self):
        super()._setup_var_data()
        self.var_setup_count += 1

    def add_mult(self, inp, mult, out):
        self.mults((inp, mult, out))

    def setup(self):
        all_ins = set([inp for inp, _, _ in self.mults])
        all_outs = set([out for _, _, out in self.mults])
        common = sorted(all_ins.intersection(all_outs))
        if common:
            raise RuntimeError(f"{common} are both inputs and outputs.")

        out_list = [o for _, _, o in self.mults]
        if len(all_outs) < len(out_list):
            raise RuntimeError(f"Some outputs appear more than once.")

        for inp, _, out in self.mults:
            self.add_input(inp, val=self.inits.get(inp, 1.))
            self.add_output(out, val=self.inits.get(out, 1.))

    def compute(self, inputs, outputs):
        for inp, mult, out in self.mults:
            outputs[out] = mult * inputs[inp]


class ConfigGroup(om.Group):
    """
    This group can add IO vars and promotes during configure. It also keeps track of how many
    times _setup_var_data is called.
    """
    def __init__(self, parallel=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfgproms = []
        self.cfg_group_ins = []
        self.cfgio = {}
        self.cfg_invars = []
        self.cfg_outvars = []
        self.io_results = {}
        self.var_setup_count = 0

        if parallel:
            self._mpi_proc_allocator.parallel = True

    def _setup_var_data(self):
        super()._setup_var_data()
        self.var_setup_count += 1

    def add_config_prom(self, child, prom):
        self.cfgproms.append((child, prom))

    def add_input_defaults(self, name, val=None, units=None):
        self.cfg_group_ins.append((name, val, units))

    def add_var_input(self, name, val=None, units=None):
        self.cfg_invars.append((name, val, units))

    def add_var_output(self, name, val=None, units=None):
        self.cfg_outvars.append((name, val, units))

    def add_get_io(self, child, **kwargs):
        if child in self.cfgio:
            raise RuntimeError(f"Can't set more than 1 call to get_io_metadata for child {child}.")

        self.cfgio[child] = kwargs

    def configure(self):
        # retrieve metadata
        for child, kwargs in self.cfgio.items():
            kid = self._get_subsystem(child)
            if kid is not None:
                self.io_results[child] = kid.get_io_metadata(**kwargs)
            else:
                print(f"'{kid}' not found locally.")

        # promotes
        for child, prom in self.cfgproms:
            if '.' in child:
                parent, child = child.rsplit('.', 1)
                s = self._get_subsystem(parent)
                if s is None:
                    print(f"'{parent}' not found locally.")
                    continue
            else:
                s = self
            s.promotes(child, any=prom)

        # add inputs
        for vpath, val, units in self.cfg_invars:
            if '.' in vpath:
                parent, vname = vpath.rsplit('.', 1)
                s = self._get_subsystem(parent)
                if s is None:
                    print(f"'{parent}' not found locally.")
                    continue
                s.add_input(vname, val, units=units)
            else:
                raise RuntimeError("tried to add input var to a Group.")

        # add outputs
        for vpath, val, units in self.cfg_outvars:
            if '.' in vpath:
                parent, vname = vpath.rsplit('.', 1)
                s = self._get_subsystem(parent)
                if s is None:
                    print(f"'{parent}' not found locally.")
                    continue
                s.add_output(vname, val, units=units)
            else:
                raise RuntimeError("tried to add output var to a Group.")

        # set input defaults
        for name, val, units in self.cfg_group_ins:
            self.set_input_defaults(name, val=val, units=units)


class Test3Deep(unittest.TestCase):
    """
    This creates a system tree with two levels of subgroups below model to allow testing of various
    changes during configure that may change descendant systems that are not direct children.
    """
    cfg_par = False
    sub_par = False

    def build_model(self):
        p = om.Problem(model=ConfigGroup())

        minprocs = 3 if self.cfg_par else 1
        cfg = p.model.add_subsystem('cfg', ConfigGroup(parallel=self.cfg_par), min_procs=minprocs)
        cfg.add_subsystem('C1', MultComp([('x', 2., 'y')]))
        cfg.add_subsystem('C2', MultComp([('x', 3., 'y')]))

        minprocs = 2 if self.sub_par else 1
        sub = cfg.add_subsystem('sub', ConfigGroup(parallel=self.sub_par), min_procs=minprocs)
        sub.add_subsystem('C3', MultComp([('x', 4., 'y')]))
        sub.add_subsystem('C4', MultComp([('x', 5., 'y')]))

        return p

    def get_matching_var_setup_counts(self, p, count):
        """
        Return pathnames of any systems that have a var_setup_count that matches 'count'.
        """
        result = set()
        for s in p.model.system_iter(include_self=True):
            if hasattr(s, 'var_setup_count') and s.var_setup_count == count:
                result.add(s.pathname)

        if p.model.comm.size > 1:
            newres = set()
            for res in p.model.comm.allgather(result):
                newres.update(res)
            result = newres

        return sorted(result)

    def get_io_results(self, p, parent, path):
        """
        Retrieve results of get_io_metadata calls that occurred during config.
        Results are retrieved from all procs.
        """
        s = p.model._get_subsystem(parent)
        if s is None:
            raise RuntimeError(f"No parent named {parent}.")
        res = s.io_results[path]
        if s.comm.size > 1:
            allres = {}
            for procres in s.comm.allgather(res):
                allres.update(procres)
            res = allres
        return res

    def check_vs_meta(self, p, parent, meta_dict):
        """
        Compare the given metadata dict to the internal metadata dicts of the given parent.
        """
        system = p.model._get_subsystem(parent)
        metas = (system._var_allprocs_abs2meta['input'], system._var_allprocs_abs2meta['output'],
                 system._var_abs2meta['input'], system._var_abs2meta['output'])
        for vname, meta in meta_dict.items():
            for key, val in meta.items():
                for mymeta in metas:
                    if key in mymeta:
                        if (isinstance(val, np.ndarray) and not np.testing.assert_allclose(val, mymeta[key])) or val != mymeta[key]:
                            raise RuntimeError(f"{val} != {mymeta[key]}")
                        break

    def test_io_meta(self):
        p = self.build_model()
        p.model.cfg.add_get_io('C1', return_rel_names=False)
        p.model.cfg.add_get_io('C2')
        p.model.cfg.add_get_io('sub')

        p.setup()

        res = self.get_io_results(p, 'cfg', 'C1')
        expected = {'cfg.C1.x', 'cfg.C1.y'}
        self.assertEqual({n for n in res}, expected)
        self.check_vs_meta(p, 'cfg', res)

        res = self.get_io_results(p, 'cfg', 'C2')
        expected = {'x', 'y'}
        self.assertEqual({n for n in res}, expected)
        self.check_vs_meta(p, 'cfg', res)

        res = self.get_io_results(p, 'cfg', 'sub')
        expected = {'C3.x', 'C4.x', 'C3.y', 'C4.y'}
        self.assertEqual({n for n in res}, expected)
        self.check_vs_meta(p, 'cfg', res)

        names = self.get_matching_var_setup_counts(p, 1)
        expected = {'', 'cfg', 'cfg.C1', 'cfg.C2', 'cfg.sub', 'cfg.sub.C3', 'cfg.sub.C4'}
        self.assertEqual(names, sorted(expected))

    def test_io_meta_local_bad_meta_key(self):
        p = self.build_model()
        p.model.cfg.add_get_io('sub', metadata_keys=('val', 'foo'))
        with self.assertRaises(Exception) as cm:
            p.setup()

        self.assertEqual(cm.exception.args[0],
           "'cfg.sub' <class ConfigGroup>: ['foo'] are not valid metadata entry names.")

    def test_promote_descendant(self):
        p = self.build_model()
        p.model.cfg.add_config_prom('sub.C3', ['x'])
        p.model.cfg.add_config_prom('sub.C4', ['y'])
        p.setup()

        names = self.get_matching_var_setup_counts(p, 1)
        expected = {'', 'cfg', 'cfg.C1', 'cfg.C2', 'cfg.sub.C3', 'cfg.sub.C4'}
        self.assertEqual(names, sorted(expected))

        names = self.get_matching_var_setup_counts(p, 2)
        expected = {'cfg.sub'}
        self.assertEqual(names, sorted(expected))

    def test_promote_child(self):
        p = self.build_model()
        p.model.cfg.add_config_prom('C1', ['x'])
        p.model.cfg.add_config_prom('C2', ['y'])
        p.model.cfg.sub.add_config_prom('C3', ['x'])
        p.model.cfg.sub.add_config_prom('C4', ['y'])
        p.setup()

        names = self.get_matching_var_setup_counts(p, 1)
        expected = {'', 'cfg', 'cfg.C1', 'cfg.C2', 'cfg.sub', 'cfg.sub.C3', 'cfg.sub.C4'}
        self.assertEqual(names, sorted(expected))

    def test_add_input_to_child(self):
        p = self.build_model()
        p.model.cfg.sub.add_var_input('C3.ivar0', 3.0, units='ft')

        p.setup()

        names = self.get_matching_var_setup_counts(p, 1)
        expected = {'', 'cfg', 'cfg.C1', 'cfg.C2', 'cfg.sub', 'cfg.sub.C4'}
        self.assertEqual(names, sorted(expected))

        names = self.get_matching_var_setup_counts(p, 2)
        expected = {'cfg.sub.C3'}
        self.assertEqual(names, sorted(expected))

    def test_add_output_to_child(self):
        p = self.build_model()
        p.model.cfg.sub.add_var_output('C3.ovar0', 3.0, units='ft')

        p.setup()

        names = self.get_matching_var_setup_counts(p, 1)
        expected = {'', 'cfg', 'cfg.C1', 'cfg.C2', 'cfg.sub', 'cfg.sub.C4'}
        self.assertEqual(names, sorted(expected))

        names = self.get_matching_var_setup_counts(p, 2)
        expected = {'cfg.sub.C3'}
        self.assertEqual(names, sorted(expected))

    def test_add_input_to_descendant(self):
        p = self.build_model()
        p.model.cfg.add_var_input('sub.C3.ivar0', 3.0, units='ft')
        p.model.add_var_input('cfg.sub.C3.ivar1', 4.0, units='inch')

        p.setup()

        names = self.get_matching_var_setup_counts(p, 1)
        expected = {'', 'cfg.C1', 'cfg.C2', 'cfg.sub.C4'}
        self.assertEqual(names, sorted(expected))

        names = self.get_matching_var_setup_counts(p, 2)
        expected = {'cfg'}
        self.assertEqual(names, sorted(expected))

        names = self.get_matching_var_setup_counts(p, 3)
        expected = {'cfg.sub', 'cfg.sub.C3'}
        self.assertEqual(names, sorted(expected))

    def test_add_output_to_descendant(self):
        p = self.build_model()
        p.model.cfg.add_var_output('sub.C3.ovar0', 3.0, units='ft')
        p.model.add_var_output('cfg.sub.C3.ovar1', 4.0, units='inch')

        p.setup()

        names = self.get_matching_var_setup_counts(p, 1)
        expected = {'', 'cfg.C1', 'cfg.C2', 'cfg.sub.C4'}
        self.assertEqual(names, sorted(expected))

        names = self.get_matching_var_setup_counts(p, 2)
        expected = {'cfg'}
        self.assertEqual(names, sorted(expected))

        names = self.get_matching_var_setup_counts(p, 3)
        expected = {'cfg.sub', 'cfg.sub.C3'}
        self.assertEqual(names, sorted(expected))


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestInConfigMPIpar(Test3Deep):
    N_PROCS = 2
    sub_par = True

    def test_io_meta_remote(self):
        p = self.build_model()
        p.model.add_get_io('cfg', metadata_keys=('val', 'src_indices', 'shape'), get_remote=True)
        p.model.cfg.add_get_io('sub')

        p.setup()

        res = p.model.io_results['cfg']
        expected = {'sub.C3.x', 'sub.C3.y', 'sub.C4.x', 'sub.C4.y', 'C1.x', 'C1.y', 'C2.x', 'C2.y'}
        self.assertEqual(sorted(res), sorted(expected))
        self.check_vs_meta(p, 'cfg', res)

        res = p.model.cfg.io_results['sub']
        if p.model.comm.rank == 0:
            expected = {'C3.y', 'C3.x'}
        else:
            expected = {'C4.y', 'C4.x'}
        self.assertEqual(sorted(res), sorted(expected))
        self.check_vs_meta(p, 'cfg.sub', res)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestInConfigMPIparpar(Test3Deep):
    N_PROCS = 4
    cfg_par = True
    sub_par = True


#
# Feature Tests
#

class TestFeatureAddSubsystem(unittest.TestCase):

    def test_group_simple(self):

        p = om.Problem()
        p.model.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0))

        p.setup()

        self.assertEqual(p.get_val('comp1.a'), 3.0)
        self.assertEqual(p.get_val('comp1.b'), 6.0)

    def test_group_simple_promoted(self):

        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('a', 3.0),
                              promotes_outputs=['a'])
        p.model.add_subsystem('comp1', om.ExecComp('b=2.0*a'),
                              promotes_inputs=['a'])

        p.setup()
        p.run_model()

        self.assertEqual(p.get_val('a'), 3.0)
        self.assertEqual(p.get_val('comp1.b'), 6.0)

    def test_group_nested(self):

        p = om.Problem()
        p.model.add_subsystem('G1', om.Group())
        p.model.G1.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0))
        p.model.G1.add_subsystem('comp2', om.ExecComp('b=3.0*a', a=4.0, b=12.0))

        p.setup()

        self.assertEqual(p.get_val('G1.comp1.a'), 3.0)
        self.assertEqual(p.get_val('G1.comp1.b'), 6.0)
        self.assertEqual(p.get_val('G1.comp2.a'), 4.0)
        self.assertEqual(p.get_val('G1.comp2.b'), 12.0)

    def test_group_nested_promoted1(self):

        # promotes from bottom level up 1
        p = om.Problem()
        g1 = p.model.add_subsystem('G1', om.Group())
        g1.add_subsystem('comp1', om.ExecComp('b=2.0*a', a=3.0, b=6.0),
                         promotes_inputs=['a'], promotes_outputs=['b'])
        g1.add_subsystem('comp2', om.ExecComp('b=3.0*a', a=4.0, b=12.0),
                         promotes_inputs=['a'])
        g1.set_input_defaults('a', val=3.5)
        p.setup()

        # output G1.comp1.b is promoted
        self.assertEqual(p.get_val('G1.b'), 6.0)
        # output G1.comp2.b is not promoted
        self.assertEqual(p.get_val('G1.comp2.b'), 12.0)

        # use unpromoted names for the following 2 promoted inputs
        self.assertEqual(p.get_val('G1.comp1.a'), 3.5)
        self.assertEqual(p.get_val('G1.comp2.a'), 3.5)

    def test_group_nested_promoted2(self):

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
        self.assertEqual(p.get_val('comp1.b'), 6.0)
        # output G1.comp2.b is promoted
        self.assertEqual(p.get_val('comp2.b'), 12.0)

        # access both promoted inputs using unpromoted names.
        self.assertEqual(p.get_val('G1.comp1.a'), 3.0)
        self.assertEqual(p.get_val('G1.comp2.a'), 4.0)

    def test_group_rename_connect(self):

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

        self.assertEqual(p.get_val('comp1.b'), 6.0)
        self.assertEqual(p.get_val('comp2.b'), 9.0)

    def test_promotes_any(self):

        class SimpleGroup(om.Group):

            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))

            def configure(self):
                self.promotes('comp1', any=['*'])

        top = om.Problem(model=SimpleGroup())
        top.setup()

        self.assertEqual(top.get_val('x'), 5)

    def test_promotes_inputs_and_outputs(self):

        class SimpleGroup(om.Group):

            def setup(self):
                self.add_subsystem('comp1', om.IndepVarComp('x', 5.0))
                self.add_subsystem('comp2', om.ExecComp('b=2*a'))

            def configure(self):
                self.promotes('comp2', inputs=['a'], outputs=['b'])

        top = om.Problem(model=SimpleGroup())
        top.setup()

        self.assertEqual(top.get_val('a'), 1)
        self.assertEqual(top.get_val('b'), 1)


class TestFeatureConnect(unittest.TestCase):

    def test_basic_connect_units(self):

        p = om.Problem()

        p.model.set_input_defaults('x', np.ones(5), units='ft')

        exec_comp = om.ExecComp('y=sum(x)',
                                x={'val': np.zeros(5), 'units': 'inch'},
                                y={'units': 'inch'})

        p.model.add_subsystem('comp1', exec_comp, promotes_inputs=['x'])

        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('x', units='ft'), np.ones(5))
        assert_near_equal(p.get_val('comp1.x'), np.ones(5)*12.)
        assert_near_equal(p.get_val('comp1.y'), 60.)

    def test_connect_1_to_many(self):

        p = om.Problem()

        p.model.add_subsystem('C1', om.ExecComp('y=sum(x)*2.0', x=np.zeros(5)), promotes_inputs=['x'])
        p.model.add_subsystem('C2', om.ExecComp('y=sum(x)*4.0', x=np.zeros(5)), promotes_inputs=['x'])
        p.model.add_subsystem('C3', om.ExecComp('y=sum(x)*6.0', x=np.zeros(5)), promotes_inputs=['x'])

        p.setup()
        p.set_val('x', np.ones(5))
        p.run_model()

        assert_near_equal(p.get_val('C1.y'), 10.)
        assert_near_equal(p.get_val('C2.y'), 20.)
        assert_near_equal(p.get_val('C3.y'), 30.)

    def test_connect_src_indices(self):

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

        assert_near_equal(p['C1.x'], np.ones(3))
        assert_near_equal(p['C1.y'], 6.)
        assert_near_equal(p['C2.x'], np.ones(2))
        assert_near_equal(p['C2.y'], 8.)

    def test_connect_src_indices_noflat(self):

        p = om.Problem()

        p.model.add_subsystem('indep', om.IndepVarComp('x', np.arange(12).reshape((4, 3))))
        p.model.add_subsystem('C1', om.ExecComp('y=sum(x)*2.0', x=np.zeros((2, 2))))

        # connect C1.x to entries (0,0), (-1,1), (2,1), (1,1) of indep.x
        p.model.connect('indep.x', 'C1.x',
                        src_indices=[[0, -1, 2, 1],[0, 1, 1, 1]], flat_src_indices=False)

        p.setup()
        p.run_model()

        assert_near_equal(p['indep.x'], np.array([[0., 1., 2.],
                                                  [3., 4., 5.],
                                                  [6., 7., 8.],
                                                  [9., 10., 11.]]))

        assert_near_equal(p['C1.x'], np.array([[0., 10.],
                                               [7., 4.]]))

        assert_near_equal(p['C1.y'], 42.)


class TestFeatureSrcIndices(unittest.TestCase):

    def test_promote_src_indices(self):

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

        assert_near_equal(p.get_val('C1.x'), np.ones(3))
        assert_near_equal(p.get_val('C1.y'), 6.)
        assert_near_equal(p.get_val('C2.x'), np.ones(2))
        assert_near_equal(p.get_val('C2.y'), 8.)

    def test_promote_src_indices_nonflat(self):

        class MyComp(om.ExplicitComponent):
            def setup(self):
                # We want to pull the following 4 values out of the source:
                # [(0,0), (3,1), (2,1), (1,1)].
                self.add_input('x', np.ones((2, 2)),
                               src_indices=[[0,3,2,1],[0,1,1,1]],
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

        assert_near_equal(p.get_val('C1.x'),
                         np.array([[0., 10.],
                                   [7., 4.]]))
        assert_near_equal(p.get_val('C1.y'), 21.)

    def test_group_promotes_src_indices(self):

        class MyComp1(om.ExplicitComponent):
            """ multiplies input array by 2. """
            def setup(self):
                self.add_input('x', np.ones(3))
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*2.0

        class MyComp2(om.ExplicitComponent):
            """ multiplies input array by 4. """
            def setup(self):
                self.add_input('x', np.ones(2))
                self.add_output('y', 1.0)

            def compute(self, inputs, outputs):
                outputs['y'] = np.sum(inputs['x'])*4.0

        class MyGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp1', MyComp1())
                self.add_subsystem('comp2', MyComp2())

            def configure(self):
                # splits input via promotes using src_indices
                self.promotes('comp1', inputs=['x'], src_indices=[0, 1, 2])
                self.promotes('comp2', inputs=['x'], src_indices=[3, 4])

        p = om.Problem()

        p.model.set_input_defaults('x', np.ones(5))
        p.model.add_subsystem('G1', MyGroup(), promotes_inputs=['x'])

        p.setup()
        inp = np.random.random(5)
        p.set_val('x', inp)
        p.run_model()

        assert_near_equal(p.get_val('G1.comp1.x'), inp[:3])
        assert_near_equal(p.get_val('G1.comp2.x'), inp[3:])
        assert_near_equal(p.get_val('G1.comp1.y'), np.sum(inp[:3]*2))
        assert_near_equal(p.get_val('G1.comp2.y'), np.sum(inp[3:]*4))


class TestFeatureSetOrder(unittest.TestCase):

    def test_set_order(self):

        class ReportOrderComp(om.ExplicitComponent):
            """Adds name to list."""

            def __init__(self, order_list):
                super().__init__()
                self._order_list = order_list

            def compute(self, inputs, outputs):
                self._order_list.append(self.pathname)

        # this list will record the execution order of our C1, C2, and C3 components
        order_list = []

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('C1', ReportOrderComp(order_list))
        model.add_subsystem('C2', ReportOrderComp(order_list))
        model.add_subsystem('C3', ReportOrderComp(order_list))

        prob.setup()
        prob.run_model()

        self.assertEqual(order_list, ['C1', 'C2', 'C3'])

        # reset the shared order list
        order_list[:] = []

        prob.setup()
        # now swap C2 and C1 in the order
        model.set_order(['C2', 'C1', 'C3'])

        # after changing the order, we must call setup again
        prob.setup()
        prob.run_model()

        self.assertEqual(order_list, ['C2', 'C1', 'C3'])


class TestFeatureGetSubsystem(unittest.TestCase):

    def test_group_getsystem_top(self):

        p = om.Problem(model=BranchGroup())
        p.setup()

        c1 = p.model.Branch1.G1.G2.comp1
        self.assertEqual(c1.pathname, 'Branch1.G1.G2.comp1')

        c2 = p.model.Branch2.G3.comp2
        self.assertEqual(c2.pathname, 'Branch2.G3.comp2')


class TestFeatureConfigure(unittest.TestCase):

    def test_system_configure(self):

        class ImplSimple(om.ImplicitComponent):

            def setup(self):
                self.add_input('a', val=1.)
                self.add_output('x', val=0.)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = np.exp(outputs['x']) - \
                    inputs['a']**2 * outputs['x']**2

            def linearize(self, inputs, outputs, jacobian):
                jacobian['x', 'x'] = np.exp(outputs['x']) - \
                    2 * inputs['a']**2 * outputs['x']
                jacobian['x', 'a'] = -2 * inputs['a'] * outputs['x']**2

        class Sub(om.Group):
            def setup(self):
                self.add_subsystem('comp', ImplSimple())

            def configure(self):
                # This solver won't solve the system. We want
                # to override it in the parent.
                self.nonlinear_solver = om.NonlinearBlockGS()

        class Super(om.Group):
            def setup(self):
                self.add_subsystem('sub', Sub())

            def configure(self):
                # This will solve it.
                self.sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.sub.linear_solver = om.ScipyKrylov()

        top = om.Problem(model=Super())

        top.setup()

        self.assertTrue(isinstance(top.model.sub.nonlinear_solver, om.NewtonSolver))
        self.assertTrue(isinstance(top.model.sub.linear_solver, om.ScipyKrylov))

    def test_configure_set_input_defaults(self):
        class ConfigGroup(om.Group):
            def configure(self):
                self.set_input_defaults('x', val=99.)

        p = om.Problem(model=ConfigGroup())
        C1 = p.model.add_subsystem('C1', om.ExecComp('y=2*x'), promotes_inputs=['x'])
        C2 = p.model.add_subsystem('C2', om.ExecComp('y=3*x'), promotes_inputs=['x'])

        p.setup()
        self.assertEqual(p['x'], 99.)

    def test_configure_add_input_output(self):
        """
        A simple example to compute the resultant force on an aircraft using data
        from an external source. Demonstrates adding I/O in the 'configure' method.
        """

        class FlightDataComp(om.ExplicitComponent):
            """
            Simulate data generated by an external source/code
            """
            def setup(self):
                # number of points may not be known a priori
                n = 3

                # The vector represents forces at n time points (rows) in 2 dimensional plane (cols)
                self.add_output(name='thrust', shape=(n, 2), units='kN')
                self.add_output(name='drag', shape=(n, 2), units='kN')
                self.add_output(name='lift', shape=(n, 2), units='kN')
                self.add_output(name='weight', shape=(n, 2), units='kN')

            def compute(self, inputs, outputs):
                outputs['thrust'][:, 0] = [500, 600, 700]
                outputs['drag'][:, 0]  = [400, 400, 400]
                outputs['weight'][:, 1] = [1000, 1001, 1002]
                outputs['lift'][:, 1]  = [1000, 1000, 1000]


        class ForceModel(om.Group):
            def setup(self):
                self.add_subsystem('flightdatacomp', FlightDataComp(),
                                   promotes_outputs=['thrust', 'drag', 'lift', 'weight'])

                self.add_subsystem('totalforcecomp', om.AddSubtractComp())

            def configure(self):
                # Some models that require self-interrogation need to be able to add
                # I/O in components from the configure method of their containing groups.
                # In this case, we can only determine the 'vec_size' for totalforcecomp
                # after flightdatacomp has been setup.

                meta = self.flightdatacomp.get_io_metadata('output', includes='thrust')
                data_shape = meta['thrust']['shape']

                self.totalforcecomp.add_equation('total_force',
                                                 input_names=['thrust', 'drag', 'lift', 'weight'],
                                                 vec_size=data_shape[0], length=data_shape[1],
                                                 scaling_factors=[1, -1, 1, -1], units='kN')

                self.connect('thrust', 'totalforcecomp.thrust')
                self.connect('drag', 'totalforcecomp.drag')
                self.connect('lift', 'totalforcecomp.lift')
                self.connect('weight', 'totalforcecomp.weight')


        p = om.Problem(model=ForceModel())
        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('totalforcecomp.total_force', units='kN'),
                         np.array([[100, 200, 300], [0, -1, -2]]).T)

    def test_configure_add_input_output_list_io_group(self):
        """
        Like the example above but system we're calling list_outputs on is a Group.
        """

        class FlightDataComp(om.ExplicitComponent):
            """
            Simulate data generated by an external source/code
            """
            def setup(self):
                # number of points may not be known a priori
                n = 3

                # The vector represents forces at n time points (rows) in 2 dimensional plane (cols)
                self.add_output(name='thrust', shape=(n, 2), units='kN')
                self.add_output(name='drag', shape=(n, 2), units='kN')
                self.add_output(name='lift', shape=(n, 2), units='kN')
                self.add_output(name='weight', shape=(n, 2), units='kN')

            def compute(self, inputs, outputs):
                outputs['thrust'][:, 0] = [500, 600, 700]
                outputs['drag'][:, 0]  = [400, 400, 400]
                outputs['weight'][:, 1] = [1000, 1001, 1002]
                outputs['lift'][:, 1]  = [1000, 1000, 1000]


        class ForceModel(om.Group):
            def setup(self):
                fdgroup = om.Group()
                fdgroup.add_subsystem('flightdatacomp', FlightDataComp(),
                                      promotes_outputs=['thrust', 'drag', 'lift', 'weight'])
                self.add_subsystem('flightdatagroup', fdgroup,
                                   promotes_outputs=['thrust', 'drag', 'lift', 'weight'])

                self.add_subsystem('totalforcecomp', om.AddSubtractComp())

            def configure(self):
                # Some models that require self-interrogation need to be able to add
                # I/O in components from the configure method of their containing groups.
                # In this case, we can only determine the 'vec_size' for totalforcecomp
                # after flightdatagroup has been setup.

                flight_data = dict(self.flightdatagroup.list_outputs(shape=True, prom_name=True,
                                                                     out_stream=None))
                data_shape = flight_data['flightdatacomp.thrust']['shape']

                self.totalforcecomp.add_equation('total_force',
                                                 input_names=['thrust', 'drag', 'lift', 'weight'],
                                                 vec_size=data_shape[0], length=data_shape[1],
                                                 scaling_factors=[1, -1, 1, -1], units='kN')

                self.connect('thrust', 'totalforcecomp.thrust')
                self.connect('drag', 'totalforcecomp.drag')
                self.connect('lift', 'totalforcecomp.lift')
                self.connect('weight', 'totalforcecomp.weight')


        p = om.Problem(model=ForceModel())
        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('totalforcecomp.total_force', units='kN'),
                         np.array([[100, 200, 300], [0, -1, -2]]).T)

    def test_configure_dyn_shape(self):

        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape_by_conn=True, copy_shape='y')
                self.add_output('y', shape_by_conn=True, copy_shape='x')

            def compute(self, inputs, outputs):
                outputs['y'] = 3*inputs['x']

        class MyGroup(om.Group):
            def setup(self):
                self.add_subsystem('comp', MyComp())

            def configure(self):
                meta = self.comp.get_io_metadata('output', includes='y')

        p = om.Problem()
        p.model.add_subsystem("G", MyGroup())
        p.model.add_subsystem("sink", om.ExecComp('y=5*x'))
        p.model.connect('G.comp.y', 'sink.x')
        # this used to raise an exception
        p.setup()


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestConfigureMPI(unittest.TestCase):
    N_PROCS = 2

    def test_sorting_bug(self):
        class MyComp(om.ExplicitComponent):
            def __init__(self, count, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.count = count

            def setup(self):
                for i in range(self.count):
                    self.add_input(f"x{i+1}", np.ones(i + 1))
                    self.add_output(f"y{i+1}", np.ones(i + 1))

            def compute(self, inputs, outputs):
                pass

        class MyGroup(om.Group):
            def __init__(self, count, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.count = count

            def setup(self):
                for i in range(self.count):
                    self.add_subsystem(f"C{i+1}", MyComp(i + 1, distributed=True), promotes_inputs=['*'])

            def configure(self):
                for s in self._subsystems_myproc:
                    s.add_output('b')

        class MyGroupUpper(om.Group):
            def setup(self):
                self.add_subsystem("G1", MyGroup(1), promotes_inputs=['*'])
                self.add_subsystem("G2", MyGroup(2), promotes_inputs=['*'])

            def configure(self):
                for s in self._subsystems_myproc:
                    s.promotes('C1', inputs=['*'])

        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp(distributed=True))
        indep.add_output("x1", np.ones(1))
        indep.add_output("x2", np.ones(2))

        p.model.add_subsystem('G', MyGroupUpper())

        p.model.connect("indep.x1", "G.x1")
        p.model.connect("indep.x2", "G.x2")

        p.setup()


class TestFeatureGuessNonlinear(unittest.TestCase):

    def test_guess_nonlinear(self):

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
                # Check residuals
                if np.abs(residuals['x']) > 1.0E-2:
                    # inputs are addressed using full path name, regardless of promotion
                    external_input = inputs['comp1.external_input']

                    # balance drives x**2 = 2*external_input
                    x_guess = (2*external_input)**.5

                    # outputs are addressed by the their promoted names
                    outputs['x'] = x_guess # perfect guess should converge in 0 iterations

        p = om.Problem()

        p.model.add_subsystem('discipline', Discipline(), promotes_inputs=['external_input'])

        p.setup()
        p.set_val('external_input', 1.)
        p.run_model()

        self.assertEqual(p.model.nonlinear_solver._iter_count, 0)

        assert_near_equal(p.get_val('discipline.x'), 1.41421356, 1e-6)


class TestNaturalNaming(unittest.TestCase):

    def test_buried_proms(self):
        p = om.Problem()
        model = p.model
        g1 = model.add_subsystem('g1', om.Group())
        g2 = g1.add_subsystem('g2', om.Group(), promotes=['*'])
        g3 = g2.add_subsystem('g3', om.Group())
        g4 = g3.add_subsystem('g4', om.Group(), promotes=['*'])
        c1 = g4.add_subsystem('c1', om.ExecComp('y=2.0*x', x=7., y=9.), promotes=['x','y'])
        p.setup()

        full_in = 'g1.g2.g3.g4.c1.x'
        full_out = 'g1.g2.g3.g4.c1.y'

        prom_ins = ['g1.g2.g3.g4.x', 'g1.g2.g3.x', 'g1.g3.x']
        for prom in prom_ins:
            self.assertEqual(name2abs_names(model, prom), [full_in])

        prom_outs = ['g1.g2.g3.g4.y', 'g1.g2.g3.y', 'g1.g3.y']
        for prom in prom_outs:
            self.assertEqual(name2abs_names(model, prom), [full_out])

        # check setting/getting before final setup

        for name in prom_ins + [full_in]:
            self.assertEqual(p[name], 7.)

        self.assertEqual(g3.get_val('x', get_remote=True), 7.)

        # we allow 'g1.g3.x' here even though it isn't relative to g3,
        # because it maps to an absolute name that is contained in g3.
        self.assertEqual(g3.get_val('g1.g3.x', get_remote=True), 7.)

        for name in prom_outs + [full_out]:
            self.assertEqual(p[name], 9.)

        incount = 0
        for name in prom_ins + [full_in]:
            incount += 1
            p[name] = 77. + incount
            self.assertEqual(p[name], 77. + incount)

        outcount = 0
        for name in prom_outs + [full_out]:
            outcount += 1
            p[name] = 99. + outcount
            self.assertEqual(p[name], 99. + outcount)

        p.final_setup()

        # now check after final setup

        for name in prom_ins + [full_in]:
            self.assertEqual(p[name], 77. + incount)

        self.assertEqual(g3.get_val('x', get_remote=True), 77. + incount)

        for name in prom_outs + [full_out]:
            self.assertEqual(p[name], 99. + outcount)

        incount = 0
        for name in prom_ins + [full_in]:
            incount += 1
            p[name] = 7. + incount
            self.assertEqual(p[name], 7. + incount)

        outcount = 0
        for name in prom_outs + [full_out]:
            outcount += 1
            p[name] = 9. + outcount
            self.assertEqual(p[name], 9. + outcount)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestNaturalNamingMPI(unittest.TestCase):
    N_PROCS = 2

    def test_buried_proms(self):
        p = om.Problem()
        model = p.model
        par = model.add_subsystem('par', om.ParallelGroup())
        g1 = par.add_subsystem('g1', om.Group())
        g2 = g1.add_subsystem('g2', om.Group(), promotes=['*'])
        g3 = g2.add_subsystem('g3', om.Group())
        g4 = g3.add_subsystem('g4', om.Group(), promotes=['*'])
        c1 = g4.add_subsystem('c1', om.ExecComp('y=2.0*x', x=7., y=9.), promotes=['x','y'])

        g1a = par.add_subsystem('g1a', om.Group())
        g2a = g1a.add_subsystem('g2', om.Group(), promotes=['*'])
        g3a = g2a.add_subsystem('g3', om.Group())
        g4a = g3a.add_subsystem('g4', om.Group(), promotes=['*'])
        c1 = g4a.add_subsystem('c1', om.ExecComp('y=2.0*x', x=7., y=9.), promotes=['x','y'])

        p.setup()

        for gtop in ['par.g1', 'par.g1a']:
            full_in = f'{gtop}.g2.g3.g4.c1.x'
            full_out = f'{gtop}.g2.g3.g4.c1.y'

            prom_ins = [f'{gtop}.g2.g3.g4.x', f'{gtop}.g2.g3.x', f'{gtop}.g3.x']
            for prom in prom_ins:
                self.assertEqual(name2abs_names(model, prom), [full_in])

            prom_outs = [f'{gtop}.g2.g3.g4.y', f'{gtop}.g2.g3.y', f'{gtop}.g3.y']
            for prom in prom_outs:
                self.assertEqual(name2abs_names(model, prom), [full_out])

            # check setting/getting before final setup

            for name in prom_ins + [full_in]:
                self.assertEqual(p.get_val(name, get_remote=True), 7.)

            for name in prom_outs + [full_out]:
                self.assertEqual(p.get_val(name, get_remote=True), 9.)

            incount = 0
            for name in prom_ins + [full_in]:
                incount += 1
                p[name] = 77. + incount
                p.model.comm.barrier()
                self.assertEqual(p.get_val(name, get_remote=True), 77. + incount)

            outcount = 0
            for name in prom_outs + [full_out]:
                outcount += 1
                p[name] = 99. + outcount
                p.model.comm.barrier()
                self.assertEqual(p.get_val(name, get_remote=True), 99. + outcount)

        p.final_setup()

        # now check after final setup

        for gtop in ['par.g1', 'par.g1a']:
            full_in = f'{gtop}.g2.g3.g4.c1.x'
            full_out = f'{gtop}.g2.g3.g4.c1.y'

            for name in prom_ins + [full_in]:
                self.assertEqual(p.get_val(name, get_remote=True), 77. + incount)

            for name in prom_outs + [full_out]:
                self.assertEqual(p.get_val(name, get_remote=True), 99. + outcount)

        for gtop in ['par.g1', 'par.g1a']:
            full_in = f'{gtop}.g2.g3.g4.c1.x'
            full_out = f'{gtop}.g2.g3.g4.c1.y'

            incount = 0
            for name in prom_ins + [full_in]:
                incount += 1
                p[name] = 7. + incount
                p.model.comm.barrier()
                self.assertEqual(p.get_val(name, get_remote=True), 7. + incount)

            outcount = 0
            for name in prom_outs + [full_out]:
                outcount += 1
                p[name] = 9. + outcount
                p.model.comm.barrier()
                self.assertEqual(p.get_val(name, get_remote=True), 9. + outcount)

        self.assertEqual(set(p.model._vars_to_gather),
                         {'par.g1.g2.g3.g4.c1.x', 'par.g1a.g2.g3.g4.c1.x', 'par.g1.g2.g3.g4.c1.y', 'par.g1a.g2.g3.g4.c1.y'})


class TestConfigureUpdate(unittest.TestCase):

    def setUp(self):
        class Middle(om.ParallelGroup):

            def setup(self):
                b = self.add_subsystem("bottom1", om.ExplicitComponent())
                b.add_output("a", val=1, tags=["my_tag"])
                b = self.add_subsystem("bottom2", om.ExplicitComponent())
                b.add_output("a", val=1, tags=["my_tag"])

            def configure(self):
                self.bottom1.add_output("b", val=1, tags=["my_tag"])
                self.bottom2.add_output("b", val=1, tags=["my_tag"])

            def nom_get_metadata(self):
                for system in self._subsystems_myproc:
                    metadata = system.get_io_metadata(iotypes=["output"], metadata_keys=["tags"], tags="my_tag", get_remote=True)
                    assert(list(metadata.keys()) == ['a', 'b', 'c'])

                metadata = self.get_io_metadata(iotypes=["output"], metadata_keys=["tags"], tags="my_tag", get_remote=True)
                assert(list(metadata.keys()) == ['bottom1.a', 'bottom1.b', 'bottom1.c',
                                                 'bottom2.a', 'bottom2.b', 'bottom2.c'])

            def nom_add_output(self):
                self.bottom1.add_output("c", val=1, tags=["my_tag"])
                self.bottom2.add_output("c", val=1, tags=["my_tag"])

        class Top(om.Group):

            def setup(self):
                self.add_subsystem("middle", Middle())

            def configure(self):
                self.middle.nom_add_output()
                self.middle.nom_get_metadata()

        p = om.Problem()
        p.model = Top()
        self.problem = p

    def test_config_update(self):
        p = self.problem
        p.setup()


class TestConfigureUpdateMPI(TestConfigureUpdate):
    """
    Runs test in TestConfigureUpdate with 2 procs to make sure the metadata is gathered correctly.
    """
    N_PROCS = 2


if __name__ == "__main__":
    unittest.main()
