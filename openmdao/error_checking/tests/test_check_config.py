import errno
import os
import unittest
from tempfile import mkdtemp
from shutil import rmtree

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2
from openmdao.error_checking.check_config import get_sccs_topo
from openmdao.utils.assert_utils import assert_warning, assert_no_warning
from openmdao.utils.logger_utils import TestLogger


class MyComp(om.ExecComp):
    def __init__(self):
        super().__init__(["y = 2.0*a", "z = 3.0*b"])


class TestCheckConfig(unittest.TestCase):

    def test_dataflow_1_level(self):
        p = om.Problem()
        root = p.model

        root.add_subsystem("indep", om.IndepVarComp('x', 1.0))
        root.add_subsystem("C1", MyComp())
        root.add_subsystem("C2", MyComp())
        root.add_subsystem("C3", MyComp())
        root.add_subsystem("C4", MyComp())

        root.connect("C4.y", "C2.a")
        root.connect("C4.y", "C3.a")
        root.connect("C2.y", "C1.a")
        root.connect("C1.y", "C4.a")

        # make sure no system has dangling inputs so we avoid that warning
        root.connect("indep.x", "C1.b")
        root.connect("indep.x", "C2.b")
        root.connect("indep.x", "C3.b")
        root.connect("indep.x", "C4.b")

        # set iterative solvers since we have cycles
        root.linear_solver = om.LinearBlockGS()
        root.nonlinear_solver = om.NonlinearBlockGS()

        testlogger = TestLogger()
        p.setup(check=['cycles', 'out_of_order'], logger=testlogger)
        p.final_setup()

        expected_info = (
            "The following groups contain cycles:\n"
            "   Group '' has the following cycles: [['C1', 'C2', 'C4']]\n"
        )

        expected_warning = (
            "The following systems are executed out-of-order:\n"
            "   System 'C3' executes out-of-order with respect to its source systems ['C4']\n"
        )

        testlogger.find_in('info', expected_info)
        testlogger.find_in('warning', expected_warning)

    def test_parallel_group_order(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 1.0))
        model.add_subsystem('p2', om.IndepVarComp('x', 1.0))

        parallel = model.add_subsystem('parallel', om.ParallelGroup())
        parallel.add_subsystem('c1', om.ExecComp(['y=-2.0*x']))
        parallel.add_subsystem('c2', om.ExecComp(['y=5.0*x']))
        parallel.connect('c1.y', 'c2.x')

        parallel = model.add_subsystem('parallel_copy', om.ParallelGroup())
        parallel.add_subsystem('comp1', om.ExecComp(['y=-2.0*x']))
        parallel.add_subsystem('comp2', om.ExecComp(['y=5.0*x']))
        parallel.connect('comp1.y', 'comp2.x')

        model.add_subsystem('c3', om.ExecComp(['y=3.0*x1+7.0*x2']))
        model.add_subsystem('c4', om.ExecComp(['y=3.0*x_copy_1+7.0*x_copy_2']))

        model.connect("parallel.c1.y", "c3.x1")
        model.connect("parallel.c2.y", "c3.x2")
        model.connect("parallel_copy.comp1.y", "c4.x_copy_1")
        model.connect("parallel_copy.comp2.y", "c4.x_copy_2")

        model.connect("p1.x", "parallel.c1.x")
        model.connect("p1.x", "parallel_copy.comp1.x")

        testlogger = TestLogger()
        prob.setup(check=True, mode='fwd', logger=testlogger)

        msg = "Need to attach NonlinearBlockJac, NewtonSolver, or BroydenSolver to 'parallel' when " \
              "connecting components inside parallel groups"

        with assert_warning(UserWarning, msg):
            prob.run_model()

        expected_warning = ("The following systems are executed out-of-order:\n"
                            "   System 'parallel.c2' executes out-of-order with respect to its source systems ['parallel.c1']\n"
                            "   System 'parallel_copy.comp2' executes out-of-order with respect to its source systems ['parallel_copy.comp1']\n")

        testlogger.find_in('warning', expected_warning)

    def test_serial_in_parallel(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 1.0))

        parallel = model.add_subsystem('parallel', om.ParallelGroup())
        parallel.add_subsystem('c1', om.ExecComp(['y=-2.0*x']))

        parallel2 = model.add_subsystem('parallel_copy', om.ParallelGroup())
        parallel2.add_subsystem('comp1', om.ExecComp(['y=-2.0*x']))

        model.add_subsystem('con', om.ExecComp('y = 3.0*x'))

        model.connect("p1.x", "parallel.c1.x")
        model.connect('parallel.c1.y', 'parallel_copy.comp1.x')
        model.connect('parallel_copy.comp1.y', 'con.x')

        prob.setup(check=True)

        msg = ("The following systems are executed out-of-order:\n"
               "   System 'parallel.c2' executes out-of-order with respect to its source systems ['parallel.c1']\n")

        with assert_no_warning(UserWarning, msg):
            prob.run_model()

    def test_single_parallel_group_order(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 1.0))
        model.add_subsystem('p2', om.IndepVarComp('x', 1.0))

        parallel = model.add_subsystem('parallel', om.ParallelGroup())
        parallel.add_subsystem('c1', om.ExecComp(['y=-2.0*x']))
        parallel.add_subsystem('c2', om.ExecComp(['y=5.0*x']))
        parallel.connect('c1.y', 'c2.x')

        model.add_subsystem('c3', om.ExecComp(['y=3.0*x1+7.0*x2']))

        model.connect("parallel.c1.y", "c3.x1")
        model.connect("parallel.c2.y", "c3.x2")

        model.connect("p1.x", "parallel.c1.x")

        testlogger = TestLogger()
        prob.setup(check=True, mode='fwd', logger=testlogger)

        msg = "Need to attach NonlinearBlockJac, NewtonSolver, or BroydenSolver to 'parallel' when " \
              "connecting components inside parallel groups"

        with assert_warning(UserWarning, msg):
            prob.run_model()

        expected_warning = ("The following systems are executed out-of-order:\n"
                            "   System 'parallel.c2' executes out-of-order with respect to its source systems ['parallel.c1']\n")

        testlogger.find_in('warning', expected_warning)

    def test_no_connect_parallel_group(self):
        prob = om.Problem()
        model = prob.model

        traj = model.add_subsystem('traj', om.ParallelGroup())

        burn1 = traj.add_subsystem('burn1', om.Group())
        burn1.add_subsystem('p1', om.IndepVarComp('x', 1.0))
        burn1.add_subsystem('burn_eq1', om.ExecComp(['y=-2.0*x']))
        burn1.connect('p1.x', 'burn_eq1.x')

        burn2 = traj.add_subsystem('burn2', om.Group())
        burn2.add_subsystem('p2', om.IndepVarComp('x', 1.0))
        burn2.add_subsystem('burn_eq2', om.ExecComp(['y=5.0*x']))
        burn2.connect('p2.x', 'burn_eq2.x')

        testlogger = TestLogger()
        prob.setup(check=True, mode='fwd', logger=testlogger)

        msg = "Need to attach NonlinearBlockJac, NewtonSolver, or BroydenSolver to 'parallel' when " \
              "connecting components inside parallel groups"

        with assert_no_warning(UserWarning, msg):
            prob.run_model()

    def test_dataflow_multi_level(self):
        p = om.Problem()
        root = p.model

        root.add_subsystem("indep", om.IndepVarComp('x', 1.0))

        G1 = root.add_subsystem("G1", om.Group())

        G1.add_subsystem("C1", MyComp())
        G1.add_subsystem("C2", MyComp())

        root.add_subsystem("C3", MyComp())
        root.add_subsystem("C4", MyComp())

        root.connect("C4.y", "G1.C2.a")
        root.connect("C4.y", "C3.a")
        root.connect("G1.C2.y", "G1.C1.a")
        root.connect("G1.C1.y", "C4.a")

        # make sure no system has dangling inputs so we avoid that warning
        root.connect("indep.x", "G1.C1.b")
        root.connect("indep.x", "G1.C2.b")
        root.connect("indep.x", "C3.b")
        root.connect("indep.x", "C4.b")

        # set iterative solvers since we have cycles
        root.linear_solver = om.LinearBlockGS()
        root.nonlinear_solver = om.NonlinearBlockGS()

        testlogger = TestLogger()
        p.setup(check=['cycles', 'out_of_order'], logger=testlogger)
        p.final_setup()

        expected_info = (
            "The following groups contain cycles:\n"
            "   Group '' has the following cycles: [['G1', 'C4']]\n"
        )

        expected_warning = (
            "The following systems are executed out-of-order:\n"
            "   System 'C3' executes out-of-order with respect to its source systems ['C4']\n"
            "   System 'G1.C1' executes out-of-order with respect to its source systems ['G1.C2']\n"
        )

        testlogger.find_in('info', expected_info)
        testlogger.find_in('warning', expected_warning)

        # test comps_only cycle check
        graph = root.compute_sys_graph(comps_only=True)
        sccs = [sorted(s) for s in get_sccs_topo(graph) if len(s) > 1]
        self.assertEqual([['C4', 'G1.C1', 'G1.C2']], sccs)

    def test_out_of_order_repeat_bug_and_dup_inputs(self):
        p = om.Problem()
        p.model.add_subsystem("indep", om.IndepVarComp('x', 1.0))
        p.model.add_subsystem("C1", om.ExecComp(["y = 2.0*a", "z = 3.0*b"]))
        p.model.add_subsystem("C2", om.ExecComp("y = 2.0*a"))

        # create 2 out of order connections from C2 to C1
        p.model.connect("C2.y", "C1.a")
        p.model.connect("C2.y", "C1.b")

        # make sure no system has dangling inputs so we avoid that warning
        p.model.connect("indep.x", "C2.a")

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        expected_warning_1 = (
            "The following systems are executed out-of-order:\n"
            "   System 'C1' executes out-of-order with respect to its source systems ['C2']\n"
        )

        expected_warning_2 = (
            "The following components have multiple inputs connected to the same source, "
            "which can introduce unnecessary data transfer overhead:\n"
            "   C1 has inputs ['a', 'b'] connected to C2.y\n"
        )

        testlogger.find_in('warning', expected_warning_1)
        testlogger.find_in('warning', expected_warning_2)

    def test_multi_cycles(self):
        p = om.Problem()
        root = p.model

        root.add_subsystem("indep", om.IndepVarComp('x', 1.0))

        def make_cycle(root, start, end):
            # systems within a cycle will be declared out of order, but
            # should not be reported since they're internal to a cycle.
            for i in range(end, start-1, -1):
                root.add_subsystem("C%d" % i, MyComp())

            for i in range(start, end):
                root.connect("C%d.y" % i, "C%d.a" % (i+1))
            root.connect("C%d.y" % end, "C%d.a" % start)

        G1 = root.add_subsystem('G1', om.Group())

        make_cycle(G1, 1, 3)

        G1.add_subsystem("N1", MyComp())

        make_cycle(G1, 11, 13)

        G1.add_subsystem("N2", MyComp())

        make_cycle(G1, 21, 23)

        G1.add_subsystem("N3", MyComp())

        G1.connect("N1.z", "C12.b")
        G1.connect("C13.z", "N2.b")
        G1.connect("N2.z", "C21.b")
        G1.connect("C23.z", "N3.b")
        G1.connect("N3.z", "C2.b")
        G1.connect("C11.z", "C3.b")

        # set iterative solvers since we have cycles
        root.linear_solver = om.LinearBlockGS()
        root.nonlinear_solver = om.NonlinearBlockGS()

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        expected_warning_1 = (
            "The following systems are executed out-of-order:\n"
            "   System 'G1.C2' executes out-of-order with respect to its source systems ['G1.N3']\n"
            "   System 'G1.C3' executes out-of-order with respect to its source systems ['G1.C11']\n"
        )

        testlogger.find_in('warning', expected_warning_1)

    def test_multi_cycles_non_default(self):
        p = om.Problem()
        root = p.model

        root.add_subsystem("indep", om.IndepVarComp('x', 1.0))

        def make_cycle(root, start, end):
            # systems within a cycle will be declared out of order, but
            # should not be reported since they're internal to a cycle.
            for i in range(end, start-1, -1):
                root.add_subsystem("C%d" % i, MyComp())

            for i in range(start, end):
                root.connect("C%d.y" % i, "C%d.a" % (i+1))
            root.connect("C%d.y" % end, "C%d.a" % start)

        G1 = root.add_subsystem('G1', om.Group())

        make_cycle(G1, 1, 3)

        G1.add_subsystem("N1", MyComp())

        make_cycle(G1, 11, 13)

        G1.add_subsystem("N2", MyComp())

        make_cycle(G1, 21, 23)

        G1.add_subsystem("N3", MyComp())

        G1.connect("N1.z", "C12.b")
        G1.connect("C13.z", "N2.b")
        G1.connect("N2.z", "C21.b")
        G1.connect("C23.z", "N3.b")
        G1.connect("N3.z", "C2.b")
        G1.connect("C11.z", "C3.b")

        # set iterative solvers since we have cycles
        root.linear_solver = om.LinearBlockGS()
        root.nonlinear_solver = om.NonlinearBlockGS()

        testlogger = TestLogger()
        p.setup(check=['cycles', 'out_of_order', 'unconnected_inputs'], logger=testlogger)
        p.final_setup()

        expected_info = (
            "The following groups contain cycles:\n"
            "   Group 'G1' has the following cycles: "
            "[['C13', 'C12', 'C11'], ['C23', 'C22', 'C21'], ['C3', 'C2', 'C1']]\n"
        )

        expected_warning_1 = (
            "The following systems are executed out-of-order:\n"
            "   System 'G1.C2' executes out-of-order with respect to its source systems ['G1.N3']\n"
            "   System 'G1.C3' executes out-of-order with respect to its source systems ['G1.C11']\n"
        )

        testlogger.find_in('info', expected_info)
        testlogger.find_in('warning', expected_warning_1)

    def test_unconnected_auto_ivc(self):
        class SellarMDAConnect(om.Group):

            def setup(self):
                cycle = self.add_subsystem('cycle', om.Group(), promotes_inputs=['x', 'z'])
                cycle.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z'])
                cycle.add_subsystem('d2', SellarDis2(), promotes_inputs=['z'])
                cycle.connect('d1.y1', 'd2.y1')

                cycle.nonlinear_solver = om.NonlinearBlockGS()

                self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                          z=np.array([0.0, 0.0]), x=0.0),
                                   promotes_inputs=['x', 'z'])

                self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
                self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

                self.connect('cycle.d1.y1', ['obj_cmp.y1', 'con_cmp1.y1'])
                self.connect('cycle.d2.y2', ['obj_cmp.y2', 'con_cmp2.y2'])

        prob = om.Problem(model=SellarMDAConnect())

        prob.model.add_design_var('z', lower=0, upper=10)
        prob.model.add_objective('obj_cmp.obj')
        prob.model.add_constraint('con_cmp1.con1', upper=0)
        prob.model.add_constraint('con_cmp2.con2', upper=0)

        testlogger = TestLogger()
        prob.setup(check=['unconnected_inputs'], logger=testlogger)
        prob.final_setup()

        expected_warning = (
            "The following inputs are not connected:\n"
            "  cycle.d1.y2 (cycle.d1.y2)\n"
            "  x (cycle.d1.x)\n"
            "  x (obj_cmp.x)\n"
        )

        testlogger.find_in('warning', expected_warning)

    def test_comp_has_no_outputs(self):
        p = om.Problem()
        root = p.model

        root.add_subsystem("indep", om.IndepVarComp('x', 1.0))

        comp1 = root.add_subsystem("comp1", om.ExplicitComponent())
        comp1.add_input('x', val=0.)

        root.connect('indep.x', 'comp1.x')

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        expected = (
            "The following Components do not have any outputs:\n"
            "   comp1\n"
        )

        testlogger.find_in('warning', expected)

    def test_initial_condition_order(self):
        # Makes sure we set vars to their initial condition before running checks.

        class TestComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', 37.0)
                self.add_output('y', 45.0)

            def check_config(self, logger):
                x = self._vectors['input']['nonlinear']['x']
                if x != 75.0:
                    raise ValueError('Check config is being called before initial conditions are set.')

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                outputs['y'] = inputs['x']

        prob = om.Problem()
        prob.model.add_subsystem('comp', TestComp())

        prob.setup(check='all')

        prob['comp.x'] = 75.0

        prob.final_setup()


class TestRecorderCheckConfig(unittest.TestCase):

    def setUp(self):
        self.orig_dir = os.getcwd()
        self.temp_dir = mkdtemp()
        os.chdir(self.temp_dir)

        self.filename = os.path.join(self.temp_dir, "sqlite_test")
        self.recorder = om.SqliteRecorder(self.filename)

    def tearDown(self):
        os.chdir(self.orig_dir)
        try:
            rmtree(self.temp_dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_check_no_recorder_set(self):
        p = om.Problem()

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        expected_warning = "The Problem has no recorder of any kind attached"
        testlogger.find_in('warning', expected_warning)

    def test_check_driver_recorder_set(self):
        p = om.Problem()
        p.driver.add_recorder(self.recorder)

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)

    def test_check_system_recorder_set(self):
        p = om.Problem()
        p.model.add_recorder(self.recorder)

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)

    def test_check_linear_solver_recorder_set(self):
        p = om.Problem()
        p.model.nonlinear_solver.add_recorder(self.recorder)

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()
        p.cleanup()

        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)


if __name__ == "__main__":
    unittest.main()
