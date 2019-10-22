import errno
import os
import unittest
from tempfile import mkdtemp
from shutil import rmtree

from six.moves import range

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ExplicitComponent, \
    LinearBlockGS, NonlinearBlockGS, SqliteRecorder

from openmdao.utils.logger_utils import TestLogger
from openmdao.error_checking.check_config import get_sccs_topo


class MyComp(ExecComp):
    def __init__(self):
        super(MyComp, self).__init__(["y = 2.0*a", "z = 3.0*b"])


class TestCheckConfig(unittest.TestCase):

    def test_hanging_inputs(self):
        p = Problem()
        root = p.model

        G1 = root.add_subsystem("G1", Group(), promotes=['*'])
        G2 = G1.add_subsystem("G2", Group(), promotes=['*'])
        G2.add_subsystem("C2", IndepVarComp('x', 1.0), promotes=['*'])
        G2.add_subsystem("C1", ExecComp('y=x*2.0+w'), promotes=['*'])

        G3 = root.add_subsystem("G3", Group())
        G4 = G3.add_subsystem("G4", Group())
        G4.add_subsystem("C3", ExecComp('y=x*2.0+u'), promotes=['*'])
        G4.add_subsystem("C4", ExecComp('y=x*2.0+v'))

        testlogger = TestLogger()
        p.setup(check='all', logger=testlogger)
        p.final_setup()

        expected = (
            "The following inputs are not connected:\n"
            "   G3.G4.C4.v\n"
            "   G3.G4.C4.x\n"
            "   G3.G4.u: ['G3.G4.C3.u']\n"
            "   G3.G4.x: ['G3.G4.C3.x']\n"
            "   w: ['G1.G2.C1.w']\n"
        )

        self.assertTrue(testlogger.contains('warning', expected))

    def test_dataflow_1_level(self):
        p = Problem()
        root = p.model

        root.add_subsystem("indep", IndepVarComp('x', 1.0))
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
        root.linear_solver = LinearBlockGS()
        root.nonlinear_solver = NonlinearBlockGS()

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

        self.assertTrue(testlogger.contains('info', expected_info))
        self.assertTrue(testlogger.contains('warning', expected_warning))

    def test_dataflow_multi_level(self):
        p = Problem()
        root = p.model

        root.add_subsystem("indep", IndepVarComp('x', 1.0))

        G1 = root.add_subsystem("G1", Group())

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
        root.linear_solver = LinearBlockGS()
        root.nonlinear_solver = NonlinearBlockGS()

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

        self.assertTrue(testlogger.contains('info', expected_info))
        self.assertTrue(testlogger.contains('warning', expected_warning))

        # test comps_only cycle check
        graph = root.compute_sys_graph(comps_only=True)
        sccs = [sorted(s) for s in get_sccs_topo(graph) if len(s) > 1]
        self.assertEqual([['C4', 'G1.C1', 'G1.C2']], sccs)

    def test_out_of_order_repeat_bug_and_dup_inputs(self):
        p = Problem()
        p.model.add_subsystem("indep", IndepVarComp('x', 1.0))
        p.model.add_subsystem("C1", ExecComp(["y = 2.0*a", "z = 3.0*b"]))
        p.model.add_subsystem("C2", ExecComp("y = 2.0*a"))

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

        self.assertTrue(testlogger.contains('warning', expected_warning_1))
        self.assertTrue(testlogger.contains('warning', expected_warning_2))

    def test_multi_cycles(self):
        p = Problem()
        root = p.model

        root.add_subsystem("indep", IndepVarComp('x', 1.0))

        def make_cycle(root, start, end):
            # systems within a cycle will be declared out of order, but
            # should not be reported since they're internal to a cycle.
            for i in range(end, start-1, -1):
                root.add_subsystem("C%d" % i, MyComp())

            for i in range(start, end):
                root.connect("C%d.y" % i, "C%d.a" % (i+1))
            root.connect("C%d.y" % end, "C%d.a" % start)

        G1 = root.add_subsystem('G1', Group())

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
        root.linear_solver = LinearBlockGS()
        root.nonlinear_solver = NonlinearBlockGS()

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        expected_warning_1 = (
            "The following systems are executed out-of-order:\n"
            "   System 'G1.C2' executes out-of-order with respect to its source systems ['G1.N3']\n"
            "   System 'G1.C3' executes out-of-order with respect to its source systems ['G1.C11']\n"
        )

        self.assertTrue(testlogger.contains('warning', expected_warning_1))

    def test_multi_cycles_non_default(self):
        p = Problem()
        root = p.model

        root.add_subsystem("indep", IndepVarComp('x', 1.0))

        def make_cycle(root, start, end):
            # systems within a cycle will be declared out of order, but
            # should not be reported since they're internal to a cycle.
            for i in range(end, start-1, -1):
                root.add_subsystem("C%d" % i, MyComp())

            for i in range(start, end):
                root.connect("C%d.y" % i, "C%d.a" % (i+1))
            root.connect("C%d.y" % end, "C%d.a" % start)

        G1 = root.add_subsystem('G1', Group())

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
        root.linear_solver = LinearBlockGS()
        root.nonlinear_solver = NonlinearBlockGS()

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

        expected_warning_2 = (
            "The following inputs are not connected:\n"
            "   G1.C1.b\n"
            "   G1.C11.b\n"
            "   G1.C13.b\n"
            "   G1.C22.b\n"
            "   G1.C23.b\n"
            "   G1.N1.a\n"
            "   G1.N1.b\n"
            "   G1.N2.a\n"
            "   G1.N3.a\n"
        )

        self.assertTrue(testlogger.contains('info', expected_info))
        self.assertTrue(testlogger.contains('warning', expected_warning_1))
        self.assertTrue(testlogger.contains('warning', expected_warning_2))

    def test_comp_has_no_outputs(self):
        p = Problem()
        root = p.model

        root.add_subsystem("indep", IndepVarComp('x', 1.0))

        comp1 = root.add_subsystem("comp1", ExplicitComponent())
        comp1.add_input('x', val=0.)

        root.connect('indep.x', 'comp1.x')

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        expected = (
            "The following Components do not have any outputs:\n"
            "   comp1\n"
        )

        self.assertTrue(testlogger.contains('warning', expected))

    def test_initial_condition_order(self):
        # Makes sure we set vars to their initial condition before running checks.

        class TestComp(ExplicitComponent):

            def setup(self):
                self.add_input('x', 37.0)
                self.add_output('y', 45.0)

            def check_config(self, logger):
                x = self._vectors['input']['nonlinear']['x']
                if x != 75.0:
                    raise ValueError('Check config is being called before initial conditions are set.')

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                outputs['y'] = inputs['x']

        prob = Problem()
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
        self.recorder = SqliteRecorder(self.filename)

    def tearDown(self):
        os.chdir(self.orig_dir)
        try:
            rmtree(self.temp_dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_check_no_recorder_set(self):
        p = Problem()

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        expected_warning = "The Problem has no recorder of any kind attached"
        self.assertTrue(testlogger.contains('warning', expected_warning))

    def test_check_driver_recorder_set(self):
        p = Problem()
        p.driver.add_recorder(self.recorder)

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)

    def test_check_system_recorder_set(self):
        p = Problem()
        p.model.add_recorder(self.recorder)

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()

        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)

    def test_check_linear_solver_recorder_set(self):
        p = Problem()
        p.model.nonlinear_solver.add_recorder(self.recorder)

        testlogger = TestLogger()
        p.setup(check=True, logger=testlogger)
        p.final_setup()
        p.cleanup()

        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)


if __name__ == "__main__":
    unittest.main()
