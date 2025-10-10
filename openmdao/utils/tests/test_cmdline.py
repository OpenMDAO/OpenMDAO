
import os
import sys
import unittest
import subprocess
import re

from openmdao.utils.testing_utils import use_tempdirs
import openmdao.core.tests.test_coloring as coloring_test_mod

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

try:
    import psutil
except ImportError:
    psutil = None

try:
    import matplotlib
except ImportError:
    matplotlib = None

try:
    import tornado
except ImportError:
    tornado = None

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

try:
    import pydot
except ImportError:
    pydot = None

try:
    import graphviz
except ImportError:
    graphviz = None


dname = os.path.dirname

scriptdir = os.path.join(dname(dname(dname(os.path.abspath(__file__)))), 'test_suite', 'scripts')

counter = 0

def _test_func_name(func, num, param):
    # test name is the command with spaces, colons and backslashes replaced by underscore
    return func.__name__ + '_' + re.sub(r'[ \:\\]', '_', param.args[0])

cmd_tests = [
    # tuple of (command line, dict of dependencies that might not be installed)
    ('python -m openmdao -h', {}),
    ('openmdao --help', {}),
    ('openmdao --view_reports {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao call_tree openmdao.components.exec_comp.ExecComp.setup', {}),
    ('openmdao check {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao comm_info {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao cite {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao clean --dryrun {}'.format(scriptdir), {}),
    ('python -m openmdao clean --dryrun {}'.format(scriptdir), {}),
    ('openmdao compute_entry_points openmdao', {}),
    ('openmdao graph --no-display {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --type=tree {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('python -m openmdao graph --no-display --show-vars {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --show-vars --no-recurse {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --group=circuit {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --group=circuit --show-vars {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --group=circuit --show-vars --no-recurse {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao iprof --no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
        {'tornado': tornado}),
    ('openmdao iprof_totals {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao list_installed component command nl_solver lin_solver driver', {}),
    ('python -m openmdao list_installed component -d', {}),
    ('openmdao list_pre_post {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao n2 --no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao n2 --no_browser {} -- -f bar'.format(os.path.join(scriptdir, 'circle_coloring_needs_args.py')), {}),
    ('openmdao partial_coloring {}'.format(os.path.join(scriptdir, 'circle_coloring_dynpartials.py')), {}),
    ('openmdao rtplot {}'.format(os.path.join(scriptdir, 'circle_opt_with_driver_recording.py')), {}),
    ('openmdao scaffold -b ExplicitComponent -c Foo', {}),
    ('python -m openmdao scaffold -b ImplicitComponent -c Foo', {}),
    ('openmdao scaffold -p blahpkg --cmd=hello', {}),
    ('openmdao scaling --no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao summary {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao timing -v no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao total_coloring {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao trace {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao tree -c {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao view_connections --no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao view_dyn_shapes --no_display {}'.format(os.path.join(scriptdir, 'dyn_system.py')),
        {'matplotlib': matplotlib}),
    ('openmdao mem {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
        {'psutil': psutil}),
    ('openmdao mem --tree {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
        {'psutil': psutil}),
    ('openmdao trace -m {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
        {'psutil': psutil})
]

@use_tempdirs
class CmdlineTestCase(unittest.TestCase):
    @parameterized.expand(cmd_tests, name_func=_test_func_name)
    def test_cmd(self, cmd, dependencies):
        # skip any commands for which we do not have required dependencies
        not_installed = [n for n, inst in dependencies.items() if not inst]
        if not_installed:
            raise unittest.SkipTest(f"{not_installed} is not installed")

        # this only tests that a given command line tool returns a 0 return code. It doesn't
        # check the expected output at all.  The underlying functions that implement the
        # commands should be tested seperately.
        try:
            subprocess.check_output(cmd.split(),
                                    stderr=subprocess.STDOUT)  # nosec: trusted input
        except subprocess.CalledProcessError as err:
            self.fail(f"Command '{cmd}' failed.  Return code: {err.returncode}: "
                      f"Output was: \n{err.output.decode('utf-8')}")

    @unittest.skipIf(sys.platform == 'win32', 'problematic on Windows due to the interaction between python and the OS')
    @unittest.skipIf(sys.platform == 'darwin', 'problematic on MacOS due to the interaction between python and the OS')
    def test_clean(self):
        import openmdao.api as om

        for om_cmd in ('openmdao', 'python -m openmdao'):

            with self.subTest('Test using command line `{om_cmd}`'):

                p1 = om.Problem()
                p1.model.add_subsystem('exec', om.ExecComp('y = a + b'))
                p1.setup()
                p1.run_model()

                p2 = om.Problem()
                p2.model.add_subsystem('exec', om.ExecComp('z = a * b'))
                p2.setup()
                p2.run_model()

                p1_outdir = os.path.basename(str(p1.get_outputs_dir(mkdir=True)))
                p2_outdir = os.path.basename(str(p2.get_outputs_dir(mkdir=True)))

                subdirs = os.listdir(os.getcwd())
                self.assertIn(p1_outdir, subdirs)
                self.assertIn(p2_outdir, subdirs)

                proc = subprocess.Popen(f'{om_cmd} clean -f'.split(),  # nosec: trusted input
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    outs, errs = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    outs, errs = proc.communicate()

                subdirs = os.listdir(os.getcwd())
                self.assertNotIn(p1_outdir, subdirs)
                self.assertNotIn(p2_outdir, subdirs)

    def test_outdir(self):
        env_vars = os.environ.copy()
        env_vars["OPENMDAO_REPORTS"] = "1"
        env_vars["TESTFLO_RUNNING"] = "0"

        cmd = f"openmdao {os.path.join(scriptdir, 'circle_opt.py')}"
        print('Command:', cmd)
        print('Current working dir:', os.getcwd())
        proc = subprocess.Popen(cmd.split(),  # nosec: trusted input
                                env=env_vars,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()

        print('Output')
        print('------')
        print(outs.decode())
        print('Errors')
        print('------')
        print(errs.decode())

        self.assertTrue(os.path.exists('circle_opt_out'))

    def test_n2_err(self):
        # command should raise exception but still produce an n2 html file
        cmd = f'openmdao n2 --no_browser {scriptdir}/bad_connection.py'
        workdir = os.getcwd()
        n2file = os.path.join(workdir, 'n2.html')
        if os.path.isfile(n2file):
            os.remove(n2file)
        proc = subprocess.Popen(cmd.split(),  # nosec: trusted input
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()

        if not os.path.isfile(n2file):
            self.fail(f"File '{n2file}' was not created: {errs}")

        lines = errs.splitlines()
        for i, line in enumerate(lines):
            if b"Collected errors for problem" in line:
                self.assertEqual(lines[i+1],
                    b"   'sub' <class Group>: Attempted to connect from 'tgt.x' to 'cmp.x', but 'tgt.x' is an input. All connections must be from an output to an input.")
                break
        else:
            self.fail("Didn't find expected err msg in output.")


class CmdlineTestCaseCheck(unittest.TestCase):
    def test_auto_ivc_warnings_check(self):
        cmd = 'openmdao check -c auto_ivc_warnings {}'.format(os.path.join(scriptdir, 'auto_ivc_warnings.py'))
        msg = "WARNING: Groups 'G1' and 'G1.G2' called set_input_defaults for the input 'x' with conflicting 'value'. The value (14.0) from 'G1' will be used."

        output = subprocess.Popen(cmd.split(),  # nosec: trusted input
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = output.communicate()
        for i in out.decode('utf-8').split("\n"):
            if "WARNING:" in i:
                self.assertEqual(i, msg)


col_test_file = coloring_test_mod.__file__

test_cmd_tests = [
    'openmdao tree {}:SimulColoringScipyTestCase.test_dynamic_total_coloring_auto'.format(col_test_file),
    'openmdao tree -c {}:SimulColoringScipyTestCase.test_dynamic_total_coloring_auto'.format(coloring_test_mod.__name__),
]

@use_tempdirs
class CmdlineTestfuncTestCase(unittest.TestCase):
    @parameterized.expand(test_cmd_tests, name_func=_test_func_name)
    def test_cmd(self, cmd):
        # this only tests that a given command line tool returns a 0 return code. It doesn't
        # check the expected output at all.  The underlying functions that implement the
        # commands should be tested seperately.
        try:
            subprocess.check_output(cmd.split())  # nosec: trusted input
        except subprocess.CalledProcessError as err:
            self.fail(f"Command '{cmd}' failed.  Return code: {err.returncode} "
                      f"Output was: \n{err.output.decode('utf-8')}")


test_cmd_err = [
    f"openmdao -scaling {os.path.join(scriptdir, 'circle_opt.py')}",
]

@use_tempdirs
class CmdlineTestErrTestCase(unittest.TestCase):
    @parameterized.expand(test_cmd_err, name_func=_test_func_name)
    def test_cmd(self, cmd):
        proc = subprocess.run(cmd.split(),   # nosec: trusted input
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding='UTF-8')
        if 'argument : invalid choice:' not in proc.stderr:
            self.fail(f"Command '{cmd}' didn't fail in the expected way.\n"
                      f"Return code: {proc.returncode}.\nstderr: {proc.stderr}\nstdout: {proc.stdout}")


if __name__ == '__main__':
    unittest.main()
