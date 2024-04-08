
import os
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
    return func.__name__ + '_' + re.sub('[ \\:\\\]', '_', param.args[0])

cmd_tests = [
    # tuple of (command line, dict of dependencies that might not be installed)
    ('openmdao call_tree openmdao.components.exec_comp.ExecComp.setup', {}),
    ('openmdao check {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao comm_info {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao cite {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao compute_entry_points openmdao', {}),
    ('openmdao graph --no-display {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --type=tree {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --show-vars {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --show-vars --no-recurse {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --group=circuit {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --group=circuit --show-vars {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao graph --no-display --group=circuit --show-vars --no-recurse {}'.format(os.path.join(scriptdir, 'circuit_analysis.py')), {'pydot': pydot, 'graphviz': graphviz}),
    ('openmdao iprof --no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
        {'tornado': tornado}),
    ('openmdao iprof_totals {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao list_installed component command nl_solver lin_solver driver', {}),
    ('openmdao list_installed component -d', {}),
    ('openmdao list_pre_post {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao n2 --no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')), {}),
    ('openmdao n2 --no_browser {} -- -f bar'.format(os.path.join(scriptdir, 'circle_coloring_needs_args.py')), {}),
    ('openmdao partial_coloring {}'.format(os.path.join(scriptdir, 'circle_coloring_dynpartials.py')), {}),
    ('openmdao scaffold -b ExplicitComponent -c Foo', {}),
    ('openmdao scaffold -b ImplicitComponent -c Foo', {}),
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
            output = subprocess.check_output(cmd.split(),
                                             stderr=subprocess.STDOUT)  # nosec: trusted input
        except subprocess.CalledProcessError as err:
            self.fail(f"Command '{cmd}' failed.  Return code: {err.returncode}: "
                      f"Output was: \n{err.output.decode('utf-8')}")

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
            output = subprocess.check_output(cmd.split())  # nosec: trusted input
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
