
import os
import unittest
import subprocess

from openmdao.utils.testing_utils import use_tempdirs
import openmdao.core.tests.test_coloring as coloring_test_mod

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

dname = os.path.dirname

scriptdir = os.path.join(dname(dname(dname(os.path.abspath(__file__)))), 'test_suite', 'scripts')

counter = 0

def _test_func_name(func, num, param):
    return func.__name__ + '_' + '_'.join(param.args[0].split()[1:-1])

cmd_tests = [
    'openmdao call_tree openmdao.components.exec_comp.ExecComp.setup',
    'openmdao check {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao cite {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao compute_entry_points openmdao',
    'openmdao iprof --no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao iprof_totals {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao list_installed component command nl_solver lin_solver driver',
    'openmdao list_installed component -d',
    'openmdao n2 --no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao n2 --no_browser {}'.format(os.path.join(scriptdir, 'bad_connection.py')),
    'openmdao n2 --no_browser {} -- -f bar'.format(os.path.join(scriptdir, 'circle_coloring_needs_args.py')),
    'openmdao partial_coloring {}'.format(os.path.join(scriptdir, 'circle_coloring_dynpartials.py')),
    'openmdao scaffold -b ExplicitComponent -c Foo',
    'openmdao scaffold -b ImplicitComponent -c Foo',
    'openmdao scaffold -p blahpkg --cmd=hello',
    'openmdao summary {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao total_coloring {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao trace {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao tree -c {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao view_connections --no_browser {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
]


mem_cmd_tests = [
    'openmdao mem {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao mem --tree {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
    'openmdao trace -m {}'.format(os.path.join(scriptdir, 'circle_opt.py')),
]


try:
    import psutil
except ImportError:
    psutil = None


@use_tempdirs
class CmdlineTestCase(unittest.TestCase):
    @parameterized.expand(cmd_tests, name_func=_test_func_name)
    def test_cmd(self, cmd):
        # this only tests that a given command line tool returns a 0 return code. It doesn't
        # check the expected output at all.  The underlying functions that implement the
        # commands should be tested seperately.
        try:
            output = subprocess.check_output(cmd.split())
        except subprocess.CalledProcessError as err:
            self.fail("Command '{}' failed.  Return code: {}".format(cmd, err.returncode))

    @parameterized.expand(mem_cmd_tests, name_func=_test_func_name)
    @unittest.skipIf(psutil is None, 'psutil must be installed to run mem commands')
    def test_cmd2(self, cmd):
        # this only tests that a given command line tool returns a 0 return code. It doesn't
        # check the expected output at all.  The underlying functions that implement the
        # commands should be tested seperately.
        try:
            output = subprocess.check_output(cmd.split())
        except subprocess.CalledProcessError as err:
            self.fail("Command '{}' failed.  Return code: {}".format(cmd, err.returncode))


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
            output = subprocess.check_output(cmd.split())
        except subprocess.CalledProcessError as err:
            self.fail("Command '{}' failed.  Return code: {}".format(cmd, err.returncode))


if __name__ == '__main__':
    unittest.main()
