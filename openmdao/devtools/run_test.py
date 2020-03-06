"""
This allows you to run a specific test function by itself from an importable module.

The function can be either a method of a unittest.TestCase subclass, or just a function defined
at module level.  This is useful when running an individual test under mpirun for debugging
purposes.

To specify the test to run, use the following forms:

<modpath>:<testcase name>.<funcname>   OR   <modpath>:<funcname>

where <modpath> is either the dotted module name or the full filesystem path of the python file.

for example:

    mpirun -n 4 run_test mypackage.mysubpackage.mymod:MyTestCase.test_foo

    OR

    mpirun -n 4 run_test /foo/bar/mypackage/mypackage/mysubpackage/mymod.py:MyTestCase.test_foo
"""

import sys
import importlib
from openmdao.utils.file_utils import get_module_path


def run_test():
    """
    Run individual test(s).
    """

    sys.path.append('.')
    if len(sys.argv) > 1:
        testspec = sys.argv[1]
        parts = testspec.split(':')

    if len(sys.argv) != 2 or len(parts) != 2:
        print('Usage: run_test my_mod_path:my_test_case.test_func_name\n'
              '            OR\n'
              '       run_test my_mod_path:test_func_name')
        sys.exit(-1)

    modpath, funcpath = parts
    if modpath.endswith('.py'):
        modpath = get_module_path(modpath)

    mod = importlib.import_module(modpath)

    parts = funcpath.split('.', 1)
    if len(parts) == 2:
        tcase_name, method_name = parts
        testcase = getattr(mod, tcase_name)(methodName=method_name)
        setup = getattr(testcase, 'setUp', None)
        if setup is not None:
            setup()
        getattr(testcase, method_name)()
        teardown = getattr(testcase, 'tearDown', None)
        if teardown:
            teardown()
    else:
        funcname = parts[0]
        getattr(mod, funcname)()


if __name__ == '__main__':
    run_test()
