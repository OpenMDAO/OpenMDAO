from __future__ import print_function

import unittest
import os.path
import importlib
import inspect
from six import PY3

from numpydoc.docscrape import NumpyDocString

directories = [
    'assemblers',
    'core',
    'jacobians',
    'matrices',
    'proc_allocators',
    'solvers',
    'utils',
    'vectors',
]


class LintTestCase(unittest.TestCase):

    def check_method(self, dir_name, file_name,
                     class_name, method_name, method):
        if PY3:
            argspec = inspect.getfullargspec(method)
        else:
            argspec = inspect.getargspec(method)
        doc = inspect.getdoc(method)

        fail_msg = '{0}, {1} : {2} {3}'.format(dir_name, file_name, class_name,
                                               method_name)
        # Check if docstring is missing
        if doc is None:
            self.fail(fail_msg + '... missing docstring')

        # Check if docstring references another method
        if doc[:3] == 'See':
            return

        nds = NumpyDocString(doc)

        if len(argspec.args) > 1:

            if not nds['Parameters']:
                self.fail(fail_msg + '... does not have a Parameters section')

            # Check formatting
            for entry in nds['Parameters']:
                name = entry[0]
                type_ = entry[1]
                desc = '\n'.join(entry[2])
                if ':' in name:
                    self.fail(fail_msg + '...colon after parameter '
                                         'name \'{0}\' and before type must '
                                         'be surrounded by '
                                         'spaces'.format(name.split(':')[0]))
                if type_ == '':
                    self.fail(fail_msg + '...no type info given for '
                                         '{0}'.format(name))
                if desc == '':
                    self.fail(fail_msg + '...no description given for '
                                         '{0}'.format(name))

            documented_arg_set = set(item[0] for item in nds['Parameters'])
            arg_set = set(argspec.args)

            # Require documentation of *args and **kwargs
            if argspec.varargs:
                arg_set |= set([argspec.varargs])
            if argspec.keywords:
                arg_set |= set([argspec.keywords])

            # Don't require documentation of self
            if 'self' in arg_set:
                arg_set.remove('self')

            # Arguments that aren't documented
            undocumented = arg_set - documented_arg_set
            if undocumented:
                self.fail(fail_msg + '... is missing documentation for: '
                                     '{0}'.format(str(list(undocumented))))

            # Arguments that are documented but don't exist
            overdocumented = documented_arg_set - arg_set
            if overdocumented:
                self.fail(fail_msg + '... documents nonexisting parameters: '
                                     '{0}'.format(str(list(overdocumented))))

    def test_docstrings(self):
        topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        print_info = False

        # Loop over directories
        for dir_name in directories:
            dirpath = os.path.join(topdir, dir_name)
            if print_info:
                print('-'*len(dir_name))
                print(dir_name)
                print('-'*len(dir_name))

            # Loop over files
            for file_name in os.listdir(dirpath):
                if file_name != '__init__.py' and file_name[-3:] == '.py':
                    if print_info:
                        print(file_name)

                    module_name = 'openmdao.%s.%s' % (dir_name, file_name[:-3])
                    try:
                        mod = importlib.import_module(module_name)
                    except ImportError as err:
                        if print_info:
                            print('Skipped:', err)
                        # e.g. PETSc is not installed
                        continue

                    # Loop over classes
                    classes = [x for x in dir(mod)
                               if inspect.isclass(getattr(mod, x)) and
                               getattr(mod, x).__module__ == module_name]
                    for class_name in classes:
                        if print_info:
                            print(' '*4, class_name)
                        clss = getattr(mod, class_name)

                        # skip namedtuples
                        if issubclass(clss, tuple):
                            continue

                        # Loop over methods
                        methods = [x for x in dir(clss)
                                   if inspect.ismethod(getattr(clss, x)) and
                                   x in clss.__dict__]
                        for method_name in methods:
                            if print_info:
                                print(' '*8, method_name)
                            method = getattr(clss, method_name)

                            self.check_method(dir_name, file_name, class_name,
                                              method_name, method)


if __name__ == '__main__':
    unittest.main()
