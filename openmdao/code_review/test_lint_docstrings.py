from __future__ import print_function

import unittest
import os.path
import importlib
import inspect
from six import PY3

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

        # Check if docstring is missing
        if doc is None:
            self.fail('%s, %s : %s.%s ' %
                (dir_name, file_name, class_name,
                 method_name) +
                '... missing docstring')

        # Check if docstring references another method
        if doc[:3] == 'See':
            return

        # Check if there are args
        if len(argspec.args) > 1:
            loc = doc.find('Args\n----')

            # Check if args section exists in docstrings
            if loc == -1:
                self.fail('%s, %s : %s.%s ' %
                    (dir_name, file_name, class_name,
                     method_name) +
                    '... missing Args section in docstring')

            # Read the Args section in the docstring
            istart = loc + 10
            index = doc[istart:].find('\n\n')
            if index == -1:
                iend = len(doc)
            else:
                iend = istart + index
            entries = doc[istart:iend].split('\n')

            num_args = len(argspec.args) - 1 + \
                int(argspec.varargs is not None) + \
                int(argspec.keywords is not None)
            index = 0
            # Check the Args section line-by-line
            for line in entries:
                if line[:4] == ' '*4 and line[4] != ' ':
                    pass
                else:
                    # If currently on one of the args
                    if index < len(argspec.args) - 1:
                        arg = argspec.args[index+1]
                        ind = len(arg)
                        valid = line[:ind] == arg
                        valid = valid and line[ind:ind+3] == ' : '
                        if not valid:
                            self.fail('%s, %s : %s.%s , %s ' %
                                (dir_name, file_name, class_name,
                                 method_name, arg) +
                                '... formatting incorrect')
                        index += 1
                    # If currently on varargs or kwargs
                    elif index < num_args:
                        index += 1
                    # If we've exceeded the counter
                    else:
                        self.fail('%s, %s : %s.%s ' %
                            (dir_name, file_name, class_name,
                             method_name) +
                            '... formatting incorrect ' +
                            'or too many arg docstrings')
            # If we haven't reached the end
            if index < num_args:
                self.fail('%s, %s : %s.%s ' %
                    (dir_name, file_name, class_name,
                     method_name) +
                    'missing arg docstrings')

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
                    if print_info: print(file_name)

                    module_name = 'openmdao.%s.%s' % (dir_name, file_name[:-3])
                    try:
                        mod = importlib.import_module(module_name)
                    except ImportError as err:
                        if print_info: print('Skipped:', err)
                        # e.g. PETSc is not installed
                        continue

                    # Loop over classes
                    classes = [x for x in dir(mod)
                               if inspect.isclass(getattr(mod, x)) and
                               getattr(mod, x).__module__ == module_name]
                    for class_name in classes:
                        if print_info: print(' '*4, class_name)
                        clss = getattr(mod, class_name)

                        # Loop over methods
                        methods = [x for x in dir(clss)
                                   if inspect.ismethod(getattr(clss, x)) and
                                   x in clss.__dict__]
                        for method_name in methods:
                            if print_info: print(' '*8, method_name)
                            method = getattr(clss, method_name)

                            self.check_method(dir_name, file_name,
                                class_name, method_name, method)


if __name__ == '__main__':
    unittest.main()
