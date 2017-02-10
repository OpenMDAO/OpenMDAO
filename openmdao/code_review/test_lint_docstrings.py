from __future__ import print_function

import ast
import unittest
import os.path
import importlib
import inspect
import re
import textwrap
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


class ReturnFinder(ast.NodeVisitor):
    """
    An implementation of node visitor only intended to visit a single
    function/method and not recurse through nested functions.  To use this
    we have to use ast.parse on the source code of a method.  Since ast.parse
    is expected full python files, we have to dedent the source first.

    Attributes
    ----------
    has_return : bool
        When visit is called, this NodeVisitor will recurse through the
        syntax tree and find any instance of return that is not in a nested
        function.  If it finds return, it is set to True, otherwise it is
        False.
    passes : bool
        Set to True if the method does nothing but pass (i.e. is not yet
        implemented or is a 'virtual' method in a base class)

    """
    def __init__(self):
        self.has_return = False
        self.passes = False
        self._function_depth = 0
        self._depth = 0

    def visit(self, node):
        """ Visit nodes in the syntax tree to find instances of return. """

        self._depth += 1

        is_function_def = isinstance(node, ast.FunctionDef)

        # Increase function_depth if node is a FunctionDef
        if is_function_def:
            self._function_depth += 1

        # If node is a Return, and we're at a function depth of 1,
        # and the value attribute is not None, then it
        # returns meaningful (non None) values.
        if isinstance(node, ast.Return) and self._function_depth == 1:
            if node.value is not None:
                self.has_return = True

        if hasattr(node, 'body'):
            # If the top level function does nothing but pass, note it.
            if is_function_def and self._depth == 2 and len(node.body) <= 2 \
                     and isinstance(node.body[-1], ast.Pass):
                self.passes = True
            # Recurse through subnodes
            for subnode in node.body:
                self.visit(subnode)

        # If the node is an If it will have an orelse section to visit
        if hasattr(node, 'orelse'):
            for subnode in node.orelse:
                self.visit(subnode)

        # Reduce function_depth on exit if this is a FunctionDef
        if is_function_def:
            self._function_depth -= 1

        self._depth -= 1


class LintTestCase(unittest.TestCase):

    def check_method_parameters(self, dir_name, file_name, class_name,
                                method_name, argspec, numpy_doc_string,
                                failures):
        """

        Parameters
        ----------
        dir_name : str
            The name of the directory in which the method is defined.
        file_name : str
            The name of the file in which the method is defined.
        class_name : str
            The name of the class to which the method belongs
        method_name : str
            The name of the method
        argspec : namedtuple
            Method argument information from inspect.getargspec (python2) or
            inspect.getfullargspec (python3)
        numpy_doc_string : numpydoc.docscrape.NumpyDocString
            An instance of the NumpyDocString parsed from the method
        failures : dict
            The failures encountered by the method.  These are all stored
            so that we can fail once at the end of the check_method method
            with information about every failure. Form is
            { 'dir_name/file_name:class_name.method_name': [ messages ] }

        Returns
        -------
        bool
            True if *no* failures were encountered for the method, otherwise
            False.

        """
        new_failures = []

        if len(argspec.args) > 1:
            if not numpy_doc_string['Parameters']:
                new_failures.append('does not have a Parameters section')
                #self.fail(fail_msg + '... does not have a Parameters section')

            # Check formatting
            for entry in numpy_doc_string['Parameters']:
                name = entry[0]
                type_ = entry[1]
                desc = '\n'.join(entry[2])
                if ':' in name:
                    new_failures.append('colon after parameter '
                                         'name \'{0}\' and before type must '
                                         'be surrounded by '
                                         'spaces'.format(name.split(':')[0]))
                if type_ == '':
                    new_failures.append('no type info given for '
                                        'Parameter {0}'.format(name))
                if desc == '':
                    new_failures.append('no description given for '
                                         'Parameter {0}'.format(name))

            documented_arg_set = set(item[0] for item in
                                     numpy_doc_string['Parameters'])
            arg_set = set(argspec.args)

            # Require documentation of *args and **kwargs
            if argspec.varargs:
                arg_set |= {argspec.varargs}
            if argspec.keywords:
                arg_set |= {argspec.keywords}

            # Don't require documentation of self or cls
            if 'self' in arg_set:
                arg_set.remove('self')
            if 'cls' in arg_set:
                arg_set.remove('cls')

            # Arguments that aren't documented
            undocumented = arg_set - documented_arg_set
            if undocumented:
                new_failures.append('missing documentation for: '
                                     '{0}'.format(str(list(undocumented))))

            # Arguments that are documented but don't exist
            overdocumented = documented_arg_set - arg_set
            if overdocumented:
                new_failures.append('documents nonexisting parameters: '
                                     '{0}'.format(str(list(overdocumented))))

        if new_failures:
            key = '{0}/{1}:{2}.{3}'.format(dir_name, file_name, class_name,
                                           method_name)
            failures[key] = new_failures


    def check_method_returns(self, dir_name, file_name, class_name,
                             method_name, method, numpy_doc_string, failures):
        new_failures = []

        method_src = inspect.getsource(method)
        dedented_src = textwrap.dedent(method_src)

        f = ReturnFinder()
        f.visit(ast.parse(dedented_src))

        # If the function does nothing but pass, return
        if f.passes:
            return

        returns = numpy_doc_string['Returns']

        if returns and not f.has_return:
            new_failures.append('method returns no value but found '
                                'unnecessary \'Returns\' sections '
                                'in docstring')
        elif f.has_return and not returns:
            new_failures.append('method returns value(s) but found '
                                'no \'Returns\' sections in docstring')
        elif f.has_return and returns:
            # Check formatting
            for entry in returns:
                name = entry[0]
                desc = '\n'.join(entry[2])
                if not name:
                    new_failures.append('no detectable name for Return '
                                        'value'.format(name))
                if desc == '':
                    new_failures.append('no description given for Return '
                                        '{0}'.format(name))
        if new_failures:
            key = '{0}/{1}:{2}.{3}'.format(dir_name, file_name, class_name,
                                           method_name)
            failures[key] = new_failures

    def check_method(self, dir_name, file_name,
                     class_name, method_name, method, failures):
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

        self.check_method_parameters(dir_name, file_name, class_name,
                                     method_name, argspec, nds, failures)

        self.check_method_returns(dir_name, file_name, class_name, method_name,
                                  method, nds, failures)

    def test_docstrings(self):
        topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        print_info = False

        failures = {}

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
                                              method_name, method, failures)

            if failures:
                msg = '\n'
                for key in failures:
                    msg += '{0}\n'.format(key)
                    for failure in failures[key]:
                        msg += '    {0}\n'.format(failure)

                self.fail(msg)


if __name__ == '__main__':
    unittest.main()
