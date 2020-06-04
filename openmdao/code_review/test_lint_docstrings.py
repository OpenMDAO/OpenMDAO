import ast
import unittest
import os.path
import importlib
import inspect
import textwrap
import collections
import re

try:
    from numpydoc.docscrape import NumpyDocString
except ImportError:
    NumpyDocString = None


# directories in which we do not wish to lint for docstrings/parameters.
exclude = [
    'code_review',
    'devtools',
    'docs',
    'test_suite',
    'tests',
    'test',
]

# we will build a list of dirs in which to do linting.
directories = []

top = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for root, dirs, files in os.walk(top, topdown=True):
    dirs[:] = [d for d in dirs if d not in exclude]
    for di in dirs:
        directories.append(os.path.join(root, di))


def _is_context_manager(func):
    """
    Detect if the given method or function is decorated with @contextmanager.

    Parameters
    ----------
    func : function or method
        The function or method to be tested.

    Returns
    -------
    bool
        True if the function or method is has the @contextmanager decorator,
        otherwise False.

    """
    src = inspect.getsource(func).lstrip()
    return 'return GeneratorContextManager' in src or src.startswith('@contextmanager')


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
        self._func_depth = 0
        self.is_context_manager = False
        self._depth = 0

    def visit(self, node):
        """
        Visit nodes in the syntax tree to find instances of return.
        """
        self._depth += 1

        is_func_def = isinstance(node, ast.FunctionDef)

        # Increase function_depth if node is a FunctionDef
        if is_func_def:
            self._func_depth += 1

        # If node is a Return, and we're at a function depth of 1,
        # and the value attribute is not None, then it
        # returns meaningful (non None) values.
        if isinstance(node, ast.Return) and self._func_depth == 1:
            if node.value is not None:
                self.has_return = True

        if hasattr(node, 'body') and isinstance(node.body, collections.Iterable):
            # If the top level function does nothing but pass, note it.
            if is_func_def and self._depth == 2 and len(node.body) <= 2 \
                     and isinstance(node.body[-1], ast.Pass):
                self.passes = True
            # Recurse through subnodes
            for subnode in node.body:
                self.visit(subnode)

        # If the node is an If it will have an orelse section to visit
        if hasattr(node, 'orelse'):
            for subnode in node.orelse:
                self.visit(subnode)

        # If we're in a context manager top-level function, ignore its return
        if is_func_def and self._func_depth == 1 \
                and hasattr(node, 'decorator_list') and node.decorator_list:
            try:
                wrapper = node.body[0].value.func.id
                if 'ContextManager' in wrapper:
                    self.is_context_manager = True
            except AttributeError:
                pass

        # Reduce function_depth on exit if this is a FunctionDef
        if is_func_def:
            self._func_depth -= 1

        self._depth -= 1


class DecoratorFinder(ast.NodeVisitor):
    """
    An implementation of node visitor used to find decorators on a
    FunctionDef and record their names

    Attributes
    ----------
    decorators : dict
        The dict where the keys are function names and the values are
        the corresponding decorators.
    """

    def __init__(self):
        self.decorators = {}

    def visit_FunctionDef(self, node):
        """
        Called when a FunctionDef node is visited.  If decorators are found,
        record them in self.decorators.

        Parameters
        ----------
        node : node
            The node being visited

        """
        self.decorators[node.name] = []
        for n in node.decorator_list:
            name = ''
            if isinstance(n, ast.Call):
                name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
            else:
                name = n.attr if isinstance(n, ast.Attribute) else n.id
            self.decorators[node.name].append(name)


@unittest.skipUnless(NumpyDocString, "requires 'numpydoc', install openmdao[test]")
class LintTestCase(unittest.TestCase):

    def check_summary(self, numpy_doc_string):
        """
        Parameters
        ----------
        numpy_doc_string : numpydoc.docscrape.NumpyDocString
            An instance of the NumpyDocString parsed from the method

        Returns
        -------
        failures : dict
            The failures encountered by the method.  These are all stored
            so that we can fail once at the end of the check_method method
            with information about every failure. Form is
            { 'dir_name/file_name:class_name.method_name': [ messages ] }
        """
        new_failures = []
        summary = numpy_doc_string['Summary']

        # Check that summary is present
        if not summary:
            return ['is missing a summary.']

        # Summary should be a single line.
        if len(summary) > 1:
            new_failures.append('summary should be only one line.')
        summary = summary[0]
        # Summary should have no leading/trailing whitespace.
        if summary[0].isspace() or summary[-1].isspace():
            new_failures.append('summary should not contain leading or '
                                'trailing whitespace.')
        # Summary should end with a period.
        if not summary.endswith('.'):
            new_failures.append('Summary should end with a period.')

        return new_failures

    def check_parameters(self, argspec, numpy_doc_string):
        """
        Check that the parameters section is correct.

        Parameters
        ----------
        argspec : namedtuple
            Method argument information from inspect.getargspec (python2) or
            inspect.getfullargspec (python3)
        numpy_doc_string : numpydoc.docscrape.NumpyDocString
            An instance of the NumpyDocString parsed from the method

        Returns
        -------
        failures : dict
            The failures encountered by the method.  These are all stored
            so that we can fail once at the end of the check_method method
            with information about every failure. Form is
            { 'dir_name/file_name:class_name.method_name': [ messages ] }
        """
        new_failures = []

        arg_set = set(argspec.args)

        # Don't require documentation of self or cls
        if 'self' in arg_set:
            arg_set.remove('self')
        if 'cls' in arg_set:
            arg_set.remove('cls')

        # Do require documentation of *args and **kwargs
        if argspec.varargs:
            arg_set |= {'*' + argspec.varargs}
        if argspec.varkw:
            arg_set |= {'**' + argspec.varkw}

        if len(arg_set) >= 1:
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

        return new_failures

    def check_returns(self, func, numpy_doc_string, name_required=False):
        """
        Check that the returns section is correct.

        Parameters
        ----------
        func : method or function
            The method being checked
        numpy_doc_string : numpydoc.docscrape.NumpyDocString
            An instance of the NumpyDocString parsed from the method
        name_required : bool
            If True, a name is required for the return value.

        Returns
        -------
        failures : dict
            The failures encountered by the method.  These are all stored
            so that we can fail once at the end of the check_method method
            with information about every failure. Form is
            { 'dir_name/file_name:class_name.method_name': [ messages ] }
        """
        new_failures = []

        method_src = inspect.getsource(func)
        dedented_src = textwrap.dedent(method_src)

        f = ReturnFinder()
        f.visit(ast.parse(dedented_src))

        # If the function does nothing but pass, return
        if f.passes:
            return []

        doc_returns = numpy_doc_string['Returns']
        doc_yields = numpy_doc_string['Yields']

        # TODO:  Enforce Yields in docs for contextmanagers
        if _is_context_manager(func):
            pass
        elif doc_returns and not f.has_return:
            new_failures.append('method returns no value but found '
                                'unnecessary \'Returns\' section '
                                'in docstring')
        elif f.has_return and not doc_returns:
            new_failures.append('method returns value(s) but found '
                                'no \'Returns\' section in docstring')
        elif f.has_return and doc_returns:
            # Check formatting
            for (name, typ, desc) in doc_returns:
                if name_required and not name:
                    new_failures.append('no detectable name for Return '
                                        'value'.format(name))
                if desc == '':
                    new_failures.append('no description given for Return '
                                        '{0}'.format(name))

        return new_failures

    def check_method(self, dir_name, file_name,
                     class_name, method_name, method, failures):
        """
        Perform docstring checks on each method.

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
        method : instancemethod
            The method being tested.
        failures : dict
            The failures encountered by the method.  These are all stored
            so that we can fail once at the end of the check_method method
            with information about every failure. Form is
            { 'dir_name/file_name:class_name.method_name': [ messages ] }
        """
        argspec = inspect.getfullargspec(method)
        doc = inspect.getdoc(method)

        new_failures = []

        # If the method is decorated with @contextmanager, skip it for now
        if _is_context_manager(method):
            return
        # Check if docstring is missing
        if doc is None:
            new_failures.append('is missing docstring')
            return

        if method.__doc__ is not None and not method.__doc__.startswith('\n'):
            new_failures.append('docstring should start with a new line')

        # Check if docstring references another method
        if doc[:3] == 'See':
            return

        nds = NumpyDocString(doc)

        new_failures.extend(self.check_summary(nds))

        new_failures.extend(self.check_parameters(argspec, nds))

        new_failures.extend(self.check_returns(method, nds))

        if new_failures:
            key = '{0}/{1}:{2}.{3}'.format(dir_name, file_name, class_name,
                                           method_name)
            if key in failures:
                failures[key] += new_failures
            else:
                failures[key] = new_failures

    def check_class(self, dir_name, file_name, class_name, clss, failures):
        """
        Perform docstring checks on a class.

        Parameters
        ----------
        dir_name : str
            The name of the directory in which the method is defined.
        file_name : str
            The name of the file in which the method is defined.
        class_name : str
            The name of the class being checked.
        clss : class
            The class being tested.
        failures : dict
            The failures encountered by the method.  These are all stored
            so that we can fail once at the end of the check_method method
            with information about every failure. Form is
            { 'dir_name/file_name:class_name.method_name': [ messages ] }
        """

        new_failures = []
        doc = inspect.getdoc(clss)

        # Check if docstring is missing
        if doc is None:
            new_failures.append('is missing docstring')
            return

        if clss.__doc__ is not None and not clss.__doc__.startswith('\n'):
            new_failures.append('docstring should start with a new line')

        if new_failures:
            key = '{0}/{1}:{2}'.format(dir_name, file_name, class_name)
            if key in failures:
                failures[key] += new_failures
            else:
                failures[key] = new_failures

    def check_function(self, dir_name, file_name, func_name, func, failures):
        """
        Perform docstring checks on a function.

        Parameters
        ----------
        dir_name : str
            The name of the directory in which the method is defined.
        file_name : str
            The name of the file in which the method is defined.
        func_name : str
            The name of the function being checked
        func : function
            The function being tested.
        failures : dict
            The failures encountered by the method.  These are all stored
            so that we can fail once at the end of the check_method method
            with information about every failure. Form is
            { 'dir_name/file_name:class_name.method_name': [ messages ] }
        """

        argspec = inspect.getfullargspec(func)
        doc = inspect.getdoc(func)

        new_failures = []

        # If the method is decorated with @contextmanager, skip it for now
        if _is_context_manager(func):
            return

        # Check if docstring is missing
        if doc is None:
            new_failures.append('is missing docstring')
            return

        if not func.__doc__.startswith('\n'):
            new_failures.append('docstring should start with a new line')

        # Check if docstring references another function
        if doc[:3] == 'See':
            return

        nds = NumpyDocString(doc)

        new_failures.extend(self.check_summary(nds))

        new_failures.extend(self.check_parameters(argspec, nds))

        new_failures.extend(self.check_returns(func, nds))

        if new_failures:
            key = '{0}/{1}:{2}'.format(dir_name, file_name, func_name)
            if key in failures:
                failures[key] += new_failures
            else:
                failures[key] = new_failures

    def test_docstrings(self):
        print_info = False

        failures = {}

        # Loop over directories
        for dir_name in directories:
            dirpath = dir_name
            if print_info:
                print('-'*len(dir_name))
                print(dir_name)
                print('-'*len(dir_name))

            # Loop over files
            for file_name in os.listdir(dirpath):
                if not file_name.startswith("_") and file_name[-3:] == '.py' and not os.path.isdir(file_name):
                    if print_info:
                        print(file_name)

                    # to construct module name, remove part of abs path that
                    # precedes 'openmdao', and then replace '/' with '.' in the remainder.
                    mod1 = re.sub(r'.*openmdao', 'openmdao', dir_name).replace('/', '.')

                    # then, get rid of the '.py' to get final part of module name.
                    mod2 = file_name[:-3]

                    module_name = '{}.{}'.format(mod1, mod2)

                    try:
                        mod = importlib.import_module(module_name)
                    except ImportError as err:
                        if print_info:
                            print('Skipped:', err)
                        # e.g. PETSc is not installed
                        continue

                    # Loop over classes
                    classes = [x for x in dir(mod)
                               if not x.startswith('_') and inspect.isclass(getattr(mod, x)) and
                               getattr(mod, x).__module__ == module_name]

                    for class_name in classes:
                        if print_info:
                            print(' '*4, class_name)
                        clss = getattr(mod, class_name)

                        # skip namedtuples
                        if issubclass(clss, tuple):
                            continue

                        self.check_class(dir_name, file_name, class_name, clss,
                                         failures)

                        # Loop over methods
                        methods = [x for x in dir(clss)
                                   if (inspect.ismethod(getattr(clss, x)) or inspect.isfunction(getattr(clss, x))) and
                                   x in clss.__dict__]
                        for method_name in methods:
                            if print_info:
                                print(' '*8, method_name)
                            method = getattr(clss, method_name)

                            self.check_method(dir_name, file_name, class_name,
                                              method_name, method, failures)

                    # Loop over functions
                    tree = ast.parse(inspect.getsource(mod))

                    if hasattr(tree, 'body'):
                        funcs = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
                    else:
                        funcs = []

                    for func_name in funcs:
                        if not func_name.startswith('_'):
                            func = getattr(mod, func_name)
                            self.check_function(dir_name, file_name, func_name,
                                                func, failures)

        if failures:
            msg = '\n'
            count = 0
            for key in failures:
                msg += '{0}\n'.format(key)
                count += len(failures[key])
                for failure in failures[key]:
                    msg += '    {0}\n'.format(failure)
            msg += 'Found {0} issues in docstrings'.format(count)
            self.fail(msg)


if __name__ == '__main__':
    unittest.main()
