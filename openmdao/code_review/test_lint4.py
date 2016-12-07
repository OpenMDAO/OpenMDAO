from __future__ import print_function

import unittest
import os.path
import importlib
import inspect
import re

directories = [
    'assemblers',
    'core',
    'jacobians',
    'matrices',
    'proc_allocators',
    'solvers',
    'utils',
    'vectors'
]


class LintReturnsTestCase(unittest.TestCase):

    """ Check that for each method, if the code has 'return ___' in it, then a 'Returns' section
    must exist with return types and descriptions in the method docstring.

    Parses the source code for all methods, searching for any RHS after a 'return' statement
    and ensure that a complete 'Returns' docstring section exists.
    """
    def test_returns(self):
        topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print_info = False

        # Regex to grab the returns documentation section from the method.
        # This section stops at end of string or first empty line.
        methoddoc_re = re.compile('(?<=Returns\\n-------\\n)([^\\n].*?\\n\\n|[^\\n].*)', re.DOTALL)

        # Regex to grab a return type name from the Returns documentation section from the method.
        # Start at the beginning of a line that does not start with whitespace, grab everything
        # up to the end of the line.
        methoddoc_returntypes_re = re.compile('^([^\\s].*)', re.MULTILINE)

        # Regex to grab everything after "return[space]" in the method code section
        # Example: grab "foo" from:
        # return foo
        method_afterreturn_re = re.compile('(?<=return\\s)(.+)')

        # Regex to see if the word 'pass' exists on a single line of source
        method_pass_re = re.compile('^([\\s]*pass[\\s]*)$', re.MULTILINE)

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
                    if print_info: print('File: %s' % (file_name))

                    module_name = 'openmdao.%s.%s' % (dir_name, file_name[:-3])
                    if print_info: print(' Module: %s' % (module_name))
                    try:
                        mod = importlib.import_module(module_name)
                    except ImportError:
                        print('Error: could not import %s' % (module_name))


                    # Loop over classes
                    classes = [x for x in dir(mod)
                               if inspect.isclass(getattr(mod, x)) and
                               getattr(mod, x).__module__ == module_name]
                    for class_name in classes:
                        if print_info : print('  Class:', class_name)
                        clss = getattr(mod, class_name)


                        # Loop over methods
                        methods = [x for x in dir(clss)
                                   if( (inspect.isfunction(getattr(clss, x)) or inspect.ismethod(getattr(clss, x))) and x in clss.__dict__ )]
                        for method_name in methods:
                            if print_info: print('   Method:', method_name)
                            method = getattr(clss, method_name)

                            method_doc = inspect.getdoc(method)
                            method_src = inspect.getsource(method)


                            methoddoc_matches = methoddoc_re.findall(method_doc)
                            if(len(methoddoc_matches) > 1):
                                self.fail('%s/%s : Class %s : Method %s... multiple \'Returns\' sections in docstring'
                                    % (dir_name, file_name, class_name, method_name))
                            elif(len(methoddoc_matches) == 1):
                                methoddoc_returntypes_matches = methoddoc_returntypes_re.findall(methoddoc_matches[0])
                                if(len(methoddoc_returntypes_matches) == 0):
                                    self.fail('%s/%s : Class %s : Method %s... invalid \'Returns\' section in docstring'
                                        % (dir_name, file_name, class_name, method_name))
                                if(print_info and len(methodsrc_returnvar_matches) > 0):
                                    print("    Docstring return types: %s" % (methoddoc_returntypes_matches))


                            methodsrc_returnvar_matches = method_afterreturn_re.findall(method_src)
                            if(print_info and len(methodsrc_returnvar_matches) > 0):
                                print("    return RHS: %s" % (methodsrc_returnvar_matches))

                            if(len(methodsrc_returnvar_matches) > 0 and len(methoddoc_matches) == 0):
                                self.fail('%s/%s : Class %s : Method %s... method returns value(s) %s but no \'Returns\' section in docstring'
                                    % (dir_name, file_name, class_name, method_name, methodsrc_returnvar_matches))

                            pass_lines = method_pass_re.findall(method_src)
                            if(print_info and len(methodsrc_returnvar_matches) == 0 and len(methoddoc_matches) == 1 and len(pass_lines) == 0):
                                print('%s/%s : Class %s : Method %s... method returns no value but found unnecessary \'Returns\' sections in docstring'
                                    % (dir_name, file_name, class_name, method_name))


if __name__ == '__main__':
    unittest.main()
