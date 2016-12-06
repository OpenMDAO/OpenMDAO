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
    'vectors',
]


class LintAttributesTestCase(unittest.TestCase):

    """ Check that for each class, all attributes are documented in the class docstring.

    Parses the source code for the __init__ methods, searching for attribute declarations in every self.* =
    and ensure these attributes are in the 'Attributes' class docstring.
    """
    def test_attributes(self):
        topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print_info = False

        # Regex to grab the attributes documentation section from the class.
        # This section stops at end of string or first empty line.
        classdoc_re = re.compile('(?<=Attributes\\n----------\\n)(.*?\\n\\n|.*)', re.DOTALL)

        # Regex to grab a variable name from the attributes documentation section from the class.
        # Start at the beginning of a line that does not start with whitespace, grab everything
        # up to the first space or colon. Example: grab _system from:
        # _system : System
        classdoc_varnames_re = re.compile('^([^\\s].*?[^\\s\\:]*)', re.MULTILINE)

        # Regex to grab a variable name from the __init__ section of every self.* =.
        # Example: grab _system from:
        # self._system = None
        member_var_re = re.compile('(?<=self\\.)(.*?)(?=\\=)')

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
                        class_doc = inspect.getdoc(clss)
                        classdoc_matches = classdoc_re.findall(class_doc)
                        if(len(classdoc_matches) > 1):
                            self.fail('%s/%s : Class %s' % (dir_name, file_name, class_name) + '... multiple Attributes section in docstring')
                        if(len(classdoc_matches) == 1):
                            classdoc_varnames_matches = classdoc_varnames_re.findall(classdoc_matches[0])


                        # There is a valid __init__ section in the class
                        if('__init__' in clss.__dict__ and '__init__' in dir(clss) and inspect.isfunction(getattr(clss, '__init__'))):
                            method = getattr(clss, '__init__')
                            mysrc = inspect.getsource(method)
                            all_member_vars = list(set([m.strip() for m in member_var_re.findall(mysrc) if ("." not in m)]))

                            # no 'self.*=' statements in __init__ but there is an Attributes section
                            if(len(all_member_vars) == 0 and len(classdoc_matches) == 1):
                                self.fail('%s/%s : Class %s... Attributes section not required' % (dir_name, file_name, class_name))

                            # there are 'self.*=' statements in __init__ but there is no Attributes section
                            if(len(all_member_vars) > 0 and len(classdoc_matches) == 0):
                                self.fail('%s/%s : Class %s... Attributes section missing but found %d member vars' % (dir_name, file_name, class_name, len(all_member_vars)))


                            for v in all_member_vars:
                                if v not in classdoc_varnames_matches:
                                    self.fail('%s/%s : Class %s, Member %s not documented in Attributes section of docstring'
                                                % (dir_name, file_name, class_name, v))
                                else:
                                    if(print_info): print("    Documented member \'%s\'" % (v))
                        else: #no init section
                            if(len(classdoc_matches) == 0): # no Attributes section
                                if(print_info): print('    Skipping Class %s... missing Attributes and init' % (class_name))
                            else: # one Attributes section
                                self.fail('%s/%s : Class %s... Attributes section in docstring but no __init__ function' % (dir_name, file_name, class_name))

if __name__ == '__main__':
    unittest.main()
