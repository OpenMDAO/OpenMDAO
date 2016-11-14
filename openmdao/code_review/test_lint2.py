from __future__ import print_function

import os.path
import importlib
import inspect

directories = [
    'assemblers',
    'core',
    'jacobians',
    'proc_allocators',
    'solvers',
    'utils',
    'vectors',
]

def _get_files():
    """A generator of files to check for pep8/pep257 violations."""
    topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for dir_name in directories:
        dirpath = os.path.join(topdir, dir_name)

        for file_name in  os.listdir(dirpath):
            if file_name != '__init__.py' and file_name[-3:] == '.py':
                yield dirpath, file_name
#                yield os.path.join(dirpath, file_name)

def import_file(full_path_to_module):
    try:
        import os
        module_dir, module_file = os.path.split(full_path_to_module)
        module_name, module_ext = os.path.splitext(module_file)
        save_cwd = os.getcwd()
        os.chdir(module_dir)
        module_obj = __import__(module_name)
        module_obj.__file__ = full_path_to_module
        globals()[module_name] = module_obj
        os.chdir(save_cwd)
    except:
        raise ImportError

topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print_info = 0
print_error = 1
print_dir = 'core'
print_file = 'component.py'
print_class = 'Component'
print_method = '_setup_connections'

for dir_name in directories:
    dirpath = os.path.join(topdir, dir_name)
    if print_info:
        print('-'*len(dir_name))
        print(dir_name)
        print('-'*len(dir_name))

    for file_name in os.listdir(dirpath):
        if file_name != '__init__.py' and file_name[-3:] == '.py':
            if print_info: print(file_name)
            module_name = 'openmdao.%s.%s' % (dir_name, file_name[:-3])
            mod = importlib.import_module(module_name)

            classes = [x for x in dir(mod)
                       if inspect.isclass(getattr(mod, x)) and
                       getattr(mod, x).__module__ == module_name]
            for class_name in classes:
                if print_info: print(' '*4, class_name)
                clss = getattr(mod, class_name)

                methods = [x for x in dir(clss)
                           if inspect.ismethod(getattr(clss, x)) and
                           x in clss.__dict__]
                for method_name in methods:
                    if print_info: print(' '*8, method_name)
                    method = getattr(clss, method_name)

                    argspec = inspect.getargspec(method)
                    doc = inspect.getdoc(method)

                    if doc is None:
                        if print_error or \
                                print_dir == dir_name and \
                                print_file == file_name and \
                                print_class == class_name and \
                                print_method == method_name:
                            print(dir_name, file_name,
                                  class_name, method_name,
                                  'no docs defined.')
                        continue

                    if doc[:3] == 'See':
                        continue

                    if len(argspec.args) > 1:
                        loc = doc.find('Args\n----')
                        if loc == -1:
                            if print_error or \
                                    print_dir == dir_name and \
                                    print_file == file_name and \
                                    print_class == class_name and \
                                    print_method == method_name:
                                print(dir_name, file_name,
                                      class_name, method_name,
                                      'no args defined.', argspec.args)
                                continue
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
                        for line in entries:
                            if line[:4] == ' '*4 and line[4] != ' ':
                                pass
                            else:
                                if index < len(argspec.args) - 1:
                                    arg = argspec.args[index+1]
                                    ind = len(arg)
                                    valid = line[:ind] == arg
                                    valid = valid and line[ind:ind+3] == ' : '
                                    if not valid:
                                        print('%s, %s : %s.%s , %s ' %
                                            (dir_name, file_name, class_name,
                                             method_name, arg) +
                                            '... formatting incorrect')
                                        break
                                    index += 1
                                elif index < num_args:
                                    index += 1
                                else:
                                    print(dir_name, file_name,
                                          class_name, method_name,
                                          'too many arg docstrings')
                                    break
                        if index < len(argspec.args) - 1:
                            print(dir_name, file_name,
                                  class_name, method_name,
                                  'missing arg docstrings')
