"""
Function that returns the source code of a method or class.
The docstrings are stripped from the code
"""
import importlib
import inspect

from openmdao.docs.utils.docutil import remove_docstrings

# pylint: disable=C0103

def get_source_code_of_class_or_method(class_or_method_path):
    '''The function to be called a the custom Sphinx directive code
    that includes the source code of a class or method.
    '''

    # the class_or_method_path could be either to a class or method

    # first assume class and see if it works
    try:
        module_path = '.'.join(class_or_method_path.split('.')[:-1])
        module_with_class = importlib.import_module(module_path)
        class_name = class_or_method_path.split('.')[-1]
        cls = getattr(module_with_class, class_name)
        source = inspect.getsource(cls)
    except ImportError:
        # else assume it is a path to a method
        module_path = '.'.join(class_or_method_path.split('.')[:-2])
        module_with_method = importlib.import_module(module_path)
        class_name = class_or_method_path.split('.')[-2]
        method_name = class_or_method_path.split('.')[-1]
        cls = getattr(module_with_method, class_name)
        meth = getattr(cls, method_name)
        source = inspect.getsource(meth)

    # Remove docstring from source code
    source_minus_docstrings = remove_docstrings(source)

    return source_minus_docstrings

if __name__ == "__main__":
    # Get source of method
    method_source = get_source_code_of_class_or_method(
                    'openmdao.solvers.ln_direct.DirectSolver.__call__')
    print(90*'-')
    print(method_source)
    print(90*'=')
    # Get source of class
    class_source = get_source_code_of_class_or_method('openmdao.solvers.ln_direct.DirectSolver')
    print(90*'-')
    print(class_source)
    print(90*'=')
    # Try something that should fail
    nonexistent_source = get_source_code_of_class_or_method(
                        'openmdao.solvers.ln_direct.NonexistentSolver')
