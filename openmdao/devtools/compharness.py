"""
Functions to make it easier to test or work with a single component without having to build a model.
"""

import sys
from importlib import import_module

import numpy as np

from openmdao.core.problem import Problem
from openmdao.core.indepvarcomp import IndepVarComp


def get_class(classpath):
    modpath, cname = classpath.rsplit('.', 1)
    import_module(modpath)
    mod = sys.modules[modpath]
    return getattr(mod, cname)


def get_setup_comp_instance(classpath, run_model=True, *args, **kwargs):
    """
    Given a module path to a class, create a fully set-up component instance.

    Parameters
    ----------
    classpath : str
        The module path to the class definition (includes module and class name).

    Returns
    -------
    Component
        An instance of the specified component class.
    """
    class_ = get_class(classpath)
    p = Problem()
    comp = p.model.add_subsystem('comp', class_(*args, **kwargs))
    p.setup()
    if run_model:
        p.run_model()
    else:
        p.final_setup()

    return comp



if __name__ == '__main__':
    if len(sys.argv) > 1:
        cpath = sys.argv[1]
    else:
        cpath = 'openmdao.test_suite.components.sellar.SellarDis1'
    comp = get_setup_comp_instance(cpath)
    inputs = {}
    for path in comp._inputs._views:
        parts = path.rsplit('.', 1)
        if len(parts) == 1:
            inp = path
        else:
            inp = parts[-1]
        inputs[inp] = np.array(comp._inputs._views[path])

    import pprint
    pprint.pprint(inputs)



