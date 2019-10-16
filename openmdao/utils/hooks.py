"""
Functions for handling runtime function hooks.
"""

import sys
import os
from collections import defaultdict
from functools import wraps
import inspect
import warnings


# global dict of hooks
# {fname : { class_name: { inst_id: {}}}}
_hooks = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))))

# ids of objects that have already had their methods wrapped
_hooks_done = set()

# classes found here are known to contain no hooks
_hook_skip_classes = set()

# _setup_hooks is a noop if this is False
use_hooks = False


def _setup_hooks(obj):
    """
    Wrap object's methods with a hook checking function if hooks are defined for any of them.

    This is typically called from the __init__ method, because otherwise it will be too early
    to apply the decorator (the hooks may not be defined yet).

    Parameters
    ----------
    obj : object
        The object whose methods may be wrapped.
    """
    if use_hooks and obj.__class__ not in _hook_skip_classes and id(obj) not in _hooks_done:

        classes = inspect.getmro(obj.__class__)
        for c in classes:
            if c.__name__ in _hooks:
                hk = _hooks[c.__name__]
                break
        else:
            # didn't find any matching hooks for this class or any base class, so skip in future
            _hook_skip_classes.update(classes)
            return

        ident = obj._get_inst_id()
        if ident in hk:
            hk = hk[ident]
        elif None in hk:
            hk = hk[None]
        else:
            return

        for name, method in inspect.getmembers(obj, inspect.ismethod):
            if name in hk:
                hk = hk[name]
                setattr(obj, name, _hook_decorator(method, obj, hk))

        _hooks_done.add(id(obj))


def _hook_decorator(f, inst, hookmeta):
    """
    Wrap a method with pre and/or post hook checks.

    Parameters
    ----------
    f : method
        The method being wrapped.
    inst : object
        The instance that owns the method.
    hookmeta : dict
        A dict with information about the hooks.
    """
    cond = hookmeta.get('cond')

    @wraps(f)
    def check_hooks(*args, **kwargs):
        if hookmeta['pre']:
            if cond is None or cond(inst):
                hookmeta['hookfunc'](inst)

        f(*args, **kwargs)

        if hookmeta['post']:
            if cond is None or cond(inst):
                hookmeta['hookfunc'](inst)

    return check_hooks


def _register_hook(hookfunc, fname, class_name, inst_id=None, pre=False, post=False, cond=None):
    """
    Register a hook function.

    Parameters
    ----------
    hookfunc : function
        A function to execute in the designated hook location.
    fname : str
        The name of the function where the pre and/or post hook will be applied.
    class_name : str
        The name of the class owning the method where the hook will be applied.
    inst_id : str, optional
        The name of the instance owning the method where the hook will be applied.
    pre : bool, optional
        If True, the hook will be activated before the named function runs.
    post : bool, optional
        If True, the hook will be activated after the named function runs.
    cond : function, optional
        A additional function that must return True before the hook will activate.
    """
    hook = _hooks[class_name][inst_id][fname]
    hook['pre'] = hook.get('pre', False) | pre
    hook['post'] = hook.get('post', False) | post
    hook['cond'] = cond
    hook['hookfunc'] = hookfunc


def _can_remove_hook(hook):
    return not (hook['pre'] or hook['post'] or hook['cond'] is not None)


def _unregister_hook(fname, class_name, inst_id=None, pre=False, post=False, cond=None):
    """
    Register a hook function.

    Parameters
    ----------
    fname : str
        The name of the function where the pre and/or post hook will be applied.
    class_name : str
        The name of the class owning the method where the hook will be applied.
    inst_id : str, optional
        The name of the instance owning the method where the hook will be applied.
    pre : bool, optional
        If True, the hook will be activated before the named function runs.
    post : bool, optional
        If True, the hook will be activated after the named function runs.
    cond : function, optional
        A additional function that must return True before the hook will activate.
    """
    hookdict = _hooks[class_name][inst_id]
    if fname in hookdict:
        hook = hookdict[fname]
        if pre and hook['pre']:
            hook['pre'] = False
        if post and hook['post']:
            hook['post'] = False
        if cond and hook['cond']:
            hook['cond'] = None

        if _can_remove_hook(hook):
            del hookdict[fname]
            if not hookdict:  # we just removed the last hook entry for this inst
                del _hooks[class_name][inst_id]
                if not _hooks[class_name]:  # removed last entry for this class
                    del _hooks[class_name]
    else:
        warnings.warn("No hook found for method '{}' for class '{}' and instance '{}'.".format(
            fname, class_name, inst_id
        ))
