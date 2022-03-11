"""
Functions for handling runtime function hooks.
"""

from collections import defaultdict
from functools import wraps
import inspect
import warnings
import sys


# global dict of hooks
# {class_name: { inst_id: {fname: [pre_hooks, post_hooks]}}}
# At class_name level, there might be a 'None' entry, which means that all instances of
# that class will have have the hooks specified in the 'None' dict.  Otherwise, entries
# there will be specific instance names.
# pre_hooks and post_hooks are lists of functions to call before or after the function named
# 'fname'.
_hooks = {}

# classes found here are known to contain no hooks within themselves or their ancestors
_hook_skip_classes = set()

# global switch that turns hook machinery on/off. But need it on for reporting system
use_hooks = True


def _reset_all_hooks():
    global _hooks, _hook_skip_classes
    _hooks = {}
    _hook_skip_classes = set()


def _setup_hooks(obj):
    """
    Wrap object's methods with a hook checking function if hooks are defined for any of them.

    Parameters
    ----------
    obj : object
        The object whose methods may be wrapped.
    """
    global _hooks, _hook_skip_classes

    # _setup_hooks should be called after 'obj' can return a valid name from _get_inst_id().
    # For example, in Problem, it can happen in __init__, but in Component and Group it shouldn't
    # happen until _setup_procs because that's the earliest point where the component/group has a
    # valid pathname.
    if use_hooks and obj.__class__ not in _hook_skip_classes:

        classes = inspect.getmro(obj.__class__)
        for c in classes:
            if c.__name__ in _hooks:
                classmeta = _hooks[c.__name__]
                break
        else:
            # didn't find any matching hooks for this class or any base class, so skip in future
            _hook_skip_classes.update(classes)
            return

        # any object where we register hooks must define the '_get_inst_id' method.
        ident = obj._get_inst_id()

        if ident in classmeta:
            instmeta = classmeta[ident]
        elif None in classmeta:  # ident of None applies to all instances of a class
            instmeta = classmeta[None]
        else:
            return

        for funcname in instmeta:
            method = getattr(obj, funcname, None)
            if method is not None:
                # if _hook_ attr is present, we've already wrapped this method
                if not hasattr(method, '_hook_'):
                    setattr(obj, funcname, _hook_decorator(method, obj, instmeta[funcname]))


def _run_hooks(hooks, inst):
    """
    Run the given list of hooks.

    Parameters
    ----------
    hooks : list
        List of hook data.
    inst : object
        Object instance to pass to hook functions.
    """
    for i, (hook, ncalls, ex, kwargs) in enumerate(hooks):
        if ncalls is None or ncalls > 0:
            hook(inst, **kwargs)
            if ex:
                sys.exit()
            if ncalls is not None:
                ncalls -= 1
                hooks[i] = (hook, ncalls, ex, kwargs)


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
    pre_hooks, post_hooks = hookmeta

    # args and kwargs are arguments to the method that is being wrapped
    def execute_hooks(*args, **kwargs):
        _run_hooks(pre_hooks, inst)
        ret = f(*args, **kwargs)
        _run_hooks(post_hooks, inst)
        return ret

    execute_hooks._hook_ = True  # to prevent multiple decoration of same function

    return wraps(f)(execute_hooks)


def _get_hook_lists(class_name, inst_id, fname):
    """
    Retrieve the pre and post hook lists for the given class, instance, and function name.

    Parameters
    ----------
    class_name : str
        The name of the class owning the method where the hook will be applied.
    inst_id : str, optional
        The name of the instance owning the method where the hook will be applied.
    fname : str
        The name of the function where the pre and/or post hook will be applied.
    """
    global _hooks
    if class_name in _hooks:
        cmeta = _hooks[class_name]
    else:
        _hooks[class_name] = cmeta = {}
    if inst_id in cmeta:
        imeta = cmeta[inst_id]
    else:
        cmeta[inst_id] = imeta = {}
    if fname in imeta:
        return imeta[fname]
    imeta[fname] = [[], []]
    return imeta[fname]


def _register_hook(fname, class_name, inst_id=None, pre=None, post=None, ncalls=None, exit=False,
                   **kwargs):
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
    pre : function (None)
        If not None, this hook will run before the named function runs.
    post : function (None)
        If not None, this hook will run after the named function runs.
    ncalls : int or None
        Auto-remove the hook function after this many calls.  If None, never auto-remove.
        If both pre and post are registered, this will affect both.
    exit : bool
        If True, run sys.exit() after calling the hook function.  If post is registered, this
        affects only post, else it will affect pre.
    **kwargs : dict of keyword arguments
        Keyword arguments that will be passed to the hook function.
    """
    if pre is None and post is None:
        raise RuntimeError("In _register_hook you must specify pre or post.")

    pre_hooks, post_hooks = _get_hook_lists(class_name, inst_id, fname)
    if pre is not None and (ncalls is None or ncalls > 0):
        pre_hooks.append((pre, ncalls, exit and post is None, kwargs))
    if post is not None and (ncalls is None or ncalls > 0):
        post_hooks.append((post, ncalls, exit, kwargs))


def _remove_hook(to_remove, hooks, class_name, fname, hook_loc):
    """
    Remove a hook function.

    Parameters
    ----------
    to_remove : bool or function
        If True, all hook functions in 'hooks' will be removed.  If a function, any function
        in 'hooks' that matches will be removed.
    hooks : list
        List of (hook_func, ncalls, exit) tuples.
    class_name : str
        The name of the class owning the method where the hook will be removed.
    fname : str
        The name of the function where the hooks are located.
    hook_loc : str
        Either 'pre' or 'post', indicating the hooks run before or after respectively the
        function specified by fname.
    """
    if to_remove and hooks:
        if to_remove is True:
            hooks[:] = []
        else:
            for hook in hooks:
                p, _, _, _ = hook
                if p is to_remove:
                    hooks.remove(hook)
                    break
            else:
                raise RuntimeError(f"Couldn't find the given '{hook_loc}' function in the "
                                   f"{hook_loc} hooks for {class_name}.{fname}.")


def _unregister_hook(fname, class_name, inst_id=None, pre=True, post=True):
    """
    Unregister a hook function.

    By default, both pre and post hooks will be removed if they are present. To avoid
    removal of pre or post, you must set the pre or post arg to False.

    Parameters
    ----------
    fname : str
        The name of the function where the pre and/or post hook will be removed.
    class_name : str
        The name of the class owning the method where the hook will be removed.
    inst_id : str, optional
        The name of the instance owning the method where the hook will be removed.
    pre : bool or function, (True)
        If True, hooks that run before the named function runs will be removed. If a
        function then that function, if found, will be removed from the pre list, else
        an exception will be raised.
    post : bool or function, (True)
        If True, hooks that run after the named function runs will be removed. If a
        function then that function, if found, will be removed from the post list, else
        an exception will be raised.
    """
    try:
        hookdict = _hooks[class_name][inst_id]
    except KeyError:
        warnings.warn(f"No hooks found for class '{class_name}' and instance '{inst_id}'.")
        return

    if fname in hookdict:
        pre_hooks, post_hooks = hookdict[fname]
        _remove_hook(pre, pre_hooks, class_name, fname, 'pre')
        _remove_hook(post, post_hooks, class_name, fname, 'post')

        if not (pre_hooks or post_hooks):
            del hookdict[fname]
            if not hookdict:  # we just removed the last hook entry for this inst
                del _hooks[class_name][inst_id]
                if not _hooks[class_name]:  # removed last entry for this class
                    del _hooks[class_name]
    else:
        warnings.warn(f"No hook found for method '{fname}' for class '{class_name}' and instance "
                      f"'{inst_id}'.")
