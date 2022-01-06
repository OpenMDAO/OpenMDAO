"""
Functions for handling runtime function hooks.
"""

from collections import defaultdict
from functools import wraps
import inspect
import warnings


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

# global switch that turns hook machinery on/off
use_hooks = False


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
    def execute_hooks(*args, **kwargs):
        pre_hooks, post_hooks = hookmeta
        for hook in pre_hooks:
            hook(inst)

        ret = f(*args, **kwargs)

        for hook in post_hooks:
            hook(inst)

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


def _register_hook(fname, class_name, inst_id=None, pre=None, post=None):
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
    """
    if pre is None and post is None:
        raise RuntimeError("In _register_hook you must specify pre or post.")

    pre_hooks, post_hooks = _get_hook_lists(class_name, inst_id, fname)
    if pre is not None:
        pre_hooks.append(pre)
    if post is not None:
        post_hooks.append(post)


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
        if pre and pre_hooks:
            if pre is True:
                pre_hooks[:] = []
            else:
                for p in pre_hooks:
                    if p is pre:
                        pre_hooks.remove(pre)
                        break
                else:
                    raise RuntimeError("Couldn't find the given 'pre' function in the pre hooks "
                                       f"for {class_name}.{fname}.")
        if post and post_hooks:
            if post is True:
                post_hooks[:] = []
            else:
                for p in post_hooks:
                    if p is post:
                        post_hooks.remove(post)
                        break
                else:
                    raise RuntimeError("Couldn't find the given 'post' function in the post hooks "
                                       f"for {class_name}.{fname}.")

        if not (pre_hooks or post_hooks):
            del hookdict[fname]
            if not hookdict:  # we just removed the last hook entry for this inst
                del _hooks[class_name][inst_id]
                if not _hooks[class_name]:  # removed last entry for this class
                    del _hooks[class_name]
    else:
        warnings.warn(f"No hook found for method '{fname}' for class '{class_name}' and instance "
                      f"'{inst_id}'.")
