"""
Functions for handling runtime function hooks.
"""

from collections import defaultdict
from functools import wraps
import inspect
import warnings


def _reset_all_hooks():
    global _hooks, _hook_skip_classes
    _hooks = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    )
    _hook_skip_classes = set()


# global dict of hooks
# {class_name: { inst_id: {fname: <hook_meta_dict>}}}
# At class_name level, there might be a 'None' entry, which means that all instances of
# that class will have have the hooks specified in the 'None' dict.  Otherwise, entries
# there will be specific instance names.
# The <hook_meta_dict> contains the following:
# {
#    'pre': funct or None,   # hook function for pre-execution
#    'post': funct or None,  # hook function for post-execution
# }
_hooks = None
_reset_all_hooks()

# classes found here are known to contain no hooks within themselves or their ancestors
_hook_skip_classes = set()

# global switch that turns hook machinery on/off
use_hooks = False


def _setup_hooks(obj):
    """
    Wrap object's methods with a hook checking function if hooks are defined for any of them.

    Parameters
    ----------
    obj : object
        The object whose methods may be wrapped.
    """
    # _setup_hooks should be called after 'obj' can return a valid name from _get_inst_id().
    # For example, in Problem, it can happen in __init__, but in Component and Group it shouldn't
    # happen until _setup_procs because that's the earliest point where the component/group has a
    # valid pathname.
    if use_hooks and obj.__class__ not in _hook_skip_classes:

        classes = inspect.getmro(obj.__class__)
        for c in classes:
            if c.__name__ in _hooks:
                hk = _hooks[c.__name__]
                break
        else:
            # didn't find any matching hooks for this class or any base class, so skip in future
            _hook_skip_classes.update(classes)
            return

        # any object where we register hooks must define the '_get_inst_id' method.
        ident = obj._get_inst_id()

        if ident in hk:
            hk = hk[ident]
        elif None in hk:
            hk = hk[None]
        else:
            return

        for name, hook in hk.items():
            method = getattr(obj, name, None)
            if method is not None:
                # if _hook_ attr is present, we've already wrapped this method
                if hasattr(method, '_hook_'):
                    method = method.f  # unwrap the method prior to re-wrapping
                setattr(obj, name, _hook_decorator(method, obj, hook))


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
    def check_hooks(*args, **kwargs):
        pre = hookmeta['pre']
        if pre:
            pre(inst)

        f(*args, **kwargs)

        post = hookmeta['post']
        if post:
            post(inst)

    check_hooks._hook_ = True  # to prevent multiple decoration of same function

    return wraps(f)(check_hooks)


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

    hook = _hooks[class_name][inst_id][fname]
    if pre is not None:
        hook['pre'] = pre
    if post is not None:
        hook['post'] = post


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
    pre : bool (True)
        If True, the hook that runs before the named function runs will be removed.
    post : bool, (True)
        If True, the hook that runs after the named function runs will be removed.
    """
    hookdict = _hooks[class_name][inst_id]
    if fname in hookdict:
        hook = hookdict[fname]
        if pre and hook['pre']:
            hook['pre'] = None
        if post and hook['post']:
            hook['post'] = None

        if not (hook['pre'] or hook['post']):
            del hookdict[fname]
            if not hookdict:  # we just removed the last hook entry for this inst
                del _hooks[class_name][inst_id]
                if not _hooks[class_name]:  # removed last entry for this class
                    del _hooks[class_name]
    else:
        warnings.warn("No hook found for method '{}' for class '{}' and instance '{}'.".format(
            fname, class_name, inst_id
        ))
