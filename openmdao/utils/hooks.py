"""
Functions for handling runtime function hooks.
"""

import sys
import warnings
import weakref
from inspect import getmro

from openmdao.utils.om_warnings import issue_warning


# global dict of hooks
# {class_name: { inst_id: {fname: [pre_hooks, post_hooks]}}}
# At class_name level, there might be a 'None' entry, which means that all instances of
# that class will have have the hooks specified in the 'None' dict.  Otherwise, entries
# there will be specific instance names.
# pre_hooks and post_hooks are lists of functions to call before or after the function named
# 'fname'.
_hooks = {}

# use this to keep hook ordering consistent across multiple calls to register_hook
_hook_counter = 0

# global switch that turns hook machinery on/off. Need it on in general for the default
# reporting system
use_hooks = True


def _reset_all_hooks():
    global _hooks
    _hooks = {}


class _HookMeta(object):
    def __init__(self, class_name, inst_id, hook, ncalls=None, exit=False, **kwargs):
        global _hook_counter
        self._stamp = _hook_counter
        _hook_counter += 1

        self.class_name = class_name
        self.inst_id = inst_id
        self.hook = hook
        self.ncalls = ncalls
        self.exit = exit
        self.kwargs = kwargs
        self.children = []  # if we're a 'None' inst_id hook, keep track of our child hooks

    def __repr__(self):
        return f"<_HookMeta {self.class_name} {self.inst_id} {self.hook} {self.ncalls} "\
               f"{self.exit} {self.kwargs}>"

    def __call__(self, inst):
        if self.ncalls is None:
            self.hook(inst, **self.kwargs)
            if self.exit:
                sys.exit()
        else:
            if self.ncalls > 0:
                self.ncalls -= 1
                self.hook(inst, **self.kwargs)
                if self.exit:
                    sys.exit()

    def copy(self):
        hm = _HookMeta(self.class_name, self.inst_id, self.hook, self.ncalls,
                       self.exit, **self.kwargs)
        # keep the same stamp so that the order of hooks doesn't change
        hm._stamp = self._stamp
        if self.inst_id is None:
            self.children.append(hm)

        return hm

    def deactivate(self):
        self.ncalls = 0
        for child in self.children:
            child.deactivate()


class _HookDecorator(object):
    def __init__(self, inst, func, hooks):
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

        self.inst = weakref.ref(inst)
        self.func = func
        self.pre_hooks = []
        self.post_hooks = []
        self.add_hooks(hooks)

    def add_hooks(self, hooks):
        """
        Add additional pre and/or post hooks to this method.

        Parameters
        ----------
        hooks : list
            List of hook data.
        """
        # copy any hooks with inst_id of None so we can modify them without modifying the original
        for prehook, posthook in hooks:
            if prehook is not None:
                if prehook.inst_id is None:
                    prehook = prehook.copy()
                self.pre_hooks.append(prehook)
            if posthook is not None:
                if posthook.inst_id is None:
                    posthook = posthook.copy()
                self.post_hooks.append(posthook)

        # put the hooks in order by stamp
        self.pre_hooks = sorted(self.pre_hooks, key=lambda x: x._stamp)
        self.post_hooks = sorted(self.post_hooks, key=lambda x: x._stamp)

    def _run_hooks(self, hooks):
        """
        Run the given list of hooks.

        Parameters
        ----------
        hooks : list
            List of hook data.
        inst : object
            Object instance to pass to hook functions.
        """
        inst = self.inst()
        if inst is None:
            return

        for hook in hooks:
            hook(inst)

    def __call__(self, *args, **kwargs):
        """
        Run the function with any pre and post hooks.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        object
            The return value of the function.
        """
        self._run_hooks(self.pre_hooks)
        ret = self.func(*args, **kwargs)
        self._run_hooks(self.post_hooks)
        return ret


def _setup_hooks(obj):
    """
    Wrap object's methods with a hook checking function if hooks are defined for any of them.

    Parameters
    ----------
    obj : object
        The object whose methods may be wrapped.
    """
    global _hooks

    # _setup_hooks should be called after 'obj' can return a valid name from _get_inst_id().
    # For example, in Problem, it can happen in __init__, but in Component and Group it shouldn't
    # happen until _setup_procs because that's the earliest point where the component/group has a
    # valid pathname.
    if use_hooks:

        # any object where we register hooks must define the '_get_inst_id' method.
        ident = obj._get_inst_id()

        if ident is None:
            raise RuntimeError(f"Object {obj} must define a '_get_inst_id' method that returns "
                               "a unique identifier for the object.")

        all_hooks = {}

        for c in getmro(obj.__class__):
            if c.__qualname__ in _hooks:
                classmeta = _hooks[c.__qualname__]

                if ident in classmeta:
                    instmeta = classmeta[ident]
                    for funcname, hooklist in instmeta.items():
                        method = getattr(obj, funcname, None)
                        if method is not None and callable(method):
                            if funcname not in all_hooks:
                                all_hooks[funcname] = []
                            all_hooks[funcname].extend(hooklist)

                if None in classmeta:
                    instmeta = classmeta[None]
                    for funcname, hooklist in instmeta.items():
                        method = getattr(obj, funcname, None)
                        if method is not None and callable(method):
                            if funcname not in all_hooks:
                                all_hooks[funcname] = []
                            all_hooks[funcname].extend(hooklist)

        for funcname, hooks in all_hooks.items():
            method = getattr(obj, funcname)
            if isinstance(method, _HookDecorator):
                method.add_hooks(hooks)
            else:
                setattr(obj, funcname, _HookDecorator(obj, method, hooks))


def _register_hook(fname, class_name, inst_id=None, pre=None, post=None, ncalls=None, exit=False,
                   **kwargs):
    """
    Register a hook function.

    Note that the 'class_name' arg should be the __qualname__ of the class, so for a nested class,
    the name would include the names of any containing classes as well, with each class name
    separated by a '.'.

    Parameters
    ----------
    fname : str
        The name of the function where the pre and/or post hook will be applied.
    class_name : str
        The name of the class owning the method where the hook will be applied.
    inst_id : str or None
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

    global _hooks
    if class_name in _hooks:
        cmeta = _hooks[class_name]
    else:
        _hooks[class_name] = cmeta = {}

    if inst_id in cmeta:
        imeta = cmeta[inst_id]
    else:
        cmeta[inst_id] = imeta = {}

    if fname not in imeta:
        imeta[fname] = []

    if pre is None:
        pre_hook = None
    else:
        pre_exit = exit if post is None else False
        pre_hook = _HookMeta(class_name, inst_id, pre, ncalls, pre_exit, **kwargs)

    if post is None:
        post_hook = None
    else:
        post_hook = _HookMeta(class_name, inst_id, post, ncalls, exit, **kwargs)

    imeta[fname].append((pre_hook, post_hook))


def _deactivate_hook(to_remove, hook):
    """
    Deactivate a hook function.

    Parameters
    ----------
    to_remove : bool or function
        If True, all hook functions in 'hooks' will be removed.  If a function, any function
        in 'hooks' that matches will be removed.
    hook : _HookMeta or None
        Hook metadata object.
    """
    if to_remove and hook is not None:
        if to_remove is True:
            hook.deactivate()
        else:  # to_remove is a hook function
            if to_remove is hook.hook:
                hook.deactivate()


def _unregister_hook(fname, class_name, inst_id=None, pre=True, post=True):
    """
    Unregister a hook function.

    By default, both pre and post hooks will be deactivated if they are present. To avoid
    removal of pre or post, you must set the pre or post arg to False.

    Parameters
    ----------
    fname : str
        The name of the function where the pre and/or post hook will be deactivated.
    class_name : str
        The name of the class owning the method where the hook will be deactivated.
    inst_id : str, optional
        The name of the instance owning the method where the hook will be deactivated.
    pre : bool or function, (True)
        If True, hooks that run before the named function runs will be deactivated. If pre is a
        function, then that function will have its hook(s) deactivated.
    post : bool or function, (True)
        If True, hooks that run after the named function runs will be deactivated. If post is a
        function, then that function will have its hook(s) deactivated.
    """
    try:
        classhooks = _hooks[class_name]
    except KeyError:
        return

    for instkey, hookdict in classhooks.items():
        if not (inst_id is None or instkey == inst_id):
            continue
        if fname in hookdict:
            for pre_hook, post_hook in hookdict[fname]:
                _deactivate_hook(pre, pre_hook)
                _deactivate_hook(post, post_hook)
        else:
            warnings.warn(f"No hook found for method '{fname}' for class '{class_name}' and "
                          f"instance '{inst_id}'.")
