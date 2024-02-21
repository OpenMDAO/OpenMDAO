"""
Functions for handling runtime function hooks.
"""

from inspect import getmro
import warnings
import sys

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
    def __init__(self, class_name, inst_id, pre=None, post=None, ncalls=None, exit=False, **kwargs):
        global _hook_counter
        self._id = _hook_counter
        _hook_counter += 1

        self.class_name = class_name
        self.inst_id = inst_id
        self.pre = pre
        self.post = post
        self.ncalls = ncalls
        self.exit = exit
        self.kwargs = kwargs

    def __repr__(self):
        return f"<_HookMeta {self.class_name} {self.inst_id} {self.pre} {self.post} {self.ncalls} "\
               f"{self.exit} {self.kwargs}>"

    def copy(self):
        return _HookMeta(self.class_name, self.inst_id, self.pre, self.post, self.ncalls,
                         self.exit, **self.kwargs)


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
                            # make instance versions of the 'None' hooks
                            hooklist = [h.copy() for h in hooklist]
                            all_hooks[funcname].extend(hooklist)

        for funcname, hooks in all_hooks.items():
            method = getattr(obj, funcname)
            if isinstance(method, HookDecorator):
                raise RuntimeError(f"Method '{funcname}' of class '{c.__qualname__}' has already "
                                   "been wrapped with a hook decorator.")
            else:
                setattr(obj, funcname, HookDecorator(hooks))


class HookDecorator:
    def __init__(self, hooks):
        self.hooks = []
        self.add_hooks(hooks)

    def add_hooks(self, hooks):
        """
        Add additional pre and/or post hooks to this method.

        Parameters
        ----------
        hooks : list
            List of hook data.
        """
        self.hooks.extend(hooks)
        self.hooks = sorted(self.hooks, key=lambda x: x._id)  # put them in chronological order

    def pre_hook_iter(self):
        """
        Iterate over all pre_hooks for this method.

        Yields
        ------
        list of (function, int, bool, dict)
            Pre-hook metadata.
         """
        for ident, class_name in self.hooks:
            try:
                pre_hooks, _ = _hooks[class_name][ident][self.__name__]
            except KeyError:
                pass
            else:
                yield from pre_hooks

    def post_hook_iter(self):
        """
        Iterate over all post_hooks for this method.

        Yields
        ------
        list of (function, int, bool, dict)
            Post-hook metadata.
        """
        for ident, class_name in self.hooks:
            try:
                _, post_hooks = _hooks[class_name][ident][self.__name__]
            except KeyError:
                pass
            else:
                yield from post_hooks

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
        for hookmeta in hooks:
            hook, ncalls, ex, kwargs, _ = hookmeta
            if ncalls is None:
                hook(inst, **kwargs)
                if ex:
                    sys.exit()
            else:
                inst_id = inst._get_inst_id()
                if inst_id not in ncalls:
                    # must have been registered with 'None', meaning all instances, so get initial value
                    # for this instance
                    ncalls[inst_id] = ncalls[None]

                if ncalls[inst_id] > 0:
                    ncalls[inst_id] -= 1
                    hook(inst, **kwargs)
                    if ex:
                        sys.exit()

    def __call__(self, *args, **kwargs):
        _run_hooks(self.pre_hook_iter(), self.inst)
        ret = self.func(*args, **kwargs)
        _run_hooks(self.post_hook_iter(), self.inst)
        return ret


# def _hook_decorator(f, inst, pre_post_hooks):
#     """
#     Wrap a method with pre and/or post hook checks.

#     Parameters
#     ----------
#     f : method
#         The method being wrapped.
#     inst : object
#         The instance that owns the method.
#     pre_post_hooks : tuple of (list, list)
#         Pre and post hook lists.
#     """
#     pre_hooks, post_hooks = pre_post_hooks

#     # args and kwargs are arguments to the method that is being wrapped
#     def execute_hooks(*args, **kwargs):
#         _run_hooks(pre_hooks, inst)
#         ret = f(*args, **kwargs)
#         _run_hooks(post_hooks, inst)
#         return ret

#     execute_hooks._hashook_ = True  # to prevent multiple decoration of same function

#     return wraps(f)(execute_hooks)


# def _get_hook_list_iters(class_name, inst_id, fname):
#     """
#     Retrieve the pre and post hook list iterators for the given class, instance, and function name.

#     They are iterators of lists because under some circumstances, e.g., when adding a 'None'
#     hook after non-None instance hooks were already added, the 'None' hook will need to be added
#     to *all* of the existing non-None instance hook lists.

#     Parameters
#     ----------
#     class_name : str
#         The name of the class owning the method where the hook will be applied.
#     inst_id : str, optional
#         The name of the instance owning the method where the hook will be applied.
#     fname : str
#         The name of the function where the pre and/or post hook will be applied.

#     Yields
#     ------
#     tuple of (list, list)
#         Pre and post hook lists.
#     """
#     global _hooks
#     if class_name in _hooks:
#         cmeta = _hooks[class_name]
#     else:
#         _hooks[class_name] = cmeta = {}

#     if inst_id in cmeta:
#         imeta = cmeta[inst_id]
#     else:
#         cmeta[inst_id] = imeta = {}

#     if fname not in imeta:
#         # check for any existing None hooks because we need to add those first
#         nonehooks = None
#         if None in cmeta:
#             nonemeta = cmeta[None]
#             if fname in nonemeta:
#                 nonehooks = nonemeta[fname]

#         if nonehooks is None:
#             imeta[fname] = ([], [])
#         else:
#             imeta[fname] = ([h.copy() for h in nonehooks[0]], [h.copy() for h in nonehooks[1]])

#     if inst_id is None:
#         # special case where we have to add the None hook to all existing non-None hook lists
#         # that match the fname
#         for n, meta in cmeta.items():
#             if fname in meta:
#                 yield meta[fname]
#         return

#     yield imeta[fname]


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

    imeta[fname].append(_HookMeta(class_name, inst_id, pre, post, ncalls, exit, **kwargs))


def _find_hooks(class_name, inst_id, fname):
    """
    Find the pre and post hooks for the given class, instance, and function name.

    Parameters
    ----------
    class_name : str
        The name of the class owning the method where the hook will be applied.
    inst_id : str or None
        The name of the instance owning the method where the hook will be applied.
    fname : str
        The name of the function where the pre and/or post hook will be applied.

    Returns
    -------
    tuple of (list, list)
        Pre and post hook lists.
    """
    global _hooks
    if class_name not in _hooks:
        return (), ()

    cmeta = _hooks[class_name]

    if inst_id not in cmeta:
        return (), ()

    imeta = cmeta[inst_id]

    return imeta.get(fname, (None, None))


def _remove_hook(to_remove, hooks, class_name, fname, hook_loc, inst_id):
    """
    Remove a hook function.

    Parameters
    ----------
    to_remove : bool or function
        If True, all hook functions in 'hooks' will be removed.  If a function, any function
        in 'hooks' that matches will be removed.
    hooks : list
        List of (hook_func, ncalls, exit, kwargs, inst_id) tuples.
    class_name : str
        The name of the class owning the method where the hook will be removed.
    fname : str
        The name of the function where the hooks are located.
    hook_loc : str
        Either 'pre' or 'post', indicating the hooks run before or after respectively the
        function specified by fname.
    inst_id : str or None
        The name of the instance owning the method where the hook will be applied.
    """
    if to_remove and hooks:
        if to_remove is True:
            hooks[:] = []
        else:
            for hook in hooks:
                p, _, _, _, iid = hook
                if p is to_remove and iid == inst_id:
                    hooks.remove(hook)
                    break
            else:
                issue_warning(f"Couldn't find the given '{hook_loc}' function in the "
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
        classhooks = _hooks[class_name]
    except KeyError:
        return

    todel = []
    for instkey, hookdict in classhooks.items():
        if not (inst_id is None or instkey == inst_id):
            continue
        if fname in hookdict:
            pre_hooks, post_hooks = hookdict[fname]
            _remove_hook(pre, pre_hooks, class_name, fname, 'pre', inst_id)
            _remove_hook(post, post_hooks, class_name, fname, 'post', inst_id)

            if not (pre_hooks or post_hooks):
                del hookdict[fname]
                if not hookdict:  # we just removed the last hook entry for this inst
                    todel.append(inst_id)
        else:
            warnings.warn(f"No hook found for method '{fname}' for class '{class_name}' and "
                          f"instance '{inst_id}'.")

    if todel:
        for name in todel:
            try:
                del classhooks[name]
            except KeyError:
                pass
        if not classhooks:  # removed last entry for this class
            del _hooks[class_name]
