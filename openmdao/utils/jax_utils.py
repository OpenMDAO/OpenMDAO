"""
Utilities for the use of jax in combination with OpenMDAO.
"""
from collections.abc import Callable

try:
    import jax
except ImportError:
    jax = None


def jit_stub(f, *args, **kwargs):
    """
    Provide a dummy jit decorator for use if jax is not available.

    Parameters
    ----------
    f : Callable
        The function or method to be wrapped.
    *args : list
        Positional arguments.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    Callable
        The decorated function.
    """
    return f


def register_jax_component(comp_class):
    """
    Provide a class decorator that registers the given class as a pytree_node.

    This allows jax to use jit compilation on the methods of this class if they
    reference attributes of the class itself, such as `self.options`.

    Note that this decorator is not necessary if the given class does not reference
    `self` in any methods to which `jax.jit` is applied.

    Parameters
    ----------
    comp_class : class
        The decorated class.

    Returns
    -------
    object
        The same class given as an argument.

    Raises
    ------
    NotImplementedError
        If this class does not define the `_tree_flatten` and _tree_unflatten` methods.
    RuntimeError
        If jax is not available.
    """
    if jax is None:
        raise RuntimeError("jax is not available. "
                           "Try 'pip install openmdao[jax]' with Python>=3.8.")

    if not hasattr(comp_class, '_tree_flatten'):
        raise NotImplementedError(f'class {comp_class} does not implement method _tree_flatten.'
                                  f'\nCannot register {comp_class} as a jax jit-compatible '
                                  f'component.')

    if not hasattr(comp_class, '_tree_unflatten'):
        raise NotImplementedError(f'class {comp_class} does not implement method _tree_unflatten.'
                                  f'\nCannot register class {comp_class} as a jax jit-compatible '
                                  f'component.')

    jax.tree_util.register_pytree_node(comp_class,
                                       comp_class._tree_flatten,
                                       comp_class._tree_unflatten)
    return comp_class
