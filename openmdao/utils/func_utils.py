"""
Utilities for working with function objects or function source code.
"""

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

import inspect
import ast
import textwrap
import numpy as np

from openmdao.utils.code_utils import get_function_deps
from openmdao.utils.general_utils import shape2tuple
from openmdao.utils.om_warnings import issue_warning


def _get_annotations(func):
    """
    Retrieve annotation data for function inputs and return values.

    Parameters
    ----------
    func : function
        The function object.

    Returns
    -------
    dict
        Input metadata dictionary.
    dict
        Return value metadata dictionary.
    """
    annotations = getattr(func, '__annotations__', None)
    inmeta = {}
    outmeta = {}
    if annotations is not None:
        ret = None
        # get input info
        for name, meta in annotations.items():
            if name == 'return':
                ret = meta
            else:
                inmeta[name] = meta

        if ret is not None:  # we have output annotations
            for name, meta in ret:
                outmeta[name] = meta

    return inmeta, outmeta


def _get_outnames_from_code(func):
    """
    Use ast to determine (if possible) the names of the given function's return values.

    For example, if the given function has a line `return a, b, c`, then this will return
    ['a', 'b', 'c'].  If the function instead returns something other than simple names,
    those will be replaced with None.  For example, a function containing the line
    `return a, b+1, sin(c)` would cause this to return ['a', None, None].

    If the given function contains more than one return line, this will raise an exception.

    If an ast cannot be constructed for the function, if for example it is a wrapper for
    compiled code, an exception will be raised.

    Parameters
    ----------
    func : function
        A reference to the function whose source code will be parsed for return value names.
    """
    src = textwrap.dedent(inspect.getsource(func))
    scanner = _ReturnNamesCollector()
    scanner.visit(ast.parse(src, mode='exec'))
    return scanner._ret_names


def compute_out_shapes(func, ins, outs):
    """
    Compute the shapes of outputs based on those of the inputs.

    Parameters
    ----------
    func : function
        The function whose outputs' shapes will be determined.
    ins : dict
        Dict of input metadata containing input shapes.
    outs : dict
        Dict of output metadata that will be updated with shape information.
    """
    need_shape = False
    for ometa in outs.values():
        try:
            ometa['shape']
        except KeyError:
            need_shape = True
            break

    args = []
    for name, meta in ins.items():
        if meta['val'] is not None:
            args.append(meta['val'])
        else:
            try:
                shp = meta['shape']
            except KeyError:
                raise RuntimeError(f"Can't determine shape of input '{name}'.")
            if jax is not None:
                args.append(jax.ShapedArray(shape2tuple(shp), dtype=np.float64))

    if need_shape:  # output shapes weren't provided by annotations
        if jax is None:
            raise RuntimeError("Some return values have unknown shape, and jax is required to "
                               "(possibly) determine the output shapes based on the input shapes, "
                               "but jax was not found.  Either install jax (pip install jax), or "
                               "add return value annotations that define the shapes of the "
                               "return values.")

    if jax is not None:  # compute shapes as a check against annotated value (if any)
        if 'np' in func.__globals__:
            func.__globals__['np'] = jnp
        try:
            v = jax.make_jaxpr(func)(*args)
        except Exception as err:
            if need_shape:
                raise RuntimeError("Jax failed to determine the output shapes based on the input "
                                   f"shapes. The error was: {err}.")
            issue_warning("Jax failed to determine the output shapes based on the input "
                          "shapes in order to check the provided annotated values.  The jax "
                          f"error was: {err}.")
        else:
            for val, name in zip(v.out_avals, outs):
                oldshape = outs[name].get('shape')
                if oldshape is not None and oldshape != val.shape:
                    raise RuntimeError(f"Annotated shape for return value '{name}' of {oldshape} "
                                       f"doesn't match computed shape of {val.shape}.")
                outs[name]['shape'] = val.shape
        finally:
            if 'np' in func.__globals__:
                func.__globals__['np'] = np


def get_func_info(func, comp_meta=None):
    """
    Retrieve metadata associated with function inputs and return values.

    Return value metadata can come from annotations or (shape only) can be determined
    using jax if the input shapes or values are known.  Return value names can be defined
    in annotations or can be determined from the function itself provided that the return
    values are internal function variable names.

    Parameters
    ----------
    func : function
        The function to be queried for input and return value info.
    comp_meta : dict or None
        Dict containing component wide options like shape and units.

    Returns
    -------
    dict
        Dictionary of metdata for inputs.
    dict
        Dictionary of metadata for return values.
    """
    # TODO: get func source and re-exec with redifined globals to replace numpy with jax numpy so
    # functions defined with regular numpy stuff internally will still work.
    ins = {}

    sig = inspect.signature(func)

    comp_shape = None if comp_meta is None else comp_meta['shape']
    if comp_shape is not None:
        comp_shape = shape2tuple(comp_shape)

    # first, retrieve inputs from the function signature
    for name, p in sig.parameters.items():
        ins[name] = meta = {}
        if p.annotation is not inspect.Parameter.empty:
            if isinstance(p.annotation, dict):
                meta.update(p.annotation)
            else:
                raise TypeError(f"Input '{name}' annotation should be a dict, but is type "
                                f"'{type(p.annotation).__name__}'.")
        meta['val'] = p.default if p.default is not inspect._empty else None
        if meta['val'] is not None:
            if np.isscalar(meta['val']):
                shape = ()
            else:
                shape = meta['val'].shape
            meta_shape = meta['shape'] if 'shape' in meta else shape
            cmpshape = shape if comp_shape is None else comp_shape
            shapes = set([shape, meta_shape, cmpshape])

            if len(shapes) > 1:
                if shape != meta_shape:
                    raise ValueError(f"Input '{name}' default value has shape {shape}, but shape "
                                     f"was specified as {meta_shape} in annotation.")
                if cmpshape != shape:
                    raise ValueError(f"Input '{name}' default value has shape {shape}, but shape "
                                     f"was specified as {cmpshape} in the component.")
            meta['shape'] = shape
        else:
            if comp_shape is not None:
                if'shape' not in meta or meta['shape'] is None:
                    meta['shape'] = comp_shape
                elif comp_shape != meta['shape']:
                    raise ValueError(f"Input '{name}' has annotated shape {meta['shape']}, but "
                                     f"shape was specified as {comp_shape} in the component.")

    outmeta = {}
    if sig.return_annotation is not inspect.Signature.empty:
        outmeta.update(sig.return_annotation)

    # Parse the function code to possibly identify the names of the return values and input/output
    # dependencies. Return names will be non-None only if they are simple name, e.g.,
    #  return a, b, c
    outlist = []
    try:
        ret_info = get_function_deps(func)
    except RuntimeError:
        #  this could happen if function is compiled or has multiple return lines
        if not outmeta:
            raise RuntimeError("Couldn't determine function return names or number of return "
                               "values based on AST and no return value annotations were supplied.")
        issue_warning("Couldn't determine function return names based on AST.  Assuming number of "
                      "return values matches number of return value annotations.")
        outlist = list(outmeta.items())
    else:
        for o, deps in ret_info:
            if o is not None and '.' in o:  # don't allow dots in return value names
                issue_warning(f"Ignoring return name '{o}' because names containing '.' are not "
                              "supported.")
                o = None
            outlist.append([o, {'deps': deps}])

    notfound = []
    for i, (oname, ometa) in enumerate(outmeta.items()):
        for n, meta in outlist:
            if n == oname:
                if meta is not ometa:
                    meta.update(ometa)
                break
        else:  # didn't find oname
            notfound.append(oname)

    if notfound:  # try to fill in the unnamed slots with annotated output data
        inones = [i for i, (n, m) in enumerate(outlist) if n is None]  # indices with name of None
        if len(notfound) != len(inones):
            raise RuntimeError(f"Number of unnamed return values ({len(inones)}) doesn't match "
                               f"number of unmatched annotated return values ({len(notfound)}).")

        # number of None return slots equals number of annotated entries not found in outlist
        for i_olist, name_notfound in zip(inones, notfound):
            annotated_meta = outmeta[name_notfound]
            _, ret_meta = outlist[i_olist]
            ret_meta.update(annotated_meta)
            outlist[i_olist] = (name_notfound, ret_meta)

    outs = {n: m for n, m in outlist}

    compute_out_shapes(func, ins, outs)

    return ins, outs
