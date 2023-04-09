"""
This package contains various functions that may be useful in component compute methods,
along with implementations of their derivatives to make it easier for users to write
their corresponding compute partials methods.

Functions
---------

Functions may take one or more numeric arguments and some optional keyword arguments
that affect the behavior of the function. For example, axis is a common keyword for
a function like sum.

Functions may return more than one output.

Where possible, these functions mimic their numpy counterparts, though often with fewer options.
These functions are written to be relatively easy to wrap with om.ExplicitFuncComp.

Derivatives
-----------
Derivatives of function outputs with respect to their inputs are returned by functions named
`f'd_{func_name}'`. For instance, `'d_sum(x, axis=None)'` returns the jacobian matrix for
the sum function with respect to its single argument, x.

A function will return a derivative for each output and each input. For instance, say we have
a function named `foo(x, y, z)` that returns two values, `a` and `b`.

The corresponding `d_foo(x, y, z)` will return 6 elements. Think of them as being in row-major
order where there is one row for each output and one column for each argument. Then we have

```
da_dx, da_dy, da_dz, db_dx, db_dy, db_dz = d_foo(x, y, z)
```

By convention, functions may allow the user to omit calculations of derivatives wrt
some inputs through optional arguments named `d_{arg_name}`. If this argument is
`False`, then the corresponding return value in the output will return `None`.
For instance, below z is a fixed value and we don't need to compute derivatives wrt to it.

```
da_dx, da_dy, da_dz, db_dx, db_dy, db_dz = d_foo(x, y, z=1.0, d_z=False)
```

Now, da_dz and db_dz will both be None.
"""

from .smooth import act_tanh, d_act_tanh, smooth_max, d_smooth_max, smooth_min, d_smooth_min, \
    smooth_abs, d_smooth_abs

from .cs_safe import abs, d_abs, arctanh, d_arctanh, arctan2, d_arctan2, norm, d_norm

from .numpy import arcsin, d_arcsin, arccos, d_arccos, arccosh, d_arccosh, \
    arcsinh, d_arcsinh, arctan, d_arctan, cos, d_cos, cosh, d_cosh, cumsum, d_cumsum, dot, d_dot, \
    erf, d_erf, erfc, d_erfc, exp, d_exp, log, d_log, log10, d_log10, sin, d_sin, sinh, d_sinh, \
    sqrt, d_sqrt, sum, d_sum, tan, d_tan, tanh, d_tanh
