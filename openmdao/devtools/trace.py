import os
import sys

from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.vectors.vector import Vector, Transfer
from openmdao.core.problem import Problem
from openmdao.core.driver import Driver
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent

_class_tuple = (
    Group, ExplicitComponent, ImplicitComponent,
    #Vector,
    Transfer,
    Problem,
    Driver
)

_method_set = set()
_method_counts = {}

for klass in _class_tuple:
    _method_set.update(n for n in klass.__dict__ if not n.startswith('__'))


_TRACE_DICT = {
    'openmdao': _method_set,
    'solvers': {
        'solve_nonlinear',
        '_solve_nonlinear',
        'solve_linear',
        '_solve_linear',
        '_iter_initialize',
        '_iter_execute',
        '_linearize_children',
        'solve',
        '_iter_get_norm',
    },
    'linear_solve': {  # value of OPENMDAO_TRACE
        'apply_linear',
        '_apply_linear',
        'solve_linear',
        '_solve_linear',
        'compute_partial_derivs',
        'compute_jacvec_product',
        '_transfer',
        '_negate_jac',
        '_linearize',
        'linearize',
        'solve',
        'compute_total_derivs',
        '_iter_execute',
        '_iter_initialize',
        '_iter_get_norm',
    }
}

try:
    _active_traces = _TRACE_DICT[os.environ['OPENMDAO_TRACE']]
except KeyError:
    pass
else:
    _callstack = []
    def trace_return(frame, event, arg):
        if event is not 'return':
            return
        func_name = frame.f_code.co_name
        if func_name in _active_traces:
            _callstack.pop()

    def trace_calls(frame, event, arg):
        if event is 'call':
            func_name = frame.f_code.co_name
            if _active_traces and func_name in _active_traces:
                _callstack.append(func_name)
                loc = frame.f_locals
                print '   ' * len(_callstack),
                if 'self' in loc:
                    if isinstance(loc['self'], System):
                        fullname = '.'.join((loc['self'].pathname,
                                             func_name))
                    elif hasattr(loc['self'], '__class__'):
                        fullname = '.'.join(("<%s %d>" %
                                             (loc['self'].__class__.__name__,
                                             id(loc['self'])),
                                             func_name))
                else:
                    fullname = func_name
                if fullname in _method_counts:
                    _method_counts[fullname] += 1
                else:
                    _method_counts[fullname] = 1
                print "%s (%d)" % (fullname, _method_counts[fullname])

                return trace_return


    sys.settrace(trace_calls)
