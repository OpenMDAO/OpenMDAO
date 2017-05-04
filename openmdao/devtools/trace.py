from __future__ import print_function

import os
import sys
import gc

from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.vectors.vector import Vector, Transfer
from openmdao.core.problem import Problem
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver
from openmdao.core.component import Component
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent

trace = os.environ.get('OPENMDAO_TRACE')
trace_mem = os.environ.get('OPENMDAO_TRACE_MEM')

if trace_mem:
    from openmdao.devtools.debug import mem_usage

_class_tuple = (
    System,
    Component,
    ExplicitComponent,
    ImplicitComponent,
    Group,
    Vector,
    Transfer,
    Problem,
    Solver,
    Driver
)

_method_set = set()
_method_counts = {}
_mem_changes = {}

for klass in _class_tuple:
    _method_set.update(n for n, obj in klass.__dict__.items()
                       if callable(obj))


_TRACE_DICT = {
    'openmdao': (_method_set, _class_tuple),
    'setup': ({
        '_setup_partials',
        '_setup_vars',
        '_setup_var_data',
        '_setup_var_index_ranges',
        '_setup_var_sizes',
        '_setup_global_connections',
        '_setup_connections',
        '_setup_global',
        '_setup_transfers',
        '_setup_var_index_maps',
        '_setup_vectors',
        '_setup_bounds',
        '_setup_scaling',
        '_setup_jacobians',
        '_setup_solvers',
    }, object),
    'solvers': ({
        'solve_nonlinear',
        '_solve_nonlinear',
        'solve_linear',
        '_solve_linear',
        '_iter_initialize',
        '_iter_execute',
        '_linearize_children',
        'solve',
        '_iter_get_norm',
    }, _class_tuple),
    'linear_solve': ({
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
    }, _class_tuple)
}


if trace:
    _active_traces, insts = _TRACE_DICT[trace]
elif trace_mem:
    _active_traces, insts = _TRACE_DICT[trace_mem]

    def print_totals():
        items = sorted(_mem_changes.items(), key=lambda x: x[1])
        for n, delta in items:
            if delta > 0.0:
                print("%s %g" % (n, delta))
    import atexit
    atexit.register(print_totals)
else:
    _active_traces = None

if _active_traces:
    start_mem = 0

    _callstack = []
    def trace_return(frame, event, arg):
        if event is not 'return':
            return
        func_name = frame.f_code.co_name
        if func_name in _active_traces:
            data = _callstack.pop()

            if trace_mem:
                fullname, mem_start = data
                delta = mem_usage() - mem_start
                if fullname not in _mem_changes:
                    _mem_changes[fullname] = 0
                if delta > 0.0:
                    _mem_changes[fullname] += delta

                if len(_callstack) == 0:
                    gc.enable()

    def trace_calls(frame, event, arg):
        if event is 'call':
            func_name = frame.f_code.co_name
            if _active_traces and func_name in _active_traces:
                loc = frame.f_locals
                if 'self' in loc:
                    self = loc['self']
                    if isinstance(self, insts):
                        fullname = '.'.join((self.__class__.__name__, func_name))
                    else:
                        return
                else:
                    return

                # turn off the gc if we're in the middle of any memory trace
                if trace_mem and gc.isenabled() and len(_callstack) == 0:
                    gc.disable()

                if trace_mem:
                    _callstack.append((fullname, mem_usage()))
                else:
                    _callstack.append(func_name)
                    if fullname in _method_counts:
                        _method_counts[fullname] += 1
                    else:
                        _method_counts[fullname] = 1
                    print('   ' * len(_callstack),
                          "%s (%d)" % (fullname, _method_counts[fullname]))

                return trace_return


    sys.settrace(trace_calls)
