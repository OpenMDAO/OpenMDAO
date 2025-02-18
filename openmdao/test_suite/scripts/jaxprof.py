"""
This script is used to compare performance of JAX sparsity comp vs. plain sparsity comp.

Cmd line args control the following:

color: use coloring
prof: use profiler
rev: use reverse mode
check: check partials
jax: use JAX sparsity comp
sparse: use sparse partials
fd: use finite difference partials
show: show sparsity

"""

import time
import sys

import jax.numpy as jnp
import numpy as np

import openmdao.api as om
from openmdao.devtools.debug import profiling
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.test_suite.components.sparsity_comp import SparsityComp, JaxSparsityComp
from openmdao.utils.array_utils import rand_sparsity
from openmdao.utils.general_utils import do_nothing_context

args = sys.argv[1:]

use_coloring = 'color' in args
use_prof = 'prof' in args
rev = 'rev' in args
check = 'check' in args
use_jax = 'jax' in args
show = 'show' in args
use_fd = 'fd' in args
use_sparse = 'sparse' in args

if rev:
    nrows = 100
    ncols = 1000
else:
    nrows = 1000
    ncols = 100


def main():
    rng = np.random.default_rng(66)
    p = om.Problem()
    klass = JaxSparsityComp if use_jax else SparsityComp
    sparsity = rand_sparsity((nrows, ncols), 0.01, rng=rng)
    if not use_sparse:
        sparsity = sparsity.toarray()
    comp = p.model.add_subsystem('comp', klass(sparsity=sparsity))
    if use_coloring:
        comp.declare_coloring(show_summary=True, show_sparsity=show)
    if use_fd:
        comp.options['derivs_method'] = 'fd'

    print("Performance for args: ", args)

    t0 = time.perf_counter()
    p.setup()
    setup_time = time.perf_counter() - t0
    print(f'setup time: {setup_time}')

    t0 = time.perf_counter()
    p.run_model()
    run_time = time.perf_counter() - t0
    print(f'run_model time: {run_time}')

    if check:
        t0 = time.perf_counter()
        assert_check_partials(comp.check_partials(method='fd', show_only_incorrect=True))
        check_time = time.perf_counter() - t0
        print(f'check_partials time: {check_time}')

    if use_prof:
        profname = 'color' if use_coloring else 'nocolor'
        if use_jax:
            profname = 'jax_' + profname
        if rev:
            profname = profname + '_rev'
        if use_fd:
            profname = profname + '_fd'
        if not use_sparse:
            profname = profname + '_dense'

        ctx = profiling(profname + '.prof')
    else:
        ctx = do_nothing_context()

    reps = 1000
    t0 = time.perf_counter()
    with ctx:
        for i in range(reps):
            comp._linearize()  # force coloring to be computed
    t1 = time.perf_counter()
    print(f'linearize time: {t1 - t0} for {reps} reps')


if __name__ == '__main__':
    main()

