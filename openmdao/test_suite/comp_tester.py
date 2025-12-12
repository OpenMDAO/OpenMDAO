import traceback
import sys
from io import StringIO
import textwrap
from timeit import timeit

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals, assert_check_partials
from openmdao.utils.general_utils import indent_context


class ComponentTester(object):
    """
    A class to run a component through its paces.

    Parameters
    ----------
    comp_class : class
        The component class to test.
    comp_args : tuple
        The arguments to pass to the component constructor.
    comp_kwargs : dict
        The keyword arguments to pass to the component constructor.
    out_stream : file
        The stream to write the output to.
    """
    def __init__(self, comp_class, comp_args=(), comp_kwargs=None, reuse=False,
                 out_stream=sys.stdout):
        self._comp_class = comp_class
        self._comp_args = comp_args
        self._comp_kwargs = comp_kwargs if comp_kwargs is not None else {}
        self._reuse = reuse
        self._out_stream = out_stream
        self._p = None
        self._comp = None

        self._test_methods = {
            'partials': self.test_partials,
            'partials_complex': self.test_partials_complex,
            'totals': self.test_totals,
            'totals_complex': self.test_totals_complex,
            'coloring': self.test_coloring,
            'sparsity': self.test_sparsity,
            # 'fd': self.test_fd,
            # 'config': self.test_config,
        }

    def instance(self):
        return self._comp_class(*self._comp_args, **self._comp_kwargs)

    def wrapped_instance(self, setup_kwargs=None, final_setup=True, run_model=True):
        if not self._reuse or self._p is None:
            p = om.Problem()
            comp = p.model.add_subsystem('comp', self.instance())
            if setup_kwargs is None:
                setup_kwargs = {}

            self._p = p
            self._comp = comp

            p.setup(**setup_kwargs)

            if final_setup and not run_model:
                p.final_setup()
            if run_model:
                p.run_model()

        return self._p, self._comp

    def run(self, tests=None):
        if tests is None:
            tests = self._test_methods.keys()

        failed = []
        if self._out_stream is not None and tests:
            print(f"\nTesting class {self._comp_class.__name__}", file=self._out_stream)

        for test in tests:
            with indent_context(self._out_stream) as outer_stream:
                print(f'Running test {test}')
                try:
                    with indent_context(outer_stream) as inner_stream:
                        self._test_methods[test]()
                except Exception:
                    print(f'{test} failed:\n{traceback.format_exc()}')
                    failed.append(test)

        if failed:
            raise Exception(f'{len(failed)} tests failed: {", ".join(failed)}')

        return self

    def test_partials(self):
        _, comp = self.wrapped_instance()
        result, _= comp.check_partials(method='fd', show_only_incorrect=True, compact_print=True,
                                       out_stream=self._out_stream)
        assert_check_partials(result)

    def test_partials_complex(self):
        _, comp = self.wrapped_instance(setup_kwargs={'force_alloc_complex': True})
        result, _ = comp.check_partials(method='cs', show_only_incorrect=True, compact_print=True,
                                        out_stream=self._out_stream)
        assert_check_partials(result)

    def test_totals(self):
        p, comp = self.wrapped_instance()
        ofs = ['comp.' + n for n in comp._outputs.keys()]
        wrts = ['comp.' + n for n in comp._inputs.keys()]
        result = p.check_totals(of=ofs, wrt=wrts,
                                method='fd', show_only_incorrect=True, compact_print=True,
                                out_stream=self._out_stream)
        assert_check_totals(result)

    def test_totals_complex(self):
        p, comp = self.wrapped_instance(setup_kwargs={'force_alloc_complex': True})
        ofs = ['comp.' + n for n in comp._outputs.keys()]
        wrts = ['comp.' + n for n in comp._inputs.keys()]
        result = p.check_totals(of=ofs, wrt=wrts,
                                method='cs', show_only_incorrect=True, compact_print=True,
                                out_stream=self._out_stream)
        assert_check_totals(result)

    def test_sparsity(self):
        _, comp = self.wrapped_instance()
        msg = StringIO()
        comp.check_sparsity(method='fd', out_stream=msg)
        msg = textwrap.indent(msg.getvalue(), '      ')
        print(msg, file=self._out_stream)

    def test_coloring(self, nlinearize=2, show_sparsity=False):
        save = self._reuse
        self._reuse = False
        try:
            p_colored, comp_colored = self.wrapped_instance(final_setup=False, run_model=False)
            comp_colored.declare_coloring(show_summary=True, show_sparsity=show_sparsity)
            p_colored.run_model()
            comp_colored._linearize()  # force coloring to be computed

            if comp_colored._coloring_info.coloring is None:
                print(f'{comp_colored.msginfo}: Coloring failed to be computed.')
                return

            _, comp = self.wrapped_instance()
            t = timeit(lambda: comp._linearize(), number=nlinearize)
            print(f'\nuncolored linearize time: {t/nlinearize:10.4e} s')
            t_colored = timeit(lambda: comp_colored._linearize(), number=nlinearize)
            print(f'colored linearize time: {t_colored/nlinearize:10.4e} s')

        finally:
            self._reuse = save
