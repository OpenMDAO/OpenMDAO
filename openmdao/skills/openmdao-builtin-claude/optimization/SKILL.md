# OpenMDAO Optimization Skills

## When to use this skill

Use when the user is setting up or debugging an optimization with OpenMDAO —
defining design variables, constraints, and objectives; choosing a driver;
or diagnosing why an optimization is not converging.

## Core pattern

```python
import openmdao.api as om

prob = om.Problem()
prob.model.add_subsystem('comp', MyComponent())

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('comp.x', lower=-10, upper=10)
prob.model.add_objective('comp.f')
prob.model.add_constraint('comp.g', upper=0.0)

prob.setup()
prob.run_driver()
```

## Key concepts

- **Design variables** — declared with `add_design_var()`; can have lower and/or upper bounds.
- **Objectives** — declared with `add_objective()`
- **Constraints** — declared with `add_constraint()`; can be equality (`equals=`) or inequality (`upper=`/`lower=`).
- **Drivers** — `ScipyOptimizeDriver` (SLSQP, COBYLA) for smaller problems; `pyOptSparseDriver` for larger problems.
- **Derivatives** — gradient-based optimizers require accurate total derivatives; use `prob.check_totals()` to verify.

## Key file locations

| What | Path |
|------|------|
| First optimization tutorial | `{{OPENMDAO_DOCS}}/basic_user_guide/single_disciplinary_optimization/first_optimization.ipynb` |
| Sellar MDO optimization | `{{OPENMDAO_DOCS}}/basic_user_guide/multidisciplinary_optimization/sellar_opt.ipynb` |
| Paraboloid example | `{{OPENMDAO_EXAMPLES}}/basic_opt_paraboloid.py` |
| Sellar opt example | `{{OPENMDAO_EXAMPLES}}/test_sellar_opt.py` |
| Beam optimization example | `{{OPENMDAO_DOCS}}/examples/beam_optimization_example.ipynb` |
| Scaling theory | `{{OPENMDAO_DOCS}}/theory_manual/scaling.ipynb` |
| Total derivatives theory | `{{OPENMDAO_DOCS}}/theory_manual/total_derivs_theory.ipynb` |

## Common debugging steps

1. Run `prob.check_totals()` — inaccurate derivatives are the most common cause of optimizer failure.
2. Check variable scaling — poorly scaled problems mislead gradient-based drivers.
3. Verify bounds — infeasible initial points prevent SLSQP from starting.
4. Check constraint signs

## Planned enhancements (coming in future releases)

- Guidance on pyOptSparseDriver and SNOPT/IPOPT
- Parallel derivative coloring
- Multipoint optimization patterns
- Surrogate-based optimization with MetaModelStructuredComp
