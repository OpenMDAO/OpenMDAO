# OpenMDAO Skills for Claude Code

This skill file provides Claude Code with context about the OpenMDAO framework
installed on this machine. Richer guidance is being added incrementally —
skills covering components, solvers, derivatives, and optimization are coming
in future releases.

## When to use these skills

Trigger this skill when the user is working with OpenMDAO — building models,
running analyses, setting up optimizations, or debugging convergence issues.

## Key concepts Claude should know

- An OpenMDAO **model** is a tree of `System` objects (`ExplicitComponent`,
  `ImplicitComponent`, `Group`).
- Variables are declared in `setup()` via `add_input` / `add_output`.
- Connections are made with `connect()` or via promoted names.
- Derivatives are declared in `setup_partials()` and computed (or approximated)
  in `compute_partials()` / `linearize()`.
- The `Problem` object owns the model and the driver. Call `setup()` then
  `run_model()` or `run_driver()`.

## Key file locations

| What | Path |
|------|------|
| Package root | `{{OPENMDAO_PATH}}` |
| Docs (Jupyter Book source) | `{{OPENMDAO_DOCS}}` |
| Getting started | `{{OPENMDAO_DOCS}}/getting_started/getting_started.ipynb` |
| First analysis | `{{OPENMDAO_DOCS}}/basic_user_guide/single_disciplinary_optimization/first_analysis.ipynb` |
| First optimization | `{{OPENMDAO_DOCS}}/basic_user_guide/single_disciplinary_optimization/first_optimization.ipynb` |
| Component types overview | `{{OPENMDAO_DOCS}}/basic_user_guide/single_disciplinary_optimization/component_types.ipynb` |
| Multidisciplinary (Sellar) optimization| `{{OPENMDAO_DOCS}}/basic_user_guide/multidisciplinary_optimization/sellar.ipynb` |
| Linking variables | `{{OPENMDAO_DOCS}}/basic_user_guide/multidisciplinary_optimization/linking_vars.ipynb` |
| Recording | `{{OPENMDAO_DOCS}}/basic_user_guide/reading_recording/basic_recording_example.ipynb` |
| Runnable examples | `{{OPENMDAO_EXAMPLES}}` |

## Planned skills (coming in future releases)

- **Components** — ExplicitComponent, ImplicitComponent, setup patterns, units
- **Groups and connections** — promotion, connect(), auto_ivc
- **Solvers** — Newton, Broyden, NLBGS, linear solvers (DirectSolver, LNBGS, PETSc)
- **Derivatives** — compute_partials, check_totals, complex step (CS), finite difference (FD)
- **Optimization** — driver setup, design variables, constraints, objectives
- **Debugging** — check_setup, N2 diagram, check_totals workflow
