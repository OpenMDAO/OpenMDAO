# OpenMDAO CLAUDE.md file

The contents of this file will be inserted into either the user global CLAUDE.md file or the project level CLAUDE.md file by using the `openmdao skills` subcommands.

## OpenMDAO installed at

`{{OPENMDAO_PATH}}`

## Quick reference

```python
import openmdao.api as om

prob = om.Problem()
prob.model.add_subsystem('comp', MyComponent())
prob.setup()
prob.run_model()
```

## Docs and examples

- Docs (Jupyter Book source): `{{OPENMDAO_DOCS}}`
- Runnable examples: `{{OPENMDAO_EXAMPLES}}`
- Getting started: `{{OPENMDAO_DOCS}}/getting_started/getting_started.ipynb`
- First analysis: `{{OPENMDAO_DOCS}}/basic_user_guide/single_disciplinary_optimization/first_analysis.ipynb`

## Note

Richer per-topic guidance (components, solvers, derivatives, optimization) is
being added in future OpenMDAO releases. Upgrade OpenMDAO and re-run
`openmdao skills install` or `openmdao skills --global install` to get the latest skills.
