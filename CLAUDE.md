# Development Guidelines

Unless otherwise directed...

- Use the "dev" environment in the pixi.toml file in the root of this directory.

- Use `pixi run test` to invoke tests.

- Do not invoke git commands, let the developer handle that.

# Code Style

- Generated code should pass `ruff check`. Rules for ruff are established in pyproject.toml.

- Prefer single quotes to double quotes, except to define docstrings. When nesting quotes, prefer double quotes for the outer string.

- Documentation is in the numpy doc style.

- Do not use emojis.
