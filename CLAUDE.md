# Project

OpenMDAO provides analysis and optimization of multidisciplinary systems using the modular approach to unified derivatives.

# Development Guidelines

- Use the "dev" environment in the pixi.toml file in the root of this project.

- Do not invoke git commands, let the developer handle that, but suggest a commit message.

# Code Style

- Prefer single quotes to double quotes, except to define docstrings. When nesting quotes, prefer double quotes for the outer string.

- Documentation is in the numpy doc style.

- Do not use emojis.

# Validation

- Generated code should pass `ruff check`. Rules for ruff are established in pyproject.toml.

- Use `pixi run test <path>` to invoke tests in the relevant scope. Only run `pixi run test` if the user asks for the full test suite.
