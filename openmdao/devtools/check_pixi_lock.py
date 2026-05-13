#!/usr/bin/env python
"""Pre-commit hook to prevent accidental pixi.lock commits without pixi.toml changes.

Lockfile updates should only occur when dependencies in pixi.toml change.
This prevents committing pixi.lock alone, which usually indicates
an accidental commit rather than an intentional dependency update.

To bypass this check for intentional standalone lockfile commits:
    git commit --no-verify
"""

import subprocess
import sys


def get_staged_files():
    """Get list of staged files."""
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only'],
        capture_output=True,
        text=True
    )
    return result.stdout.strip().split('\n') if result.stdout.strip() else []


def main():
    """Check if pixi.lock is staged without pixi.toml."""
    staged_files = get_staged_files()

    has_pixi_lock = 'pixi.lock' in staged_files
    has_pixi_toml = 'pixi.toml' in staged_files

    if has_pixi_lock and not has_pixi_toml:
        print('ERROR: pixi.lock is staged but pixi.toml is not.')
        print()
        print('Lockfile updates should only happen when updating dependencies in pixi.toml.')
        print('The lockfile is typically updated via the automated workflow on the 15th of each month.')
        print()
        print('If this is an intentional standalone lockfile update, you can bypass this check with:')
        print('  git commit --no-verify')
        print()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
