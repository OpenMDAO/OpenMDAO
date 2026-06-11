"""
Execute all documentation notebooks using papermill.

MPI-dependent notebooks (those with "mpi": true in their top-level metadata) are
executed under mpiexec. All others run as plain papermill subprocesses.

Usage (from openmdao/docs/):
    python execute_notebooks.py

Output notebooks are written to _executed_book/, mirroring the source tree layout.
All failures are collected and reported at the end; execution continues even if a
notebook fails so you see the full set of broken notebooks in one pass.
"""
import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path('openmdao_book')
OUT_DIR = Path('_executed_book')
MPI_RANKS = 4


def is_mpi_notebook(nb_path):
    """Return True if the notebook requests MPI execution."""
    try:
        meta = json.loads(nb_path.read_text(encoding='utf-8')).get('metadata', {})
    except Exception:
        return False
    return bool(meta.get('mpi', False))


def main():
    """Execute all documentation notebooks using papermill."""
    notebooks = sorted(SRC_DIR.glob('**/*.ipynb'))
    # Exclude checkpoints
    notebooks = [nb for nb in notebooks if '.ipynb_checkpoints' not in nb.parts]

    print(f'Found {len(notebooks)} notebooks to execute.')

    failed = []

    for nb in notebooks:
        rel = nb.relative_to(SRC_DIR)
        out = OUT_DIR / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        use_mpi = is_mpi_notebook(nb)
        prefix = '[MPI] ' if use_mpi else '      '
        print(f'  {prefix}{rel}', flush=True)

        cmd = (
            ['mpiexec', '-n', str(MPI_RANKS), '--oversubscribe'] if use_mpi else []
        ) + [
            'papermill',
            str(nb),
            str(out),
            '--no-progress-bar',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f'\n  FAILED: {rel}', file=sys.stderr)
            if result.stdout.strip():
                print(result.stdout, file=sys.stderr)
            if result.stderr.strip():
                print(result.stderr, file=sys.stderr)
            failed.append(rel)

    if failed:
        print(f'\n{len(failed)} notebook(s) failed:', file=sys.stderr)
        for f in failed:
            print(f'  {f}', file=sys.stderr)
        sys.exit(1)

    print(f'\nAll {len(notebooks)} notebooks executed successfully.')


if __name__ == '__main__':
    main()
