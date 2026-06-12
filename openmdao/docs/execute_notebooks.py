"""
Execute all documentation notebooks using papermill.

MPI-dependent notebooks (those with "mpi": true in their top-level metadata) are
executed serially, one at a time, outside the parallel pool. They manage their own
MPI processes internally via subprocess.run(['mpiexec', ...]). All other notebooks
are executed in parallel using a multiprocessing pool.

Notebooks that need OpenMDAO reports (those with "reports": true in their top-level
metadata) are executed with OPENMDAO_REPORTS=1. All others run with OPENMDAO_REPORTS=0
to avoid the overhead of generating HTML reports for every notebook.

Usage (from openmdao/docs/):
    python execute_notebooks.py [--workers N] [--force] [--no-serial] [--no-mpi]

    --workers N   Number of parallel workers for non-MPI notebooks (default: cpu_count).
    --force       Re-execute all notebooks even if the output is up to date.
    --no-serial   Skip execution of serial (non-MPI) notebooks.
    --no-mpi      Skip execution of MPI notebooks.

Output notebooks are written to _executed_book/, mirroring the source tree layout.
A notebook is skipped when its output already exists and is newer than the source.
All failures are collected and reported at the end; execution continues even if a
notebook fails so you see the full set of broken notebooks in one pass.
"""
import argparse
import json
import os
import subprocess
import sys
from multiprocessing.pool import Pool
from pathlib import Path

SRC_DIR = Path('openmdao_book')
OUT_DIR = Path('_executed_book')

try:
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                SpinnerColumn, TaskProgressColumn, TextColumn,
                                TimeElapsedColumn, TimeRemainingColumn)
    _RICH = True
except ImportError:
    _RICH = False


def _make_progress(total, description):
    """Return a Rich Progress context manager, or None if rich is unavailable."""
    if not _RICH:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn('[bold blue]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn('[cyan]elapsed:'),
        TimeElapsedColumn(),
        TextColumn('[cyan]remaining:'),
        TimeRemainingColumn(),
    )


def is_up_to_date(src, out):
    """Return True if out exists and is newer than src."""
    return out.exists() and out.stat().st_mtime >= src.stat().st_mtime


def _nb_meta(nb_path):
    """Return the top-level metadata dict for a notebook, or {} on error."""
    try:
        return json.loads(nb_path.read_text(encoding='utf-8')).get('metadata', {})
    except Exception:
        return {}


def is_mpi_notebook(nb_path):
    """Return True if the notebook requests MPI execution."""
    return bool(_nb_meta(nb_path).get('mpi', False))


def is_reports_notebook(nb_path):
    """Return True if the notebook requires OpenMDAO reports to be enabled."""
    return bool(_nb_meta(nb_path).get('reports', False))



def _nb_env(nb_path):
    """Return the subprocess environment for executing a notebook."""
    env = os.environ.copy()
    env['OPENMDAO_REPORTS'] = '1' if is_reports_notebook(nb_path) else '0'
    return env


def _run_notebook(args):
    """Execute a single notebook with papermill; return (rel_path, returncode, stdout, stderr)."""
    nb, out = args
    rel = nb.relative_to(SRC_DIR)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ['papermill', str(nb), str(out), '--no-progress-bar', '--kernel', 'python3',
           '--cwd', str(out.parent)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_nb_env(nb))
    return rel, result.returncode, result.stdout, result.stderr



def main():
    """Execute all documentation notebooks using papermill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of parallel workers for non-MPI notebooks.')
    parser.add_argument('--force', action='store_true',
                        help='Re-execute all notebooks even if output is up to date.')
    parser.add_argument('--no-serial', action='store_true',
                        help='Skip execution of serial (non-MPI) notebooks.')
    parser.add_argument('--no-mpi', action='store_true',
                        help='Skip execution of MPI notebooks.')
    args = parser.parse_args()

    notebooks = sorted(SRC_DIR.glob('**/*.ipynb'))
    notebooks = [nb for nb in notebooks
                 if '.ipynb_checkpoints' not in nb.parts and '_build' not in nb.parts]

    mpi_notebooks = [nb for nb in notebooks if is_mpi_notebook(nb)]
    serial_notebooks = [nb for nb in notebooks if not is_mpi_notebook(nb)]

    if not args.force:
        skipped = sum(1 for nb in notebooks
                      if is_up_to_date(nb, OUT_DIR / nb.relative_to(SRC_DIR)))
        mpi_notebooks = [nb for nb in mpi_notebooks
                         if not is_up_to_date(nb, OUT_DIR / nb.relative_to(SRC_DIR))]
        serial_notebooks = [nb for nb in serial_notebooks
                           if not is_up_to_date(nb, OUT_DIR / nb.relative_to(SRC_DIR))]
    else:
        skipped = 0

    if args.no_serial:
        serial_notebooks = []
    if args.no_mpi:
        mpi_notebooks = []

    total = len(serial_notebooks) + len(mpi_notebooks)
    print(f'Found {len(notebooks)} notebooks '
          f'({len(serial_notebooks)} serial, {len(mpi_notebooks)} MPI, {skipped} up-to-date).')

    serial_failed = []
    mpi_failed = []

    if total == 0:
        print('All notebooks are up to date.')
        return

    # --- Serial notebooks: execute in parallel ---
    if serial_notebooks:
        worker_count = min(args.workers, len(serial_notebooks))
        print(f'\nExecuting {len(serial_notebooks)} serial notebooks '
              f'with {worker_count} workers...')

        pool_args = [(nb, OUT_DIR / nb.relative_to(SRC_DIR)) for nb in serial_notebooks]

        progress = _make_progress(len(serial_notebooks), 'Serial notebooks')

        if progress is not None:
            with progress:
                task_id = progress.add_task('Serial notebooks', total=len(serial_notebooks))
                with Pool(processes=worker_count) as pool:
                    for rel, rc, stdout, stderr in pool.imap_unordered(_run_notebook, pool_args):
                        progress.advance(task_id)
                        if rc != 0:
                            progress.print(f'  [red]FAILED[/red] {rel}')
                            if stdout.strip():
                                progress.print(stdout)
                            if stderr.strip():
                                progress.print(stderr)
                            serial_failed.append(rel)
        else:
            done = 0
            with Pool(processes=worker_count) as pool:
                for rel, rc, stdout, stderr in pool.imap_unordered(_run_notebook, pool_args):
                    done += 1
                    status = 'FAILED' if rc != 0 else 'ok'
                    print(f'  [{done}/{len(serial_notebooks)}] [{status}] {rel}', flush=True)
                    if rc != 0:
                        if stdout.strip():
                            print(stdout, file=sys.stderr)
                        if stderr.strip():
                            print(stderr, file=sys.stderr)
                        serial_failed.append(rel)

        if serial_failed:
            print(f'\nSerial summary: {len(serial_failed)} failed, '
                  f'{len(serial_notebooks) - len(serial_failed)} succeeded.')
            for f in serial_failed:
                print(f'  FAILED: {f}')
        else:
            print(f'\nSerial summary: all {len(serial_notebooks)} notebooks succeeded.')

    # --- MPI notebooks: execute serially under mpiexec ---
    if mpi_notebooks:
        print(f'\nExecuting {len(mpi_notebooks)} MPI notebooks serially...')

        progress = _make_progress(len(mpi_notebooks), 'MPI notebooks')

        if progress is not None:
            with progress:
                task_id = progress.add_task('MPI notebooks', total=len(mpi_notebooks))
                for nb in mpi_notebooks:
                    rel = nb.relative_to(SRC_DIR)
                    out = OUT_DIR / rel
                    out.parent.mkdir(parents=True, exist_ok=True)

                    progress.print(f'  [MPI] {rel}')
                    cmd = ['papermill', str(nb), str(out), '--no-progress-bar',
                           '--kernel', 'python3', '--cwd', str(out.parent)]
                    result = subprocess.run(cmd, capture_output=True, text=True, env=_nb_env(nb))
                    progress.advance(task_id)

                    if result.returncode != 0:
                        progress.print(f'  [red]FAILED[/red] {rel}')
                        if result.stdout.strip():
                            progress.print(result.stdout)
                        if result.stderr.strip():
                            progress.print(result.stderr)
                        mpi_failed.append(rel)
        else:
            for i, nb in enumerate(mpi_notebooks, 1):
                rel = nb.relative_to(SRC_DIR)
                out = OUT_DIR / rel
                out.parent.mkdir(parents=True, exist_ok=True)

                print(f'  [{i}/{len(mpi_notebooks)}] [MPI] {rel}', flush=True)
                cmd = ['papermill', str(nb), str(out), '--no-progress-bar',
                       '--kernel', 'python3', '--cwd', str(out.parent)]
                result = subprocess.run(cmd, capture_output=True, text=True, env=_nb_env(nb))

                if result.returncode != 0:
                    print(f'  FAILED: {rel}', file=sys.stderr)
                    if result.stdout.strip():
                        print(result.stdout, file=sys.stderr)
                    if result.stderr.strip():
                        print(result.stderr, file=sys.stderr)
                    mpi_failed.append(rel)

        if mpi_failed:
            print(f'\nMPI summary: {len(mpi_failed)} failed, '
                  f'{len(mpi_notebooks) - len(mpi_failed)} succeeded.')
            for f in mpi_failed:
                print(f'  FAILED: {f}')
        else:
            print(f'\nMPI summary: all {len(mpi_notebooks)} notebooks succeeded.')

    failed = serial_failed + mpi_failed
    if failed:
        print(f'\nOverall: {len(failed)} notebook(s) failed:', file=sys.stderr)
        for f in failed:
            print(f'  {f}', file=sys.stderr)
        sys.exit(1)

    print(f'\nOverall: {total} notebook(s) executed successfully'
          f'{f", {skipped} skipped (up to date)" if skipped else ""}.')


if __name__ == '__main__':
    main()
