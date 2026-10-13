"""
Build or clean the OpenMDAO documentation.

Usage:
    python -m openmdao.devtools.build_docs build             # full build
    python -m openmdao.devtools.build_docs build --no-exec   # skip notebook execution
    python -m openmdao.devtools.build_docs build --fast      # parallel Sphinx, no warnings-as-errors
    python -m openmdao.devtools.build_docs clean             # remove all generated files

Output: openmdao/docs/_executed_book/_build/html/index.html

Notebook execution details
--------------------------
MPI-dependent notebooks (those with "mpi": true in their top-level metadata) are
executed serially, one at a time, outside the parallel pool. They manage their own
MPI processes internally via mpi_exec(). All other notebooks are executed in parallel
using a multiprocessing pool.

Notebooks that need OpenMDAO reports (those with "reports": true in their top-level
metadata) are executed with OPENMDAO_REPORTS=1. All others run with OPENMDAO_REPORTS=0
to avoid the overhead of generating HTML reports for every notebook.

A notebook is skipped when its output in _executed_book/ already exists and is newer
than the source. All failures are collected and reported at the end.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from multiprocessing.pool import Pool
from pathlib import Path

HERE = Path(__file__).parent.parent.parent / 'openmdao' / 'docs'
SRC_DIR = HERE / 'openmdao_book'
OUT_DIR = HERE / '_executed_book'

try:
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                SpinnerColumn, TaskProgressColumn, TextColumn,
                                TimeElapsedColumn, TimeRemainingColumn)
    _RICH = True
except ImportError:
    _RICH = False


def _banner(msg):
    bar = '=' * 61
    print(f'\n{bar}')
    print(msg)
    print(bar, flush=True)


def _make_progress(total, description, no_rich=False):
    """Return a Rich Progress context manager, or None if rich is unavailable or disabled."""
    if not _RICH or no_rich:
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


def _nb_meta(nb_path):
    """Return the top-level metadata dict for a notebook, or {} on error."""
    try:
        return json.loads(Path(nb_path).read_text(encoding='utf-8')).get('metadata', {})
    except Exception:
        return {}


def _has_code_cells(nb_path):
    """Return True if the notebook contains at least one code cell."""
    try:
        cells = json.loads(Path(nb_path).read_text(encoding='utf-8')).get('cells', [])
        return any(c.get('cell_type') == 'code' for c in cells)
    except Exception:
        return True


def _is_mpi_notebook(nb_path):
    """Return True if the notebook requests MPI execution."""
    return bool(_nb_meta(nb_path).get('mpi', False))


def _is_reports_notebook(nb_path):
    """Return True if the notebook requires OpenMDAO reports to be enabled."""
    return bool(_nb_meta(nb_path).get('reports', False))


def _is_up_to_date(src, out):
    """Return True if out exists and is newer than src."""
    return out.exists() and out.stat().st_mtime >= src.stat().st_mtime


def _nb_env(nb_path):
    """Return the subprocess environment for executing a notebook."""
    env = os.environ.copy()
    env['OPENMDAO_REPORTS'] = '1' if _is_reports_notebook(nb_path) else '0'
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


def _execute_notebooks(workers, force, no_serial, no_mpi, no_rich):
    """Execute all documentation notebooks using papermill."""
    all_notebooks = sorted(SRC_DIR.glob('**/*.ipynb'))
    all_notebooks = [nb for nb in all_notebooks
                     if '.ipynb_checkpoints' not in nb.parts and '_build' not in nb.parts]

    markdown_only = [nb for nb in all_notebooks if not _has_code_cells(nb)]
    notebooks = [nb for nb in all_notebooks if _has_code_cells(nb)]

    mpi_notebooks = [nb for nb in notebooks if _is_mpi_notebook(nb)]
    serial_notebooks = [nb for nb in notebooks if not _is_mpi_notebook(nb)]

    if not force:
        skipped = sum(1 for nb in notebooks
                      if _is_up_to_date(nb, OUT_DIR / nb.relative_to(SRC_DIR)))
        mpi_notebooks = [nb for nb in mpi_notebooks
                         if not _is_up_to_date(nb, OUT_DIR / nb.relative_to(SRC_DIR))]
        serial_notebooks = [nb for nb in serial_notebooks
                            if not _is_up_to_date(nb, OUT_DIR / nb.relative_to(SRC_DIR))]
    else:
        skipped = 0

    if no_serial:
        serial_notebooks = []
    if no_mpi:
        mpi_notebooks = []

    total = len(serial_notebooks) + len(mpi_notebooks)
    print(f'Found {len(all_notebooks)} notebooks '
          f'({len(serial_notebooks)} serial, {len(mpi_notebooks)} MPI, '
          f'{skipped} up-to-date, {len(markdown_only)} markdown-only).')

    serial_failed = []
    mpi_failed = []

    if total == 0:
        print('All notebooks are up to date.')
        return

    # --- Serial notebooks: execute in parallel ---
    if serial_notebooks:
        worker_count = min(workers, len(serial_notebooks))
        print(f'\nExecuting {len(serial_notebooks)} serial notebooks '
              f'with {worker_count} workers...')

        pool_args = [(nb, OUT_DIR / nb.relative_to(SRC_DIR)) for nb in serial_notebooks]
        progress = _make_progress(len(serial_notebooks), 'Serial notebooks', no_rich)

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

    # --- MPI notebooks: execute serially ---
    if mpi_notebooks:
        print(f'\nExecuting {len(mpi_notebooks)} MPI notebooks serially...')
        progress = _make_progress(len(mpi_notebooks), 'MPI notebooks', no_rich)

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
        raise RuntimeError(f'{len(failed)} notebook(s) failed.')

    print(f'\nOverall: {total} notebook(s) executed successfully'
          f'{f", {skipped} skipped (up to date)" if skipped else ""}.')


def cmd_view(args):
    """Serve the built docs on a local HTTP server and open in the default browser."""
    import functools
    import threading
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    if args.path is not None:
        # User supplied a path — resolve to the directory containing index.html.
        p = Path(args.path).resolve()
        if p.is_file():
            p = p.parent
        html_dir = p
    else:
        html_dir = OUT_DIR / '_build' / 'html'

    if not (html_dir / 'index.html').exists():
        if args.path is not None:
            print(f'index.html not found in {html_dir}', file=sys.stderr)
        else:
            print('Docs have not been built yet. Run: python -m openmdao.devtools.build_docs build',
                  file=sys.stderr)
        sys.exit(1)

    port = args.port
    url = f'http://localhost:{port}/index.html'

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(html_dir))
    server = HTTPServer(('', port), handler)
    print(f'Serving docs at {url}  (Ctrl-C to stop)')
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')


def cmd_clean(args):
    """Remove all files generated by the build."""
    for path in [OUT_DIR, SRC_DIR / '_srcdocs']:
        if path.exists():
            _banner(f'Removing {path.relative_to(HERE)}')
            shutil.rmtree(path)
    print('\nDone.')


def cmd_build(args):
    """Run the full documentation build pipeline."""
    os.environ.setdefault('PYDEVD_DISABLE_FILE_VALIDATION', '1')
    os.environ.setdefault('OMPI_MCA_rmaps_base_oversubscribe', '1')
    os.environ.setdefault('PRTE_MCA_rmaps_default_mapping_policy', ':oversubscribe')

    os.chdir(HERE)

    _banner('Disable SNOPT cells')
    subprocess.run(
        [sys.executable, 'openmdao_book/other/disable_snopt_cells.py'], check=True)

    _banner('Build source docs (API reference)')
    subprocess.run([sys.executable, 'build_source_docs.py'], check=True)

    _banner('Copy source tree to _executed_book/')
    # Copy all non-notebook supporting files (data files, scripts, static assets)
    # so they are present alongside the notebooks when papermill executes them.
    # Notebooks with code cells are NOT copied here — papermill writes them directly
    # to _executed_book/, which allows _execute_notebooks to use timestamps to skip
    # up-to-date notebooks.
    # Notebooks with no code cells (markdown-only, e.g. index.ipynb and _srcdocs stubs)
    # ARE copied here since papermill never touches them.
    for item in SRC_DIR.rglob('*'):
        if not item.is_file():
            continue
        rel = item.relative_to(SRC_DIR)
        if '_build' in rel.parts or '.ipynb_checkpoints' in rel.parts or 'tests' in rel.parts:
            continue
        if item.suffix == '.ipynb' and _has_code_cells(item):
            continue
        target = OUT_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)

    if not args.no_exec:
        _banner('Execute notebooks')
        _execute_notebooks(
            workers=args.workers if args.workers is not None else os.cpu_count(),
            force=False,
            no_serial=args.no_serial,
            no_mpi=args.no_mpi,
            no_rich=args.no_rich,
        )
    else:
        print('Skipping notebook execution (--no-exec)')

    if args.no_serial or args.no_mpi:
        print('\nSkipping Sphinx build (partial notebook execution).')
        return

    _banner('Build HTML with Sphinx')
    sphinx_cmd = ['sphinx-build', '-b', 'html', '--keep-going']
    if args.fast:
        sphinx_cmd += ['-j', 'auto']
    else:
        sphinx_cmd += ['-W']
    sphinx_cmd += [str(OUT_DIR), str(OUT_DIR / '_build' / 'html')]
    subprocess.run(sphinx_cmd, check=True)

    _banner('Copy build artifacts')
    subprocess.run([sys.executable, 'copy_build_artifacts.py'], check=True)

    print('\nDone. Docs available at: openmdao/docs/_executed_book/_build/html/index.html')


def main():
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='subcommand')

    build_p = sub.add_parser('build', help='Build the documentation (default).')
    build_p.add_argument('--no-exec', action='store_true',
                         help='Skip notebook execution; use existing _executed_book/.')
    build_p.add_argument('--no-serial', action='store_true',
                         help='Skip execution of serial (non-MPI) notebooks.')
    build_p.add_argument('--no-mpi', action='store_true',
                         help='Skip execution of MPI notebooks.')
    build_p.add_argument('--fast', action='store_true',
                         help='Parallel Sphinx build (-j auto); skips warnings-as-errors.')
    build_p.add_argument('--workers', type=int, default=None,
                         help='Number of parallel workers (default: cpu_count).')
    build_p.add_argument('--no-rich', action='store_true',
                         help='Use plain-text progress output; recommended for CI.')

    sub.add_parser('clean', help='Remove all generated files.')

    view_p = sub.add_parser('view', help='Serve the built docs on a local HTTP server.')
    view_p.add_argument('--port', type=int, default=8000,
                        help='Port to serve on (default: 8000).')
    view_p.add_argument('--path', default=None,
                        help='Path to a built docs directory or its index.html file. '
                             'Use this to serve a downloaded CI artifact.')

    args = parser.parse_args()

    if args.subcommand is None or args.subcommand == 'build':
        if args.subcommand is None:
            args.no_exec = False
            args.no_serial = False
            args.no_mpi = False
            args.fast = False
            args.workers = None
            args.no_rich = False
        cmd_build(args)
    elif args.subcommand == 'clean':
        cmd_clean(args)
    elif args.subcommand == 'view':
        cmd_view(args)


if __name__ == '__main__':
    main()
