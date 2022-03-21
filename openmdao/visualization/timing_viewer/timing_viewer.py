
"""Define a function to view driver timing."""
import os
import json
import pickle

from openmdao.utils.webview import webview
from openmdao.utils.general_utils import default_noraise
from openmdao.utils.mpi import MPI


def _timing_iter(timing_managers):
    for probname, tmanager in timing_managers.items():
        for sysname, timers in tmanager._timers.items():
            for t in timers:
                if t.ncalls > 0:
                    yield probname, sysname, t.name, t.ncalls, t.avg(), t.min, t.max, t.tot


def _timing_file_iter(timing_file):
    with open(timing_file, 'rb') as f:
        yield from _timing_iter(pickle.load(f))


def view_timing(timing_file, outfile='timing_report.html', show_browser=True, title=''):
    """
    Generate a self-contained html file containing a table of timing data.

    Optionally pops up a web browser to view the file.

    Parameters
    ----------
    timing_file : str
        The name of the file contining the timing data.
    outfile : str, optional
        The name of the output html file.  Defaults to 'connections.html'.
    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.
    title : str, optional
        Sets the title of the web page.

    Returns
    -------
    dict
        Data to used to generate html file.
    """
    timing_table = []

    idx = 1  # unique ID for use by Tabulator

    # set up timing table data
    for pname, sname, method, ncalls, avgtime, mintime, maxtime, tottime in \
            _timing_file_iter(timing_file):

        dct = {
            'id': idx,
            'probname': pname,
            'sysname': sname,
            'method': method,
            'ncalls': ncalls,
            'avgtime': avgtime,
            'mintime': mintime,
            'maxtime': maxtime,
            'tottime': tottime,
        }

        timing_table.append(dct)

        idx += 1

    data = {
        'title': title,
        'timing_table': timing_table,
    }


    if MPI is None or MPI.COMM_WORLD.comm.rank == 0:

        viewer = 'timing_table.html'

        code_dir = os.path.dirname(os.path.abspath(__file__))
        libs_dir = os.path.join(os.path.dirname(code_dir), 'common', 'libs')
        style_dir = os.path.join(os.path.dirname(code_dir), 'common', 'style')

        with open(os.path.join(code_dir, viewer), "r") as f:
            template = f.read()

        with open(os.path.join(libs_dir, 'tabulator.min.js'), "r") as f:
            tabulator_src = f.read()

        with open(os.path.join(style_dir, 'tabulator.min.css'), "r") as f:
            tabulator_style = f.read()

        with open(os.path.join(libs_dir, 'd3.v6.min.js'), "r") as f:
            d3_src = f.read()

        jsontxt = json.dumps(data, default=default_noraise)

        with open(outfile, 'w') as f:
            s = template.replace("<tabulator_src>", tabulator_src)
            s = s.replace("<tabulator_style>", tabulator_style)
            s = s.replace("<d3_src>", d3_src)
            s = s.replace("<timing_data>", jsontxt)
            f.write(s)

        if show_browser:
            webview(outfile)

    return data
