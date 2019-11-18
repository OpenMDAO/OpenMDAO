
import os
import pstats
import sys
import traceback
import time
import webbrowser
import fnmatch
import threading
from six import StringIO
import tornado.ioloop
import tornado.web
import subprocess
from networkx.drawing.nx_pydot import write_dot
from graphviz import Digraph

from openmdao.utils.hooks import _register_hook
from openmdao.core.group import Group
from openmdao.utils.mpi import MPI


def launch_browser(port):
    time.sleep(1)
    for browser in ['chrome', 'firefox', 'chromium', 'safari']:
        try:
            webbrowser.get(browser).open('http://localhost:%s' % port)
        except:
            pass
        else:
            break


def startThread(fn):
    thread = threading.Thread(target=fn)
    thread.setDaemon(True)
    thread.start()
    return thread


class Application(tornado.web.Application):
    """
    An application that allows a user to view the system graph for any group in a model.
    """
    def __init__(self, problem, port, engine='dot'):
        """
        Inialize the app.

        Parameters
        ----------
        problem : Problem
            The Problem containing the model.
        port : int
            The server port.
        engine : str
            The graphviz layout engine to use.  Should be one of ['dot', 'fdp', 'circo'].
        """
        handlers = [
            (r"/", Index),
            (r"/sysgraph/", Index),
            (r"/sysgraph/([_a-zA-Z][_a-zA-Z0-9.]*)", SysGraph),
        ]

        settings = dict(
             template_path=os.path.join(os.path.dirname(__file__), "templates"),
             static_path=os.path.join(os.path.dirname(__file__), "static"),
        )

        super(Application, self).__init__(handlers, **settings)

        self.prob = problem
        self.port = port
        self.engine = engine


def get_graph_info(group, title, engine='dot'):
    """
    Get the system graph from the give group and return the graphviz generated SVG string.

    Parameters
    ----------
    group : Group
        Retrieve the graph SVG for this Group.
    title : str
        The title shown at the top of the rendered graph.
    engine : str
        The graphviz layout engine to use to layout the graph. Should be one of
        ['dot', 'fdp', 'circo'].

    Returns
    -------
    str
        The SVG string describing the graph.
    """
    graph = group.compute_sys_graph()

    f = Digraph(title, filename=title + '.gv', format='svg', engine=engine)
    f.attr(rankdir='LR', size='600, 600')

    # display groups as double circles
    f.attr('node', shape='doublecircle')
    groupset = set(group._subgroups_myproc)
    for g in groupset:
        f.node(g.name)

    # components as regular circles
    f.attr('node', shape='circle')
    for s in group._subsystems_myproc:
        if s not in groupset:
            f.node(s.name)
    
    for u, v in graph.edges():
        f.edge(u, v)

    svg = f.pipe()

    svg = str(svg.replace(b'\n', b''))
    return svg[1:].strip("'"), group._subgroups_myproc


class SysGraph(tornado.web.RequestHandler):
    def get(self, pathname=''):
        self.write_graph(pathname)

    def write_graph(self, pathname):
        app = self.application
        model = app.prob.model

        if pathname:
            system = model._get_subsystem(pathname)
            title = pathname
        else:
            system = model
            title = 'Model'

        if not isinstance(system, Group):
            self.write("Components don't have graphs.")
            return
        
        svg, subgroups = get_graph_info(system, title, app.engine)
        pathname = system.pathname
        parent = '' if not pathname else pathname.rsplit('.', 1)[0]
        if parent == pathname:
            parent = ''
        subgroups = [g.name for g in subgroups]

        self.write("""\
    <html>
    <head>
    <style>
    </style>
    <script src="https://d3js.org/d3.v4.min.js"></script>
    <script>

    var width = 960,
        height = 800;

    var pathnames = %s;
    var subgroups = %s;

    function d3_setup() {
        var svg = d3.select("svg");
        // register click event handler on all the graphviz SVG node elements
        svg.selectAll(".node")
            .on("click", function(d, i) {
                var txt = d3.select(this).select("text").text();
                if (subgroups.includes(txt)) {
                    var ptext = pathnames[0] + "." + txt;
                    if (ptext.startsWith(".")) {
                        ptext = txt;
                    }
                    window.location = "/sysgraph/" + ptext;
                }
            });

        window.onresize = function() {
            width = window.innerWidth * .98;
            d3.select("svg").attr("width", width);
        }
    }

    window.onload = function() {
        width = window.innerWidth * .98;
        svg = d3.select("svg")
            .attr("width", width)
            .attr("height", height);
        d3_setup();
    };

    </script>
    </head>
    <body>
        <input type="button" onclick="location.href='/';" value="Home" />
        <input type="button" onclick="location.href='/sysgraph/%s';" value="Up" />
        <h1>%s</h1>
    %s
    </body>
    </html>
    """ % ([pathname], subgroups, parent, title, svg))


class Index(SysGraph):
    def get(self):
        self.write_graph('')


def _view_graphs_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao view_graphs' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('-p', '--port', action='store', dest='port',
                        default=8009, type=int,
                        help='port used for web server')
    parser.add_argument('--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('-g', '--group', action='append', default=[], dest='groups',
                        help='Display the graph for the given group.')
    parser.add_argument('-e', '--engine', action='store', dest='engine',
                        default='dot', help='Specify graph layout engine (dot, circo, fdp)')
    parser.add_argument('file', metavar='file', nargs=1,
                        help='profile file to view.')


def _view_graphs_cmd(options):
    """
    Return the post_setup hook function for 'openmdao graphs'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    if options.engine not in {'dot', 'circo', 'fdp'}:
        raise RuntimeError("Graph layout engine '{}' not supported.".format(options.engine))

    def _view_graphs(prob):
        if not MPI or MPI.COMM_WORLD.rank == 0:
            view_graphs(prob, progname=options.file[0], port=options.port, 
                        groups=options.groups, engine=options.engine)
        exit()

    # register the hook
    _register_hook('final_setup', class_name='Problem', inst_id=options.problem, post=_view_graphs)

    return _view_graphs


def view_graphs(prob, progname, port=8009, groups=(), engine='dot'):
    """
    Start an interactive graph viewer for an OpenMDAO model.

    Parameters
    ----------
    prob : Problem
        The Problem to be viewed.
    progname: str
        Name of model file.
    port: int
        Port number used by web server.
    """
    app = Application(prob, port, engine)
    app.listen(port)

    print("starting server on port %d" % port)

    serve_thread  = startThread(tornado.ioloop.IOLoop.current().start)
    launch_thread = startThread(lambda: launch_browser(port))

    while serve_thread.isAlive():
        serve_thread.join(timeout=1)

if __name__ == '__main__':
    cmd_view_graphs()
