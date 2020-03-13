
import os
import sys
import time
import webbrowser
import threading
import json

try:
    import tornado
    import tornado.ioloop
    import tornado.web
except ImportError:
    tornado = None

from collections import defaultdict, deque
from itertools import groupby

from openmdao.devtools.iprofile import _process_profile, _iprof_py_file
from openmdao.devtools.iprof_utils import func_group, _setup_func_group
from openmdao.utils.mpi import MPI


def _launch_browser(port):
    """
    Open the default web browser to localhost:<port>
    """
    time.sleep(1)
    webbrowser.get().open('http://localhost:%s' % port)


def _startThread(fn):
    """
    Start a daemon thread running the given function.
    """
    thread = threading.Thread(target=fn)
    thread.setDaemon(True)
    thread.start()
    return thread


def _parent_key(d):
    """
    Return the function path of the parent of function specified by 'id' in the given dict.
    """
    parts = d['id'].rsplit('|', 1)
    if len(parts) == 1:
        return ''
    return parts[0]


def _stratify(call_data, sortby='time'):
    """
    Group node data by depth and sort within a depth by parent and 'sortby'.
    """
    depth_groups = []
    node_list = []  # all nodes in a single list
    depthfunc=lambda d: d['depth']
    for key, group in groupby(sorted(call_data.values(), key=depthfunc), key=depthfunc):
        # now further group each group by parent, then sort those in descending order
        # by 'sortby'
        depth_groups.append({
            key: sorted(sub, key=lambda d: d[sortby], reverse=True)
                for key, sub in groupby(sorted(group, key=_parent_key), key=_parent_key)
        })

    max_depth = len(depth_groups)
    delta_y = 1.0 / max_depth
    y = 0
    max_x = call_data['$total'][sortby]

    for depth, pardict in enumerate(depth_groups):
        y0 = delta_y * depth
        y1 = y0 + delta_y

        for parent, children in pardict.items():
            if not parent:
                end_x = 0
            else:
                end_x = call_data[parent]['x0'] * max_x

            for i, node in enumerate(children):
                start_x = end_x
                end_x += node[sortby]
                node['x0'] = start_x / max_x
                node['x1'] = end_x / max_x
                node['y0'] = y0
                node['y1'] = y1
                node['idx'] = len(node_list)
                node_list.append(node)

    return depth_groups, node_list

def _iprof_setup_parser(parser):
    if not func_group:
        _setup_func_group()

    parser.add_argument('-p', '--port', action='store', dest='port',
                        default=8009, type=int,
                        help='port used for web server')
    parser.add_argument('--no_browser', action='store_true', dest='noshow',
                        help="Don't pop up a browser to view the data.")
    parser.add_argument('-t', '--title', action='store', dest='title',
                        default='Profile of Method Calls by Instance',
                        help='Title to be displayed above profiling view.')
    parser.add_argument('-g', '--group', action='store', dest='methods',
                        default='openmdao',
                        help='Determines which group of methods will be tracked. Current '
                             'options are: %s and "openmdao" is the default' %
                              sorted(func_group.keys()))
    parser.add_argument('-m', '--maxcalls', action='store', dest='maxcalls',
                        default=15000, type=int,
                        help='Maximum number of calls displayed at one time.  Default=15000.')
    parser.add_argument('file', metavar='file', nargs='+',
                        help='Raw profile data files or a python file.')


if tornado is None:
    def _iprof_exec(options, user_args):
        """
        Called from a command line to instance based profile data in a web page.
        """
        raise RuntimeError("The 'iprof' function requires the 'tornado' package.  "
                           "You can install it using 'pip install tornado'.")

else:
    class _Application(tornado.web.Application):
        def __init__(self, options):
            self.call_data, _ = _process_profile(options.file)
            self.depth_groups, self.node_list = _stratify(self.call_data)
            self.options = options

            # assemble our call_data nodes into a tree structure, where each
            # entry contains that node's call data and a dict containing each
            # child keyed by call path.
            self.call_tree = tree = defaultdict(lambda : [None, {}])
            for path, data in self.call_data.items():
                data['id'] = path
                parts = path.rsplit('|', 1)
                # add our node to our parent
                if len(parts) > 1:
                    tree[parts[0]][1][path] = data
                tree[path][0] = data

            handlers = [
                (r"/", _Index),
                (r"/func/([0-9]+)", _Function),
            ]

            settings = dict(
                 template_path=os.path.join(os.path.dirname(__file__), "templates"),
                 static_path=os.path.join(os.path.dirname(__file__), "static"),
            )

            super(_Application, self).__init__(handlers, **settings)

        def get_nodes(self, idx):
            """
            Yield all children of the given root up to a maximum number stored in options.maxcalls.
            """
            if idx == 0:
                root = self.call_tree['$total']
            else:
                root = self.node_list[idx]
                root = self.call_tree[root['id']]

            maxcalls = self.options.maxcalls
            stack = deque()
            stack.appendleft(root)
            callcount = 1
            stop_adding = False
            while stack:
                parent, children = stack.pop()
                yield parent
                if not stop_adding:
                    callcount += len(children)
                    if callcount <= maxcalls:
                        for child in children.values():
                            stack.appendleft(self.call_tree[child['id']])
                    else:
                        stop_adding = True


    class _Index(tornado.web.RequestHandler):
        def get(self):
            """
            Load the page template and request call data nodes starting at idx=0.
            """
            app = self.application
            self.render("iprofview.html", title=app.options.title)


    class _Function(tornado.web.RequestHandler):
        def get(self, idx):
            """
            Request an updated list of call data nodes, rooted at the node specified by idx.
            """
            app = self.application
            dump = json.dumps(list(app.get_nodes(int(idx))))
            self.set_header('Content-Type', 'application/json')
            self.write(dump)


    def _iprof_exec(options, user_args):
        """
        Called from a command line to instance based profile data in a web page.
        """
        if options.file[0].endswith('.py'):
            if len(options.file) > 1:
                print("iprofview can only process a single python file.", file=sys.stderr)
                sys.exit(-1)

            _iprof_py_file(options, user_args)
            if MPI:
                options.file = ['iprof.%d' % i for i in range(MPI.COMM_WORLD.size)]
            else:
                options.file = ['iprof.0']

        if not options.noshow and (not MPI or MPI.COMM_WORLD.rank == 0):
            app = _Application(options)
            app.listen(options.port)

            print("starting server on port %d" % options.port)

            serve_thread = _startThread(tornado.ioloop.IOLoop.current().start)
            launch_thread = _startThread(lambda: _launch_browser(options.port))

            while serve_thread.isAlive():
                serve_thread.join(timeout=1)
