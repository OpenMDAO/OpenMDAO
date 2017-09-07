
# This is a modified version of:
#    http://sourceforge.net/p/imvu/code/HEAD/tree/imvu_open_source/tools/pstats_viewer.py


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


def launch_browser(port):
    time.sleep(1)
    webbrowser.get().open('http://localhost:%s' % port)

def startThread(fn):
    thread = threading.Thread(target=fn)
    thread.setDaemon(True)
    thread.start()
    return thread

def htmlquote(fn):
    return fn.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def shrink(s):
    if len(s) < 40:
        return s
    return s[:20] + '...' + s[-20:]

def formatfunc(func):
    file, line, func_name = func
    return '%s:%s:%s' % (os.path.basename(file), line, htmlquote(shrink(func_name)))

def formatTime(dt):
    return '%.2fs' % dt

def formatTimeAndPercent(dt, total):
    percent = "(%.1f%%)" % (100.0 * dt / total)
    if percent == '(0.0%)':
        percent = ''
    return '%s&nbsp;<font color=#808080>%s</a>' % (formatTime(dt), percent)

def wrapTag(tag, body, klass=''):
    if klass:
        klass = 'class=%s' % klass
    return '<%s %s>%s</%s>' % (tag, klass, body, tag)


class Application(tornado.web.Application):
    def __init__(self, stats, selector):
        self.stats = stats
        self.stats.stream = StringIO()
        self.stats.calc_callees()
        self.total_time = self.stats.total_tt
        self.filename = self.stats.files[0]
        width, self._full_print_list = self.stats.get_print_list(())

        if selector is None:
            self.width, self.print_list = width, self._full_print_list
        else:
            self.width, self.print_list = self.stats.get_print_list((selector,))

        self.func_to_id = {}
        self.id_to_func = {}

        for i, func in enumerate(self.print_list):
            self.id_to_func[i] = func
            self.func_to_id[func] = i

        if selector is not None:
            for i, func in enumerate(self._full_print_list):
                if func not in self.func_to_id:
                    self.func_to_id[func] = -(i+1)
                    self.id_to_func[-(i+1)] = func

        self.tabstyle = """
        table {
            font-family:Arial, Helvetica, sans-serif;
            color: black;
            font-size:14px;
            text-shadow: 1px 1px 0px #fff;
            background:#eaebec;
            margin:10px;
            border: black 1px solid;

            -moz-border-radius:3px;
            -webkit-border-radius:3px;
            border-radius:3px;
        }

        table th {
            padding-right: 5px;
            padding-left: 5px;
            background: #ededed;
        }
        table td {
            border-top: 1px solid #ffffff;
            border-bottom:1px solid #e0e0e0;
            border-left: 1px solid #e0e0e0;

            background: #e6e6e6;
            text-align: center;
        }

        table td:first-child {
            text-align: left;
            padding-right: 10px;
        }

        table tr.even td {
            background: #f2f2f2;
        }

        table tr:hover td {
            background: #ccffff;
        }
        """

        handlers = [
            (r"/", Index),
            (r"/func/([0-9]+)", Function),
        ]

        settings = dict(
             template_path=os.path.join(os.path.dirname(__file__), "templates"),
             static_path=os.path.join(os.path.dirname(__file__), "static"),
        )

        super(Application, self).__init__(handlers, **settings)

    def getFunctionLink(self, func):
        _, _, func_name = func
        title = func_name

        fid = self.func_to_id[func]
        if fid < 0:
            return '<label>%s</label>' % formatfunc(func)
        else:
            return '<a title="%s" href="/func/%s">%s</a>' % (title, fid, formatfunc(func))


class Index(tornado.web.RequestHandler):
    def get(self):
        app = self.application

        table = []

        sort_index = ['cc', 'nc', 'tt', 'ct'].index(self.get_argument('sort', 'ct'))

        app.print_list.sort(
            key=lambda func: app.stats.stats[func][sort_index],
            reverse=True)

        for i, func in enumerate(app.print_list):
            file, line, func_name = func
            primitive_calls, total_calls, exclusive_time, inclusive_time, callers = app.stats.stats[func]

            if primitive_calls == 0:
                extime = exclusive_time
                inctime = inclusive_time
            else:
                extime = exclusive_time / primitive_calls
                inctime = inclusive_time / primitive_calls

            if i % 2 == 0:
                klass = 'even'
            else:
                klass = None
            row = wrapTag('tr', ''.join(wrapTag('td', cell) for cell in (
                app.getFunctionLink(func),
                formatTimeAndPercent(exclusive_time, app.total_time),
                formatTimeAndPercent(inclusive_time, app.total_time),
                primitive_calls,
                total_calls,
                formatTime(extime),
                formatTime(inctime))), klass=klass)

            table.append(row)

        self.write('''\
    <html>
    <head>
    <style>%s</style>
    </head>
    <body>
    <h1>%s</h1>
    <h2>Total time: %s</h2>
    <table>
    <tr>
      <th>file:line:function</th>
      <th><a href="?sort=tt">Exclusive time</a></th>
      <th><a href="?sort=ct">Inclusive time</a></th>
      <th><a href="?sort=cc">Primitive calls</a></th>
      <th><a href="?sort=nc">Total calls</a></th>
      <th>Exclusive per call</th>
      <th>Inclusive per call</th>
    </tr>
    %s
    </table>
    </body>
    </html>
    ''' % (app.tabstyle, app.filename, formatTime(app.total_time), '\n'.join(table)))


class Function(tornado.web.RequestHandler):
    def get(self, func_id):
        app = self.application
        func_id = int(func_id)

        if func_id < 0:
            func = app.id_to_unslected_func[-func_id]
        else:
            func = app.id_to_func[func_id]

        f_cc, f_nc, f_tt, f_ct, callers = app.stats.stats[func]
        callees = app.stats.all_callees[func]

        def sortedByInclusive(items):
            sortable = [(ct, (f, (cc, nc, tt, ct))) for f, (cc, nc, tt, ct) in items]
            return [y for x, y in reversed(sorted(sortable))]

        def buildFunctionTable(items):
            callersTable = []
            for i, (caller, (cc, nc, tt, ct)) in enumerate(sortedByInclusive(items)):
                if i % 2 == 0:
                    klass = 'even'
                else:
                    klass = None
                callersTable.append(wrapTag('tr', ''.join(wrapTag('td', cell)
                                                          for cell in (
                    app.getFunctionLink(caller),
                    formatTimeAndPercent(tt, app.total_time),
                    formatTimeAndPercent(ct, app.total_time),
                    cc,
                    nc,
                    formatTime(tt / cc),
                    formatTime(ct / cc))), klass=klass))
            return '\n'.join(callersTable)

        caller_stats = [(c, app.stats.stats[c][:4]) for c in callers]
        callersTable = buildFunctionTable(caller_stats)
        calleesTable = buildFunctionTable(callees.items())

        self.write('''\
    <html>
    <head>
    <style>%s</style>
    </head>
    <body>
    <a href="/">Home</a>
    <h1>%s</h1>
    <table>
    <tr><th align="left">Primitive Calls</th><td>%s</td></tr>
    <tr><th align="left">Total calls</th><td>%s</td></tr>
    <tr><th align="left">Exclusive time</th><td>%s</td></tr>
    <tr><th align="left">Inclusive time</th><td>%s</td></tr>
    </table>
    <h2>Callers</h2>
    <table>
    <tr>
    <th>Function</th>
    <th>Exclusive time</th>
    <th>Inclusive time</th>
    <th>Primitive calls</th>
    <th>Total calls</th>
    <th>Exclusive per call</th>
    <th>Inclusive per call</th>
    </tr>
    %s
    </table>
    <h2>Callees</h2>
    <table>
    <tr>
    <th>Function</th>
    <th>Exclusive time</th>
    <th>Inclusive time</th>
    <th>Primitive calls</th>
    <th>Total calls</th>
    <th>Exclusive per call</th>
    <th>Inclusive per call</th>
    </tr>
    %s
    </table>
    </body>
    </html>
    ''' % (app.tabstyle, formatfunc(func), f_cc, f_nc, f_tt, f_ct,
           callersTable, calleesTable))


def view_pstats(prof_pattern, selector=None, port=8009):
    """
    Start an interactive web viewer for profiling data.

    Parameters
    ----------
    prof_pattern: str
        Name of profile data file or glob pattern.
    selector : str, optional
        Portion of filename used to select funtions.
    port: int
        Port number used by web server.
    """

    prof_files = sorted(fnmatch.filter(os.listdir('.'), prof_pattern))
    if prof_files:
        stats = pstats.Stats(prof_files[0])
        for pfile in prof_files[1:]:
            stats.add(pfile)

    app = Application(stats, selector)
    app.listen(port)

    print("starting server on port %d" % port)

    serve_thread  = startThread(tornado.ioloop.IOLoop.current().start)
    launch_thread = startThread(lambda: launch_browser(port))

    while serve_thread.isAlive():
        serve_thread.join(timeout=1)


def cmd_view_pstats(args=None):
    """
    Allows calling of view_pstats from a console script.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', action='store', dest='port',
                        default=8009, type=int,
                        help='port used for web server')
    parser.add_argument('--filter', action='store', dest='filter',
                        default=None,
                        help='portion of filename used to filter displayed functions.')
    parser.add_argument('file', metavar='file', nargs=1,
                        help='profile file to view.')

    options = parser.parse_args(args)
    view_pstats(options.file[0], port=options.port, selector=options.filter)

if __name__ == '__main__':
    cmd_view_pstats()
