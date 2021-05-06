"""A script wrapper for the Wing IDE."""


import os
import os.path
from os.path import exists, abspath, dirname, join
import sys
import fnmatch
import logging
from subprocess import Popen
from configparser import ConfigParser
from optparse import OptionParser


def find_up(name, path=None):
    """Find the named file or directory in a parent directory.

    Search upward from the starting path (or the current directory)
    until the given file or directory is found. The given name is
    assumed to be a basename, not a path.  Returns the absolute path
    of the file or directory if found, or None otherwise.

    Parameters
    ----------
    name : str
        Base name of the file or directory being searched for.

    path : str, optional
        Starting directory.  If not supplied, current directory is used.
    """
    if not path:
        path = os.getcwd()
    if not exists(path):
        return None
    while path:
        if exists(join(path, name)):
            return abspath(join(path, name))
        else:
            pth = path
            path = dirname(path)
            if path == pth:
                return None
    return None


def _modify_wpr_file(template, outfile, version):
    config = ConfigParser()
    config.read(template)
    if sys.platform == 'darwin':
        config.set('user attributes', 'proj.pyexec',
                   str(dict({None: ('custom', sys.executable)})))
        config.set('user attributes', 'proj.pypath',
                   str(dict({None: ('custom',
                                          os.pathsep.join(sys.path))})))

    with open(outfile, 'w') as fp:
        fp.write('#!wing\n#!version=%s\n' % version)
        config.write(fp)


def _find_wing():
    if sys.platform == 'win32':
        wname = 'wing.exe'
        tdir = r'C:\Program Files (x86)'
        try:
            locs = [os.path.join(tdir, p, 'bin') for p in
                    fnmatch.filter(os.listdir(tdir), r'Wing IDE ?.?')]
        except:
            locs = []
        tdir = r'C:\Program Files'
        try:
            locs.extend([os.path.join(tdir, p, 'bin') for p in
                         fnmatch.filter(os.listdir(tdir), r'Wing IDE ?.?')])
        except:
            pass
    elif sys.platform == 'darwin':
        wname = 'wing'
        locs = ['/Applications/WingIDE.app/Contents/MacOS',
                '/Applications/Wing/WingIDE.app/Contents/MacOS']
    else:
        wname = 'wing?.?'
        locs = ['/usr/bin', '/usr/sbin', '/usr/local/bin']

    try:
        pathvar = os.environ['PATH']
    except KeyError:
        pathvar = ''

    all_locs = [p for p in pathvar.split(os.pathsep) if p.strip()] + locs
    for path in all_locs:
        try:
            matches = fnmatch.filter(os.listdir(path), wname)
        except:
            continue
        if matches:
            return os.path.join(path, sorted(matches)[-1])

    raise OSError("%s was not found in PATH or in any of the common places." %
                  wname)


def run_wing():
    """Run the Wing IDE using our template project file."""
    parser = OptionParser()
    parser.add_option("-w", "--wingpath", action="store", type="string",
                      dest="wingpath", help="location of WingIDE executable")
    parser.add_option("-p", "--projpath", action="store", type="string",
                      dest="projpath", default='',
                      help="location of WingIDE project file")
    parser.add_option("-v", "--version", action="store", type="string",
                      dest="version", default='7.0',
                      help="version of WingIDE")
    (options, args) = parser.parse_args(sys.argv[1:])

    wingpath = options.wingpath
    projpath = options.projpath
    version = options.version
    if len(version) == 1:
        version = version + '.0'

    if not os.path.isfile(projpath):
        wingproj_file = 'wing_proj_template.wpr'

        mydir = os.path.dirname(os.path.abspath(__file__))
        proj_template = os.path.join(mydir, wingproj_file)
        projpath = os.path.join(mydir, 'wingproj.wpr')

        _modify_wpr_file(proj_template, projpath, version)

    # in order to find all of our shared libraries,
    # put their directories in LD_LIBRARY_PATH
    env = {}
    env.update(os.environ)
    if sys.platform == 'darwin':
        libpname = 'DYLD_LIBRARY_PATH'
        libext = '*.dyld'
    elif not sys.platform.startswith('win'):
        libpname = 'LD_LIBRARY_PATH'
        libext = '*.so'
    else:
        libpname = None

    if libpname:
        libs = env.get(libpname, '').split(os.pathsep)
        rtop = find_up('.git')
        if rtop:
            rtop = os.path.dirname(rtop)

        sodirs = set()
        for path, dirlist, filelist in os.walk(rtop):
            for name in [f for f in filelist if fnmatch.fnmatch(f, libext)]:
                sodirs.add(os.path.dirname(join(path, name)))

        libs.extend(sodirs)
        env[libpname] = os.pathsep.join(libs)

    if sys.platform == 'darwin':
        cmd = ['open', projpath]
    else:
        if not wingpath:
            wingpath = _find_wing()
        cmd = [wingpath, projpath]
    try:
        print("wing command: ", ' '.join(cmd))
        Popen(cmd, env=env)
    except Exception as err:
        print("Failed to run command '%s'." % ' '.join(cmd))

if __name__ == '__main__':
    run_wing()
