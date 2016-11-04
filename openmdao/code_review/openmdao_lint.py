from __future__ import print_function

import os
import sys
import os.path
import subprocess

directories = [
    'assemblers',
    'core',
    'jacobians',
    'proc_allocators',
    'solvers',
    'utils',
    'vectors',
]

retcode = 0
topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for dir_name in directories:
    dirpath = os.path.join(topdir, dir_name)

    for file_name in  os.listdir(dirpath):
        if file_name != '__init__.py' and file_name[-3:] == '.py':
            for check in ['pep8', 'pep257']:
                path = os.path.join(dirpath, file_name)
                print ('-' * 79)
                print (check, path)
                ret = subprocess.call([check, path])
                if retcode == 0:
                    retcode = ret

print()
print()

sys.exit(retcode)
