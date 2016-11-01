import os
import os.path
import subprocess

directories = [
    'assemblers',
    'core',
    'jacobians',
    'proc_allocators',
    'vectors',
]

for dir_name in directories:
    file_names = os.listdir('../' + dir_name)

    for file_name in file_names:
        if file_name != '__init__.py' and file_name[-3:] == '.py':
            for check in ['pep8', 'pep257']:
                path = '../%s/%s' % (dir_name, file_name)
                print '-' * 79
                print check, path
                subprocess.call([check, path])

print
print
