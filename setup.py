import os
import fnmatch

from distutils.core import setup
from distutils.extension import Extension

USE_CYTHON = os.environ.get("OPENMDAO_USE_CYTHON")
USE_SPEEDUPS = os.environ.get("OPENMDAO_USE_SPEEDUPS")

if USE_SPEEDUPS or USE_CYTHON:
    pattern = '*.pyx' if USE_CYTHON else '*.c'

    mydir = os.path.dirname(os.path.abspath(__file__))
    spd_dir = os.path.join(mydir, "openmdao", "utils", "speedups")
    sources = fnmatch.filter(os.listdir(spd_dir), pattern)

    extensions = [
        Extension("openmdao.utils.speedups.%s" % os.path.splitext(s)[0],
                  [os.path.join(spd_dir, s)])
            for s in sources
    ]

    if USE_CYTHON:
        from Cython.Build import cythonize
        extensions = cythonize(extensions)
else:
    extensions = []


setup(name='openmdao',
      version='2.0.0',
      description="OpenMDAO v2.0 framework infrastructure",
      long_description="""\
      """,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
      ],
      keywords='optimization multidisciplinary multi-disciplinary analysis',
      author='OpenMDAO Team',
      author_email='openmdao@openmdao.org',
      url='http://openmdao.org',
      #download_url='http://github.com/OpenMDAO/OpenMDAO/tarball/1.7.2',
      license='Apache License, Version 2.0',
      packages=[
          'openmdao',
          'openmdao.approximation_schemes',
          'openmdao.code_review',
          'openmdao.components',
          'openmdao.core',
          'openmdao.devtools',
          'openmdao.devtools.problem_viewer',
          'openmdao.docs',
          'openmdao.drivers',
          'openmdao.error_checking',
          'openmdao.jacobians',
          'openmdao.matrices',
          'openmdao.proc_allocators',
          'openmdao.solvers',
          'openmdao.test_suite',
          'openmdao.utils',
          'openmdao.utils.speedups',
          'openmdao.vectors',
          'openmdao.surrogate_models',
          'openmdao.surrogate_models.nn_interpolators'
      ],
      package_data={
          'openmdao.devtools': ['*.wpr', '*.html'],
          'openmdao.devtools.problem_viewer': ['*.css', '*.js', '*.template',
                                               '*.woff'],
          'openmdao.docs': ['*.py', '_utils/*.py'],
          'openmdao.utils': ['unit_library.ini'],
          'openmdao.test_suite': ['*.py', '*/*.py'],
          'openmdao': ['*/tests/*.py', '*/*/tests/*.py', '*/*/*/tests/*.py']
      },
      ext_modules=extensions,
      install_requires=[
        'six', 'numpydoc', #'numpy>=1.9.2',
        'scipy',
        'sqlitedict',
        'pycodestyle', 'pydocstyle',
        'testflo',
        'parameterized',
        'networkx',
      ],
      #scripts=['bin/om-pylint.sh']
      entry_points="""
      [console_scripts]
      wingproj=openmdao.devtools.wingproj:run_wing
      1to2=openmdao.devtools.compat:convert_file
      webview=openmdao.devtools.webview:webview_argv
      iprofview=openmdao.devtools.iprofile:prof_view
      iproftotals=openmdao.devtools.iprofile:prof_totals
      iprofdump=openmdao.devtools.iprofile:prof_dump
      """
)
