from distutils.core import setup

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
          'openmdao.recorders',
          'openmdao.solvers',
          'openmdao.solvers.linear',
          'openmdao.solvers.linesearch',
          'openmdao.solvers.nonlinear',
          'openmdao.test_suite',
          'openmdao.test_suite.components',
          'openmdao.test_suite.groups',
          'openmdao.test_suite.test_examples',
          'openmdao.utils',
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
      install_requires=[
        'six', 'numpydoc',  # 'numpy>=1.9.2',
        'scipy',
        'sqlitedict',
        'pycodestyle', 'pydocstyle',
        'testflo',
        'parameterized',
        'networkx',
        'sphinx',
        'redbaron',
        'mock',
        'requests_mock',
        'tornado',
      ],
      # scripts=['bin/om-pylint.sh']
      entry_points="""
      [console_scripts]
      wingproj=openmdao.devtools.wingproj:run_wing
      webview=openmdao.devtools.webview:webview_argv
      iprofview=openmdao.devtools.iprofile_app.iprofile_app:_prof_view
      iproftotals=openmdao.devtools.iprofile:_prof_totals
      iprofmem=openmdao.devtools.iprof_mem:_profile_py_file
      icalltrace=openmdao.devtools.itrace:_trace_py_file
      """
)
