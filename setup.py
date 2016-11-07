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
          'openmdao.core',
          'openmdao.solvers',
          'openmdao.drivers',
          'openmdao.vectors',
          'openmdao.proc_allocators',
          'openmdao.assemblers',
          'openmdao.tests',
          'openmdao.jacobians',
          'openmdao.devtools',
      ],
      # package_data={
      #     'openmdao.units': ['unit_library.ini'],
      #     'openmdao.devtools': ['*.template', '*.html'],
      #     'openmdao.util': ['*.html'],
      # },
      install_requires=[
        'six', 'numpydoc', #'numpy>=1.9.2',
        'scipy', 'sqlitedict',
        'pep8', 'pep257',
      ],
      #scripts=['bin/om-pylint.sh']
      entry_points="""
      [console_scripts]
      wingproj=openmdao.devtools.wingproj:run_wing
      1to2=openmdao.devtools.1to2:convert
      """
      # webview=openmdao.devtools.webview:webview_argv
      # view_profile=openmdao.util.profile:prof_view
      # proftotals=openmdao.util.profile:prof_totals
      # profdump=openmdao.util.profile:prof_dump
      # """
)
