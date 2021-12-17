from distutils.core import setup

setup(name='deepgl',
      version=0.02,
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'sklearn'],
      py_modules=['deepgl', 'deepgl_utils'])
