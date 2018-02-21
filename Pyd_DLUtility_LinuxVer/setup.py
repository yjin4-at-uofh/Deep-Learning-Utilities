from distutils.core import setup, Extension
import numpy as np

BASE_SRC_DIR = 'DLUtility'
PYTHON_INC_DIR = '/usr/include/python3.5'
NUMPY_DIR = '/usr/local/lib/python3.5/dist-packages/numpy/core'
PYTHON_LIB_DIR='/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu'
TARGET='dlUtility'

module1 = Extension(name = TARGET,
                    language = 'c++',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    extra_compile_args = ['-std=c++11'],
                    include_dirs = [PYTHON_INC_DIR, np.get_include(), BASE_SRC_DIR],
                    libraries = ['python3.5', 'npymath'],
                    library_dirs = [PYTHON_LIB_DIR, '{0}/lib'.format(NUMPY_DIR)],
                    sources = ['{0}/DLUErrorLog.cpp'.format(BASE_SRC_DIR), '{0}/DLUSettings.cpp'.format(BASE_SRC_DIR), '{0}/dllmain.cpp'.format(BASE_SRC_DIR)])

setup (name = 'dlUtility',
       version = '0.5',
       description = 'deep learning utilities.',
       author = 'Yuchen Jin',
       author_email = 'cainmagi@gmail.com',
       url = 'mailto:cainmagi@gmail.com',
       long_description = '''
A collection of useful utilities in deep learning.
''',
       ext_modules = [module1])
