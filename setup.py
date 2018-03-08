#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Much of the build system code was adapted from work done by the pandas
developers [1], which was in turn based on work done in pyzmq [2] and lxml [3].

[1] http://pandas.pydata.org
[2] http://zeromq.github.io/pyzmq/
[3] http://lxml.de/
"""
import os
import shutil
import sys
import subprocess
from distutils.version import LooseVersion
from setuptools import setup, Command, find_packages
import pkg_resources

# versioning
import versioneer
cmdclass = versioneer.get_cmdclass()


curdir = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(curdir, "README.md")).read()
CYTHON_EXCLUSION_FILE = 'cythonize_exclusions.dat'

DISTNAME = 'sm2'
DESCRIPTION = 'Bugfix Fork of statsmodels'
LONG_DESCRIPTION = README
MAINTAINER = 'Brock Mendel'
MAINTAINER_EMAIL = 'jbrockmendel@gmail.com'
# URL = 'http://www.statsmodels.org/'
LICENSE = 'BSD License'
DOWNLOAD_URL = ''

classifiers = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Programming Language :: Cython',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Operating System :: OS Independent',
               'Intended Audience :: End Users/Desktop',
               'Intended Audience :: Developers',
               'Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: BSD License',
               'Topic :: Scientific/Engineering']

# FIXME: Don't use these
ISRELEASED = False


extras = {'docs': ['sphinx>=1.3.5',
                   'matplotlib',
                   'numpydoc>=0.6.0']}
min_versions = {'numpy': '1.13.0',
                'scipy': '0.19.0',
                'pandas': '0.22.0',
                'patsy': '0.4.0'}

setuptools_kwargs = {
    "zip_safe": False,
    'install_requires': [
        "numpy >= {version}".format(version=min_versions['numpy']),
        "scipy >= {version}".format(version=min_versions['scipy']),
        "pandas >= {version}".format(version=min_versions['pandas']),
        "patsy >= {version}".format(version=min_versions['patsy'])
    ],
    'setup_requires': [
        'numpy >= {version}'.format(version=min_versions['numpy'])
    ]}

# TODO: Can we just put this with the next (only) use of CYTHON_INSTALLED?
min_cython_ver = '0.24'
try:
    import Cython
    ver = Cython.__version__
    _CYTHON_INSTALLED = ver >= LooseVersion(min_cython_ver)
except ImportError:
    _CYTHON_INSTALLED = False


# TODO: What is this?  Is it needed?
no_frills = (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                     sys.argv[1] in ('--help-commands',
                                                     'egg_info', '--version',
                                                     'clean')))


# These imports need to be here; setuptools needs to be imported first.
from distutils.extension import Extension  # noqa:E402
from distutils.command.build import build  # noqa:E402
from distutils.command.build_ext import build_ext as _build_ext  # noqa:E402


try:
    if not _CYTHON_INSTALLED:
        raise ImportError('No supported version of Cython installed.')
    try:
        from Cython.Distutils.old_build_ext import old_build_ext as _build_ext  # noqa:F811,E501
    except ImportError:
        # Pre 0.25
        from Cython.Distutils import build_ext as _build_ext
    cython = True
except ImportError:
    cython = False


class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if (hasattr(ext, 'include_dirs') and
                    numpy_incl not in ext.include_dirs):
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


'''
def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                          os.path.join(cwd, 'tools', 'cythonize.py'),
                          'sm2'],
                         cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def init_cython_exclusion(filename):
    with open(filename, 'w') as f:
        pass


def append_cython_exclusion(path, filename):
    with open(filename, 'a') as f:
        f.write(path + "\n")
'''


class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""
    # Identical to pandas version except for self._clean_exclude

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = ["bspline_ext.c",
                               "bspline_impl.c"]

        for root, dirs, files in list(os.walk('sm2')):
            for f in files:
                if f in self._clean_exclude:
                    continue
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(os.path.join(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(os.path.join(root, d))

        for d in ('build', 'dist'):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                shutil.rmtree(clean_tree)
            except Exception:
                pass


# we need to inherit from the versioneer
# class as it encodes the version info
sdist_class = cmdclass['sdist']


class CheckingBuildExt(build_ext):
    """Subclass build_ext to get clearer report if Cython is necessary."""
    # effectively identical to pandas version
    def check_cython_extensions(self, extensions):
        for ext in extensions:
            for src in ext.sources:
                if not os.path.exists(src):
                    print("{}: -> [{}]".format(ext.name, ext.sources))
                    raise Exception("""Cython-generated file '{src}' not found.
                Cython is required to compile sm2 from a development branch.
                Please install Cython or download a release package of sm2.
                """.format(src=src))

    def build_extensions(self):
        self.check_cython_extensions(self.extensions)
        build_ext.build_extensions(self)


class CythonCommand(build_ext):
    """Custom distutils command subclassed from Cython.Distutils.build_ext
    to compile pyx->c, and stop there. All this does is override the
    C-compile method build_extension() with a no-op."""
    # Copied verbatim from pandas setup.py
    def build_extension(self, ext):
        pass


class DummyBuildSrc(Command):
    """ numpy's build_src command interferes with Cython's build_ext.
    """
    # identical to pandas version
    user_options = []

    def initialize_options(self):
        self.py_modules_dict = {}

    def finalize_options(self):
        pass

    def run(self):
        pass


cmdclass = {'clean': CleanCommand,
            'build': build,
            'build_ext': CheckingBuildExt}

if cython:
    suffix = '.pyx'
    cmdclass['cython'] = CythonCommand
else:
    suffix = '.c'
    cmdclass['build_src'] = DummyBuildSrc

'''


# some linux distros require it
# NOTE: we are not currently using this but add it to Extension, if needed.
# libraries = ['m'] if 'win32' not in sys.platform else []

from numpy.distutils.misc_util import get_info

# Reset the cython exclusions file
init_cython_exclusion(CYTHON_EXCLUSION_FILE)

npymath_info = get_info("npymath")
ext_data = {
    "kalman_loglike": {
        "name": "sm2/tsa/kalmanf/kalman_loglike.c",
        "depends" : ["sm2/src/capsule.h"],
        "include_dirs": ["sm2/src"],
        "sources" : []}}

'''
extensions = []
'''
for name, data in ext_data.items():
    data['sources'] = data.get('sources', []) + [data['name']]

    destdir = ".".join(os.path.dirname(data["name"]).split("/"))
    data.pop('name')

    filename = data.pop('filename', name)

    obj = Extension('%s.%s' % (destdir, filename), **data)

    extensions.append(obj)
'''


def get_data_files():
    # this adds *.csv and *.dta files in datasets folders
    # and *.csv and *.txt files in test/results folders
    data_files = {}
    root = os.path.join(curdir, "sm2", "datasets")
    if not os.path.exists(root):
        return []
    for i in os.listdir(root):
        if i is "tests":
            continue
        path = os.path.join(root, i)
        if os.path.isdir(path):
            key = os.path.relpath(path, start=curdir).replace(os.sep, ".")
            data_files[key] = ["*.csv", "*.dta"]

    # add all the tests and results files
    for r, ds, fs in os.walk(os.path.join(curdir, "sm2")):
        relpath = os.path.relpath(r, start=curdir)
        if relpath.endswith('results'):
            key = relpath.replace(os.sep, ".")
            data_files[key] = ["*.csv", "*.txt", "*.dta"]

    # Manually fill in the rest.  Upstream this was done _outside_ this call
    data_files["sm2.datasets.tests"].append("*.zip")
    data_files["sm2.iolib.tests.results"].append("*.dta")
    data_files["sm2.stats.tests.results"].append("*.json")
    data_files["sm2.tsa.vector_ar.tests.results"].append("*.npz")
    # data files that don't follow the tests/results pattern. should fix.
    data_files["sm2.stats.tests"] = ["*.txt"]
    data_files["sm2.stats.libqsturng"] = ["*.r", "*.txt", "*.dat"]
    data_files["sm2.stats.libqsturng.tests"] = ["*.csv", "*.dat"]
    data_files["sm2.tsa.vector_ar.data"] = ["*.dat"]
    data_files["sm2.tsa.vector_ar.data"] = ["*.dat"]
    # temporary, until moved:
    data_files["sm2.sandbox.regression.tests"] = ["*.dta", "*.csv"]
    return data_files


#TODO: deal with this. Not sure if it ever worked for bdists
#('docs/build/htmlhelp/statsmodelsdoc.chm',
# 'sm2/statsmodelsdoc.chm')

cwd = os.path.abspath(os.path.dirname(__file__))
#if not os.path.exists(os.path.join(cwd, 'PKG-INFO')) and not no_frills:
#    # Generate Cython sources, unless building from source release
#    generate_cython()

setup(name=DISTNAME,
      version=versioneer.get_version(),
      maintainer=MAINTAINER,
      ext_modules=extensions,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      # url=URL,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      classifiers=classifiers,
      platforms='any',
      cmdclass=cmdclass,
      packages=find_packages(),
      # package_data=get_data_files(),
      include_package_data=False,  # True will install all files in repo
      extras_require=extras,
      **setuptools_kwargs)
