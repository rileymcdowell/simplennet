from __future__ import print_function

from setuptools import setup, find_packages, Command

class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys
        import subprocess 
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno) 

setup( name='simplennet'
     , version='0.0.1'
     , description='Simple neural network implementation that leverages numpy.'
     , author='Riley McDowell'
     , author_email='mcdori02_at_luther_dot_edu'
     , url='https://github.com/rileymcdowell/simplennet'
     , packages=find_packages()
     , cmdclass = { 'test': PyTest }
     )

