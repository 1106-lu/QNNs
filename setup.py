import pathlib
from setuptools import find_packages, setup

__version__ = ''
exec(open('qnn/_version.py').read())
assert __version__, 'Version string cannot be empty'

required_packages = pathlib.Path('requirements.txt').read_text().split('\n')
INSTALL_PACKAGES = [pkg for pkg in required_packages if pkg and not pkg.startswith('#')]

setup(name='qnn',
      version=__version__,
      url='https://github.com/TomiTeam/QNNs',
      author='Tomi',
      author_email='ockier1@gmail.com',
      python_requires='>=3.6.5',
      install_requires=INSTALL_PACKAGES,
      license='null',
      description="",
      long_description=open('README.md', encoding='utf-8').read(),
      packages=find_packages()
      )