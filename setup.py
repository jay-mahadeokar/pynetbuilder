from setuptools import setup

VERSION = "1.0"

setup(
    name='pynetbuilder',
    description="Python module for prototxt generation including standard network architecture building blocks",
    author_email='jaym@yahoo-inc.com',
    version=VERSION,
    packages=['netbuilder',
              'netbuilder.lego',
              'netbuilder.tools',
              'netbuilder.nets']
)
