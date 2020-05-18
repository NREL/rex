"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
from warnings import warn

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(here, "rex", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}"
                 .format(e))

        develop.run(self)


with open("requirements.txt") as f:
    install_requires = f.readlines()

test_requires = ["pytest>=5.2", ]
description = ("National Renewable Energy Laboratory's (NREL's) REsource "
               "eXtraction tool: rex")

setup(
    name="NREL-rex",
    version=version,
    description=description,
    long_description=readme,
    author="Michael Rossol",
    author_email="michael.rossol@nrel.gov",
    url="https://nrel.github.io/rex/",
    packages=find_packages(),
    package_dir={"rex": "rex"},
    entry_points={
        "console_scripts": ["rex=rex.resource_cli:main",
                            "NSRDB=rex.solar_cli:main",
                            "WIND=rex.wind_cli:main",
                            "rechunk=rex.rechunk_h5.rechunk_cli:main"],
    },
    include_package_data=True,
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="rex",
    python_requires='>=3.7',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="tests",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "dev": test_requires + ["flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
