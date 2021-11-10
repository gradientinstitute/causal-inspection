"""Setup tools file for causal_inspection.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To use a consistent encoding
from codecs import open
from os import path

# Always prefer setuptools over distutils
from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="causal_inspection",

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.0.1",

    description="Scikit-learn inspired inspection utilities for causal models",
    long_description=long_description,

    # The project"s main homepage.
    url="https://github.com/gradientinstitute/causal_inspection",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",

        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Statistics :: Causal",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3"
    ],

    # What does your project relate to?
    keywords="causality inspection interpretability",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=["cinspect"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip"s
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "multimethod"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-flake8",
            "pytest-mock",
            "flake8-bugbear",
            "flake8-builtins",
            "pytest-cov",
            "flake8-comprehensions",
            "flake8-docstrings",
            "flake8-quotes",
            "mypy",
            "mypy_extensions",
            "networkx"
        ]
    }
)
