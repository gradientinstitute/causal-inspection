Causal Inspection
=================

A Scikit-learn inspired inspection module for causal models.

Installation
------------

To just install the cinspect package, clone it from github and then in the
cloned directory,

    pip install .

To also install the extra packages required for development and simulation,
install in the following way,

    pip install -e .[dev]

You can then run the simulations in the `simulations` directory.


Modules
-------

- `cinspect`: the tools for inspecting estimated causal effects.
- `simulations`: a data generation class and some simulations for demonstrating
    and testing the tools.
