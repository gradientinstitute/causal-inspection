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


License
-------

Copyright 2022 Gradient Institute

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
