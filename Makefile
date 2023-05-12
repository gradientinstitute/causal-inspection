# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
help:
	@echo "clean - clean all artefacts"
	@echo "clean-build - remove build artefacts"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

typecheck:
	mypy ./cinspect

lint:
	flake8 ./cinspect

isort:
	isort .

test:
	pytest . --cov=cinspect tests/

test-ci:
	pytest . --cov=cinspect tests/ --hypothesis-profile "ci"

# shortcut for making html docs
doc:
	$(MAKE) html -C docs