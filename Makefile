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
	py.test --flake8 ./cinspect -p no:regtest --cache-clear

isort:
	isort .

test:
	pytest . --cov=cinspect tests/	
