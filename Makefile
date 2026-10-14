# Developer tasks. CI runs `make install-ci lint test`. Recipes assume a POSIX shell (Linux,
# Colab, macOS, Git Bash on Windows); README.md lists the plain commands behind each target.

PYTHON ?= python
# Packages from requirements-ml.txt that the fast (not slow) tests need. CI installs only these on
# top of requirements-core.txt; extend the list when a phase's fast tests import more.
CI_ML_PACKAGES := pandas|pyarrow|requests|pytest|ruff|matplotlib

.PHONY: install-core install-ml install-ci install-colab lint format test test-all tiny stages

install-core:
	$(PYTHON) -m pip install -r requirements-core.txt

install-ml:
	$(PYTHON) -m pip install -r requirements-core.txt -r requirements-ml.txt

# Colab keeps its preinstalled torch (docs/decisions.md D21): everything else from the pins.
install-colab:
	grep -v '^torch==' requirements-ml.txt > .colab-requirements.txt
	$(PYTHON) -m pip install -q -r requirements-core.txt -r .colab-requirements.txt
	rm -f .colab-requirements.txt

install-ci:
	grep -E '^($(CI_ML_PACKAGES))==' requirements-ml.txt > .ci-requirements.txt
	$(PYTHON) -m pip install -r requirements-core.txt -r .ci-requirements.txt
	rm -f .ci-requirements.txt

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test:
	$(PYTHON) -m pytest -m "not slow"

test-all:
	$(PYTHON) -m pytest

tiny:
	$(PYTHON) -m tests.fixtures.tiny.make_tiny --out local/tiny

stages:
	$(PYTHON) -m ml.stages list
