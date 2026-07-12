check:
	ruff check .

fixcheck:
	ruff check . --fix

test:
	pytest

install:
	pip install -e .

gui-install:
	pip install -e '.[gui]'

test-install:
	pip install --group test

profiling-install:
	pip install --group profiling