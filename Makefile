check:
	ruff check .

fixcheck:
	ruff check . --fix

install:
	pip install -e .

gui-install:
	pip install -e '.[gui]'