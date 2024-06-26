.PHONY: format

format:
	ruff format jsonthat/*.py tests/*.py bump_version.py

build:
	poetry build

patch:
	poetry version patch
	python bump_version.py
