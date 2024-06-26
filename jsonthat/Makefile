.PHONY: format

format:
	ruff format jsonthat/*.py tests/*.py

build:
	poetry build

patch:
	poetry version patch
	python bump_version.py
