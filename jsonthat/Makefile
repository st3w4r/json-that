.PHONY: format patch build publish

format:
	ruff format jsonthat/*.py tests/*.py bump_version.py

patch:
	poetry version patch
	python bump_version.py

minor:
	poetry version minor
	python bump_version.py

build:
	poetry build

publish:
	poetry publish

