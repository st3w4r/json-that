# contributing

```
export PYPI_TOKEN=<pypi-token>

poetry config pypi-token.pypi $PYPI_TOKEN

poetry version patch
poetry build
poetry publish
```
