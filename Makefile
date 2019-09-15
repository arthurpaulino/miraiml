.PHONY: install develop flake docs tests major minor patch release upload_test upload_pypi help

install:
	@pip install .

develop:
	@pip install -e ".[dev]"

flake:
	@python -m flake8

docs:
	@cd docs && make clean && make html
	@echo -e 'ctrl+click -> \e]8;;file://${PWD}/docs/_build/html/index.html\aMiraiML Docs\e]8;;\a'

tests:
	pytest tests/*

major:
	@python versioning.py major

minor:
	@python versioning.py minor

patch:
	@python versioning.py patch

release:
	@rm -rf dist && python setup.py sdist

upload_test:
	@twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload_pypi:
	@twine upload dist/*

help:
	@echo '	install'
	@echo '		Builds and installs MiraiML'
	@echo '	develop'
	@echo '		Builds and installs MiraiML along with development packages'
	@echo '	flake'
	@echo '		Runs flake8 checks'
	@echo '	docs'
	@echo '		Cleans and builds html files for MiraiML Docs'
	@echo '		Outputs a clickable link for the local index page'
	@echo '	tests'
	@echo '		Runs all tests'
	@echo '	major'
	@echo '		Increments the MAJOR version identifier'
	@echo '	minor'
	@echo '		Increments the MINOR version identifier'
	@echo '	patch'
	@echo '		Increments the PATCH version identifier'
	@echo '	release'
	@echo '		Creates a dist directory for release'
	@echo '	upload_test'
	@echo '		Uploads the release to TestPyPI (requires password)'
	@echo '	upload_pypi'
	@echo '		Uploads the release to PyPI (requires password)'
