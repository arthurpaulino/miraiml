.PHONY: install docs major minor patch release upload_test upload_pypi help

install:
	@python setup.py install && rm -rf build

docs:
	@cd docs && make clean && make html
	@echo -e 'ctrl+click -> \e]8;;file://${PWD}/docs/_build/html/index.html\aMiraiML Docs\e]8;;\a'

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
	@echo '		Cleans build files'
	@echo '	docs'
	@echo '		Cleans and builds html files for MiraiML Docs'
	@echo '		Outputs a clickable link for the index page'
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
