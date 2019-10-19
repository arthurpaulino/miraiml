.PHONY: install develop flake doctest tests example docs dist release upload_test upload_pypi clean help

install:
	@pip install -r requirements.txt && pip install .

develop:
	@pip install -r requirements_develop.txt && make install

flake:
	@python -m flake8

doctest:
	@python -m doctest -v README.rst docs/*.rst miraiml/*.py

tests:
	@make doctest && python -m pytest tests/*

example:
	@python -m jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 --to markdown docs/example.ipynb

docs:
	@cd docs && make clean && make html
	@echo -e 'ctrl+click -> \e]8;;file://${PWD}/docs/_build/html/index.html\aMiraiML Docs\e]8;;\a'

clean:
	@rm -rf MANIFEST */__pycache__ dist .pytest_cache docs/_build */miraiml_local miraiml_local */.ipynb_checkpoints

dist:
	@rm -rf dist && python setup.py sdist

release:
	@make clean && make develop && make tests && make example && make docs && make dist

upload_test:
	@python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload_pypi:
	@python -m twine upload dist/*

help:
	@echo '	install'
	@echo '		Builds and installs MiraiML'
	@echo '	develop'
	@echo '		Builds and installs MiraiML along with development packages'
	@echo '	flake'
	@echo '		Runs flake8 checks'
	@echo '	doctest'
	@echo '		Runs doctest on all files that matter'
	@echo '	tests'
	@echo '		Runs all tests'
	@echo '	docs'
	@echo '		Converts example notebook to markdown'
	@echo '		Cleans and builds html files for MiraiML Docs'
	@echo '		Outputs a clickable link for the local index page'
	@echo '	clean'
	@echo '		Removes all temporary files'
