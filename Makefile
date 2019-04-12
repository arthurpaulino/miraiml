.PHONY: install docs opendocs

install:
	python setup.py install && rm -rf build

docs:
	@cd docs && make clean && make html
	@echo -e 'ctrl+click -> \e]8;;file://${PWD}/docs/_build/html/index.html\aMiraiML Docs\e]8;;\a'

help:
	@echo '	install'
	@echo '		Builds and installs MiraiML;'
	@echo '		Cleans build files.'
	@echo '	docs'
	@echo '		Cleans and builds html files for MiraiML Docs;'
	@echo '		Outputs a clickable link for the index page.'
