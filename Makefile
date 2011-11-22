PYTHON?=python

all:
	$(PYTHON) setup.py build

clean:
	rm -Rf build

install:
	$(PYTHON) setup.py build
	sudo $(PYTHON) setup.py install
