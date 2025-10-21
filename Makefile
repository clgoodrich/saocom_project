.PHONY: setup lint fix test run clean

setup:
	mamba env create -f environment.yml || conda env update -f environment.yml
	pre-commit install

lint:
	ruff check .
	black --check .
	isort --check-only .

fix:
	ruff check . --fix
	black .
	isort .

test:
	pytest

run:
	python -m src.analysis

clean:
	rm -rf data/interim/* data/processed/* outputs/*
