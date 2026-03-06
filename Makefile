.PHONY: install test lint docker-up docker-down format

install:
	pip install -r requirements.txt

test:
	pytest

lint:
	flake8 src/

docker-up:
	docker compose up -d

docker-down:
	docker compose down

format:
	black src/