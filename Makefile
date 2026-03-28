.PHONY: install test lint docker-up docker-down format

install:
	pip install -r requirements.txt

test:
	pytest

lint:
	flake8 src/

docker-up:
	docker compose -f infrastructure/docker-compose.yml up -d --build

docker-down:
	docker compose -f infrastructure/docker-compose.yml down

format:
	black src/