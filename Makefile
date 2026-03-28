.PHONY: install test lint docker-up docker-down format deploy-champion

install:
	pip install -r requirements.txt

test:
	pytest

lint:
	flake8 .

docker-up:
	docker compose -f infrastructure/docker-compose.yml up -d --build

docker-down:
	docker compose -f infrastructure/docker-compose.yml down

format:
	black src/

deploy-champion:
	python monitoring/promote_champion.py
