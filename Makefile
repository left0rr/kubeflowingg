.PHONY: install test lint docker-up docker-down grafana-up grafana-down format deploy-champion

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

grafana-up:
	docker compose -f infrastructure/docker-compose.yml up -d grafana

grafana-down:
	docker compose -f infrastructure/docker-compose.yml stop grafana

format:
	black src/

deploy-champion:
	python monitoring/promote_champion.py	
