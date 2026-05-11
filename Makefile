.PHONY: install test lint docker-up docker-down grafana-up grafana-down gateway-run retraining-check retraining-submit format deploy-champion

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

gateway-run:
	uvicorn src.api.inference_gateway:app --host 0.0.0.0 --port 8010

retraining-check:
	python -m monitoring.retraining_trigger --config retraining/retraining_config.example.yaml

retraining-submit:
	python -m monitoring.retraining_trigger --config retraining/retraining_config.example.yaml --submit

format:
	black src/

deploy-champion:
	python monitoring/promote_champion.py	
