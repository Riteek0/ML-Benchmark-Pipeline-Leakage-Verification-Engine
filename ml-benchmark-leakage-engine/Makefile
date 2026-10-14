.PHONY: test benchmark docker-build docker-run

test:
	pytest -v

benchmark:
	python3 -m src.benchmark_runner --n-samples 100000 --positive-rate 0.035 --n-trials 25

docker-build:
	docker build -t ml-benchmark-leakage-engine:2026 .

docker-run:
	docker run --rm ml-benchmark-leakage-engine:2026
