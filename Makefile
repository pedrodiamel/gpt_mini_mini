# .PHONY: default


install-dev:
	python setup.py develop

test:
	python -m pytest test -vvv

docker-build:
	docker-compose up --build -d

docker-down:
	docker-compose down

docker-start:
	docker start gptmm-dev
	docker exec -it gptmm-dev /bin/bash
