PHONY: build start help
.DEFAULT_GOAL:= help
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_UID := $(shell id -u)

help:  ## describe make commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build:  ## build image
	DOCKER_BUILDKIT=1 docker build --ssh=default \
		--build-arg USERNAME=$(USER) --build-arg USER_UID=$(CURRENT_UID) \
		-f docker/Dockerfile \
		-t yolox . 


start:  ## start containerized gpu research
	docker run --rm \
			--gpus 'device=0' \
			--shm-size=4096m \
			-v $(HOME)/.aws/credentials:/root/.aws/credentials \
			-v $(ROOT_DIR)/:/workspace/mnt/ \
			-v /:/outer_root \
			-p 8887:8887 \
			-p 5151:5151 \
			-p 6006:6006 \
			-it yolox

#:ro