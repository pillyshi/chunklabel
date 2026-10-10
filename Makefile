# Home GPU server (uv + CUDA llama.cpp). Adapted from
# https://github.com/pillyshi/gpu-makefiles (home/Makefile). Set HOST in .env.
#
#   make setup                     # uv sync + CUDA build of llama-cpp-python
#   make run CMD="uv run pytest -q"
#   make run-bg CMD="uv run --extra llamacpp python -u benchmarks/fidelity/run.py ..."
#   make attach                    # watch the run-bg session; Ctrl-b d to detach
#   make fetch                     # pull outputs/ and benchmarks/*/*.json back

.DEFAULT_GOAL := help

include common.mk

CUDA_VERSION      ?= 12.8
GCC_VERSION       ?= 13
# Keep in sync with llama-cpp-python in uv.lock. If the versions differ,
# `uv run --extra llamacpp` replaces the CUDA build with a CPU-only one.
LLAMA_CPP_VERSION ?= 0.3.28

.PHONY: help install install-llama-cpp setup

help:
	@sed -n '3,8p' Makefile | sed 's/^# \{0,1\}//'

install: sync
	$(call remote-exec,uv sync)

setup: install install-llama-cpp

# --no-cache: uv caches built wheels by version only, ignoring CMAKE_ARGS, so a cached
# CPU-only wheel would otherwise be reused silently. The last step fails if the
# result cannot offload to the GPU.
install-llama-cpp: sync
	ssh $(HOST) "bash -l -c 'cd $(REMOTE_DIR) && \
		CUDA_HOME=/usr/local/cuda-$(CUDA_VERSION) \
		PATH=/usr/local/cuda-$(CUDA_VERSION)/bin:\$$HOME/.local/bin:\$$PATH \
		CMAKE_ARGS=\"-DGGML_CUDA=on -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-$(GCC_VERSION)\" \
		CC=/usr/bin/gcc-$(GCC_VERSION) CXX=/usr/bin/g++-$(GCC_VERSION) \
		uv pip install \"llama-cpp-python==$(LLAMA_CPP_VERSION)\" --no-binary llama-cpp-python \
			--no-cache --reinstall-package llama-cpp-python && \
		.venv/bin/python -c \"import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)\" && \
		echo llama-cpp-python: CUDA build OK'"
