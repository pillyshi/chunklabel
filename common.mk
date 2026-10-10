# Shared targets for running on a remote GPU server: sync / fetch / run / run-bg / ssh / attach.
# Adapted from https://github.com/pillyshi/gpu-makefiles (common.mk), without `meta`.

-include .env

REMOTE_DIR ?= $(shell basename $(CURDIR))

define remote-exec
ssh $(HOST) "bash -l -c 'cd $(REMOTE_DIR) && $(1)'"
endef

# Start a command in a detached tmux session. The process survives an ssh disconnect;
# check on it later with `make attach SESSION=<name>`.
define remote-exec-tmux
ssh $(HOST) "bash -l -c 'cd $(REMOTE_DIR) && tmux new-session -d -s $(1) \"$(2)\"'"
endef

.PHONY: sync fetch run run-bg ssh attach guard-%

guard-%:
	$(if $(value $*),,$(error $* is not set. Define it in .env or pass $*=...))

sync: guard-HOST
	rsync -avz --exclude='.git' --exclude='docs' --filter='. .rsync-include' --filter=':- .gitignore' ./ $(HOST):$(REMOTE_DIR)/

fetch: guard-HOST
	rsync -avz --rsync-path="cd $(REMOTE_DIR) && rsync" --filter='. .rsync-fetch' --filter='- *' $(HOST):./ ./

# Sync, run CMD in the foreground, then fetch the results.
run: sync guard-CMD
	$(call remote-exec,$(CMD))
	$(MAKE) fetch

# Sync and start CMD in a tmux session (default name: run). Fetch later with `make fetch`.
SESSION ?= run
run-bg: sync guard-CMD
	$(call remote-exec-tmux,$(SESSION),$(CMD))

ssh: guard-HOST
	ssh $(HOST)

attach: guard-HOST guard-SESSION
	ssh -t $(HOST) "tmux attach -t $(SESSION)"
