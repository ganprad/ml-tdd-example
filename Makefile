# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Install exact Python and CUDA versions
conda-gpu:
	conda env update --prune -f env_configs/environment.yml

# Install without CUDA
conda-cpu:
	conda env update --prune -f env_configs/environment_cpu.yml

# Compile and install exact pip packages
pip-tools:
	pip install pip-tools
	pip-compile requirements/requirements.in && pip-compile requirements/requirements-dev.in
	pip-sync requirements/requirements.txt requirements/requirements-dev.txt

# Install Rust compiler
rust:
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
