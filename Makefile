.PHONY: gradient_job dataset sequence_data

#################################################################################
# GLOBALS                                                                       #
#################################################################################
SETTINGS_FILE = "settings_local.json"
GRADIENT_JOB = "attention_numbers__round-number-easy"
GRADIENT_PROJECT = pr1hc80pw
GRADIENT_MACHINE = GPU+
GRADIENT_TEAM = "tensor-league"
GRADIENT_API_KEY := $(shell cat ~/.paperspace/credentials | grep $(GRADIENT_TEAM) -A1 | tail -1 | cut -d= -f2-)

#################################################################################
# COMMANDS                                                                      #
#################################################################################
gradient_job:
	gradient jobs create \
	--name $(GRADIENT_JOB) \
	--projectId $(GRADIENT_PROJECT) \
	--container tensorflow/tensorflow:2.0.0a0-gpu-py3-jupyter \
	--machineType $(GRADIENT_MACHINE) \
	--command "/paperspace/run_script.sh" \
	--ignoreFiles "raw,processed,env" \
	--apiKey $(GRADIENT_API_KEY); rm deep-math.zip

dataset:
	mkdir -p data/raw/
	cd data/raw/; gsutil cp gs://mathematics-dataset/mathematics_dataset-v1.0.tar.gz .
	cd data/raw; tar -xvzf mathematics_dataset-v1.0.tar.gz
	cd data/raw; rm mathematics_dataset-v1.0.tar.gz

sequence_data:
	python src/sequences.py --settings $(SETTINGS_FILE)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')