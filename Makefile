.PHONY: gradient_job dataset sequence_data

#################################################################################
# GLOBALS                                                                       #
#################################################################################
SETTINGS_FILE = "settings_local.json"
GRADIENT_JOB = "lstm_attention"
GRADIENT_MACHINE = GPU+
GRADIENT_TEAM = "tensor-league"
GRADIENT_API_KEY := $(shell cat ~/.paperspace/credentials | grep $(GRADIENT_TEAM) -A1 | tail -1 | cut -d= -f2-)

#################################################################################
# COMMANDS                                                                      #
#################################################################################
gradient_job:
	gradient jobs create \
	--name $(GRADIENT_JOB) \
	--container tensorflow/tensorflow:2.0.0a0-gpu-py3-jupyter \
	--machineType $(GRADIENT_MACHINE) \
	--command "/paperspace/run_script.sh" \
	--ignoreFiles "data,env" \
	--apiKey $(GRADIENT_API_KEY)

dataset:
	mkdir -p data/raw/
	cd data/raw/; gsutil cp gs://mathematics-dataset/mathematics_dataset-v1.0.tar.gz .
	cd data/raw; tar -xvzf mathematics_dataset-v1.0.tar.gz
	cd data/raw; rm mathematics_dataset-v1.0.tar.gz

sequence_data:
	python src/data/sequences.py --settings $(SETTINGS_FILE)