.PHONY: gradient_job

#################################################################################
# COMMANDS                                                                      #
#################################################################################
gradient_job:
	gradient jobs create \
	--name "lstm_test" \
	--container tensorflow/tensorflow:2.0.0a0-gpu-py3-jupyter \
	--machineType GPU+ \
	--command "/paperspace/run_script.sh" \
	--ignoreFiles "data,env"