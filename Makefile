.DEFAULT_GOAL := default
#################### SETUP & PACKAGE ACTIONS ###################
env:
	@pyenv python 3.10.6 travel-home-env
	@pyenv local travel-home-env

reinstall_package:
	@pip uninstall -y travel-home || :
	@pip install -e .

run_api:
	uvicorn travel_home.api.fast:app --reload

# reset_local_files:
# 	rm -rf ${ML_DIR}
# 	mkdir -p ~/.lewagon/mlops/data/
# 	mkdir ~/.lewagon/mlops/data/raw
# 	mkdir ~/.lewagon/mlops/data/processed
# 	mkdir ~/.lewagon/mlops/training_outputs
# 	mkdir ~/.lewagon/mlops/training_outputs/metrics
# 	mkdir ~/.lewagon/mlops/training_outputs/models
# 	mkdir ~/.lewagon/mlops/training_outputs/params

run_vm:
	gcloud compute ssh ${INSTANCE}

show_bucket:
	gcloud config set project ${BUCKET_PROJECT}
	gsutil ls gs://${BUCKET_NAME}/
