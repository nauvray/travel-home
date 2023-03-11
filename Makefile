.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y travel-home || :
	@pip install -e .

run_api:
	uvicorn travel-home.api.fast:app --reload

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
	gcloud compute ssh $INSTANCE

show_bucket:
	gcloud config set project $BUCKET_PROJECT
	gsutil ls gs://$BUCKET_NAME/
