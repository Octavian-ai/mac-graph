#!/bin/sh

pipenv run python -m experiment.k8 "$@" \
	--gcs-dir k8 \
	--bucket octavian-training \
	--model-dir gs://octavian-training/k8/checkpoint