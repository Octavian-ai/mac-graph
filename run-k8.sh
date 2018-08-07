#!/bin/sh

pipenv run python -m experiment.k8 "$@" \
	--gcs-dir k8/mac-graph \
	--bucket octavian-training \
	--model-dir gs://octavian-training/k8/mac-graph/checkpoint \
	--output-dir gs://octavian-training/k8/mac-graph
	