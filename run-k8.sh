#!/bin/sh

# Google stackdriver already gives us time and hostname
export COLOREDLOGS_LOG_FORMAT='%(message)s'

pipenv run python -m experiment.k8 "$@" \
	--gcs-dir k8/mac-graph \
	--bucket octavian-training \
	--model-dir gs://octavian-training/k8/mac-graph/checkpoint \
	--output-dir gs://octavian-training/k8/mac-graph
	