#!/bin/bash

# Google stackdriver already gives us time and hostname
export COLOREDLOGS_LOG_FORMAT='{"severity":%(levelname)s, "textPayload":"%(name)s %(message)s"}'

BUCKET=octavian-training2
OUTPUT_DIR=cluster/mac-graph

pipenv run python -m experiment.k8 "$@" \
	--gcs-dir k8/mac-graph \
	--bucket $BUCKET \
	--model-dir gs://$BUCKET/$OUTPUT_DIR/checkpoint \
	--output-dir gs://$BUCKET/$OUTPUT_DIR \
	