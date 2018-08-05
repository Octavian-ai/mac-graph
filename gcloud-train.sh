#!/bin/bash

USER="dmack"
JOB_NAME="macgraph_$(date +%Y%m%d_%H%M%S)"
BUCKET_NAME="octavian-training"
REGION="us-central1"
GCS_PATH="${BUCKET_NAME}/${JOB_NAME}"
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
 
gcloud ml-engine jobs submit training "$JOB_NAME" \
    --stream-logs \
    --module-name macgraph.train \
    --package-path macgraph \
    --staging-bucket "gs://${BUCKET_NAME}" \
    --region "$REGION" \
    --runtime-version=1.8 \
    --python-version=3.5 \
    --scale-tier "BASIC_GPU" \
    -- \
    --output-dir "gs://${BUCKET_NAME}/${JOB_NAME}/" \
    --input-dir "gs://octavian-static/download/mac-graph/station-adjacency" \
    --model-dir "gs://${BUCKET_NAME}/${JOB_NAME}/checkpoint" \