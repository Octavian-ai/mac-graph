#!/bin/bash

python -m experiment.k8 \
	--master-works \
	--n-drones 1 \
	--n-workers 1 \
	--micro-step 1 \
	--macro-step 1 \
	--input-dir ./input_data/processed/sa-sp-small-100k/ \
	--disable-load \