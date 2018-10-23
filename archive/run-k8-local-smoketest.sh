#!/bin/bash

python -m experiment.k8 \
	--master-works \
	--n-drones 1 \
	--n-workers 1 \
	--micro-step 1000 \
	--macro-step 10 \
	--input-dir ./input_data/processed/sa-sp-small-100k/ \
	--disable-load \