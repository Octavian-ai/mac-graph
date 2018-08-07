#!/bin/bash

python -m experiment.k8 \
	--master-works \
	--n-drones 3 \
	--n-workers 3 \
	--input-dir ./input_data/processed/sa-sp-small-100k/