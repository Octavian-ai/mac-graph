#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--input-dir input_data/processed/sa-sp-small-100k \
	--model-dir output/model/sa_sp/$COMMIT \
	--log-level DEBUG \
	$@
	# --enable-tf-debug \