#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/ssc/16a/$COMMIT \
	--input-dir input_data/processed/ssc_small_1m \
	--control-heads 2 \
	--disable-read-cell \
	--disable-input-bilstm \
	--input-width 32 \
	--embed-width 32 \
	--mp-state-width 32 \
	--max-decode-iterations 22 \
	--output-layers 1 \
	--output-activation selu \
	--memory-transform-layers 1 \
	$@