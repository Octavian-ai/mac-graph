#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/ssc/4a/output_exp/selu/$COMMIT \
	--input-dir input_data/processed/ssc_small_1m \
	--filter-output-class 0 \
	--filter-output-class 1 \
	--filter-output-class 2 \
	--filter-output-class 3 \
	--filter-output-class 4 \
	--control-heads 2 \
	--disable-read-cell \
	--disable-input-bilstm \
	--input-width 32 \
	--embed-width 32 \
	--mp-state-width 5 \
	--max-decode-iterations 5 \
	--output-layers 1 \
	--output-activation selu \
	--disable-message-passing-node-transform \
	--memory-transform-layers 1 \
	$@