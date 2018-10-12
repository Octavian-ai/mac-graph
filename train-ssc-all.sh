#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/ssc/all/$COMMIT \
	--input-dir input_data/processed/ssc_small_1m \
	--control-heads 2 \
	--disable-read-cell \
	--disable-input-bilstm \
	--input-width 64 \
	--embed-width 64 \
	--mp-state-width 8 \
	--max-decode-iterations 16 \
	$@