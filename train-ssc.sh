#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/ssc/2a/$COMMIT \
	--input-dir input_data/processed/ssc_small_1m \
	--filter-output-class 0 \
	--filter-output-class 1 \
	--filter-output-class 2 \
	--max-decode-iterations 2 \
	--control-heads 2 \
	--disable-read-cell \
	--disable-input-bilstm \
	--input-width 64 \
	--embed-width 64 \
	--enable-lr-decay \
	$@