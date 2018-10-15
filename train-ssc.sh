#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/ssc/9a/$COMMIT \
	--input-dir input_data/processed/ssc_small_1m \
	--filter-output-class 0 \
	--filter-output-class 1 \
	--filter-output-class 2 \
	--filter-output-class 3 \
	--filter-output-class 4 \
	--filter-output-class 5 \
	--filter-output-class 6 \
	--filter-output-class 7 \
	--filter-output-class 8 \
	--filter-output-class 9 \
	--control-heads 2 \
	--disable-read-cell \
	--disable-input-bilstm \
	--input-width 64 \
	--embed-width 64 \
	--mp-state-width 10 \
	--max-decode-iterations 10 \
	--disable-message-passing-node-transform \
	$@