#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/sa/exp/$COMMIT \
	--input-dir input_data/processed/sa_small_100k \
	--max-decode-iterations 1 \
	--control-heads 2 \
	--disable-memory-cell \
	--disable-read-cell \
	--disable-input-bilstm \
	--input-width 64 \
	--mp-state-width 1 \
	$@