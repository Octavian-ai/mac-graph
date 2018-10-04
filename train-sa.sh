#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--input-dir input_data/processed/sa_small_10k\
	--model-dir output/model/sa/$COMMIT \
	--disable-dynamic-decode \
	--disable-kb-node \
	--disable-memory-cell \
	--control-heads 2 \
	--disable-summary \
	--disable-message-passing \
	$@