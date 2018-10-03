#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/sa/exp/dumb/$COMMIT \
	--input-dir input_data/processed/sa_small_10k \
	--max-decode-iterations 1 \
	--control-heads 2 \
	--disable-read-cell \
	--disable-memory-cell \
	--disable-message-passing \
	$@