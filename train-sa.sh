#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/sa/$COMMIT \
	--input-dir input_data/processed/sa_small_100k \
	--max-decode-iterations 1 \
	--control-heads 2 \
	--disable-message-passing \
	--disable-memory-cell \
	--disable-kb-node \
	$@