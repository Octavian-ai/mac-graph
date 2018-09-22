#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/sa/exp/$COMMIT \
	--input-dir input_data/processed/sa_small_10k \
	--max-decode-iterations 6 \
	--control-heads 2 \
	--read-heads 1 \
	--disable-read-cell \
	$@