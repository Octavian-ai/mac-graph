#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/s1a/$COMMIT \
	--input-dir input_data/processed/s1a_small_10k \
	--max-decode-iterations 2 \
	--disable-kb-node \
	--control-heads 2 \
	$@