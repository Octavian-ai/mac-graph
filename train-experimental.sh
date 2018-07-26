#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m mac-graph.train \
	--input-dir input_data/processed/sa_small_100k_balanced \
	--model-dir output/model/sa/$COMMIT \
	--log-level DEBUG \
	--disable-kb-node \
	--max-decode-iterations 1 \
	--disable-dynamic-decode \
	--num-input-layers 1 \
	--disable-memory-cell \
	--memory-width 8 \
	--disable-control-cell \
	--output-activation relu \
	--read-activation relu \