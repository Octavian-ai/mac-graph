#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m mac-graph.train \
	--input-dir input_data/processed/sa_small_100k_balanced \
	--model-dir output/model/sa/$COMMIT \
	--log-level DEBUG \
	--disable-kb-node \
	--disable-data-stack \
	--disable-read-comparison \
	--memory-transform-layers 13 \
	--memory-width 128 \
	--embed-width 32
