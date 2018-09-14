#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/sa/exp/$COMMIT \
	--input-dir input_data/processed/sa_sp_small_100k \
	--type-string-prefix StationAdj \
	--disable-dynamic-decode \
	--disable-memory-cell \
	$@