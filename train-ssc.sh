#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/ssc/$COMMIT \
	--input-dir input_data/processed/ssc_small_1m \
	--max-decode-iterations 1 \
	--control-heads 2 \
	--disable-input-bilstm \
	--input-width 64 \
	$@