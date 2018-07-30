#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m mac-graph.train \
	--input-dir input_data/processed/sa-sp-small-100k \
	--model-dir output/model/sa/$COMMIT \
	--log-level DEBUG \
	--max-decode-iterations 3 \
	--input-layers 3 \
	--answer-classes 86 \
	--vocab-size 86 \
	--enable-tf-debug \