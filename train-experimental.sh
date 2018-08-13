#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--log-level DEBUG \
	--model-dir output/model/sp/$COMMIT \
	--input-dir input_data/processed/sp_small_100k \
	--disable-kb-edge \
	--input-layers 1 \
	--answer-classes 110 \
	--vocab-size 110 \
	--memory-transform-layers 1 \
	--max-decode-iterations 1 \
	--output-activation mi \
	--output-layers 1 \
	--read-activation abs \
	--read-layers 1 \
	--memory-forget-activation mi \
	--control-dropout 0.0 \
	--read-dropout 0.0 \
	--disable-memory-cell \
	--disable-control-cell \
	$@