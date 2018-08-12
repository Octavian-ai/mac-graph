#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--input-dir input_data/processed/sa_small_100k_balanced \
	--model-dir output/model/sa/$COMMIT \
	--log-level DEBUG \
	--disable-kb-node \
	--max-decode-iterations 1 \
	--input-layers 1 \
	--disable-memory-cell \
	--read-indicator-rows 1 \
	--disable-control-cell \
	--disable-dynamic-decode \
	--disable-question-state \
	--read-activation abs \
	--read-from-question \
	--read-dropout 0.0 \
	--output-layers 1 \
	--answer-classes 8 \
	--input-width 64 \
	--output-activation mi \
	--learning-rate 0.001 \
	--max-gradient-norm 4.0 \
	--read-layers 1