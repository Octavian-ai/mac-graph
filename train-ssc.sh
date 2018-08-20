#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/ssc/$COMMIT \
	--input-dir input_data/processed/ssc_small_50k \
	--control-heads 2 \
	--control-width 128 \
	--disable-kb-node \
	--disable-dynamic-decode \
	--disable-memory-cell \
	--disable-question-state \
	--input-layers 1 \
	--input-width 64 \
	--learning-rate 0.001 \
	--max-decode-iterations 2 \
	--max-gradient-norm 0.4 \
	--output-activation mi \
	--output-classes 100 \
	--output-layers 1 \
	--read-activation abs \
	--read-dropout 0.0 \
	--read-indicator-rows 1 \
	--read-layers 1 \
	--disable-read-extract \
	--vocab-size 100 \