#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/sp/$COMMIT \
	--input-dir input_data/processed/sp_small_100k \
	--control-dropout 0 \
	--disable-dynamic-decode \
	--disable-memory-cell \
	--disable-question-state \
	--disable-summary \
	--enable-read-question-state \
	--input-layers 2 \
	--input-width 128 \
	--learning-rate 0.001 \
	--max-decode-iterations 1 \
	--max-gradient-norm 4.0 \
	--output-activation mi \
	--output-classes 512 \
	--output-layers 1 \
	--read-activation tanh \
	--read-dropout 0 \
	--read-from-question \
	--read-indicator-rows 1 \
	--read-layers 1 \
	--vocab-size 512 \
	$@