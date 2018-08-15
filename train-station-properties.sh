#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--log-level DEBUG \
	--model-dir output/model/sp/$COMMIT \
	--input-dir input_data/processed/sp_small_100k \
	--output-classes 512 \
	--control-dropout 0.0 \
	--disable-dynamic-decode \
	--disable-kb-edge \
	--disable-memory-cell \
	--enable-read-question-state \
	--input-layers 3 \
	--input-width 64 \
	--learning-rate 0.001 \
	--max-decode-iterations 1 \
	--max-gradient-norm 4 \
	--output-activation tanh \
	--output-layers 1 \
	--read-activation tanh \
	--read-dropout 0.0 \
	--read-layers 1 \
	--vocab-size 512 \
	$@