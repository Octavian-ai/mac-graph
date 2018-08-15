#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--input-dir input_data/processed/sa_small_100k_balanced \
	--model-dir output/model/sa/$COMMIT \
	--log-level DEBUG \
	--learning-rate 1E-7 \
	--vocab-size 512 \
	--input-layers 3 \
	--input-width 64 \
	--read-activation abs \
	--read-layers 1 \
	--read-dropout 0.0 \
	--read-from-question \
	--disable-memory-cell \
	--memory-transform-layers 1 \
	--memory-forget-activation tanh \
	--control-dropout 0.0 \
	--output-activation tanh \
	--output-layers 1 \
	--output-classes 512 \
	--learning-rate 0.001 \
	--max-gradient-norm 4 \
	--disable-dynamic-decode \
	--enable-read-question-state \
	--max-decode-iterations 1 \
	$@