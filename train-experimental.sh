#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--input-dir input_data/processed/sa_small_100k_balanced \
	--model-dir output/model/sa/$COMMIT \
	--log-level DEBUG \
	--input-layers 3 \
	--answer-classes 512 \
	--vocab-size 512 \
	--read-activation abs \
	--read-layers 1 \
	--read-dropout 0.0 \
	--memory-transform-layers 1 \
	--memory-forget-activation tanh \
	--output-activation tanh \
	--output-layers 1 \
	--control-dropout 0.0 \
	--input-width 64 \
	--learning-rate 0.001 \
	--max-gradient-norm 4 \
	--disable-dynamic-decode \
	--enable-read-question-state \
	--disable-memory-cell \
	--max-decode-iterations 1 \
	$@