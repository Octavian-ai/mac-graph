#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 1)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)


floyd run \
	--message "$COMMIT station adjacency" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-sa-sp:/input \
	--max-runtime $RUNTIME \
	"python -m macgraph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--control-dropout 0 \
		--disable-dynamic-decode \
		--disable-memory-cell \
		--disable-question-state \
		--enable-read-question-state \
		--input-layers 3 \
		--input-width 128 \
		--learning-rate 0.001 \
		--max-decode-iterations 1 \
		--max-gradient-norm 0.4 \
		--output-activation mi \
		--output-classes 110 \
		--output-layers 1 \
		--read-activation tanh_abs \
		--read-dropout 0 \
		--read-heads 1 \
		--read-from-question \
		--read-indicator-rows 1 \
		--read-layers 1 \
		--vocab-size 110 \
		--type-string-prefix StationAdjacent \
	"


floyd run \
	--message "$COMMIT station properties" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-sa-sp:/input \
	--max-runtime $RUNTIME \
	"python -m macgraph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--control-dropout 0 \
		--disable-dynamic-decode \
		--disable-memory-cell \
		--disable-question-state \
		--enable-read-question-state \
		--input-layers 3 \
		--input-width 128 \
		--learning-rate 0.001 \
		--max-decode-iterations 1 \
		--max-gradient-norm 0.4 \
		--output-activation mi \
		--output-classes 110 \
		--output-layers 1 \
		--read-activation tanh_abs \
		--read-dropout 0 \
		--read-heads 1 \
		--read-from-question \
		--read-indicator-rows 1 \
		--read-layers 1 \
		--vocab-size 110 \
		--type-string-prefix StationProperty \
	"
