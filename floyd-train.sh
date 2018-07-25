#!/bin/sh

RUNTIME=$(expr 60 \* 60 \* 1)

floyd run \
	--message "4667b13" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-adjacent:/input \
	--max-runtime $RUNTIME  \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-kb-node \
		--max-decode-iterations 1 \
		--num-input-layers 1 \
		--read-indicator-rows 1 \
		--disable-memory-cell \
		--read-dropout 0.2 \
		--control-dropout 0.2 \
	"
